import os
import json
import glob
import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
import pretrainedmodels
from torchvision import models as _models
from efficientnet_pytorch import model as enet
from core import models as panda_models
import config as panda_config


def load_config(ckpt_dir):
    config_filepath = os.path.join(ckpt_dir, 'config.json')
    if not os.path.exists(config_filepath):
        # walk up one directory to find config file.
        config_filepath = os.path.join(
            os.path.realpath(os.path.join(ckpt_dir, '..')),
            'config.json'
        )
    with open(config_filepath, 'r') as f:
        return json.load(f)


def load_best_state_dict(ckpt_dir, step=None, filename=None, device='cuda'):
    if step is None and filename is None:
        raise ValueError('`step` and `filename` cannot both be None.')
    if step is not None:
        filepaths = glob.glob(os.path.join(ckpt_dir, f'ckpt_{step:04d}_*'))
        if len(filepaths) == 0:
            raise ValueError(f'Found 0 matches for step {step} in {ckpt_dir}')
        elif len(filepaths) > 1:
            warnings.warn(f'Found 3 matches for step {step} in {ckpt_dir}. Loading the first match.')
            filepath = filepaths[0]
        else:
            filepath = filepaths[0]
    else:
        filepath = os.path.join(ckpt_dir, filename)

    print(f'Loading state_dict from {filepath}')
    state_dict = torch.load(filepath, map_location=torch.device(device))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    return state_dict


def load_model(ckpt_dir,
               step=None,
               filename=None,
               device='cuda',
               **kwargs):
    config = load_config(ckpt_dir)
    model_params = config['model']
    model_params['params']['pretrained'] = False
    model_params['params'].update(kwargs)
    state_dict = load_best_state_dict(ckpt_dir, step, filename, device=device)
    model = get_model(**model_params).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model(method, params=None):
    params = params or {}
    if method == 'concat':
        model = panda_models.ConcatModel(**params)
    elif method == 'agg':
        model = panda_models.AggModel(**params)
    elif method == 'chowder':
        model = panda_models.ChowderModel(**params)
    elif method == 'blend':
        model = panda_models.BlendModel(**params)
    else:
        raise ValueError(f'Unrecognized `method` {method}.')

    return model

def get_backbone(backbone, pretrained=True):
    if backbone in ['resnext50_32x4d_ssl', 'resnet18_ssl', 'resnet50_ssl', 'resnext101_32x4d_ssl']:
        if pretrained:
            model = torch.hub.load(panda_config.ARCH_TO_PRETRAINED[backbone], backbone)
        else:
            model = getattr(_models, backbone.split('_ssl')[0])(pretrained=pretrained)
        encoder = nn.Sequential(*list(model.children())[:-2])
        in_features = model.fc.in_features
    elif backbone in ['resnet18', 'resnet34', 'resnet50']:
        pretrained = 'imagenet' if pretrained else None
        model = getattr(_models, backbone)(pretrained=pretrained)
        in_features = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-2])
    elif backbone in ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'se_resnet101', 'se_resnet152']:
        pretrained = 'imagenet' if pretrained else None
        model = getattr(pretrainedmodels, backbone)(pretrained=pretrained)
        encoder = nn.Sequential(*list(model.children())[:-2])
        in_features = model.last_linear.in_features
    elif backbone.startswith('efficientnet'):
        encoder = enet.EfficientNet.from_name(backbone)
        if pretrained:
            encoder.load_state_dict(torch.load(panda_config.ARCH_TO_PRETRAINED[backbone]))
        in_features = encoder._fc.in_features
        encoder._fc = nn.Identity()
    elif backbone == 'inception_resnet_v2':
        pretrained = 'imagenet' if pretrained else None
        encoder = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
        in_features = encoder.last_linear.in_features
        encoder.last_linear = nn.Identity()
    elif backbone == 'inception_v4':
        pretrained = 'imagenet' if pretrained else None
        encoder = pretrainedmodels.inceptionv4(pretrained=pretrained)
        in_features = encoder.last_linear.in_features
        encoder.last_linear = nn.Identity()
    else:
        raise ValueError(f'Unrecognized backbone {backbone}')

    return encoder, in_features


def get_emb_model(model):
    emb_model = nn.Sequential(*list(model.children())[:-1])
    for param in emb_model.parameters():
        param.requires_grad = False
    return emb_model


def aggregate(x, batch_size, num_tiles):
    b, c, h, w = x.shape
    x = x.view(batch_size, num_tiles, c, h, w)
    x = x.permute(0,2,1,3,4).contiguous().view(batch_size, c, num_tiles * h, w)
    return x

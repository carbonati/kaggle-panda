import os
import torch
import torch.nn as nn
import albumentations
from torch.utils import data
from core import losses
from core import layers
from data.samplers import BatchStratifiedSampler
from evaluation import postprocess


ROOT_PATH = '/root/workspace/kaggle-panda'


SCHEDULER_MAP = {
    'one_cycle': torch.optim.lr_scheduler.OneCycleLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'step': torch.optim.lr_scheduler.StepLR
}

OPTIMIZER_MAP = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

AUGMENTATION_MAP = {
    'transpose': albumentations.Transpose,
    'vertical': albumentations.VerticalFlip,
    'horizontal': albumentations.HorizontalFlip,
    'normalize': albumentations.Normalize,
    'brightness': albumentations.RandomBrightness,
    'contrast': albumentations.RandomContrast,
    'shift': albumentations.ShiftScaleRotate
}

ARCH_TO_PRETRAINED = {
    'efficientnet-b0': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b0-08094119.pth'),
    'efficientnet-b1': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b1-dbc7070a.pth'),
    'efficientnet-b3': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b3-c8376fa2.pth'),
    'efficientnet-b4': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b4-e116e8b3.pth'),
    'efficientnet-b5': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b5-586e6cc6.pth'),
    'efficientnet-b7': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b7-dcc49843.pth'),
    'resnext50_32x4d_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnext101_32x4d_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnet18_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnet50_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
}

CRITERION_MAP = {
    'xent': nn.CrossEntropyLoss,
    'lbl_smth': losses.LabelSmoothingLoss,
    'mse': nn.MSELoss,
    'bce': nn.BCEWithLogitsLoss,
    'l1': nn.L1Loss,
    'l1_smth': nn.SmoothL1Loss,
    'cauchy': losses.CauchyLoss,
    'mse_clip': losses.ClippedMSELoss
}

CLF_CRITERION = ['xent', 'lbl_smth']

SAMPLER_MAP = {
    'random': data.RandomSampler,
    'sequential': data.SequentialSampler,
    'weighted_random': data.WeightedRandomSampler,
    'batch':  BatchStratifiedSampler
}

POOLING_MAP = {
    'concat': layers.AdaptiveConcatPool2d,
    'gem': layers.GeM
}

POSTPROCESSOR_MAP = {
    'optimized_rounder': postprocess.OptimizedRounder
}

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'mish': layers.Mish
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import model as enet
from torchvision import models as _models
from core.layers import AdaptiveConcatPool2d
from utils import model_utils
import config as panda_config


class ConcatModel(nn.Module):
    """Concat tile model."""
    def __init__(self,
                 backbone,
                 num_classes=6,
                 output_net_params=None,
                 pool_params=None,
                 pretrained=True):
        super().__init__()
        self._backbone = backbone
        self._num_classes = num_classes
        self._pool_params = pool_params
        self._output_net_params = output_net_params
        if self._pool_params is None:
            self._pool_method = 'concat'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
            self._pool_params['params'] = self._pool_params.get('params', {})
        self._pretrained = pretrained

        self.encoder, in_features = model_utils.get_backbone(self._backbone,
                                                             self._pretrained)
        self.pool_layer = panda_config.POOLING_MAP[self._pool_method](**self._pool_params['params'])

        if not backbone.startswith('efficientnet'):
            in_features = in_features * 2 if self._pool_method == 'concat' else in_features

        if self._output_net_params is not None:
            hidden_dim = self._output_net_params.get('hidden_dim', 512)
            modules = [
                nn.Linear(in_features, hidden_dim, bias=False),
                nn.ReLU()
            ]

            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm1d(hidden_dim))
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.Linear(hidden_dim, self._num_classes, bias=False))

            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Linear(in_features,
                                        self._num_classes,
                                        bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool_layer(x)
        x = self.output_net(x)
        return x


class AggModel(nn.Module):
    """Aggregate tile model."""
    def __init__(self,
                 backbone,
                 num_classes=6,
                 output_net_params=None,
                 pool_params=None,
                 pretrained=True):
        super().__init__()
        self._backbone = backbone
        self._num_classes = num_classes
        self._pool_params = pool_params
        self._output_net_params = output_net_params
        if self._pool_params is None:
            self._pool_method = 'concat'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
            self._pool_params['params'] = self._pool_params.get('params', {})
        self.pool_layer = panda_config.POOLING_MAP[self._pool_method](**self._pool_params['params'])
        self._pretrained = pretrained

        self.encoder, in_features = model_utils.get_backbone(self._backbone,
                                                             self._pretrained)
        in_features = in_features * 2 if self._pool_method == 'concat' else in_features

        if self._output_net_params is not None:
            hidden_dim = self._output_net_params.get('hidden_dim', 512)
            modules = [
                nn.Linear(in_features, hidden_dim, bias=False),
                nn.ReLU()
            ]

            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm1d(hidden_dim))
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.Linear(hidden_dim, self._num_classes, bias=False))

            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Linear(in_features,
                                        self._num_classes,
                                        bias=False)

    def forward(self, x):
        batch_size, num_tiles, h, w, c = x.shape
        x = x.view(-1, c, h, w)
        x = self.encoder(x)
        x = model_utils.aggregate(x, batch_size, num_tiles)
        x = self.pool_layer(x)
        x = self.output_net(x)
        return x


class BlendModel(nn.Module):
    """Blender."""
    def __init__(self, models, num_classes=1, hidden_dim=256, **kwargs):
        super().__init__()
        self.models = models
        self._num_models = len(self.models)
        self.branches = []
        self._in_features = 0
        for model in self.models:
            self.branches.append(model_utils.get_emb_model(model))
            self._in_features += list(model.children())[-1].in_features
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self._in_features, self._hidden_dim, bias=False)
        self.fc2 = nn.Linear(self._hidden_dim, self._num_classes, bias=False)

    def forward(self, x):
        x_output = []
        for i in range(self._num_models):
            x_output.append(self.branches[i](x[i]))
        x = F.relu(self.fc1(torch.cat(x_output, axis=1)))
        return self.fc2(x)


class ChowderModel(nn.Module):
    def __init__(self,
                 backbone,
                 num_classes=6,
                 J=1,
                 R=4,
                 output_net_params=None,
                 pool_params=None,
                 pretrained=True):
        super().__init__()
        self._backbone = backbone
        self._num_classes = num_classes
        self._pool_params = pool_params
        self._output_net_params = output_net_params
        self.J = J
        self.R = R
        if self._pool_params is None:
            self._pool_method = 'concat'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
        self.pool_layer = panda_config.POOLING_MAP[self._pool_method](**self._pool_params.get('params', {}))
        self._pretrained = pretrained

        self.encoder, in_features = model_utils.get_backbone(self._backbone,
                                                             self._pretrained)
        in_features = in_features * 2 if self._pool_method == 'concat' else in_features
        self._in_features = in_features
        self.emb_layer = nn.Conv1d(1, self.J, kernel_size=in_features)

        if self._output_net_params is not None:
            hidden_dims = self._output_net_params.get('hidden_dims', [self.R * 2])
            if not isinstance(hidden_dims, list):
                hidden_dims = [hidden_dims]

            modules = [
                nn.Linear(self.R * 2, hidden_dims[0], bias=False),
                nn.ReLU(),
                nn.Dropout(self._output_net_params.get('dropout', .5))
            ]
            for i in range(1, len(hidden_dims)):
                modules.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=False),
                    nn.ReLU(),
                    nn.Dropout(self._output_net_params.get('dropout', .5))
                ])

            modules.append(nn.Linear(hidden_dims[-1], self._num_classes, bias=False))

            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Linear(self.R * 2,
                                        self._num_classes,
                                        bias=False)

    def forward(self, x):
        batch_size = len(x)
        num_tiles, h, w, c = x[0].shape
        x_view = x.view(-1, c, h, w)
        x_enc = self.encoder(x_view)
        x_pool = self.pool_layer(x_enc)

        x_emb = self.emb_layer(x_pool.view(-1, 1, self._in_features)).view(batch_size, num_tiles)
        x_sort, _ = torch.sort(x_emb, dim=1, descending=True)
        x_minmax = torch.cat((x_sort[:, :(self.R)], x_sort[:, -self.R:]), axis=1)
        x = self.output_net(x_minmax)
        return x

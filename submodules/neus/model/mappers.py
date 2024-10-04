from torch import nn
import torch
from torch.nn import functional as F
import gin
from typing import Callable, Any
from misc.math import *
#import tinycudann as tcnn


def tcnn_linear(in_dim, out_dim):
    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1
    }
    return tcnn.Network(in_dim, out_dim, network_config)


class TCNNLinear(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super(TCNNLinear, self).__init__()
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4
        }
        network_config.update(kwargs)
        self.mlp = tcnn.Network(in_dim, out_dim, network_config)

    def forward(self, x):
        init_shape = list(x.shape[:-1]) + [-1]
        x = x.view(-1, x.shape[-1])
        y = self.mlp(x)
        return y.view(*init_shape).float()


@gin.configurable
class MLP(nn.Module):

    def __init__(self, input_dim=3, cond_dim=3, out_dim=1, cond_out_dim=3,
        use_cond: bool = True,       # use condition net for view directions
        net_depth: int = 8,          # The depth of the first part of MLP.
        net_width: int = 256,        # The width of the first part of MLP.
        skip_layer: int = 4,         # Add a skip connection to the output of every N layers.
        cond_net_depth: int = 2,     # The depth of the second part of MLP.
        cond_net_width: int = 128,   # The width of the second part of MLP.
        net_activation: Callable[..., Any] = F.relu,     # The activation function.
        out_activation: Callable[..., Any] = None,       # The activation function.
        cond_out_activation: Callable[..., Any] = None,  # The activation function.
    ):
        super(MLP, self).__init__()
        W = net_width
        M = cond_net_width
        self.skip_layer = skip_layer
        self.net_activation = net_activation
        self.out_activation = out_activation
        self.cond_out_activation = cond_out_activation
        self.use_cond = use_cond

        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, W)] +
            [nn.Linear(W + input_dim, W) if i % self.skip_layer == 0
             else nn.Linear(W, W)
             for i in range(1, net_depth)])
        self.out_linear = nn.Linear(W, out_dim)
        self.feature_linear = nn.Linear(W, W)

        if self.use_cond and cond_dim > 0:
            assert cond_out_dim > 0
            indim = lambda i: M if i > 0 else W + cond_dim
            self.cond_linears = nn.ModuleList(
                [nn.Linear(indim(i), M) for i in range(cond_net_depth - 1)] +
                [nn.Linear(indim(cond_net_depth - 1), cond_out_dim)])
        elif cond_out_dim > 0:
            self.cond_out_linear = nn.Linear(W, cond_out_dim)

    def forward(self, x, c=None):
        h = x
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.net_activation(h)
            if (i + 1) % self.skip_layer == 0 and i < len(self.linears) - 1:
                h = torch.cat([x, h], -1)

        out = self.out_linear(h)
        if self.out_activation is not None:
            out = self.out_activation(out)

        if self.use_cond:
            assert c is not None
            h = self.feature_linear(h)
            h = torch.cat([h, c], -1)

            for i, l in enumerate(self.cond_linears):
                h = self.cond_linears[i](h)
                h = self.net_activation(h)

            cond_out = h
        elif hasattr(self, 'cond_out_linear'):
            cond_out = self.cond_out_linear(h)
        else:
            return out

        if self.cond_out_activation is not None:
            cond_out = self.cond_out_activation(cond_out)

        return out, cond_out


def simple_mlp(in_dim, out_dim, layers=2, features=64):
    return MLP(in_dim, 0, out_dim, 0, False, layers, features)


@gin.configurable
class SH(nn.Module):

    def __init__(self, sh_deg=2, out_dim=3):
        super(SH, self).__init__()
        self.sh_deg = sh_deg
        self.out_dim = out_dim
        self.in_dim = (self.sh_deg + 1) ** 2 * out_dim

    def forward(self, sh, dirs):
        init_shape = sh.shape
        assert init_shape[:-1] == dirs.shape[:-1]
        dirs = dirs.reshape(-1, 3)
        sh_chan = (self.sh_deg + 1) ** 2
        sh = sh.view(-1, sh_chan, self.out_dim).permute(1, 0, 2)
        rgb = eval_sh(self.sh_deg, sh, dirs)

        rgb = rgb.view(*init_shape[:-1], self.out_dim)
        return rgb

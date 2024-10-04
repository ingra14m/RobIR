import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.embedders import *
from model.mappers import *
from misc.defs import *
from misc.utils import prox_gradients, torch_tree_map


def get_embedder(multires, input_dims=3, Embedder=PE, windowed=False):
    embedder_obj = Embedder(input_dims=input_dims, num_freq=multires)
    def embed(x, eo=embedder_obj): return eo.embed(x) if not windowed else eo.windowed_embed(x)
    return embed, embedder_obj.out_dim


def isotropic_cov(mean, var):
    init_shape = list(mean.shape[:-1])
    cov = torch.eye(3, device=mean.device) * var
    cov = cov[None, :, :].expand(mean.view(-1, 3).shape[0], -1, -1)
    return cov.reshape(init_shape + [3, 3])


def ipe_embedder(multires, var=0.005):
    ipe = IPE(max_deg=multires)
    embed_fn = lambda x: ipe(x, isotropic_cov(x, var))
    return embed_fn, ipe.feature_dim()


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=10,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            # embed_fn, input_ch = ipe_embedder(multires, var=0.0001)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        not_eval = torch.is_grad_enabled()
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=not_eval,
                retain_graph=not_eval,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)


@gin.configurable
class HashSDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 multires=12,
                 dx_curve=0.1):
        super(HashSDFNetwork, self).__init__()

        self.hash = Hash(n_levels=multires, in_dim=d_in)
        input_ch = self.hash.feature_dim()
        embed_fn = self.hash.windowed_embed
        self.embed_fn_fine = embed_fn

        self.linear = TCNNLinear(input_ch, d_out)
        self.dx = Curve(dx_curve)

    def forward(self, inputs):
        x = self.embed_fn_fine(inputs)
        x = self.linear(x)
        return torch.cat([x[:, :1], x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    # def gradient(self, x):
    #     not_eval = torch.is_grad_enabled()
    #     with torch.enable_grad():
    #         x.requires_grad_(True)
    #         y = self.sdf(x)
    #         d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #         gradients = torch.autograd.grad(
    #             outputs=y,
    #             inputs=x,
    #             grad_outputs=d_output,
    #             create_graph=not_eval,
    #             retain_graph=not_eval,
    #             only_inputs=True)[0]
    #     return gradients.unsqueeze(1)

    def gradient(self, x, dx=None):
        if dx is None:
            dx = self.dx()
        return prox_gradients(self.sdf, x, dx)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=4,
                 squeeze_out=True):
        super().__init__()

        if "raw" in mode:
            squeeze_out = False

        self.mode = mode
        self.squeeze_out = squeeze_out

        if "no" in mode:
            d_in = d_in - 3

        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if "tcnn" in mode:
            self.linear = TCNNLinear(dims[0], d_out)
        else:
            self.num_layers = len(dims)

            for l in range(0, self.num_layers - 1):
                out_dim = dims[l + 1]
                lin = nn.Linear(dims[l], out_dim)

                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(l), lin)

            self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if 'no_view_dir' in self.mode:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif 'no_normal' in self.mode:
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:       # 'idr'
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        x = rendering_input

        if "tcnn" in self.mode:
            x = self.linear(x)
        else:
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))

                x = lin(x)

                if l < self.num_layers - 2:
                    x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=10,
                 multires_view=4,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


def auto_flatten(f):
    from functools import wraps

    @wraps(f)
    def wrapper(self, x, *args, **kwargs):
        init_shape = list(x.shape[:-1]) + [-1]
        x = x.view(-1, 3)
        return f(self, x, *args, **kwargs).view(init_shape)

    return wrapper


def auto_flatten2(f):
    from functools import wraps

    @wraps(f)
    def wrapper(self, x, dirs, *args, **kwargs):
        init_shape = list(x.shape[:-1]) + [-1]
        if len(init_shape) > len(dirs.shape):
            dirs = dirs[:, None, :].expand(x.shape)

        x = x.view(-1, 3)
        dirs = dirs.reshape(-1, 3)
        rgb, a = f(self, x, dirs, *args, **kwargs)
        return rgb.view(init_shape), a.view(init_shape)

    return wrapper


@gin.register
@gin.configurable
class NeuSModel(nn.Module, ISDF):

    def __init__(self, mode='idr', hashing=False, outside=True):
        """
        mode: { [no_view_dir/no_normal/-] + [tcnn/-] + [raw/-] (+ idr) } / sh / seg
        """
        super(NeuSModel, self).__init__()

        if mode == 'sh':
            self.sh = SH()

            def wrap_color_net(x, gradients, dirs, feature_vector): return self.sh(feature_vector, dirs)

            self.color_network = wrap_color_net
            d_feat = self.sh.in_dim
        elif mode == 'seg':
            d_feat = 128
            self.seg_net = TCNNLinear(2 * d_feat, 3)

            def wrap_color_net(x, gradients, dirs, feature_vector):
                # 512 x 128 x 3, assume sample num is 128
                feature_vector = feature_vector.view(-1, 128, feature_vector.size(-1))
                feat_pairs = torch.cat([feature_vector[..., :-1, :], feature_vector[..., 1:, :]], -1)
                rgb = self.seg_net(feat_pairs)
                rgb = torch.cat([rgb, rgb[:, -1:]], dim=1)
                return rgb.view(-1, 3)

            self.color_network = wrap_color_net
        else:
            d_feat = 256
            self.color_network = RenderingNetwork(d_feature=d_feat, mode=mode, d_in=9, d_out=3, d_hidden=256, n_layers=4)

        if outside:
            self.nerf_outside = NeRF(d_in=4)
        if hashing:
            self.sdf_network = HashSDFNetwork(d_in=3, d_out=d_feat + 1)
        else:
            self.sdf_network = SDFNetwork(d_in=3, d_out=d_feat + 1, d_hidden=256, n_layers=8)
        self.deviation_network = SingleVarianceNetwork(init_val=0.3)

    def sdf(self, x):
        return self.sdf_network.sdf(x)

    def sdf_and_feat(self, x):
        out = self.sdf_network(x)
        return out[..., :1], out[..., 1:]

    def color(self, x, gradients, dirs, feature_vector):
        return self.color_network(x, gradients, dirs, feature_vector)

    @auto_flatten
    def grad(self, x):
        return self.sdf_network.gradient(x)

    def dev(self, x):
        return self.deviation_network(x)

    def radius(self):
        return 2.0

    def background(self, x, dirs):
        return self.nerf_outside(x, dirs)

    @auto_flatten2
    def forward(self, pnts, dirs, **kwargs):
        a, feat = self.sdf_and_feat(pnts)
        return self.color(pnts, self.grad(pnts), dirs, feat), a


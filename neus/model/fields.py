from model.embedders import *
from model.mappers import *
from model.vanilla import OriginNeRF
from misc.utils import prox_gradients
from misc.defs import *
from model.neus_fields import NeRF, SDFNetwork, RenderingNetwork, SingleVarianceNetwork


@gin.register
@gin.configurable
class VNeRF(nn.Module):

    def __init__(self, encoder=PE, use_sh=False, naive_version=True, alpha_dim=1, sdf_init_var=0.3):
        super(VNeRF, self).__init__()
        self.enc = encoder()
        self.use_sh = use_sh
        if use_sh:
            self.sh = SH()
            rgb_dim = self.sh.in_dim
            cond_dim = 0
        else:
            self.dir_enc = PE(3, 4)
            rgb_dim = 3
            cond_dim = self.dir_enc.feature_dim()
        if naive_version:
            if self.use_sh:
                # self.mlp = OriginNeRF(input_ch=self.enc.feature_dim(), input_ch_views=cond_dim, D=2,
                #                       output_ch=rgb_dim, use_viewdirs=False)
                self.rho_net = tcnn_linear(self.enc.feature_dim(), alpha_dim)
                self.rgb_net = tcnn_linear(self.enc.feature_dim(), rgb_dim)
                def mlp(x, y):
                    init_shape = list(x.shape[:-1]) + [-1]
                    x = x.view(-1, x.shape[-1])
                    a, rgb = self.rho_net(x), self.rgb_net(x)
                    return a.view(*init_shape).float(), rgb.view(*init_shape).float()
                self.mlp = mlp
            else:
                self.mlp = OriginNeRF(input_ch=self.enc.feature_dim(), input_ch_views=cond_dim)
        else:
            self.mlp = MLP(self.enc.feature_dim(), cond_dim, 1, rgb_dim, not use_sh)

        # SDF
        self.variance = nn.Parameter(torch.tensor(sdf_init_var))

    def density(self, pnts):
        x = self.enc(pnts)
        a = self.mlp.density(x)
        return a

    def forward(self, pnts, dirs, **kwargs):
        init_shape = list(pnts.shape[:-1]) + [-1]
        x = self.enc(pnts)

        if self.use_sh:
            if len(init_shape) == 3 and len(dirs.shape) == 2:
                dirs = dirs.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, None)
            rgb = self.sh(rgb, dirs)
        else:
            c = self.dir_enc(dirs)
            if len(init_shape) == 3 and len(c.shape) == 2:
                c = c.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, c)
        return rgb, a


@gin.register
@gin.configurable
class HashNeRF(nn.Module):

    def __init__(self, use_sh=False, use_tcnn=True, alpha_dim=1, rgb_dim=3, ends_to_seg=False, sdf_init_var=0.3):
        super(HashNeRF, self).__init__()
        self.enc = Hash()

        if ends_to_seg:
            use_sh = False

        self.use_sh = use_sh
        self.use_tcnn = use_tcnn
        self.ends_to_seg = ends_to_seg

        if use_sh:
            self.sh = SH()
            rgb_dim = self.sh.in_dim
            cond_dim = 0
        else:
            self.dir_enc = PE(3, 4)
            cond_dim = self.dir_enc.feature_dim()
        if use_tcnn:
            self.rho_net = tcnn_linear(self.enc.feature_dim(), alpha_dim)
            self.rgb_net = tcnn_linear(self.enc.feature_dim(), rgb_dim)

            def mlp(x, y):
                init_shape = list(x.shape[:-1]) + [-1]
                x = x.view(-1, x.shape[-1])
                a, rgb = self.rho_net(x), self.rgb_net(x)
                return a.view(*init_shape).float(), rgb.view(*init_shape).float()

            self.mlp = mlp
        else:
            self.mlp = OriginNeRF(input_ch=self.enc.feature_dim(), input_ch_views=cond_dim)

        if ends_to_seg:
            self.seg_net = tcnn_linear(2 * rgb_dim, 3)

            def seg_mlp(x):
                init_shape = list(x.shape[:-1]) + [-1]
                x = x.view(-1, x.shape[-1])
                seg = self.seg_net(x)
                return seg.view(*init_shape).float()

            self.seg_mlp = seg_mlp

        # SDF
        self.variance = nn.Parameter(torch.tensor(sdf_init_var))

    def gradients(self, pnts):
        dx = 0.0001
        return prox_gradients(self.get_density, pnts, dx)

    def density(self, x):
        x = self.enc(x)
        if self.use_tcnn:
            init_shape = list(x.shape[:-1]) + [-1]
            x = x.view(-1, x.shape[-1])
            a = self.rho_net(x)
            a = a.view(*init_shape).float()
        else:
            a = self.mlp.density(x)
        return a

    def forward(self, pnts, dirs, **kwargs):
        init_shape = list(pnts.shape[:-1]) + [-1]
        x = self.enc(pnts)

        if self.use_tcnn:
            if len(init_shape) == 3 and len(dirs.shape) == 2:
                dirs = dirs.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, None)
        else:
            c = self.dir_enc(dirs)
            if len(init_shape) == 3 and len(c.shape) == 2:
                c = c.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, c)
        if self.use_sh:
            rgb = self.sh(rgb, dirs)
        if self.ends_to_seg:
            rgb_2 = torch.cat([rgb[..., :-1, :], rgb[..., 1:, :]], -1)
            rgb = self.seg_mlp(rgb_2)

        return rgb, a


@gin.register
@gin.configurable
class MipNeRF(nn.Module, IMip):

    def __init__(self, naive_version=True, contract=False):
        super(MipNeRF, self).__init__()
        self.enc = IPE()
        self.dir_enc = PE(3, 4)
        if not contract:
            self.contract = lambda x, y: (x, y)
        if naive_version:
            self.mlp = OriginNeRF(input_ch=self.enc.feature_dim(), input_ch_views=self.dir_enc.feature_dim())
        else:
            self.mlp = MLP(self.enc.feature_dim(), self.dir_enc.feature_dim())

    def contract_J(self, x):

        def joint(a, b):
            return a.unsqueeze(-2) * b.unsqueeze(-1)

        nx = torch.norm(x, dim=-1, keepdim=True)
        nx_ = nx.unsqueeze(-1)
        I = torch.eye(x.size(-1), device=x.device)
        dv = I / nx_ - joint(x, x) / nx_ ** 3
        du = x / nx ** 3
        u = 2 - 1 / nx
        v = x / nx
        J = joint(du, v) + u.unsqueeze(-1) * dv
        return torch.where(nx_ <= 1, I.expand(J.shape), J)

    def contract(self, x, std):
        torch.set_default_dtype(torch.float64)
        x, std = x.double(), std.double()
        nx = torch.norm(x, dim=-1, keepdim=True)
        assert (nx >= -1e-8).all()
        f = torch.where(nx <= 1, x, (2 - 1 / nx) * (x / nx))
        J = self.contract_J(x)
        new_x, new_std = f, J @ std @ J.transpose(-2, -1)
        torch.set_default_dtype(torch.float32)
        return new_x.float(), new_std.float()

    def forward(self, means, covs, dirs, **kwargs):
        init_shape = list(means.shape[:-1]) + [-1]
        # means = means + 10
        means, covs = self.contract(means, covs)

        x = self.enc(means, covs)
        c = self.dir_enc(dirs)
        if len(init_shape) == 3 and len(c.shape) == 2:
            c = c.unsqueeze(1).expand(init_shape)
        a, rgb = self.mlp(x, c)
        return rgb, a

    def color_and_density_of_gaussian(self, means, covs, dirs):
        return self(means, covs, dirs)


# @gin.configurable
# class HashSDF(nn.Module):
#
#     def __init__(self, use_sh=False, alpha_dim=1, rgb_dim=3, grad_dx=0.1):
#         super(HashSDF, self).__init__()
#         self.enc = Hash()
#         self.use_sh = use_sh
#
#         if use_sh:
#             self.sh = SH()
#             rgb_dim = self.sh.in_dim
#         else:
#             self.dir_enc = PE(3, 4)
#
#         self.rho_net = tcnn_linear(self.enc.feature_dim(), alpha_dim)
#         self.rgb_net = tcnn_linear(self.enc.feature_dim(), rgb_dim)
#
#         def mlp(x):
#             init_shape = list(x.shape[:-1]) + [-1]
#             x = x.view(-1, x.shape[-1])
#             a, rgb = self.rho_net(x), self.rgb_net(x)
#             return a.view(*init_shape).float(), rgb.view(*init_shape).float()
#
#         self.mlp = mlp
#
#         self.dx_curve = Curve(grad_dx)
#
#     def gradients(self, pnts):
#         dx = self.dx_curve()
#         return prox_gradients(self.density, pnts, dx)
#
#     def density(self, x):
#         x = self.enc.windowed_embed(x)
#         init_shape = list(x.shape[:-1]) + [-1]
#         x = x.view(-1, x.shape[-1])
#         a = self.rho_net(x)
#         a = a.view(*init_shape).float()
#         return a
#
#     def density_and_sh(self, pnts):
#         x = self.enc(pnts)
#         a, rgb = self.mlp(x)
#         return a, rgb
#
#     def color(self, pnts, dirs, sh):
#         init_shape = list(pnts.shape[:-1]) + [-1]
#         if len(init_shape) == 3 and len(dirs.shape) == 2:
#             dirs = dirs.unsqueeze(1).expand(init_shape)
#         if self.use_sh:
#             return self.sh(sh, dirs)
#         else:
#             return sh
#
#     def forward(self, pnts, dirs, **kwargs):
#         a, sh = self.density_and_sh(pnts)
#         return self.color(pnts, dirs, sh), a


@gin.configurable
class HashSDF(nn.Module):
    def __init__(self,
                 alpha_dim=1,
                 multires=12,
                 n_features=4,
                 grad_dx=0.1):
        super(HashSDF, self).__init__()

        self.hash = Hash(n_levels=multires, n_features=n_features, in_dim=3)
        input_ch = self.hash.feature_dim()
        embed_fn = self.hash.windowed_embed
        self.embed_fn_fine = embed_fn

        self.sh = SH()
        self.linear = TCNNLinear(input_ch, self.sh.in_dim + 1)
        self.dx = Curve(grad_dx)

    def forward(self, inputs):
        x = self.embed_fn_fine(inputs)
        x = self.linear(x)
        return torch.cat([x[:, :1], x[:, 1:]], dim=-1)

    def density(self, x):
        return self.forward(x)[:, :1]

    def density_and_sh(self, pnts):
        x = self.hash(pnts)
        argb = self.linear(x)
        return argb[..., :1], argb[..., 1:]

    def gradients(self, x, dx=None):
        if dx is None:
            dx = self.dx()
        return prox_gradients(self.density, x, dx)

    def color(self, pnts, dirs, sh):
        init_shape = list(pnts.shape[:-1]) + [-1]
        if len(init_shape) == 3 and len(dirs.shape) == 2:
            dirs = dirs.unsqueeze(1).expand(init_shape)
        return self.sh(sh, dirs)


@gin.register
@gin.configurable
class HashNeuS(nn.Module, ISDF):

    def __init__(self, sdf_dim=1, outside=True):
        super(HashNeuS, self).__init__()
        if outside:
            self.nerf_outside = NeRF(d_in=4)
        self.hash_sdf = HashSDF(alpha_dim=sdf_dim)
        self.deviation_network = SingleVarianceNetwork(init_val=0.3)

    def forward(self, *args, **kwargs):
        return self.hash_sdf.forward(*args, **kwargs)

    def sdf(self, x):
        return self.hash_sdf.density(x)

    def sdf_and_feat(self, x):
        return self.hash_sdf.density_and_sh(x)

    def color(self, x, gradients, dirs, feature_vector):
        return self.hash_sdf.color(x, dirs, feature_vector)

    def grad(self, x):
        return self.hash_sdf.gradients(x)

    def dev(self, x):
        return self.deviation_network(x)

    def radius(self):
        return 4.0

    def background(self, x, dirs):
        return self.nerf_outside(x, dirs)


@gin.register
@gin.configurable
class MySDF(nn.Module):

    def __init__(self, in_dim=3, use_sh=False):
        super(MySDF, self).__init__()
        self.enc = PE(input_dims=in_dim)
        self.use_sh = use_sh
        self.dir_enc = PE(3, 4)
        cond_dim = self.dir_enc.feature_dim()
        self.mlp = OriginNeRF(input_ch=self.enc.feature_dim(), input_ch_views=cond_dim)

    def density(self, pnts):
        x = self.enc(pnts)
        a = self.mlp.density(x)
        return a

    def forward(self, pnts, dirs, **kwargs):
        init_shape = list(pnts.shape[:-1]) + [-1]
        x = self.enc(pnts)

        if self.use_sh:
            if len(init_shape) == 3 and len(dirs.shape) == 2:
                dirs = dirs.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, None)
            rgb = self.sh(rgb, dirs)
        else:
            c = self.dir_enc(dirs)
            if len(init_shape) == 3 and len(c.shape) == 2:
                c = c.unsqueeze(1).expand(init_shape)
            a, rgb = self.mlp(x, c)
        return rgb, a


@gin.register
@gin.configurable
class MyNeuS(nn.Module, ISDF):

    def __init__(self):
        super(MyNeuS, self).__init__()
        self.nerf_outside = MySDF(in_dim=4)
        self.vanilla_sdf = MySDF()
        self.deviation_network = SingleVarianceNetwork(init_val=0.3)

    def sdf(self, x):
        x = self.vanilla_sdf.enc(x)
        return self.vanilla_sdf.mlp.density(x)

    def sdf_and_feat(self, x):
        x = self.vanilla_sdf.enc(x)
        return self.vanilla_sdf.mlp.density_and_feature(x)

    def color(self, x, gradients, dirs, feature_vector):
        dirs = self.vanilla_sdf.dir_enc(dirs)
        return self.vanilla_sdf.mlp.color(x, dirs, feature_vector)

    def grad(self, x):
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

    def dev(self, x):
        return self.deviation_network(x)

    def radius(self):
        return 2.0

    def background(self, x, dirs):
        rgb, a = self.nerf_outside(x, dirs)
        return a, rgb


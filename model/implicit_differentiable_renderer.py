import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sg_envmap_material import EnvmapMaterialNetwork, fibonacci_sphere
from model.sg_render import render_envmap, render_with_all_sg

from model.neus_model import ImplicitNetworkMy
from model.color_correction import GammaCorrect
from model.octree_tracing import OctreeTracing, OctreeVisModel

TINY_NUMBER = 1e-6


class ImplicitNetwork(nn.Module):
    def __init__(self,
                 feature_vector_size,
                 d_in,
                 d_out,
                 dims,
                 geometric_init=True,
                 bias=1.0,
                 skip_in=(),
                 weight_norm=True,
                 multires=0
                 ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
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

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(self,
                 feature_vector_size,
                 mode,
                 d_in,
                 d_out,
                 dims,
                 weight_norm=True,
                 multires_view=0
                 ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            print('Applying positional encoding to view directions: ', multires_view)
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x = self.tanh(x)
        # return (x + 1.) / 2.
        return x


class IndirctIllumNetwork(nn.Module):
    def __init__(self, multires=0, dims=[128, 128, 128, 128], num_lgt_sgs=24, no_hdr=False):
        super().__init__()

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires)
            # self.embed_fn, input_dim = ipe_embedder(multires, 0.001)
        self.num_lgt_sgs = num_lgt_sgs
        self.actv_fn = nn.ReLU()
        self.use_hdr = not no_hdr
        if self.use_hdr:
            input_dim = input_dim + 1  # hdr-shift dim plus

        print('indirect illumination SG network size: ', dims)
        lobe_layer = []
        dim = input_dim
        for i in range(len(dims)):
            lobe_layer.append(nn.Linear(dim, dims[i]))
            lobe_layer.append(self.actv_fn)
            dim = dims[i]
        lobe_layer.append(nn.Linear(dim, self.num_lgt_sgs * 6))
        self.lobe_layer = nn.Sequential(*lobe_layer)

        from model.sg_envmap_material import SparseAE
        self.integral_layer = SparseAE(input_dim, 3, out_act=None, smooth_on_latent=False)
        self.integral_layer.lc_act = F.softplus

    def forward(self, points, hdr_shift):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        if self.use_hdr:
            points = torch.cat([points, hdr_shift], -1)

        batch_size = points.shape[0]
        output = self.lobe_layer(points).reshape(batch_size, self.num_lgt_sgs, 6)

        lgt_lobes = torch.sigmoid(output[..., :2])
        theta, phi = lgt_lobes[..., :1] * 2 * np.pi, lgt_lobes[..., 1:2] * np.pi  # phi in 0 ~ pi

        lgt_lobes = torch.cat(
            [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)

        lambda_mu = output[..., 2:]
        lambda_mu[..., :1] = torch.sigmoid(lambda_mu[..., :1]) * 30 + 0.1
        lambda_mu[..., 1:] = self.actv_fn(lambda_mu[..., 1:])

        lgt_sgs = torch.cat([lgt_lobes, lambda_mu], axis=-1)

        env_int = torch.abs(self.integral_layer(points)[1])

        return lgt_sgs, env_int


class VisNetwork(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, dims=[128, 128, 128, 128]):
        super().__init__()

        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)

        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = get_embedder(dirs_multires)

        self.actv_fn = nn.ReLU()

        vis_layer = []
        dim = p_input_dim + dir_input_dim
        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, 2))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, points, view_dirs):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            view_dirs = self.dir_embed_fn(view_dirs)

        vis = self.vis_layer(torch.cat([points, view_dirs], -1))

        return vis


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        use_neus = conf.get_bool("use_neus")

        self.feature_vector_size = conf.get_int('feature_vector_size')

        self.octree_ray_tracer = OctreeTracing(**conf.get_config('ray_tracer'), max_iter=32)

        if conf.get_bool('use_octree'):
            self.ray_tracer = OctreeTracing(**conf.get_config('ray_tracer'))
        else:
            self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))

        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        if use_neus:
            self.implicit_network = ImplicitNetworkMy(self.feature_vector_size, **conf.get_config('implicit_network'))
            self.rendering_network = self.implicit_network.color
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
            self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.indirect_illum_network = IndirctIllumNetwork(**conf.get_config('indirect_illum_network'),
                                                          no_hdr=conf.get_int("hdr_mode") == -1)
        self.visibility_network = VisNetwork(**conf.get_config('visibility_network'))
        self.envmap_material_network = EnvmapMaterialNetwork(**conf.get_config('envmap_material_network'))
        self.gamma = GammaCorrect(conf.get_float("gamma"), conf.get_int("hdr_mode"))

    def forward(self, input, trainstage='IDR', fun_spec=False, lin_diff=False, train_spec=False):

        if "intrinsics" in input:
            # parse model input
            intrinsics = input["intrinsics"]
            uv = input["uv"]
            pose = input["pose"]
            object_mask = input["object_mask"].reshape(-1)

            ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
            batch_size, num_pixels, _ = ray_dirs.shape

            with torch.no_grad():
                points, network_object_mask, dists = self.ray_tracer(
                    sdf=lambda x: self.implicit_network(x)[:, 0], cam_loc=cam_loc,
                    object_mask=object_mask, ray_directions=ray_dirs)
        else:
            cam_loc = input["points"].reshape(-1, 3)
            ray_dirs = input["dirs"].reshape(-1, 1, 3)
            if "object_mask" in input:
                object_mask = input["object_mask"].reshape(-1)
            else:
                object_mask = torch.ones_like(ray_dirs[..., 0, 0], dtype=torch.bool)
            batch_size, num_pixels, _ = ray_dirs.shape

            points = torch.zeros_like(cam_loc)
            network_object_mask = torch.zeros_like(object_mask)
            dists = torch.zeros_like(cam_loc[..., 0])

            with torch.no_grad():
                points[object_mask], network_object_mask[object_mask], dists[object_mask] = self.ray_tracer(
                    sdf=lambda x: self.implicit_network(x)[:, 0], cam_loc=cam_loc[object_mask],
                    object_mask=object_mask[object_mask], ray_directions=ray_dirs[object_mask])

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        return_obj = {
            'points': points,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,  # tracer mask
            'object_mask': object_mask,  # dataset mask
            'ray_dirs': ray_dirs,
        }

        surface_mask = network_object_mask
        diff_surface_points = points[surface_mask]
        indirect_sgs = torch.ones(points.shape[0], self.indirect_illum_network.num_lgt_sgs, 7).cuda()
        indirect_sgs[:, :, -3:] = 0
        indirect_integral = torch.ones(points.shape[0], 3).cuda()

        # transform illum network for hdr-shift input
        if 'hdr_shift' in input:
            if diff_surface_points.shape[0] > 0:
                indirect_sgs[surface_mask], indirect_integral[surface_mask] = self.indirect_illum_network(
                    diff_surface_points, input['hdr_shift'][surface_mask])
            return_obj['hdr_shift'] = input['hdr_shift']

        ### query indirect illumination for training
        if trainstage == 'Illum':
            return_obj.update({'indirect_sgs': indirect_sgs, 'indir_integral': indirect_integral})
            normals = torch.ones_like(points)
            if surface_mask.any():
                sg_envmap_material = self.envmap_material_network(points[surface_mask], train_spec=False,
                                                                  train_norm=True)
                normals[surface_mask] = sg_envmap_material["sg_normal_map"]
            return_obj.update({'normals': normals})  # encoder-decoder in the norm stage
            return return_obj

        bg_rgb_values = torch.ones_like(points).float().cuda()
        if self.envmap_material_network.envmap is not None:  # to relight
            bg_rgb_values = render_envmap(self.envmap_material_network.envmap, ray_dirs)

        ### peform sg render for materials training
        sg_rgb_values = torch.ones_like(points).float().cuda()
        indir_rgb_values = torch.ones_like(points).float().cuda()
        indir_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        indir_specular_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_specular_rgb_values = torch.ones_like(points).float().cuda()

        normal_values = torch.ones_like(points).float().cuda()
        diffuse_albedo_values = torch.ones_like(points).float().cuda()
        roughness_values = torch.ones_like(points).float().cuda()
        metallic_values = torch.ones_like(points[..., :1]).float().cuda()
        normal_map_values = torch.ones_like(points).float().cuda()
        vis_shadow = torch.ones_like(points).float().cuda()

        random_xi_diffuse_albedo = torch.ones_like(points).float().cuda()
        random_xi_roughness = torch.ones_like(points).float().cuda()
        random_xi_metallic = torch.ones_like(points[..., :1]).float().cuda()

        acc_values = torch.ones_like(points[..., :1]).float().cuda()
        final_t_values = torch.ones_like(points[..., :1]).float().cuda()
        gradient_error = torch.tensor(0.0).float().cuda()

        albedo_ratio = None
        if 'albedo_ratio' in input:
            albedo_ratio = input['albedo_ratio']
        tex_uv = None
        if 'tex_uv' in input:
            tex_uv = input['tex_uv'][:, surface_mask]

        hdr_shift = input['hdr_shift'][surface_mask] if 'hdr_shift' in input else None

        if diff_surface_points.shape[0] > 0:
            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            mask_indir_lgs = indirect_sgs[surface_mask]
            mask_indir_int = indirect_integral[surface_mask]
            ret = self.get_sg_render(diff_surface_points,
                                     view_dirs,
                                     mask_indir_lgs,
                                     albedo_ratio=albedo_ratio,  # None
                                     fun_spec=fun_spec,  # False
                                     lin_diff=lin_diff,  # False
                                     train_spec=train_spec,  # True
                                     indir_integral=mask_indir_int,
                                     tex_uv=tex_uv,
                                     hdr_shift=hdr_shift)  # None

            sg_rgb_values[surface_mask] = ret['sg_rgb']  # direct light rgb
            indir_rgb_values[surface_mask] = ret['indir_rgb']  # indirect light rgb

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']  # direct diffuse
            indir_diffuse_rgb_values[surface_mask] = ret['indir_diffuse_rgb']  # indirect diffuse

            if fun_spec:  # in pbr stage, the func is False

                def sg_specular_rgb_values_fn(roughness):
                    sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb'](roughness[surface_mask])
                    return sg_specular_rgb_values

                def indir_specular_rgb_values_fn(roughness):
                    indir_specular_rgb_values[surface_mask] = ret['indir_specular_rgb'](roughness[surface_mask])
                    return indir_specular_rgb_values

            else:
                sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']  # direct specular
                indir_specular_rgb_values[surface_mask] = ret['indir_specular_rgb']  # indirect specular

            normal_values[surface_mask] = ret['normals']  # from nues
            diffuse_albedo_values[surface_mask] = ret['diffuse_albedo']  # learned from brdf encoder-decoder
            roughness_values[surface_mask] = ret['roughness'].expand(-1, 3)  # same as above
            metallic_values[surface_mask] = ret['metallic']  # same as above
            normal_map_values[surface_mask] = ret['normal_map']  # learned from encoder-decoder structure in norm stage

            vis_shadow[surface_mask] = ret['vis_shadow']

            random_xi_diffuse_albedo[surface_mask] = ret['random_xi_diffuse_albedo']  # add smooth
            random_xi_roughness[surface_mask] = ret['random_xi_roughness'].expand(-1, 3)
            random_xi_metallic[surface_mask] = ret['random_xi_metallic']

            if 'acc' in ret:
                acc_values[surface_mask] = ret['acc'][..., None]
            if 'final_t' in ret:
                final_t_values[surface_mask] = ret['final_t']
            if 'gradient_error' in ret:
                gradient_error = gradient_error + ret['gradient_error']
        elif fun_spec:
            def sg_specular_rgb_values_fn(roughness):
                return sg_specular_rgb_values

            def indir_specular_rgb_values_fn(roughness):
                return indir_specular_rgb_values

        return_obj.update({
            'final_t': final_t_values,
            'gradient_error': gradient_error,
            'acc': acc_values,
            'bg_rgb': bg_rgb_values,  # None
            'sg_rgb': sg_rgb_values,
            'indir_rgb': indir_rgb_values,
            'sg_diffuse_rgb': sg_diffuse_rgb_values,
            'sg_specular_rgb': sg_specular_rgb_values if not fun_spec else sg_specular_rgb_values_fn,
            'indir_diffuse_rgb': indir_diffuse_rgb_values,
            'indir_specular_rgb': indir_specular_rgb_values if not fun_spec else indir_specular_rgb_values_fn,
            'normals': normal_values,
            'diffuse_albedo': diffuse_albedo_values,
            'roughness': roughness_values,
            'metallic': metallic_values,
            'normal_map': normal_map_values,
            'surface_mask': surface_mask,
            'vis_shadow': vis_shadow,
            'random_xi_roughness': random_xi_roughness,
            'random_xi_metallic': random_xi_metallic,
            'random_xi_diffuse_albedo': random_xi_diffuse_albedo,
        })

        return return_obj

    def get_idr_render(self, points, view_dirs=None, normal_only=False):
        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        # offset normals by normal map
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        if normal_only:
            return normals

        # idr renderer
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        return normals, idr_rgb

    def get_sg_render(self, points, view_dirs, indir_lgtSGs, albedo_ratio=None, fun_spec=False, lin_diff=False,
                      train_spec=False, **kwargs):
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # normalization
        normals = self.get_idr_render(points, view_dirs, normal_only=True)  # normal from neus
        ret = {'normals': normals, }  # neus normal

        # sg renderer
        sg_envmap_material = self.envmap_material_network(points, train_spec=train_spec)  # get brdf material

        if albedo_ratio is not None:
            sg_envmap_material['sg_diffuse_albedo'] = sg_envmap_material['sg_diffuse_albedo'] * albedo_ratio

        # no metallic
        sg_ret = render_with_all_sg(points=points.detach(),
                                    normal=normals.detach(),
                                    viewdirs=view_dirs,
                                    lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                    indir_lgtSGs=indir_lgtSGs,
                                    VisModel=self.visibility_network,
                                    fun_spec=fun_spec,
                                    lin_diff=lin_diff)
        ret.update(sg_ret)
        ret.update({'diffuse_albedo': sg_envmap_material['sg_diffuse_albedo'],
                    'roughness': sg_envmap_material['sg_roughness'],
                    'random_xi_roughness': sg_envmap_material['random_xi_roughness'],
                    'random_xi_diffuse_albedo': sg_envmap_material['random_xi_diffuse_albedo']})

        return ret

    def batch_idr_forward(self, points, viewdirs, n_pixels=4096):
        batch_size = n_pixels
        radiance = torch.zeros_like(points).cuda()
        split = []

        for i, indx in enumerate(torch.split(torch.arange(points.shape[0]).cuda(), batch_size, dim=0)):
            curr_points = points[indx]
            curr_viewdirs = viewdirs[indx]
            feature_vectors = None
            if self.feature_vector_size > 0:
                feature_vectors = self.implicit_network(curr_points)[:, 1:]
            g = self.implicit_network.gradient(curr_points).clone().detach()
            normals = g[:, 0, :] / (torch.norm(g[:, 0, :], dim=-1, keepdim=True) + TINY_NUMBER)
            radiance[indx] = self.rendering_network(curr_points, normals, curr_viewdirs, feature_vectors)

        return radiance.detach()

    def sample_dirs(self, normals, r_theta, r_phi):
        z_axis = torch.zeros_like(normals).cuda()
        z_axis[:, :, 0] = 1

        def norm_axis(x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

        normals = norm_axis(normals)
        U = norm_axis(torch.cross(z_axis, normals))
        V = norm_axis(torch.cross(normals, U))

        r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
        r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)
        sample_raydirs = U * torch.cos(r_theta) * torch.sin(r_phi) \
                         + V * torch.sin(r_theta) * torch.sin(r_phi) \
                         + normals * torch.cos(r_phi)  # [num_cam, num_samples, 3]
        return sample_raydirs

    def trace_radiance(self, input, nsamp=16, test_dir=None):
        points = input["points"]
        hdr_shift = input["hdr_shift"]
        points_mask = input["network_object_mask"]

        trace_radiance = torch.zeros(points.shape[0], nsamp, 3).cuda()
        sec_cam_loc = points[points_mask].clone()  # [hit_points, 3]
        sample_dirs = torch.zeros(sec_cam_loc.shape[0], nsamp, 3).cuda()

        gt_vis = torch.zeros(points.shape[0], nsamp, 1).bool().cuda()
        pred_vis = torch.zeros(points.shape[0], nsamp, 2).cuda()

        indir_mask = torch.zeros_like(gt_vis)
        gt_integral = torch.zeros_like(points)

        if sec_cam_loc.shape[0] > 0:

            def spherical_uniform(n_samples):
                u = torch.rand(n_samples) * 2 - 1
                t = torch.rand(n_samples) * torch.pi * 2
                x = (1 - u ** 2) ** 0.5 * torch.cos(t)
                y = (1 - u ** 2) ** 0.5 * torch.sin(t)
                z = u
                return torch.stack([x, y, z], -1).cuda()

            normals = input["normals"].detach()[points_mask][:, None, :]  # normal after smoothness
            normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-4)

            if test_dir is not None:
                sample_dirs = test_dir[None, None].expand(sample_dirs.shape)
            else:
                sample_dirs = spherical_uniform(sec_cam_loc.shape[0] * nsamp).view(sec_cam_loc.shape[0], nsamp, 3)

            back_cull_mask = (normals * sample_dirs).sum(-1) < 0  # back culling

            sec_object_mask = torch.ones(sec_cam_loc.shape[0] * nsamp).bool().cuda()

            # find second intersections
            with torch.no_grad():
                sec_points, sec_net_object_mask, sec_dist = self.octree_ray_tracer(
                    sdf=lambda x: self.implicit_network(x)[:, 0],
                    cam_loc=sec_cam_loc + normals[:, 0] * 0.005,
                    object_mask=sec_object_mask,
                    ray_directions=sample_dirs)

            if sec_net_object_mask.any():
                hit_points = sec_points[sec_net_object_mask]
                hit_viewdirs = -sample_dirs.reshape(-1, 3)[sec_net_object_mask]

                # get traced radiance
                mask_radiance = torch.zeros_like(sec_points).cuda()
                mask_radiance[sec_net_object_mask] = self.implicit_network.batch_borrow_color(hit_points, hit_viewdirs)

                def ldr2hdr(x, t=0.01):
                    return self.gamma.hdr_shift.ldr2hdr(x, t)

                shift = hdr_shift[points_mask][:, None, :].expand(-1, nsamp, 1).reshape(-1, 1)

                mask_radiance[sec_net_object_mask] = ldr2hdr(mask_radiance[sec_net_object_mask] ** 2.2,
                                                             t=shift[sec_net_object_mask])  # transfer to HDR

                mask_radiance = mask_radiance.reshape(sec_cam_loc.shape[0], nsamp, 3)
                mask_radiance[back_cull_mask] = 0.
                trace_radiance[points_mask] = mask_radiance

            # get visibility from network
            input_p = sec_cam_loc.unsqueeze(1).expand(-1, nsamp, 3)
            query_vis = self.visibility_network(input_p.reshape(-1, 3), sample_dirs.reshape(-1, 3))
            pred_vis[points_mask] = query_vis.reshape(-1, nsamp, 2)
            gt_vis[points_mask] = sec_net_object_mask.reshape(sec_cam_loc.shape[0], nsamp, 1)

            indir_mask[points_mask] = ~back_cull_mask[..., None] & gt_vis[points_mask]

            cos_dot = trace_radiance[points_mask] * torch.relu((normals * sample_dirs).sum(-1, keepdims=True))
            hemi_sphere = (~back_cull_mask).sum(-1)[..., None]
            gt_integral[points_mask] = cos_dot.sum(-2) / torch.clamp(hemi_sphere, 1e-4)

        output = {'trace_radiance': trace_radiance,
                  'sample_dirs': sample_dirs,
                  'gt_vis': gt_vis,
                  'pred_vis': pred_vis,
                  'indir_mask': indir_mask[..., 0],
                  'gt_integral': gt_integral
                  }
        return output


if __name__ == '__main__':
    from pyhocon import ConfigFactory

    conf = ConfigFactory.parse_file('../confs_sg/default.conf')
    idr = IDRNetwork(conf.get('model'))
    idr.cuda()
    inputs = {
        'points': torch.rand(1, 10000, 3).cuda(),
        'dirs': torch.rand(1, 10000, 3).cuda()
    }
    results = idr.forward(inputs, trainstage='Material')

    for r in results:
        print(r, results[r].shape)

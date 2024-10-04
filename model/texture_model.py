import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import trimesh
import xatlas
import imageio

from model.rasterizor import texture_rasterizor


def gen_uv_map(mesh_path, out_path=None):
    mesh = trimesh.load(mesh_path)  # load original ply mesh learned by NeuS
    st = time.time()  #
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    if out_path is None:
        out_path = ".".join(mesh_path.split(".")[:-1]) + ".obj"
    xatlas.export(out_path, mesh.vertices[vmapping], indices, uvs)
    print("[Parameterize]", time.time() - st, "seconds")


def erode_map(image, mask, iterations=1):
    """
    :param image: [H, W, 4]
    :param iterations: n
    :return: filtered image
    """

    def conv(img):
        pad_image = np.pad(img, 1)[..., 1:-1]
        kernels = [pad_image[:-2, :-2], pad_image[:-2, 1:-1], pad_image[:-2, 2:],
                   pad_image[1:-1, :-2], pad_image[1:-1, 1:-1], pad_image[1:-1, 2:],
                   pad_image[2:, :-2], pad_image[2:, 1:-1], pad_image[2:, 2:]]
        return np.stack(kernels, 0)

    inv_mask = mask.mean(-1) < 1  # [H, W]
    mask = mask.mean(-1) >= 1  # [H, W]
    for i in range(iterations):
        rgb = conv(image * mask[..., None])
        a = conv(mask[..., None])
        avg = rgb.sum(0) / np.clip(a.sum(0), 1e-4, 9.0)
        image[inv_mask] = avg[inv_mask]
    return image


class TextureCache:

    def __init__(self, mesh_path):
        super(TextureCache, self).__init__()
        self.cache_dir = self.init_cache_dir(mesh_path)
        self.mesh, (self.box_min, self.box_size), self.uv = self.init_mesh(mesh_path)

    def init_mesh(self, mesh_path):
        new_name = ".".join(os.path.basename(mesh_path).split(".")[:-1]) + ".obj"  # clean.obj
        new_path = os.path.join(self.cache_dir, new_name)  # new obj file name
        if not os.path.exists(new_path):
            print("[Parameterize] generate uv map")
            gen_uv_map(mesh_path, new_path)
        mesh = trimesh.load(new_path)

        vert = np.array(mesh.vertices)
        box_min = vert.min(axis=0, initial=0.0) - 1e-2
        box_max = vert.max(axis=0, initial=0.0) + 1e-2

        return mesh, (box_min, box_max - box_min), np.array(mesh.visual.uv)

    def init_cache_dir(self, mesh_path):
        cache_dir = ".".join(os.path.basename(mesh_path).split(".")[:-1]) + ".cache"  # clean.cache
        cache_dir = os.path.join(os.path.dirname(mesh_path), cache_dir)  # cache dir
        if os.path.exists(cache_dir):
            print("[Use Cache]", cache_dir)
        else:
            os.makedirs(cache_dir)
            print("[Make Cache]", cache_dir)

        return cache_dir

    def render_basics(self, resolution):
        """ Do not call this in OpenGL context !!! """
        if not os.path.exists(self.get_cache_path("vert", resolution)):
            print("[Cache] generate vertices, normals and masks")
            vert = np.array(self.mesh.vertices)
            norm = np.array(self.mesh.vertex_normals)
            mask = np.ones_like(vert)
            self.save_float("vert", resolution, vert)
            self.save_float("norm", resolution, norm)
            self.save_float("mask", resolution, mask)

    def load_basics(self, resolution):
        vert = self.load_float("vert", resolution)
        norm = self.load_float("norm", resolution)
        mask = self.load_float("mask", resolution)
        return vert[..., :3], norm[..., :3], mask[..., :3]

    def save_float(self, tag, resolution, arr):
        with texture_rasterizor(resolution) as tex_render:
            data = tex_render(self.uv, self.mesh.faces, arr)
            imageio.imwrite(self.get_cache_path(tag, resolution), data)

    def load_float(self, tag, resolution):
        return imageio.imread(self.get_cache_path(tag, resolution))

    def get_cache_path(self, tag, resolution, ext="exr"):
        return os.path.join(self.cache_dir, f"{tag}x{resolution}.{ext}")


def get_vert_norm_mask_maps(mesh_path, resolution=2048):
    tex_cache = TextureCache(mesh_path)
    tex_cache.render_basics(resolution)
    vert, norm, mask = tex_cache.load_basics(resolution)

    vert = erode_map(vert, mask, 2)
    norm = erode_map(norm, mask, 2)
    mask = erode_map(mask, mask.copy(), 2)
    vert = erode_map(vert, mask, 2)
    norm = erode_map(norm, mask, 2)

    mask = mask[..., 0] > 0.5

    return torch.tensor(vert).float().cuda().permute(2, 0, 1)[None], \
           torch.tensor(norm).float().cuda().permute(2, 0, 1)[None], \
           torch.tensor(mask).float().cuda()[None, None]


class TexSampler:

    def __init__(self, resolution=2048):
        from confs_sg.env_path import MESH_PATH
        self.resolution = resolution
        self.vert, self.norm, self.mask = get_vert_norm_mask_maps(MESH_PATH,
                                                                  self.resolution)  # get sample from the texture

    def sample(self, n):
        uv = torch.rand(n, 2).cuda()
        uv_ = uv.reshape(1, 1, -1, 2) * 2 - 1
        vert = F.grid_sample(self.vert, uv_).reshape(3, -1).permute(1, 0).reshape(-1, 3)
        norm = F.grid_sample(self.norm, uv_).reshape(3, -1).permute(1, 0).reshape(-1, 3)
        mask = F.grid_sample(self.mask, uv_).reshape(-1) > 0.1
        norm = norm / torch.clamp(torch.norm(norm, dim=-1, keepdim=True), min=1e-4)

        uv_x = uv + torch.tensor([0.001, 0]).cuda()
        uv_y = uv + torch.tensor([0, 0.001]).cuda()
        uv_x_ = uv_x.reshape(1, 1, -1, 2) * 2 - 1
        uv_y_ = uv_y.reshape(1, 1, -1, 2) * 2 - 1
        tan_x = F.grid_sample(self.vert, uv_x_).reshape(3, -1).permute(1, 0).reshape(-1, 3) - vert
        tan_y = F.grid_sample(self.vert, uv_y_).reshape(3, -1).permute(1, 0).reshape(-1, 3) - vert

        tan_x = tan_x / torch.clamp(torch.norm(tan_x, dim=-1, keepdim=True), min=1e-4)
        tan_y = tan_y / torch.clamp(torch.norm(tan_y, dim=-1, keepdim=True), min=1e-4)

        return {
            "uv": uv,
            "x": vert * 0.5,  # scaling
            "normal": norm,
            "object_mask": mask,
            "tangent_u": tan_y,
            "tangent_v": tan_x
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mcubes
import trimesh
from misc.defs import ISDF
from misc.utils import BBox
import gin


def extract_fields(bound_min, bound_max, resolution, query_func, device='cuda'):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):

    print('[Mesh] resolution: {} threshold: {}'.format(resolution, threshold))
    u = extract_fields(bound_min, bound_max, resolution,
                       lambda pts: -query_func.sdf(pts) if isinstance(query_func, ISDF) else query_func(pts))
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


@gin.configurable
def extract_mesh(model:ISDF, bbox=1.5, resolution=512, threshold=0.0, device='cuda') -> trimesh.Trimesh:
    bbox = BBox(bbox)
    bound_min, bound_max = bbox.box_min, bbox.box_min + bbox.box_size
    bound_min, bound_max = bound_min.to(device), bound_max.to(device)
    vertices, triangles = extract_geometry(bound_min, bound_max, resolution, threshold, model)
    return trimesh.Trimesh(vertices, triangles)


if __name__ == '__main__':
    fun = lambda x: (torch.norm(x, dim=-1) - 1)
    mesh = extract_mesh(fun)
    mesh.export('tmp.ply')


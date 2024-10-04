import torch
import torch.nn as nn
from utils import rend_util
from utils.octree import OctreeSDF
from torchvision.utils import save_image


class OctreeTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_rootfind_steps=8,
            max_iter=-1,
    ):
        super().__init__()
        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_rootfind_steps
        self.sdf_octree = None
        self.max_iter = max_iter

    def generate(self, sdf_fn, tex_sampler=None):
        box_min, box_max = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
        if tex_sampler is not None:
            v = tex_sampler.tex_sampler.vert.view(3, -1).permute(1, 0) * 0.5
            m = tex_sampler.tex_sampler.mask.view(-1) > 0.9
            box_max = [c.item() + 1e-3 for c in v[m].max(0)[0]]
            box_min = [c.item() - 1e-3 for c in v[m].min(0)[0]]
        print("[BOX]", box_min, box_max)

        self.sdf_octree = OctreeSDF(sdf_fn, [box_min, box_max],
                                    max_iter=self.max_iter)

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):
        """
        cam_loc: [K, 3], object_mask: [N], rays_d: [K, N, 3] -> x: [N, 3], mask: [N], dist: [N]
        """
        batch_size, num_pixels, _ = ray_directions.shape
        rays_o = cam_loc[:, None, :].expand(ray_directions.shape).reshape(-1, 3)
        rays_d = ray_directions.reshape(-1, 3)
        hit_t, is_hit = self.sdf_octree.cast(rays_o, rays_d, return_is_hit=True)
        hit_x = hit_t * rays_d + rays_o

        return hit_x.float(), \
               is_hit, \
               hit_t[..., 0].float()


class OctreeVisModel(nn.Module):

    def __init__(self, ray_tracer: OctreeTracing):
        super(OctreeVisModel, self).__init__()
        self.ray_tracer = ray_tracer
        self.ray_tracer.sdf_octree.max_iter = 32

    def intersect_sphere(self, points, view_dirs, radius=1.0):
        rays_d = view_dirs
        rays_o = points
        rays_d = rays_d / torch.clamp(rays_d.norm(dim=-1, keepdim=True), 1e-4)
        closest = (-rays_o * rays_d).sum(-1, keepdim=True) * rays_d + rays_o
        t = torch.sqrt(radius ** 2 - (closest ** 2).sum(dim=-1, keepdim=True))  # must be inside sphere
        return closest + rays_d * t

    def forward(self, points, view_dirs):
        with torch.no_grad():
            rays_d = view_dirs
            rays_o = points

            hit_t, is_hit = self.ray_tracer.sdf_octree.cast(rays_o, rays_d, return_is_hit=True)

            return torch.stack([is_hit, ~is_hit], dim=-1).float()



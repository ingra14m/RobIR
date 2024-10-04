import torch
from model.texture_model import TexSampler
from model.focus_sampler import FocusSampler, focus_sampler_from_dataset


class TexSpaceSampler:

    def __init__(self, trainer):
        self.tex_sampler = TexSampler()
        self.focus_sampler = focus_sampler_from_dataset(trainer.train_dataset)
        self.model = trainer.model

    def sample_observations(self, x, normals):
        """
        x: [N, 3] -> rgb: [100, N, 3], dirs: [100, N, 3], mask: [100, N]
        """
        rand_idx = torch.randint(self.focus_sampler.n_cameras, []).to(x.device)

        sample, gt = self.focus_sampler.scatter_sample(x)
        cam_dir = sample['view_dir'][rand_idx:rand_idx + 1]  # [100, N, 3]
        obj_mask = sample['object_mask'][rand_idx:rand_idx + 1]  # [100, N]
        cam_pos = self.focus_sampler.cam_loc[rand_idx:rand_idx + 1]  # [100, 3]

        with torch.no_grad():
            sec_points, sec_net_object_mask, sec_dist = self.model.octree_ray_tracer(
                sdf=lambda x: self.model.implicit_network(x)[:, 0],
                cam_loc=x + normals * 0.005,
                object_mask=obj_mask.permute(1, 0),
                ray_directions=-cam_dir.permute(1, 0, 2))

        vis_mask = ~sec_net_object_mask.reshape(obj_mask.shape[1], -1).permute(1, 0)
        vis_mask = torch.logical_and(obj_mask, vis_mask)
        return gt['rgb'], cam_dir, vis_mask, cam_pos

    def sample_observations_sorted(self, x, surface_mask, n_obs=1, normals=None):
        all_obs_rgb = torch.ones_like(x)
        all_obs_dirs = torch.ones_like(x)
        all_obs_mask = torch.zeros_like(x[..., 0])
        all_obs_pos = torch.zeros_like(x)
        # remove unseen cases
        obs_rgb, obs_dirs, obs_mask, obs_pos = self.sample_observations(x[surface_mask], normals[surface_mask])
        obs_pos = obs_pos[:, None, :].expand(-1, obs_mask.shape[1], -1)
        selected = obs_mask.sum(0) >= n_obs
        surface_mask[surface_mask.clone()] = selected
        obs_rgb = obs_rgb[:, selected]
        obs_dirs = obs_dirs[:, selected]
        obs_mask = obs_mask[:, selected].float()
        obs_pos = obs_pos[:, selected]

        sort_mask, idx = torch.sort(obs_mask + torch.rand_like(obs_mask) * 0.01, dim=0, descending=True)
        sort_rgb = obs_rgb.gather(0, idx[..., None].expand(-1, -1, 3))[:n_obs]
        sort_dirs = obs_dirs.gather(0, idx[..., None].expand(-1, -1, 3))[:n_obs]
        sort_pos = obs_pos.gather(0, idx[..., None].expand(-1, -1, 3))[:n_obs]

        all_obs_rgb[surface_mask] = sort_rgb
        all_obs_dirs[surface_mask] = sort_dirs
        all_obs_mask[surface_mask] = 1
        all_obs_pos[surface_mask] = sort_pos
        return all_obs_rgb, all_obs_dirs, all_obs_mask, all_obs_pos

    def data_batch(self, n):
        tex_input = self.tex_sampler.sample(n)
        x = tex_input['x']
        surface_mask = tex_input["object_mask"]
        normal = tex_input["normal"]

        obs_rgb, obs_dirs, obs_mask, obs_pos = self.sample_observations_sorted(x, surface_mask, normals=normal)

        new_inputs = {}
        new_inputs['points'] = obs_pos[None]
        new_inputs['dirs'] = obs_dirs[None]
        new_inputs['object_mask'] = obs_mask.bool()[None]
        new_inputs['tex_uv'] = tex_input['uv'][None]

        return new_inputs, normal, obs_rgb

    def simple_data_batch(self, model_inputs):
        n = model_inputs['uv'].shape[1]  # can change to 512 constant since it will out of memory
        tex_input = self.tex_sampler.sample(n)
        x = tex_input['x']
        surface_mask = tex_input["object_mask"]
        normal = tex_input["normal"]

        new_inputs = {}
        new_inputs['points'] = x
        new_inputs['normals'] = normal
        new_inputs['object_mask'] = surface_mask.bool()

        return new_inputs

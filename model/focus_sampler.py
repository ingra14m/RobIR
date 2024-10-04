import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from model.embedder import get_embedder
from datasets.syn_dataset import SynDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sg_sample(sg_lobe, sg_lambda, sg_mu, direction):
    lobe = sg_lobe / torch.clamp(torch.norm(sg_lobe, dim=-1, keepdim=True), min=1e-5)
    dot = (lobe * direction).sum(-1, keepdim=True)
    return sg_mu * torch.exp(sg_lambda * (dot - 1))


def inv_camera_params(locations, pose_inv, cam_loc, intrinsics):
    batch_size, num_samples, _ = locations.shape
    ray_dirs = F.normalize(locations - cam_loc[:, None, :], dim=-1)  # [num_cam, samples, 3]

    norm_inv = pose_inv @ torch.cat([ray_dirs + cam_loc[:, None, :], torch.ones_like(ray_dirs[..., -1:])], -1).permute(
        0, 2, 1)

    z_pos = -norm_inv[:, 2:3, :]
    ppc_inv = norm_inv / torch.where(z_pos != 0, z_pos, 1e-5 * torch.ones_like(z_pos))  # ndc
    ppc_inv[:, 1:3, :] *= -1  # reverse y&z
    uv_inv = intrinsics @ ppc_inv[:, :3, :]  # transform to image space
    uv = uv_inv.permute(0, 2, 1)[..., :2]  # [num_cam, samples, uv]

    return uv, ray_dirs


class FocusSampler:

    def __init__(self, images, masks, poses, intrinsics, img_res=None):
        cam_loc = poses[:, :3, 3]
        p = torch.eye(4).repeat(poses.shape[0], 1, 1).to(device).float()  # from (4, 4) to (image_len, 4, 4)
        p[:, :3, :4] = poses[:, :3, :4]
        self.pose_inv = p.inverse().float()
        self.cam_loc = cam_loc.to(device).float()
        self.n_cameras = len(images)
        if img_res is None:
            shape = [self.n_cameras, images[0].shape[0], images[0].shape[1], -1]
        else:
            shape = [self.n_cameras, img_res[0], img_res[1], -1]

        # [num_cam, H, W, 3] -> [num_cam, 3， H, W]
        self.images = images.view(shape).permute(0, 3, 1, 2).to(device).float()
        self.masks = masks.view(shape).permute(0, 3, 1, 2).to(device).float()  # [num_cam, 1， H, W]
        self.intrinsics = intrinsics.to(device).float()  # same in every camera
        self.img_size = torch.tensor(shape[1:3]).to(device).float()

    def sample_images(self, uv):
        uv = (uv[:, None] / self.img_size) * 2 - 1  # suit the shape of grid sample
        color = F.grid_sample(self.images, uv, align_corners=True)
        return color.permute(0, 2, 3, 1)[:, 0]  # [num_cam, sample, 3]

    def sample_masks(self, uv):
        uv = (uv[:, None] / self.img_size) * 2 - 1
        color = F.grid_sample(self.masks, uv, align_corners=True)
        return color.permute(0, 2, 3, 1)[:, 0] > 0.5

    def scatter_sample(self, x):
        assert len(x.shape) == 2
        x = x[None].expand(self.n_cameras, -1, -1)  # [n_cam, N_rays * N_Samples, 3]
        # [n_cam, N_rays * N_Samples, 2]
        uv, ray_dirs = inv_camera_params(x, self.pose_inv, self.cam_loc, self.intrinsics)
        rgb = self.sample_images(uv)  # [n_cam, N_rays * N_Samples, 3]

        uv_valid = torch.logical_and(uv >= 0, uv < self.img_size).prod(
            -1).bool()  # [n_cam, N_rays * N_Samples] must be true
        if uv_valid.any() and self.masks.numel() > 0:
            uv_valid[uv_valid.clone()] = (self.sample_masks(uv)).squeeze()[uv_valid]  # sample masks from each camera

        sample = {
            "object_mask": uv_valid,
            "uv": uv,
            "view_dir": ray_dirs,
        }

        ground_truth = {
            "rgb": rgb,
        }

        """
        M: dataset image num (or maximum num), N: sample num
        x: [N, 3] (all cuda only)
        {
            object_mask: [M, N, 1] (bool)
            uv: [M, 4, 4]
            view_dirs: [M, N, 3]
        }
        {
            rgb: [M, N, 3]
        }
        """

        if sample["uv"][sample["object_mask"]].isnan().any():
            raise RuntimeError("uv valid is nan")

        return sample, ground_truth


# just use the train data to initialize the FocusSampler
def focus_sampler_from_blender(images, poses, render_poses, hwf, i_split, dtype=torch.float32):
    images = images[i_split[0]]
    poses = poses[i_split[0]]

    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    intrinsics = K[np.newaxis].repeat(len(images), axis=0)  # from (3, 3) to (1, 3, 3) and repeat to (image_len, 3, 3)

    images = torch.tensor(images, dtype=dtype).to(device)
    poses = torch.tensor(poses, dtype=dtype).to(device)
    intrinsics = torch.tensor(intrinsics, dtype=dtype).to(device)
    focus_sampler = FocusSampler(images[..., :3], images[..., 3], poses, intrinsics)  # split rgb and alpha
    return focus_sampler


def focus_sampler_from_dataset(dataset: SynDataset):
    images = torch.stack(dataset.rgb_images).to(device)
    masks = torch.stack(dataset.object_masks).to(device)
    poses = torch.stack(dataset.pose_all).to(device)
    intrinsics = torch.stack(dataset.intrinsics_all).to(device)
    focus_sampler = FocusSampler(images, masks, poses, intrinsics, img_res=dataset.img_res)  # split rgb and alpha
    return focus_sampler


class Mixer(torch.nn.Module):  # learn from color to color

    def __init__(self, n_layers=3, width=128):
        super(Mixer, self).__init__()
        layers = []
        in_dim = 6

        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(in_dim, width),
            )
            in_dim = width
            layers.append(layer)
        self.n_layers = n_layers
        self.layers = nn.ModuleList(layers)

        self.out_linear = nn.Linear(width, 3)

    def forward(self, trained, posterior):
        """

        :param x: N, 3
        :param rgb_all: M, N, 3
        :param dirs_all: M, N, 3
        :param mask_all: M, N
        :return: N, 3
        """
        h = torch.cat([trained, posterior], dim=-1)
        for i in range(self.n_layers):
            h = self.layers[i](h)
            h = torch.relu(h)
        h = self.out_linear(h)  # [N_rays * N_samples, 3]

        return h  # just return rgb color


class Posterior(torch.nn.Module):  # learn from color to color

    def __init__(self, n_camera=100, n_layers=3, width=128):
        super(Posterior, self).__init__()
        layers = []
        self.pe_embed, embed_dim = get_embedder(10)
        self.view_embed, view_embed_dim = get_embedder(4)

        in_dim = n_camera * 3 + embed_dim
        self.n_camera = n_camera
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(in_dim, width),
            )
            in_dim = width
            layers.append(layer)
        self.n_layers = n_layers
        self.layers = nn.ModuleList(layers)
        self.spec_linear = nn.Linear(embed_dim + view_embed_dim, width)
        self.spec_linear2 = nn.Linear(width, width)
        self.spec_linear3 = nn.Linear(width, 1)
        # self.spec_out_linear = nn.Linear(width, 1)
        # self.out_linear = nn.Linear(width, n_camera)
        self.out_linear = nn.Linear(width, 3)

    def forward(self, x, rgb_all, dirs_all, mask_all, view_dir, is_log=False):
        """

        :param x: N, 3
        :param rgb_all: M, N, 3
        :param dirs_all: M, N, 3
        :param mask_all: M, N
        :return: N, 3
        """
        rgb_all[~mask_all] = 0  # filter the cam that can't see the point
        inputs = rgb_all.permute(1, 0, 2).reshape(rgb_all.shape[1], -1)  # [N_rays * N_samples, n_cam * 3]
        pos_embed = self.pe_embed(x)

        h = torch.cat([inputs, pos_embed], dim=-1)
        for i in range(self.n_layers):
            h = self.layers[i](h)
            h = torch.relu(h)
        h = self.out_linear(h)  # [N_rays * N_samples, 3]

        return h  # just return rgb color


def test_inv_sampler():
    from utils.rend_util import get_camera_params

    dataset = SynDataset("../data/lego", 5)
    focus_sampler = focus_sampler_from_dataset(dataset)
    print("[Test]", "create inv sampler with train-set...")

    print("[Test]", "inverse sampling outputs:")
    sample, gt = focus_sampler.scatter_sample(torch.rand(512, 3).to(device))
    for s in sample:
        print(s, sample[s].shape, sample[s].dtype, sample[s].device)
    print("rgb", gt["rgb"].shape, gt["rgb"].dtype, gt["rgb"].device)

    print("[Test]", "check inverse sampling by recast rays...")
    d, o = get_camera_params(sample["uv"], torch.stack(dataset.pose_all).to(device), focus_sampler.intrinsics)
    print("rays", d.shape, o.shape)

    print("[Test]", "the recasting error is:")
    diff = (d - sample["view_dir"]) ** 2
    print("diff", diff.mean().item())


def test_posterior():
    N = 115
    print("[Test]", f"create Posterior with {N} training images...")
    posterior = Posterior(N)  # for color output
    x = torch.rand(2048, 3)  # position for sampling
    rgb = torch.rand(N, 2048, 3)  # rgb for each camera
    dirs = torch.rand(N, 2048, 3)  # ray dir
    mask = torch.rand(N, 2048) > 0.5
    rgb = posterior(x, rgb, dirs, mask, x)
    print("[Test]", "output shape", rgb.shape)


if __name__ == '__main__':
    test_posterior()
    test_inv_sampler()

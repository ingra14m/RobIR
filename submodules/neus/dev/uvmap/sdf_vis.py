import torch.linalg
from torch.optim.optimizer import Optimizer

from model.neus_fields import *
from model import *
from volume_render.interface import neus_render_fn
from optimization.trainer import Trainer
from dataset.interface import load_dataset
from misc.utils import *
import imageio
from tqdm import tqdm, trange
from model.fields import tcnn_linear


class BBox(nn.Module):

    def __init__(self, box_size=None, box_min=None, box_max=None, box_center=None, dims=3):
        super(BBox, self).__init__()
        self.dims = dims
        def brd_cst(x): return torch.tensor(x).expand(dims)

        if box_size is not None:
            self.box_size = brd_cst(box_size)
            self.box_center = brd_cst(0) if box_center is None else box_center
        else:
            assert box_min is not None and box_max is not None
            self.box_size = brd_cst(box_max) - brd_cst(box_min)
            self.box_center = (brd_cst(box_max) + brd_cst(box_min)) / 2

        self.box_min = self.box_center - self.box_size / 2
        assert (self.box_size > 0).all()

    def inv(self, local_x):
        return local_x * self.box_size + self.box_min

    def forward(self, x):
        assert x.shape[-1] == self.dims
        return (x - self.box_min) / self.box_size


def parse_gin_file():

    class cd:
        """Context manager for changing the current working directory"""

        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)

    with cd('../..'):
        gin.parse_config_files_and_bindings(["config/lego.gin",
                                             r"E:\Repository\nerf\logs\lego-ngpsdf\neus-hash.gin"],
                                            ["Log.log_dir = '../../logs'",
                                             "render_image.chunk = 1024",
                                             "render_neus.n_outside = 0",]
                                            )


device = 'cuda'


class TexturedSDF(nn.Module):

    def __init__(self):
        super(TexturedSDF, self).__init__()
        trainer = Trainer()
        self.neus = trainer.model
        assert isinstance(self.neus, NeuSModel)
        enc = Hash()
        self.tex = nn.Sequential(enc, tcnn_linear(enc.feature_dim(), 3))
        self.bbox = BBox(6)
        self.chessboard = False

    def __getattr__(self, item):
        try:
            attr = super().__getattr__(item)
        except AttributeError:
            return getattr(self.neus, item)
        return attr

    def color(self, x, gradients, dirs, feature_vector):
        col = self.tex(self.bbox(x))
        return self.grid_texture(col)

    def grid_texture(self, x):
        if not self.chessboard:
            return x
        xi = torch.round(x[..., :2] * 20).type(torch.int)
        col = (xi.sum(-1, keepdims=True) % 2)
        return torch.cat([col, 1 - col, torch.ones_like(col)], -1)


class TexSphere(nn.Module, ISDF):

    def __init__(self, radius=1.0):
        super(TexSphere, self).__init__()
        self.r = 1
        self.bbox = BBox(6)
        enc = Hash()
        self.tex = nn.Sequential(enc, tcnn_linear(enc.feature_dim(), 3))
        self.chessboard = False

    def box(self, x, bsize, r=0.0):
        q = torch.abs(x) - bsize
        sdf = torch.linalg.norm(torch.maximum(q, torch.zeros_like(q)), dim=-1) + torch.minimum(
            torch.maximum(q[..., 0], torch.maximum(q[..., 1], q[..., 2])), torch.zeros_like(q[..., 0])) - r
        return sdf.unsqueeze(-1)

    def sphere(self, x, r):
        return torch.linalg.norm(x, dim=-1, keepdims=True) - r

    def torus(self, x, tx, ty):
        pyz = x[..., 1:]
        q = torch.cat([torch.linalg.norm(pyz, dim=-1, keepdims=True) - tx, x[..., :1]], dim=-1)
        return torch.linalg.norm(q, dim=-1, keepdims=True) - ty

    def grid_texture(self, x):
        if not self.chessboard:
            return x
        xi = torch.round(x[..., :2] * 5).type(torch.int)
        col = (xi.sum(-1, keepdims=True) % 2)
        return torch.cat([col, 1 - col, torch.ones_like(col)], -1)

    def sdf(self, x):
        bsize = torch.tensor([0.2, 1, 0.5]) * self.r
        return self.torus(x, self.r * 0.6, self.r * 0.2) # torch.minimum(self.sphere(x, self.r * 0.6), self.box(x, bsize, 0.1 * self.r))

    def radius(self) -> float:
        return 2.4

    def background(self, x, dirs):
        return -torch.ones_like(x[..., 0]) * 1000, torch.ones_like(x[..., :3])

    def sdf_and_feat(self, x):
        return self.sdf(x), None

    def color(self, x, gradients=None, dirs=None, feature_vector=None):
        col = self.tex(self.bbox(x))
        return self.grid_texture(col)

    def grad(self, x):
        return prox_gradients(self.sdf, x, 0.01)
        # return x / torch.linalg.norm(x, dim=-1, keepdims=True)

    def dev(self, x):
        variance = 50
        return torch.ones([len(x), 1]) * np.exp(variance * 10.0)


def render(model, rays, save_to=None):
    images = neus_render_fn(model, rays, True, 50000)
    rgb = images['rgb']
    # dist = images['dist']
    # acc = images['acc']
    # disp = 1. / torch.max(1e-10 * torch.ones_like(dist, device=dist.device), dist / (acc + 1e-8))
    if save_to is not None:
        save_image(rgb, save_to)
        # save_image(disp, save_to[:-4] + "-disp.png")
        # save_image(acc, save_to[:-4] + "-acc.png")
    return rgb


def save_image(image, filename):
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    rgb8 = to8b(image.detach().cpu().numpy())
    # if not os.path.exists(os.path.dirname(filename)):
    #     os.makedirs(os.path.dirname(filename))
    imageio.imwrite(filename, rgb8)


def train_step(i, model: TexturedSDF, optimizer: Optimizer, mode='pre'):
    bbox = BBox(6)
    batch_size = 2048
    pts0 = torch.rand(batch_size, 3).to(device)

    def local_axis(v):
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        rot = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ], dtype=torch.float)
        rot2 = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ], dtype=torch.float)
        a = rot @ rot2 @ v.unsqueeze(-1)
        a = a[..., 0]
        b = torch.cross(a, v, -1)
        b = b / torch.linalg.norm(b, dim=-1, keepdims=True)
        a = torch.cross(b, v, -1)
        a = a / torch.linalg.norm(a, dim=-1, keepdims=True)
        return a, b

    def label_loss(x, n):
        a, b = local_axis(n)
        label = x
        pred = model.color(x, None, None, None)
        return ((label - pred) ** 2).mean()

    def smooth_loss(x, n):
        a, b = local_axis(n)
        p2u = lambda p: model.color(p, None, None, None)[..., 0:1]
        p2v = lambda p: model.color(p, None, None, None)[..., 1:2]
        p2w = lambda p: model.color(p, None, None, None)[..., 2:3]
        mae = lambda f: torch.abs(f).mean()
        mse = lambda f: (f ** 2).mean()

        du = prox_tangent_gradients(p2u, x, 0.05, [a, b])
        dv = prox_tangent_gradients(p2v, x, 0.05, [a, b])

        # du_dn = prox_tangent_gradients(p2u, x, 0.05, [n])
        # dv_dn = prox_tangent_gradients(p2v, x, 0.05, [n])

        du_norm = torch.linalg.norm(du, dim=-1)
        dv_norm = torch.linalg.norm(dv, dim=-1)
        du_dv_dot = (du * dv).sum(-1)

        # du = prox_gradients(p2u, x, 0.01)
        # du_norm = torch.linalg.norm(du, dim=-1)
        # return mse(du_norm - 1)
        return mse(du_norm - 1) + mse(dv_norm - 1) + mse(p2w(x) - 1) + 0.1 * mse(du_dv_dot) # + mse(du_dn) + mse(dv_dn)

    loss_fn = label_loss if mode == "pre" else smooth_loss

    pts = bbox.inv(pts0)
    loss = loss_fn(pts, model.grad(pts))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train(model, mode='pre', max_iter=10000):
    optimizer = torch.optim.Adam(model.tex.parameters(), 1e-3, betas=(0.9, 0.99))
    pbar = trange(max_iter)
    for i in pbar:
        loss = train_step(i, model, optimizer, mode=mode)
        pbar.set_description(f"Loss: {loss.item()}")


def main():
    parse_gin_file()
    torch.random.manual_seed(20200823)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    np.random.seed(20201473)

    model = TexturedSDF()
    model.to(device)
    test_dataset = load_dataset("test")

    test_case = next(test_dataset)
    test_case = torch_tree_map(lambda x: x.to(device), test_case)
    # render(model, test_case['rays'], "sdf0.png")
    train(model, mode='pre', max_iter=500)
    train(model, mode='post')
    model.chessboard = True
    render(model, test_case['rays'], "sdf1.png")


if __name__ == '__main__':
    main()

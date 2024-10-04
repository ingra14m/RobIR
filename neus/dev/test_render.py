import torch
from absl import app
from dev.mip.volume_rendering import *
# from model import *
from misc.utils import BBox
from dataset import *


define_common_flags()


def test_vr(arg_v):
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_file, flags.FLAGS.gin_param)

    nerf = VNeRF()

    nerf.cuda()

    data = load_dataset("train")
    batch = next(data)
    rays = batch['rays']
    rays = torch_tree_map(lambda x: x.cuda(), rays)
    ret = volume_rendering(rays, nerf)

    print(torch_tree_map(lambda x: list(x.shape), ret))


def test_bbox(arg_v):
    idt = BBox(box_min=0, box_max=1)
    sim = BBox(1)
    x = sim(torch.tensor([3, 2, 1]))
    print(x)


if __name__ == '__main__':
    app.run(test_bbox)



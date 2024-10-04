from dev.misc_old.utils import *
from absl import app
from model import *


define_common_flags()


def test_embedders(arg_v):
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_file, flags.FLAGS.gin_param)

    pe = PE()
    x = torch.rand(100, 3)
    f = pe(x)
    print(f.shape)
    print(pe.feature_dim())

    grid = Grid()
    f = grid(x)
    print(f.shape)
    print(grid.feature_dim())

    ipe = IPE(1, 8, diag=False)

    mean = torch.randn(100, 3)
    half_cov = torch.randn(100, 3, 3)
    cov = half_cov @ half_cov.permute(0, 2, 1)

    f = ipe(mean, cov)
    print(f.shape)
    print(ipe.feature_dim())

    he = Hash()
    x = x.cuda()
    he.cuda()
    f = he(x)
    print(f.shape)
    print(he.feature_dim())


def test_mlp(arg_v):
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_file, flags.FLAGS.gin_param)

    pe = PE()
    x = torch.rand(100, 12, 3)
    f = pe(x)
    y = torch.rand(100, 12, 3)
    mlp = MLP(pe.feature_dim(), 3, 4, 1)
    a, rgb = mlp(f, y)
    print(a.shape)
    print(rgb.shape)
    print(mlp)

    smlp = simple_mlp(3, 4)
    print(smlp)
    print(smlp(x).shape)

    sh = SH()
    z = torch.rand(100, 12, sh.in_dim)
    z = sh(z, y)
    print(z.shape)

    nerf = VNeRF()
    rgb, a = nerf(x, y)
    print(rgb.shape)
    print(a.shape)


if __name__ == '__main__':
    app.run(test_embedders)




import matplotlib.pyplot as plt
import numpy as np
import torch
from model.energy_integral import Energy


class GammaCorrect(torch.nn.Module):

    def __init__(self, gamma=2.2, hdr_mode=0):
        super(GammaCorrect, self).__init__()
        if isinstance(gamma, torch.Tensor):
            self.gamma = gamma
        else:
            self.gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.indir_coef = torch.nn.Parameter(torch.tensor(1.0))
        self.dir_coef = torch.nn.Parameter(torch.tensor(2.0))
        self.coef = torch.nn.Parameter(torch.tensor(1.0))

        self.hdr_shift = ACESToneMapping(hdr_mode=hdr_mode)

    def forward(self, x):
        return torch.pow(x, 1 / self.gamma)
        # x = self.hdr2ldr(x)
        # x = self.ldr2hdr(x)
        # return x

    def inv(self, x):
        return torch.pow(x, self.gamma)


def aces_fn(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)


def aces_inv(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return ((0.59 * x - 0.03) + torch.sqrt((0.59 * x - 0.03) ** 2 + 4 * (2.51 - 2.43 * x) * 0.14 * x)) / (
            2 * (2.51 - 2.43 * x))


def warp_aces_inv(x, t):  # hdr with energy
    return 0.73 * aces_inv(x * t) / aces_inv(0.73 * t)


def warp_aces_fn(x, t):
    return aces_fn(aces_inv(0.73 * t) / 0.73 * x) / t


def scale_aces_inv(x, t):
    t = t ** 0.2  # most in [0.6, 1.0] is better
    return aces_inv(x * t)


def scale_aces_fn(x, t):
    t = t ** 0.2  # most in [0.6, 1.0] is better
    return aces_fn(x) / t


def identity_fn(x, t):
    return x


def ln_space_fn(x, shift):
    x = x * (0.5 + shift) / 0.5
    return x / (1 + shift * x)


def ln_space_inv(x, shift):
    y = x / (1 - shift * x)
    return y * 0.5 / (0.5 + shift)


class ACESToneMapping(torch.nn.Module):

    def __init__(self, hdr_mode=0):
        super(ACESToneMapping, self).__init__()
        self.adapt_illum = torch.nn.Parameter(torch.tensor(0.0))
        self.energy = Energy()
        if hdr_mode == 0:
            self.aces_fn = scale_aces_fn
            self.aces_inv = scale_aces_inv
        elif hdr_mode == 1:
            self.aces_fn = warp_aces_fn
            self.aces_inv = warp_aces_inv
        elif hdr_mode == 2:  # the latest version use this mode
            self.aces_fn = ln_space_fn
            self.aces_inv = ln_space_inv
        else:
            self.aces_fn = identity_fn
            self.aces_inv = identity_fn
        self.hdr_mode = hdr_mode

    def fit_data(self, data):

        def ldr2hdr(x, raw_shift):
            shift = self.make_shift(raw_shift)
            return self.aces_inv(x, shift)

        self.energy.gen_cache(data, ldr2hdr)

    def plot(self, shift=1.0):
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        a = torch.linspace(0, 1, 100, device="cuda")[..., None]
        b = self.ldr2hdr(a, shift)
        plt.plot(a.cpu().detach().numpy(), b.cpu().detach().numpy())
        plt.show()

    def scalar(self, shift):
        max_energy = self.energy(torch.ones_like(shift)).mean(-1, keepdim=True)
        return self.energy(shift).mean(-1, keepdim=True) / torch.clamp(max_energy, 1e-4, 1.0)

    def as_input(self):
        if self.hdr_mode == 2:
            return torch.clamp(self.adapt_illum * 10 + 0.5, 0, 1).view(1, 1)
        return torch.clamp(self.adapt_illum * 10 + 0.5, 0, 1).view(1, 1)

    def make_shift(self, shift):
        if shift is None:
            shift = self.as_input()
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift).cuda()
        if len(shift.shape) == 0:
            shift = shift[None]
        shift = torch.clamp(shift, 1e-4, 1)
        return shift

    def hdr2ldr(self, x, raw_shift=None):
        shift = self.make_shift(raw_shift)
        return self.aces_fn(x, shift)

    def ldr2hdr(self, x, raw_shift=None):
        shift = self.make_shift(raw_shift)
        return self.aces_inv(x, shift)


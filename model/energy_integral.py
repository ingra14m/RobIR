import torch
from torch import nn
import torch.nn.functional as F
from model.neus_model import get_embedder
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from datasets.syn_dataset import SynDataset


class EnergyInt(nn.Module):

    def __init__(self):
        super(EnergyInt, self).__init__()
        layers = []
        self.embed, dim = get_embedder(4, 1)
        dims = [128, 128, 64]
        for i in range(len(dims)):
            layers.append(nn.Linear(dim, dims[i]))
            layers.append(nn.ReLU())
            dim = dims[i]
        layers.append(nn.Linear(dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)
        x = self.mlp(x)
        return F.softplus(x)


class Energy:

    def __init__(self):
        self.net = EnergyInt()

    def __call__(self, x):
        return self.net(x)

    def plot(self):
        plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        x = torch.linspace(0, 1, 400).cuda()
        y = self(x[..., None])

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        plt.plot(x, y)
        plt.show()

    def gen_cache(self, data, ldr2hdr):
        self.net.cuda()
        self.net.train()
        images = torch.stack(data.rgb_images, 0).cuda()
        masks = torch.stack(data.object_masks, 0).cuda()
        images = images[masks]

        optimizer = torch.optim.Adam([{"lr": 0.0005, "params": self.net.parameters()}], betas=(0.9, 0.99))

        pbar = trange(1000)
        for _ in pbar:
            shift = torch.rand(512, 1).cuda()
            shift = torch.clamp(shift, 1e-4, 1 - 1e-4)
            idx = torch.tensor(np.random.choice(images.shape[0], 8192)).cuda().long()
            gt = integral(images[idx], shift, ldr2hdr)
            pred = self(shift)

            loss = ((gt - pred) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # self.plot()

        print("[Energy]", "fit_loss", loss.item())
        self.net.eval()


def integral(images, shift, ldr2hdr):
    images = images.view(-1, 3)
    images = torch.clamp(images, 1e-4, 1)
    return ldr2hdr(images[:, None, :], shift).mean(0)


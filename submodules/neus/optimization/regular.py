import torch
import gin
from functools import wraps
from torch.nn import functional as F


def regularization(reg):

    @wraps(reg)
    def f(*args, gamma=0, **kwargs):
        if gamma == 0:
            return 0
        return reg(*args, gamma=gamma, **kwargs)

    return gin.configurable(f)


@regularization
def sparsity_reg(weights, gamma):
    sp = torch.log(1 + 2 * weights ** 2)
    tot_sp = sp.sum(-1).mean()
    return tot_sp * gamma


@regularization
def similarity_reg(similarity, gamma):
    reg = (similarity - 1) ** 2
    reg = reg.sum(-1).mean()
    return reg * gamma


@regularization
def accumulate_reg(accumulate, mask, gamma):
    # reg = 1 - torch.cos(torch.clip(accumulate, 0, 1) * torch.pi * 2)
    # reg = reg.sum(-1).mean()
    reg = ((accumulate.squeeze() - mask.squeeze()) ** 2).mean()
    return reg * gamma


@regularization
def eikonal_reg(pts, gradients, gamma):
    if len(gradients.shape) < 2:
        return gradients.sum() * gamma
    # gradient error is relaxed outside the sphere so is proper to be calculated inside sdf-rendering
    raise NotImplementedError

    # pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True)
    # relax_inside_sphere = (pts_norm < 2.4).float().detach()
    # # Eikonal loss
    # gradient_error = (torch.linalg.norm(gradients.reshape(list(pts.shape[:-1]) + [-1]), ord=2, dim=-1, keepdim=True) - 1.0) ** 2
    # gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
    # return gradient_error * gamma


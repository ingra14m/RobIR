from dataset.interface import load_dataset
from optimization.log import *
from model.fields import *
from misc.utils import *
from misc.defs import *
from misc.schedule import Curve
from optimization.regular import *
from optimization.extraction import extract_mesh
from volume_render.interface import render_fns
import functools


@gin.configurable("Train")
@dataclasses.dataclass
class TrainConfig:
    lr_init: float = 5e-4  # The initial learning rate.
    lr_final: float = 5e-6  # The final learning rate.
    lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
    lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
    max_steps: int = 200000  # The number of optimization steps.
    device: str = "cuda"  # The default running device.
    weight_decay_mult: float = 0.0001  # The multiplier on weight decay.

    disable_multiscale_loss: bool = False # If True, disable multiscale loss.
    coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.


@gin.configurable
class Trainer:

    def __init__(self, model_class=VNeRF, render="mip", resume=True, reg_dict=None):
        self.dataset = load_dataset("train")
        self.test_dataset = load_dataset("test")
        self.logger = Logger()
        config = TrainConfig()
        self.model = model_class()
        self.model.to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr_init)  #, betas=(0.9, 0.99))
        self.render_fn = functools.partial(
            render_fns[render],
            self.model
        )
        self.reg_dict = reg_dict

        if resume:
            self.logger.load_state(model=self.model, optimizer=self.optimizer)
        # self.logger.skip_test_cases(self.test_dataset)

        self.learning_rate_fn = functools.partial(
            learning_rate_decay,
            lr_init=config.lr_init,
            lr_final=config.lr_final,
            max_steps=config.max_steps,
            lr_delay_steps=config.lr_delay_steps,
            lr_delay_mult=config.lr_delay_mult)

        self.config = config

    def train(self):
        logger = self.logger
        config = self.config

        for step, batch in zip(logger.trange(config.max_steps), self.dataset):
            lr = self.learning_rate_fn(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            stats = self.step(batch, step)
            logger.log_metrics(Loss=stats["loss"], PSNR=stats["psnr"], LR=lr)
            logger.log_scalars("regularization", **stats["reg"])

            # step all schedulers
            Curve.stepping(self.model, step)

            if logger.should_test():
                ret, metrics = self.image()
                logger.log_images(**ret)
                logger.log_metrics(Test_PSNR=metrics["psnr"])

            if logger.should_save():
                self.mesh()

        # reload test set
        self.test_dataset = load_dataset("test")

    def test(self):
        logger = self.logger
        config = self.config

        self.mesh()
        frames = {}

        def add_frame(**kv_pairs):
            for k in kv_pairs:
                if k not in frames:
                    frames[k] = []
                frames[k].append(kv_pairs[k])

        for i in logger.test_trange(100):
            ret, metrics = self.image()
            add_frame(rgb=ret['rgb'].cpu(), **metrics)

        def calc_avg(tag):
            mean = np.array(frames[tag]).mean().item()
            frames[f'mean_{tag}'] = mean

        [calc_avg(t) for t in ['psnr', 'mse']]
        logger.log_video(**frames)

    def profile(self):
        with torch.autograd.profiler.profile(use_cuda=True, enabled=True) as prof:
            self.image()
        print(prof)

    def mesh(self):
        if isinstance(self.model, ISDF):
            mesh = extract_mesh(self.model)
            self.logger.log_mesh(mesh)

    def model_weight_sum(self):
        def tree_sum_fn(fn):
            return torch_tree_reduce(lambda x, y: x + fn(y), list(self.model.parameters()), init=0)
        return tree_sum_fn(lambda z: torch.sum(z ** 2)) / tree_sum_fn(lambda z: torch.prod(torch.tensor(z.shape)))

    def regularize_callers(self, ret, mask):
        # try to let 2 different versions to be compatible (if reg_dict is None, use gammas)
        kwargs = {"gamma": 1} if self.reg_dict is not None else {}
        return {
            "similarity": lambda: similarity_reg(ret['sim_or_grad'], **kwargs),
            "eikonal": lambda: eikonal_reg(ret['means'], ret['sim_or_grad'], **kwargs),
            "sparsity": lambda: sparsity_reg(ret['weights'], **kwargs),
            "silhouette": lambda: accumulate_reg(ret['acc'][..., None], mask, **kwargs),
            "weight_l2": lambda: self.config.weight_decay_mult * self.model_weight_sum(),
        }

    def loss(self, ret, mask, pixels):
        rgb = ret['rgb']
        # losses = torch.cat(losses, 0)
        # loss = (config.coarse_loss_mult * torch.sum(losses[:-1]) + losses[-1] + weight_l2)
        mask_sum = mask.sum() + 1e-5
        mse_loss = (mask * (rgb - pixels[..., :3]) ** 2).sum() / mask_sum

        mask_sum = mask.sum() + 1e-5
        color_error = (rgb - pixels[..., :3]) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        l1_loss = color_fine_loss

        loss_dict = {
            "mse": mse_loss,
            "l1": l1_loss,
            "reg": {
                k: v() * (1.0 if self.reg_dict is None else self.reg_dict[k])
                for k, v in self.regularize_callers(ret, mask).items() if self.reg_dict is None or k in self.reg_dict
            },
            "psnr": mse_to_psnr(mse_loss.cpu().detach().numpy())
        }

        return loss_dict

    def step(self, batch, global_step):
        device = self.config.device
        rays = torch_tree_map(lambda x: x.to(device).float(), batch['rays'])
        pixels = torch_tree_map(lambda x: x.to(device).float(), batch['pixels'])
        ret = self.render_fn(rays, global_step=global_step)

        mask = rays.lossmult
        if self.config.disable_multiscale_loss:
            mask = torch.ones_like(mask)

        loss_dict = self.loss(ret, mask, pixels)
        # TODO: sdf uses l1 loss, but l1 has some problem
        loss = loss_dict["mse"]     # + loss_dict["l1"]
        for k in loss_dict["reg"]:
            loss = loss + loss_dict["reg"][k]

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_max_norm >= 1e-10:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_max_norm)

        self.optimizer.step()

        stats = {
            "loss": loss.item(),
            "psnr": loss_dict['psnr'],
            "reg": loss_dict["reg"]
        }

        return stats

    def image(self):
        torch.cuda.empty_cache()
        test_case = next(self.test_dataset)
        test_case = torch_tree_map(lambda x: x.to(self.config.device).float(), test_case)
        ret = self.render_fn(test_case['rays'], is_eval=True)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(ret['dist'], device=ret['dist'].device),
                                  ret['dist'] / (ret['acc'] + 1e-8))
        ret['dist'] = disp_map

        mse_loss = ((ret['rgb'] - test_case['pixels'][..., :3]) ** 2).mean().cpu()
        psnr = mse_to_psnr(mse_loss)
        metrics = {
            'mse': mse_loss.item(),
            'psnr': psnr.item(),
        }
        return ret, metrics



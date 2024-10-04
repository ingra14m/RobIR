import abc
import numpy as np
import math
from typing import *
import copy
from collections.abc import Mapping
import torch
from torch import nn


class Schedule(abc.ABC):
    """An interface for generic schedules.."""

    @abc.abstractmethod
    def get(self, step):
        """Get the value for the given step."""
        raise NotImplementedError

    def __call__(self, step):
        return self.get(step)


class ConstantSchedule(Schedule):
    """Linearly scaled scheduler."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def get(self, step):
        """Get the value for the given step."""
        return np.full_like(step, self.value, dtype=np.float32)


class LinearSchedule(Schedule):
    """Linearly scaled scheduler."""

    def __init__(self, initial_value, final_value, num_steps):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def get(self, step):
        """Get the value for the given step."""
        if self.num_steps == 0:
            return np.full_like(step, self.final_value, dtype=np.float32)
        alpha = np.minimum(step / self.num_steps, 1.0)
        return (1.0 - alpha) * self.initial_value + alpha * self.final_value


class ExponentialSchedule(Schedule):
    """Exponentially decaying scheduler."""

    def __init__(self, initial_value, final_value, num_steps, eps=1e-10):
        super().__init__()
        if initial_value <= final_value:
            raise ValueError('Final value must be less than initial value.')

        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.eps = eps

    def get(self, step):
        """Get the value for the given step."""
        if step >= self.num_steps:
            return np.full_like(step, self.final_value, dtype=np.float32)

        final_value = max(self.final_value, self.eps)
        base = final_value / self.initial_value
        exponent = step / (self.num_steps - 1)
        if step >= self.num_steps:
            return np.full_like(step, self.final_value, dtype=np.float32)
        return self.initial_value * base**exponent


class CosineEasingSchedule(Schedule):
    """Schedule that eases slowsly using a cosine."""

    def __init__(self, initial_value, final_value, num_steps):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def get(self, step):
        """Get the value for the given step."""
        alpha = np.minimum(step / self.num_steps, 1.0)
        scale = self.final_value - self.initial_value
        x = min(max(alpha, 0.0), 1.0)
        return (self.initial_value
                + scale * 0.5 * (1 + math.cos(np.pi * x + np.pi)))


class StepSchedule(Schedule):
    """Schedule that eases slowsly using a cosine."""

    def __init__(self,
                 initial_value,
                 decay_interval,
                 decay_factor,
                 max_decays,
                 final_value=None):
        super().__init__()
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.max_decays = max_decays
        if final_value is None:
            final_value = self.initial_value * self.decay_factor**self.max_decays
        self.final_value = final_value

    def get(self, step):
        """Get the value for the given step."""
        phase = step // self.decay_interval
        if phase >= self.max_decays:
            return self.final_value
        else:
            return self.initial_value * self.decay_factor**phase


class PiecewiseSchedule(Schedule):
    """A piecewise combination of multiple schedules."""

    def __init__(
            self, schedules: Iterable[Tuple[int, Union[Schedule, Iterable[Any]]]]):
        self.schedules = [from_config(s) for ms, s in schedules]
        milestones = np.array([ms for ms, s in schedules])
        self.milestones = np.cumsum(milestones)[:-1]

    def get(self, step):
        idx = np.searchsorted(self.milestones, step, side='right')
        schedule = self.schedules[idx]
        base_idx = self.milestones[idx - 1] if idx >= 1 else 0
        return schedule.get(step - base_idx)


class DelayedSchedule(Schedule):
    """Delays the start of the base schedule."""

    def __init__(self, base_schedule: Schedule, delay_steps, delay_mult):
        self.base_schedule = from_config(base_schedule)
        self.delay_steps = delay_steps
        self.delay_mult = delay_mult

    def get(self, step):
        delay_rate = (
                self.delay_mult
                + (1 - self.delay_mult)
                * np.sin(0.5 * np.pi * np.clip(step / self.delay_steps, 0, 1)))

        return delay_rate * self.base_schedule(step)


SCHEDULE_MAP = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule,
    'cosine_easing': CosineEasingSchedule,
    'step': StepSchedule,
    'piecewise': PiecewiseSchedule,
    'delayed': DelayedSchedule,
}


def from_tuple(x):
    schedule_type, *args = x
    return SCHEDULE_MAP[schedule_type](*args)


def from_dict(d):
    d = copy.copy(dict(d))
    schedule_type = d.pop('type')
    return SCHEDULE_MAP[schedule_type](**d)


def from_config(schedule):
    if isinstance(schedule, Schedule):
        return schedule
    if isinstance(schedule, Tuple) or isinstance(schedule, List):
        return from_tuple(schedule)
    if isinstance(schedule, Mapping):
        return from_dict(schedule)

    raise ValueError(f'Unknown type {type(schedule)}.')


class Curve:

    def __init__(self, config):
        if isinstance(config, (int, float)):
            self.schedule = ConstantSchedule(config)
            self.no_schedule = True
        else:
            self.schedule = from_config(config)
            self.no_schedule = False
        self.step = 0

    @staticmethod
    def stepping(module: nn.Module, step):

        def fn(m):
            for k in vars(m):
                o = getattr(m, k)
                if isinstance(o, Curve):
                    o.step = step

        module.apply(fn)

    def __call__(self, *args, **kwargs):
        return torch.tensor(self.schedule.get(self.step))

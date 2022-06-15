import logging
from typing import Iterator
from dataclasses import dataclass

import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register


@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    train_mode: str = 'custom'
    eval_period: int = 1



@register.register_optimizer('adagrad')
def adagrad_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> Adagrad:
    return Adagrad(params, lr=base_lr, weight_decay=weight_decay)


@register.register_scheduler('plateau')
def plateau_scheduler(optimizer: Optimizer, patience: int,
                      lr_decay: float) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)


@register.register_scheduler('reduce_on_plateau')
def scheduler_reduce_on_plateau(optimizer: Optimizer, reduce_factor: float, schedule_patience: int, min_lr: float,
                                train_mode: str, eval_period: int):
    if train_mode == 'standard':
        raise ValueError("ReduceLROnPlateau scheduler is not supported "
                         "by 'standard' graphgym training mode pipeline; "
                         "try setting config 'train.mode: custom'")

    if eval_period != 1:
        logging.warning("When config train.eval_period is not 1, the "
                        "optim.schedule_patience of ReduceLROnPlateau "
                        "may not behave as intended.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=reduce_factor,
        patience=schedule_patience,
        min_lr=min_lr,
        verbose=True
    )
    if not hasattr(scheduler, 'get_last_lr'):
        # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
        # pytorch1.10; we add it here for consistency with other schedulers.
        def get_last_lr(self):
            """ Return last computed learning rate by current scheduler.
            """
            return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)
        scheduler._last_lr = [group['lr']
                              for group in scheduler.optimizer.param_groups]

    return scheduler

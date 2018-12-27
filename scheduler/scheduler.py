from torch.optim import lr_scheduler

import numpy as np


def steplr(optimizer, config):
    step_size = config.SCHEDULER.STEP_SIZE
    gamma = config.SCHEDULER.GAMMA
    return lr_scheduler.StepLR(optimizer, 
                               step_size=step_size,
                               gamma=gamma)


def exponential(optimizer, config):
    gamma = config.SCHEDULER.GAMMA
    return lr_scheduler.ExponentialLR(optimizer, 
                                      gamma=gamma)


def constant(optimizer, config):
    return lr_scheduler.StepLR(optimizer, 
                               step_size=1e+8,
                               gamma=1)


def CosineAnnealingLR(optimizer, config):
    T_max = config.SCHEDULER.T_MAX
    eta_min = config.SCHEDULER.ETA_MIN
    return lr_scheduler.CosineAnnealingLR(optimizer, 
                                          T_max=T_max,
                                          eta_min=eta_min)


class CosineAnnealingLRWithRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1, factor=1.0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 0
        self._updated_cycle_len = T_max
        self._initialized = False
        super(CosineAnnealingLRWithRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


def get_optimizer(optimizer, config):
    funcs = {
        'CALR': CosineAnnealingLR,
        'CALRWR': CosineAnnealingLRWithRestarts,
    }

    name = config.OPTIMIZER.NAME

    if name in funcs:
        func = funcs[name]
    else:
        func = globals()[name]

    print('get optimizer: {}'.format(name))
    return func(optimizer, config)

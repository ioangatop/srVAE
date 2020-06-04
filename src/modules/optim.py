from torch.optim.lr_scheduler import _LRScheduler


# ----- Scheduler -----

class LowerBoundedExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, lower_bound, last_epoch=-1):
        self.gamma = gamma
        self.lower_bound = lower_bound
        super(LowerBoundedExponentialLR, self).__init__(optimizer, last_epoch)

    def _get_lr(self, base_lr):
        lr = base_lr * self.gamma ** self.last_epoch
        if lr < self.lower_bound:
            lr = self.lower_bound
        return lr

    def get_lr(self):
        return [self._get_lr(base_lr)
                for base_lr in self.base_lrs]


if __name__ == "__main__":
    pass

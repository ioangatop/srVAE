import torch
import torch.nn as nn


class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_p(self, input, **kwargs):
        return self.forward(z)

    def forward(self, input, **kwargs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


if __name__ == "__main__":
    pass

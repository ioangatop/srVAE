import math
import torch


class StandardNormal:
    def __init__(self, z_shape):
        self.z_shape = z_shape

    def sample(self, n_samples=1, **kwargs):
        return torch.randn((n_samples, *self.z_shape))

    def log_p(self, z, **kwargs):
        return self.forward(z)

    def forward(self, z, **kwargs):
        """ Outputs the log p(z).
        """
        log_probs = z.pow(2) + math.log(math.pi * 2.)
        log_probs = -0.5 * log_probs.view(z.size(0), -1).sum(dim=1)
        return log_probs

    def __call__(self, z, **kwargs):
        return self.forward(z, **kwargs)

    def __str__(self):
      return "StandardNormal"


if __name__ == "__main__":
    pass

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


from .prior import Prior
from src.modules.nn_layers import *
from src.modules.distributions import *
from src.utils import args


# Modified vertion of: https://github.com/divymurli/VAEs

class MixtureOfGaussians(Prior):
    def __init__(self, z_shape, num_mixtures=1000):
        super().__init__()
        self.z_shape = z_shape
        self.z_dim = np.prod(z_shape)
        self.k = num_mixtures

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim).to(args.device)
                                / np.sqrt(self.k * self.z_dim))

        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(self.k).to(args.device) / self.k,
                                    requires_grad=False)

    def sample_gaussian(self, m, v):
        """ Element-wise application reparameterization trick to sample from Gaussian
        """
        sample = torch.randn(m.shape).to(args.device)
        z = m + (v**0.5)*sample
        return z

    def log_sum_exp(self, x, dim=0):
        """ Compute the log(sum(exp(x), dim)) in a numerically stable manner
        """
        max_x = torch.max(x, dim)[0]
        new_x = x - max_x.unsqueeze(dim).expand_as(x)
        return max_x + (new_x.exp().sum(dim)).log()

    def log_mean_exp(self, x, dim):
        """ Compute the log(mean(exp(x), dim)) in a numerically stable manner
        """
        return self.log_sum_exp(x, dim) - np.log(x.size(dim))

    def log_normal(self, x, m, v):
        """ Computes the elem-wise log probability of a Gaussian and then sum over the
            last dim. Basically we're assuming all dims are batch dims except for the
            last dim.
        """
        const   = -0.5 * x.size(-1) * torch.log(2*torch.tensor(np.pi))
        log_det = -0.5 * torch.sum(torch.log(v), dim = -1)
        log_exp = -0.5 * torch.sum((x - m)**2/v, dim = -1)

        log_prob = const + log_det + log_exp
        return log_prob

    def log_normal_mixture(self, z, m, v):
        """ Computes log probability of a uniformly-weighted Gaussian mixture.
        """
        z = z.view(z.shape[0], 1, -1)
        log_probs = self.log_normal(z, m, v)
        log_prob = self.log_mean_exp(log_probs, 1)
        return log_prob

    def gaussian_parameters(self, h, dim=-1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def sample(self, n_samples=1, **kwargs):
        idx = torch.distributions.categorical.Categorical(self.pi).sample((n_samples,))
        m, v = self.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        m, v = m[idx], v[idx]
        z_samples = self.sample_gaussian(m, v)
        return z_samples.view(z_samples.shape[0], *self.z_shape)

    def log_p(self, z, **kwargs):
        return self.forward(z)

    def forward(self, z, dim=None, **kwargs):
        """
        Computes the mixture of Gaussian prior
        """
        m, v  = self.gaussian_parameters(self.z_pre, dim=1)
        log_p_z = self.log_normal_mixture(z, m, v)
        return log_p_z

    def __str__(self):
      return "MixtureOfGaussians"


if __name__ == "__main__":
    pass

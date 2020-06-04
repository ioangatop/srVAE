import math
import numpy as np

import torch
import torch.nn.functional as F

from src.utils import args


NMIX = 10


# ----- Helpers -----

def n_embenddings(nc, distribution=args.likelihood):

    if distribution == 'dmol':
        nmix = NMIX
        n_emb = (nc * 3 + 1) * nmix
    else:
        raise NotImplementedError
    return n_emb


def logsumexp(x, dim=None):
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


# ----- Gaussian Distribution -----

def log_normal_diag(z, z_mu, z_logvar):
    eps = 1e-12
    log_probs = z_logvar + (z - z_mu).pow(2).div(z_logvar.exp() + eps) + math.log(math.pi * 2.)
    log_probs = -0.5 * log_probs.view(z.size(0), -1).sum(dim=1)
    return log_probs


def log_normal_std(z):
    log_probs = z.pow(2) + math.log(math.pi * 2.)
    log_probs = -0.5 * log_probs.view(z.size(0), -1).sum(dim=1)
    return log_probs


# ----- Mix of Discretized logistic -----

def dmol_loss(x, output, nc=3, nmix=NMIX, nbits=8):
    """ Discretized mix of logistic distributions loss """
    bits = 2. ** nbits
    scale_min, scale_max = [0., 1.]

    bin_size   = (scale_max - scale_min) / (bits - 1.)
    eps        = 1e-12

    # unpack values
    batch_size, nmix, H, W = output[:, :nmix].size()
    logit_probs = output[:, :nmix]
    means       = output[:, nmix:(nc + 1) * nmix].view(batch_size, nmix, nc, H, W)
    logscales   = output[:, (nc + 1) * nmix:(nc * 2 + 1) * nmix].view(batch_size, nmix, nc, H, W)
    coeffs      = output[:, (nc * 2 + 1) * nmix:(nc * 2 + 4) * nmix].view(batch_size, nmix, nc, H, W)

    # activation functions and resize
    logit_probs = F.log_softmax(logit_probs, dim=1)
    logscales   = logscales.clamp(min=-7.)
    coeffs      = coeffs.tanh()

    x = x.unsqueeze(1)
    means       = means.view(batch_size, *means.size()[1:])
    logscales   = logscales.view(batch_size, *logscales.size()[1:])
    coeffs      = coeffs.view(batch_size, *coeffs.size()[1:])
    logit_probs = logit_probs.view(batch_size, *logit_probs.size()[1:])

    # channel-wise conditional modelling sub-pixels
    mean0 = means[:, :, 0]
    mean1 = means[:, :, 1] + coeffs[:, :, 0] * x[:, :, 0]
    mean2 = means[:, :, 2] + coeffs[:, :, 1] * x[:, :, 0] + coeffs[:, :, 2] * x[:, :, 1]
    means = torch.stack([mean0, mean1, mean2], dim=2)

    # compute log CDF for the normal cases (lower < x < upper)
    x_plus    = torch.exp(-logscales) * (x - means + 0.5 * bin_size)
    x_minus   = torch.exp(-logscales) * (x - means - 0.5 * bin_size)
    cdf_delta = torch.sigmoid(x_plus) - torch.sigmoid(x_minus)
    log_cdf_mid  = torch.log(cdf_delta.clamp(min=eps))

    # Extreme Case #1: x > upper (before scaling)
    upper      = scale_max - 0.5 * bin_size
    mask_upper = x.le(upper).float()
    log_cdf_up = - F.softplus(x_minus)

    # Extreme Case #2: x < lower (before scaling)
    lower       = scale_min + 0.5 * bin_size
    mask_lower  = x.ge(lower).float()
    log_cdf_low = x_plus - F.softplus(x_plus)

    # Extreme Case #3: probability on a sub-pixel is below 1e-5
    #   --> If the probability on a sub-pixel is below 1e-5, we use an approximation
    #       based on the assumption that the log-density is constant in the bin of
    #       the observed sub-pixel value
    x_in            = torch.exp(-logscales) * (x - means)
    mask_delta      = cdf_delta.gt(1e-5).float()
    log_cdf_approx  = x_in - logscales - 2. * F.softplus(x_in) + np.log(bin_size)

    # Compute log CDF w/ extrime cases
    log_cdf = log_cdf_mid * mask_delta + log_cdf_approx * (1.0 - mask_delta)
    log_cdf = log_cdf_low * (1.0 - mask_lower) + log_cdf * mask_lower
    log_cdf = log_cdf_up  * (1.0 - mask_upper) + log_cdf * mask_upper

    # Compute log loss
    loss = logsumexp(log_cdf.sum(dim=2) + logit_probs, dim=1)
    return loss.view(loss.shape[0], -1).sum(1)


def sample_from_dmol(x_mean, nc=3, nmix=NMIX, random_sample=False):
    """ Sample from Discretized mix of logistic distribution """
    scale_min, scale_max = [0., 1.]

    # unpack values
    logit_probs = x_mean[:, :nmix]                                                                      # pi
    batch_size, nmix, H, W = logit_probs.size()
    means     = x_mean[:, nmix:(nc + 1) * nmix].view(batch_size, nmix, nc, H, W)                        # mean
    logscales = x_mean[:, (nc + 1) * nmix:(nc * 2 + 1) * nmix].view(batch_size, nmix, nc, H, W)         # log_var
    coeffs    = x_mean[:, (nc * 2 + 1) * nmix:(nc * 2 + 4) * nmix].view(batch_size, nmix, nc, H, W)     # chan_coeff

    # activation functions
    logscales   = logscales.clamp(min=-7.)
    logit_probs = F.log_softmax(logit_probs, dim=1)
    coeffs      = coeffs.tanh()

    # sample mixture
    index       = logit_probs.argmax(dim=1, keepdim=True) + logit_probs.new_zeros(means.size(0), *means.size()[2:]).long()
    one_hot     = means.new_zeros(means.size()).scatter_(1, index.unsqueeze(1), 1)
    means       = (means * one_hot).sum(dim=1)
    logscales   = (logscales * one_hot).sum(dim=1)
    coeffs      = (coeffs * one_hot).sum(dim=1)
    x           = means

    if random_sample:
        # sample y from CDF
        u = means.new_zeros(means.size()).uniform_(1e-5, 1 - 1e-5)
        # from y map it to the corresponing x
        x = x + logscales.exp() * (torch.log(u) - torch.log(1.0 - u))

    # concat image channels
    x0 = (x[:, 0]).clamp(min=scale_min, max=scale_max)
    x1 = (x[:, 1] + coeffs[:, 0] * x0).clamp(min=scale_min, max=scale_max)
    x2 = (x[:, 2] + coeffs[:, 1] * x0 + coeffs[:, 2] * x1).clamp(min=scale_min, max=scale_max)
    x = torch.stack([x0, x1, x2], dim=1)
    return x


if __name__ == "__main__":
    pass

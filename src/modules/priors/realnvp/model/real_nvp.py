import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .coupling_layer import CouplingLayer, MaskType
from ..util import squeeze_2x2
from ..distributions import StandardNormal

# Modified vertion of: https://github.com/chrischute/real-nvp

class RealNVP(nn.Module):
    """RealNVP Model
    Codebase from Chris Chute:
    https://github.com/chrischute/real-nvp

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, input_shape, mid_channels=64, num_blocks=5, num_scales=2, prior='std_normal'):
        super().__init__()
        self.flows = _RealNVP(0, num_scales, input_shape[0], mid_channels, num_blocks)

        # self.nbits = 8.
        if prior=='std_normal':
            self.prior = StandardNormal(input_shape)
        elif prior=='mog':
            self.prior = MixtureOfGaussians(input_shape)


    @torch.no_grad()
    def sample(self, z_shape, n_samples, device, **kwargs):
        """Sample from RealNVP model.
        Args:
            z_shape (tuple): 
            n_samples (int): Number of samples to generate.
            device (torch.device): Device to use.
        """
        z = self.prior.sample(n_samples).to(device)
        x, _ = self.forward(z, reverse=True)
        return x


    def log_p(self, x, **kwargs):
        """ returns the log likelihood.
        """
        z, sldj = self.forward(x, reverse=False)
        ll = (self.prior.log_p(z) + sldj)


        # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # prior_ll = prior_ll.flatten(1).sum(-1) - np.log(2**self.nbits) * np.prod(z.size()[1:])
        # ll = prior_ll + sldj
        # ll = ll.mean()

        return ll


    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            sldj = 0    # we do not quintize !
            #  quintize !
            # x = (x * (2**self.nbits - 1) + torch.rand_like(x)) / (2**self.nbits)

        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj



class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)


    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj

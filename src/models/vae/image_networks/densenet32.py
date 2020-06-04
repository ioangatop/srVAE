import torch.nn as nn
import torch.nn.functional as F

from src.utils.args import args
from src.modules.nn_layers import *
from src.modules.distributions import n_embenddings


class q_z(nn.Module):
    """ Encoder q(z|x)
    """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = 2 * output_shape[0]

        self.core_nn = nn.Sequential(
            DenselyEncoder(
                in_channels=nc_in,
                out_channels=nc_out,
                growth_rate=nc_out//2,
                steps=8,
                scale_factor=2)
        )

    def forward(self, input):
        mu, logvar = self.core_nn(input).chunk(2, 1)
        return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)


class p_x(nn.Module):
    """ Dencoder p(x|z)
    """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = n_embenddings(output_shape[0])

        self.core_nn = nn.Sequential(
            DenselyDecoder(
                in_channels=nc_in,
                out_channels=nc_out,
                growth_rate=128,
                steps=6,
                scale_factor=2)
        )

    def forward(self, input):
        logits = self.core_nn(input)
        return logits


if __name__ == "__main__":
    pass

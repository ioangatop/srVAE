from functools import partial

import numpy as np

import torch
import torch.nn as nn

from src.utils import args, get_shape
from src.modules import *


# ----- NN Model Seleciton -----

if args.model == 'VAE':
    if args.network == 'densenet32':
        from .image_networks.densenet32 import *
    else:
        raise NotImplementedError("Please use 'densenet32' as 'network' argument.")


# ----- Variational AutoEncoder -----

class VAE(nn.Module):
    """
    Variational AutoEncoder.

    Author:
    Ioannis Gatopoulos.
    """
    def __init__(self, x_shape, prior=args.prior):
        super().__init__()
        self.x_shape = x_shape

        self.z_dim = args.z_dim
        self.z_shape = get_shape(self.z_dim)

        # p(z)
        self.p_z = globals()[prior](self.z_shape)

        # q(z | x)
        self.q_z = q_z(self.z_shape, self.x_shape)

        # p(x | z)
        self.p_x = p_x(self.x_shape, self.z_shape)

        # likelihood distribution
        self.recon_loss = partial(dmol_loss, nc=self.x_shape[0])
        self.sample_distribution = partial(sample_from_dmol, nc=self.x_shape[0])


    def initialize(self, dataloader):
        """ Data dependent init for weight normalization 
            (Automatically done during the first forward pass).
        """
        with torch.no_grad():
            x, _ = next(iter(dataloader))
            x = x.to(args.device)
            output = self.forward(x)
            self.calculate_elbo(x, output)
        return

    @staticmethod
    def reparameterize(z_mu, z_logvar):
        """ z ~ N(z| z_mu, z_logvar)
        """
        epsilon = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5 * z_logvar) * epsilon

    @torch.no_grad()
    def generate(self, n_samples=args.n_samples):
        # u ~ p(u)
        z = self.p_z.sample(z_shape=self.z_shape, n_samples=n_samples, device=args.device).to(args.device)
        # x ~ p(x| z)
        x_logits = self.p_x(z)
        x_hat = self.sample_distribution(x_logits, random_sample=False)
        return x_hat

    @torch.no_grad()
    def reconstruct(self, x, **kwargs):
        x_logits = self.forward(x).get('x_logits')
        x_hat = self.sample_distribution(x_logits, random_sample=False)
        return x_hat

    def calculate_elbo(self, input, outputs):
        # unpack variables
        x, x_logits = input, outputs.get('x_logits')
        z_q, z_q_mean, z_q_logvar = outputs.get('z_q'), outputs.get('z_q_mean'), outputs.get('z_q_logvar')

        # Reconstraction loss
        # N E_q [ ln p(x|z) ]
        RE = - self.recon_loss(x, x_logits).mean()

        # Regularization loss
        # N E_q [ ln q(z) - ln p(z) ]
        log_p_z = self.p_z.log_p(z_q)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar)
        KL = (log_q_z - log_p_z).mean()

        # Total negative lower bound loss
        nelbo = RE + KL

        diagnostics = {
            "bpd"   : (nelbo.item()) / (np.prod(x.shape[1:]) * np.log(2.)),
            "nelbo" : nelbo.item(),
            "RE"    : RE.mean(dim=0).item(),
            "KL"    : KL.mean(dim=0).item(),
        }
        return nelbo, diagnostics


    def forward(self, x, **kwargs):
        """ Forward pass through the inference and the generative model.
        """
        # z ~ q(z| x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        # x ~ p(x| z)
        x_logits = self.p_x(z_q)
        return {
            "z_q"        : z_q,
            "z_q_mean"   : z_q_mean,
            "z_q_logvar" : z_q_logvar,

            "x_logits"   : x_logits
        }


if __name__ == "__main__":
    pass

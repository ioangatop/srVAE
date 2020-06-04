from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from src.utils import args, get_shape
from src.modules import *


# ----- NN Model Seleciton -----

if args.model == 'srVAE':
    if args.network == 'densenet16x32':
        from .image_networks.densenet16x32 import *
    else:
        raise NotImplementedError("Please use 'densenet16x32' as 'network' argument.")


# ----- Two Staged VAE -----

class srVAE(nn.Module):
    """
    Super-Resolution Variational Auto-Encoder (srVAE).
    A Two Staged Visual Processing Variational AutoEncoder.

    Author:
    Ioannis Gatopoulos.
    """
    def __init__(self, x_shape, y_shape=(3, 16, 16), u_dim=args.u_dim, z_dim=args.z_dim, prior=args.prior, device=args.device):
        super().__init__()
        self.device = args.device
        self.x_shape = x_shape
        self.y_shape = (x_shape[0], y_shape[1], y_shape[2])

        self.u_shape = get_shape(u_dim)
        self.z_shape = get_shape(z_dim)

        # q(y|x): deterministic "compressed" transformation
        self.compressed_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.y_shape[1], self.y_shape[2])),
            transforms.ToTensor()
        ])

        # p(u)
        self.p_u = globals()[prior](self.u_shape)

        # q(u | y)
        self.q_u = q_u(self.u_shape, self.y_shape)

        # p(z | y)
        self.p_z = p_z(self.z_shape, (self.y_shape, self.u_shape))

        # q(z | x)
        self.q_z = q_z(self.z_shape, self.x_shape)

        # p(y | u)
        self.p_y = p_y(self.y_shape, self.u_shape)

        # p(x | y, z)
        self.p_x = p_x(self.x_shape, (self.y_shape, self.z_shape))

        # likelihood distribution
        self.recon_loss = partial(dmol_loss)
        self.sample_distribution = partial(sample_from_dmol)


    def compressed_transoformation(self, input):
        y = []
        for x in input:
            y.append(self.compressed_transform(x.cpu()))
        return torch.stack(y).to(self.device)


    def initialize(self, dataloader):
        """ Data dependent init for weight normalization 
            (Automatically done during the first forward pass).
        """
        with torch.no_grad():
            x, _ = next(iter(dataloader))
            x = x.to(self.device)
            output = self.forward(x)
            self.calculate_elbo(x, output)
        return


    @staticmethod
    def reparameterize(z_mean, z_log_var):
        """ z ~ N(z| z_mu, z_logvar) """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon


    @torch.no_grad()
    def generate(self, n_samples=20):
        # u ~ p(u)
        u = self.p_u.sample(self.u_shape, n_samples=n_samples, device=self.device).to(self.device)

        # p(y|u)
        y_logits = self.p_y(u)
        y_hat = self.sample_distribution(y_logits, nc=self.y_shape[0])

        # z ~ p(z|y, u)
        z_p_mean, z_p_logvar = self.p_z((y_hat, u))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_logits = self.p_x((y_hat, z_p))
        x_hat = self.sample_distribution(x_logits, nc=self.x_shape[0])
        return x_hat, y_hat


    @torch.no_grad()
    def reconstruct(self, x, **kwargs):
        outputs = self.forward(x)
        y_hat = self.sample_distribution(outputs.get('y_logits'), nc=self.y_shape[0])
        x_hat = self.sample_distribution(outputs.get('x_logits'), nc=self.x_shape[0])
        return outputs.get('y'), y_hat, x_hat


    @torch.no_grad()
    def super_resolution(self, y):
        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ p(z|y)
        z_p_mean, z_p_logvar = self.p_z((y, u_q))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_logits = self.p_x((y, z_p))
        x_hat = self.sample_distribution(x_logits)
        return x_hat


    def calculate_elbo(self, x, outputs, **kwargs):
        # unpack variables
        y, x_logits, y_logits = outputs.get('y'), outputs.get('x_logits'), outputs.get('y_logits')
        u_q, u_q_mean, u_q_logvar = outputs.get('u_q'), outputs.get('u_q_mean'), outputs.get('u_q_logvar')
        z_q, z_q_mean, z_q_logvar = outputs.get('z_q'), outputs.get('z_q_mean'), outputs.get('z_q_logvar')
        z_p_mean, z_p_logvar = outputs.get('z_p_mean'), outputs.get('z_p_logvar')

        # Reconstraction loss
        RE_x = self.recon_loss(x, x_logits, nc=self.x_shape[0])
        RE_y = self.recon_loss(y, y_logits, nc=self.y_shape[0])

        # Regularization loss
        log_p_u = self.p_u.log_p(u_q, dim=1)
        log_q_u = log_normal_diag(u_q, u_q_mean, u_q_logvar)
        KL_u = log_q_u - log_p_u

        log_p_z = log_normal_diag(z_q, z_p_mean, z_p_logvar)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar)
        KL_z = log_q_z - log_p_z

        # Total lower bound loss
        nelbo = - (RE_x + RE_y - KL_u - KL_z).mean()

        diagnostics = {
            "bpd"   : (nelbo.item()) / (np.prod(x.shape[1:]) * np.log(2.)),
            "nelbo" : nelbo.item(),

            "RE"    : - (RE_x + RE_y).mean().item(),
            "RE_x"  : - RE_x.mean().item(),
            "RE_y"  : - RE_y.mean().item(),

            "KL"    : (KL_z + KL_u).mean().item(),
            "KL_u"  : KL_u.mean().item(),
            "KL_z"  : KL_z.mean().item(),
        }
        return nelbo, diagnostics


    def forward(self, x, **kwargs):
        """ Forward pass through the inference and the generative model. """
        # y ~ f(x) (determinist)
        y = self.compressed_transoformation(x)

        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ q(z| x, y)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x ~ p(x| y, z)
        x_logits = self.p_x((y, z_q))

        # y ~ p(y| u)
        y_logits = self.p_y(u_q)

        # z ~ p(z| x)
        z_p_mean, z_p_logvar = self.p_z((y, u_q))

        return {
            'u_q_mean'   : u_q_mean,
            'u_q_logvar' : u_q_logvar,
            'u_q'        : u_q,

            'z_q_mean'   : z_q_mean,
            'z_q_logvar' : z_q_logvar,
            'z_q'        : z_q,

            'z_p_mean'   : z_p_mean,
            'z_p_logvar' : z_p_logvar,

            'y'          : y,
            'y_logits'   : y_logits,

            'x_logits'   : x_logits
        }


if __name__ == "__main__":
    pass

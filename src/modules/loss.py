import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from src.modules.distributions import logsumexp

# ----- Loss Function -----

class ELBOLoss(_Loss):
    """
    Computes negative ELBO loss and diagnostics.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, outputs, model):
        elbo = model.module.calculate_elbo if isinstance(model, nn.DataParallel) else model.calculate_elbo
        return elbo(input, outputs)

# ----- NLL -----

@torch.no_grad()
def calculate_nll(model, test_loader, criterion, args, iw_samples):
    """
        model:
        test_loader:
        iw_samples: Number of Importance Weighting samples used for approximating log-likelihood.
    """
    model.eval()

    # get data shape
    img, _ = next(iter(test_loader))
    img_shape = img.shape[1:]

    likelihood_test = []
    for i, (x_imgs, _) in enumerate(test_loader):
        iw_array = []
        for _ in range(iw_samples):
            # forward pass
            x_imgs = x_imgs.to(args.device)
            output = model(x_imgs)
            nelbo, _ = criterion(x_imgs, output, model)
            iw_array.append(nelbo.item())

        # calculate max if importance weighting samples
        nll_x = - logsumexp(torch.tensor(iw_array))
        likelihood_test.append(nll_x + np.log(len(iw_array)))
        print(i, '/', len(test_loader))

    # calculate full negative log-likelihood
    nll = - torch.tensor(likelihood_test).mean().item()
    bpd = nll / (np.prod(img_shape) * np.log(2.))
    return bpd

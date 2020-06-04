import os
import numpy as np
from argparse import Namespace
from markdown import markdown

import torch
import torch.nn as nn


from .args import args


# ----- Random Seed Control -----

def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


# ----- Print Parser -----

def print_args(ARGS):
    print('\n'+26*'='+' Configuration '+26*'=')
    for name, var in vars(ARGS).items():
        print('{} : {}'.format(name, var))
    print('\n'+25*'='+' Training Starts '+25*'='+'\n')


# ----- Data Shapes -----

def get_data_shape(data_loader):
    data_shape = tuple(next(iter(data_loader))[0].shape[1:])
    return data_shape


def get_shape(z_dim):
    """ Given the dimentionality of the latent space,
        re-shape it to an appropriate 3-D tensor.
    """
    d = 8
    if (z_dim%d==0) and (z_dim // (d*d) > 0):  # cx8x8
        H = W = d
        C = z_dim // (d*d)
        return (C, H, W)
    raise "Latent space can not mapped to a 3-D tensor. \
            Please choose another dimentionality (power of 2)."


# ----- Logging -----

def log_interval(i, len_data_loader, acc_losses, enable=args.log_interval):
    if args.log_interval:
        print('{:6d}/{:4d} batch | BPD: {:4.2f}, RE: {:4.2f}, KL: {:4.2f}'.format(
            i, len_data_loader, acc_losses['bpd']/i, acc_losses['RE']/i, acc_losses['KL']/i), end='\r')


def logging(epoch, train_losses, valid_losses, is_saved, writer=None):
    if writer is not None:
        for loss in train_losses:
            writer.add_scalar('Train Loss/' + loss, train_losses[loss], epoch)

        for loss in valid_losses:
            writer.add_scalar('Validation Loss/' + loss, valid_losses[loss], epoch)

    if is_saved:
        print('Epoch [{:4d}/{:4d}] | Train bpd: {:4.2f} | Val bpd: {:4.2f} * '.format(
            epoch, args.epochs, train_losses['bpd'], valid_losses['bpd']))
    else:
        print('Epoch [{:4d}/{:4d}] | Train bpd: {:4.2f} | Val bpd: {:4.2f} '.format(
            epoch, args.epochs, train_losses['bpd'], valid_losses['bpd']))

# ----- Parameters Counting -----

def get_params(module):
    try:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    except AttributeError:
        return 0


def n_parameters(model, writer=None):
    model = model.module if isinstance(model, nn.DataParallel) else model

    params_dict = {
        "total" : get_params(model)
    }

    if writer:
        writer.add_text('n_params', namespace2markdown(Namespace(**params_dict), title='Networks', values='Params'))    

    print(f'# Total Number of Parameters: {params_dict["total"] / 1e6:.3f}M')


# ----- Save and Load Model -----

min_loss = None
def save_model(model, optimizer, loss, epoch, pth='./src/models/'):
    """ Saves a torch model in two ways: to be retrained and/or for validation only.
    """
    global min_loss
    if min_loss is not None and loss > min_loss:
        return False
    min_loss = loss

    # create paths
    pth = os.path.join(pth, 'pretrained', args.model)
    pth = os.path.join(pth, args.dataset)
    pth_inf = os.path.join(pth, 'inference')
    pth_train = os.path.join(pth, 'trainable')
    for p in [pth, pth_inf, pth_train]:
        if not os.path.exists(p):
            os.makedirs(p)

    m = model.module if isinstance(model, nn.DataParallel) else model

    # model to be used only for Inference
    torch.save(m.state_dict(), pth_inf + '/model.pth')

    # model to be used both for Inference and Resuming Training
    torch.save({
                'epoch': epoch,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, pth_train + '/model.pth')

    return True


# ----- Tensorboard Utils -----

def namespace2markdown(args, title='Hyperparameter', values='Values'):
    txt = '<table> <thead> <tr> <td> <strong> ' + title + ' </strong> </td> <td> <strong> ' + values + ' </strong> </td> </tr> </thead>'
    txt += ' <tbody> '
    for name, var in vars(args).items():
        txt += '<tr> <td> <code>' + str(name) + ' </code> </td> ' + '<td> <code> ' + str(var) + ' </code> </td> ' + '<tr> '
    txt += '</tbody> </table>'
    return markdown(txt)


if __name__ == "__main__":
    pass

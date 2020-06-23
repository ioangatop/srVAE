import torch
import argparse

# ----- Parser -----

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    # Dataset
    PARSER.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['CIFAR10', 'CelebA', 'Imagenette', 'ImageNet32', 'ImageNet64'],
                        help="Data to be used.")
    PARSER.add_argument('--img_resize', default=32, type=int,
                        help='Change image resolution.')

    # Model
    PARSER.add_argument('--model', default='VAE', type=str,
                        choices=['VAE', 'srVAE'],
                        help="Model to be used.")
    PARSER.add_argument('--network', default='densenet32', type=str,
                        choices=['densenet32', 'densenet16x32'],
                        help="Neural Network architecture to be used.")

    # Prior
    PARSER.add_argument('--prior', default='MixtureOfGaussians', type=str,
                        choices=['StandardNormal', 'MixtureOfGaussians', 'RealNVP'],
                        help='Prior type.')
    PARSER.add_argument('--z_dim', default=1024, type=int,
                        help='Dimensionality of z latent space.')
    PARSER.add_argument('--u_dim', default=1024, type=int,
                        help='Dimensionality of z latent space.')

    # data likelihood
    PARSER.add_argument('--likelihood', default='dmol', type=str,
                        choices=['dmol'],
                        help="Type of likelihood.")
    PARSER.add_argument('--iw_test', default=512, type=int,
                        help="Number of Importance Weighting samples used for approximating the test log-likelihood.")

    # Training Parameters
    PARSER.add_argument('--batch_size', default=32, type=int,
                        help='Batch size.')
    PARSER.add_argument('--epochs', default=2000, type=int,
                        help='Number of training epochs.')

    # General Configs
    PARSER.add_argument('--seed', default=None, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--n_samples', default=8, type=int,
                        help='Number of generated samples.')
    PARSER.add_argument('--log_interval', default=True, type=bool,
                        help='Print progress on every batch.')
    PARSER.add_argument('--device', default=None, type=str,
                        choices=['cpu', 'cuda'],
                        help='Device to run the experiment.')

    PARSER.add_argument('--use_tb', default=True, type=bool,
                        help='Use TensorBoard.')
    PARSER.add_argument('--tags', default='logs', type=str,
                        help='Run tags.')


    ARGS = PARSER.parse_args()

    # Check device
    if ARGS.device is None:
        ARGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ARGS


args = parser()


if __name__ == "__main__":
    pass

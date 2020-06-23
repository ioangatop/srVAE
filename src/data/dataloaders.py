import os
import math
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.utils import args
from .datasets import CIFAR10, CelebA, Imagenette, ImageNet32, ImageNet64


ROOT = './data_root/'


# ----- Dataset Splitter -----

def get_samplers(num_train, valid_size):
    use_percentage=True if isinstance(valid_size, float) else False

    # obtain training indices that will be used for validation
    indices = list(range(num_train))
    if use_percentage:
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
    else:
        train_idx, valid_idx = indices[:-valid_size], indices[-valid_size:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


# ----- Data Transformations -----

def data_transformations(dataset):
    if dataset in ['CIFAR10', 'Imagenette', 'ImageNet32', 'ImageNet64']:
        res = args.img_resize if args.img_resize is not None else 32

        train_transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(int(math.ceil(res * 0.05)), padding_mode='edge'),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.CenterCrop(res),
            transforms.ToTensor()
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor()
        ])

    elif dataset in ['CelebA']:
        Crop = transforms.Lambda(lambda x: transforms.functional.crop(x, 40, 15, 148, 148))
        res = args.resolution

        train_transform = valid_transform = transforms.Compose([
            Crop,
            transforms.Resize((res, res)),
            transforms.ToTensor()
        ])

    else:
        raise NotImplementedError

    return train_transform, valid_transform


# ----- DataLoader -----

def dataloader(dataset=args.dataset, data_root=ROOT, batch_size=args.batch_size, num_workers=6, pin_memory=True):
    # dataset and data loader kwargs
    kwargs = {} if args.device == 'cpu' else {'num_workers': num_workers, 'pin_memory': pin_memory}
    dataset_kwargs = {'root':os.path.join(data_root, dataset), 'download':True}
    loader_kwargs = {'batch_size':batch_size, **kwargs}

    # get data transformation
    train_transform, valid_transform = data_transformations(dataset)

    # build datasets
    train_data = globals()[dataset](train=True,  transform=train_transform, **dataset_kwargs)
    valid_data = globals()[dataset](train=True,  transform=valid_transform, **dataset_kwargs)
    test_data  = globals()[dataset](train=False, transform=valid_transform, **dataset_kwargs)

    # define samplers for obtaining training and validation batches
    train_sampler, valid_sampler = get_samplers(len(train_data), 0.15)

    # Build dataloaders
    train_loader = DataLoader(train_data, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_data, sampler=valid_sampler, **loader_kwargs)
    test_loader  = DataLoader(test_data,  shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    pass

import os
import shutil
import tarfile
import requests

from PIL import Image
from torch.utils.data import Dataset

# pytorch datasets
import torchvision
from torchvision.datasets import CIFAR10


# ------- CelebA -------

class CelebA(torchvision.datasets.CelebA):
    """ Modified CelebA; replaced 'split' with bool 'train' variable. """
    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    def __init__(self, root, train=True, target_type="attr", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        
        self.split = "train" if train==True else "test"
        split = split_map[self.split]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


# ------- Imagenette -------

class Imagenette(Dataset):
    """ Imagenette: Natural Image Dataset. """
    data_size = (106551014, 205511341)  # (image folder, image folder + .tar file)
    resources = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    dir2classes = {'n01440764':'tench', 'n02102040':'English springer', 'n02979186':'cassette player', 
                   'n03000684': 'chain saw', 'n03028079': 'church', 'n03394916': 'French horn', 
                   'n03417042': 'garbage truck', 'n03425413': 'gas pump', 'n03445777': 'golf ball',
                   'n03888257': 'parachute'}

    def __init__(self, root, transform, download, train=True, **kwargs):
        super().__init__()
        self.root = os.path.join(root, 'imagenette')
        self.transform = transform

        # class attributes
        self.img_dir = ''
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if download:
            self.download()
        else:
            if not self.dataset_exists():
                raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if train:
            self.img_dir = os.path.join(self.root, 'imagenette2-160', 'train')
        else:
            self.img_dir = os.path.join(self.root, 'imagenette2-160', 'val')

        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(self.img_dir)
        self.init_dataset()

    @staticmethod
    def extract(tar_url, extract_path='.'):
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)
            if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
                extract(item.name, "./" + item.name[:item.name.rfind('/')])

    def dataset_exists(self, eps=1):
        """ Check if folder exists via folder size. """
        if not os.path.exists(self.root):
            return False

        total_size = 0
        for path, _, files in os.walk(self.root):
            for f in files:
                fp = os.path.join(path, f)
                total_size += os.path.getsize(fp)

        size1 = int(self.data_size[0]/1000000)
        size2 = int(self.data_size[1]/1000000)
        total_size = int(total_size/1000000)
        return (size1-eps <= total_size <= size1+eps) or (size2-eps <= total_size <= size2+eps)

    def download(self):
        if self.dataset_exists():
            print('Files already downloaded and verified')
            return

        # create root folder
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # download dataset
        print('{:<2} {:<4}'.format('-->', 'Downloading dataset...'))
        local_filename = os.path.join(self.root, self.resources.split('/')[-1])
        with requests.get(self.resources, stream=True) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print('{:<2} {:<4}'.format('-->', 'Downloading Complite!'))

        # extract it
        print('{:<2} {:<4}'.format('-->', 'Extracting images...'))
        self.extract(os.path.join(self.root, 'imagenette2-160.tgz'), self.root)
        print('{:<2} {:<4}'.format('-->', 'Extracting Complite!'))

    def _find_classes(self, dir):
        classes = [self.dir2classes[d.name] for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        return classes, class_to_idx, idx_to_class    

    def init_dataset(self):
        self.samples = []
        for class_dir in os.listdir(self.img_dir):
            for img in os.listdir(os.path.join(self.img_dir, class_dir)):
                if img.endswith('.JPEG'):
                    self.samples.append((os.path.join(class_dir, img), self.dir2classes[class_dir]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(self.img_dir, self.samples[index][0])).convert('RGB')
        target  = self.class_to_idx[self.samples[index][1]]

        if self.transform is not None:
            img = self.transform(img_pil)

        return img, target


# ------- ImageNet32 -------

class ImageNet32(Dataset):
    """ ImageNet32 dataset. """
    resources = 'http://www.image-net.org/small/download.php'
    def __init__(self, root, transform, train=True, **kwargs):
        super().__init__()
        root = '/mnt/data/DATASETS/unsupervised-learning-datasets/ImageNet64/'

        img_dir = 'train_32x32' if train else 'valid_32x32'
        self.root = os.path.join(root, img_dir)
        self.transform = transform

        self.init_dataset()

    def init_dataset(self):
        self.samples = []
        for img in os.listdir(self.root):
            if img.endswith('.png'):
                self.samples.append(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(self.root, self.samples[index]))

        if self.transform is not None:
            img = self.transform(img_pil)

        return img, 0


# ------- ImageNet64 -------

class ImageNet64(Dataset):
    """ ImageNet64 dataset. """
    resources = 'http://www.image-net.org/small/download.php'
    def __init__(self, root, transform, train=True, **kwargs):
        super().__init__()
        root = '/mnt/data/DATASETS/unsupervised-learning-datasets/ImageNet64/'

        img_dir = 'train_64x64' if train else 'valid_64x64'
        self.root = os.path.join(root, img_dir)
        self.transform = transform

        self.init_dataset()

    def init_dataset(self):
        self.samples = []
        for img in os.listdir(self.root):
            if img.endswith('.png'):
                self.samples.append(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(self.root, self.samples[index]))

        if self.transform is not None:
            img = self.transform(img_pil)

        return img, 0


if __name__ == "__main__":
    pass

import os
import glob
from albumentations.pytorch.transforms import ToTensorV2

import cv2
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A

__all__ = ['get_cinic10_basic', 'Cinic10']
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]
CLASSES = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


def get_cinic10_basic(root: str = '~/datasets/CINIC10'):
    """Get CINIC10 dataset with official settings from github.
    Size: (3, 32, 32)
    Number train, valid and test: 90,000

    Expect folders in this format:
    root
    |__train
        |__airplane
        ...
    |__test
    |__valid
    """
    root = os.path.expanduser(root)
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'valid')
    test_dir = os.path.join(root, 'test')
    basic_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)])

    train_dataset = ImageFolder(train_dir, transform=basic_transforms)
    val_dataset = ImageFolder(val_dir, transform=basic_transforms)
    test_dataset = ImageFolder(test_dir, transform=basic_transforms)
    return train_dataset, val_dataset, test_dataset


class Cinic10(Dataset):
    """Load CINIC10 into RAM for fast processing.
    >>> Cinic10('~/datasets/CINIC10', mode='train', transforms=basic_transform)
    """
    def __init__(self, root: str, mode: str, transforms = None):
        super().__init__()
        self.transforms = transforms
        root = os.path.expanduser(root)
        self.images = []
        self.labels = []
        mode = mode.lower()

        if mode == 'train':
            img_dir = os.path.join(root, 'train')
        elif mode == 'valid':
            img_dir = os.path.join(root, 'valid')
        elif mode == 'test':
            img_dir = os.path.join(root, 'test')
        else:
            raise ValueError(
                f'mode should be in `train`, `valid`, or `test`, your mode: {mode}')

        for k in CLASSES.keys():
            search_glob = os.path.join(img_dir, k, '*.png')
            img_dirs = glob.glob(search_glob)
            for i in img_dirs:
                img = cv2.imread(i, cv2.IMREAD_COLOR)
                assert img is not None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Keep every image to RAM for faster access.
                self.images.append(img)
            self.labels += [CLASSES[k] for _ in img_dirs]

        assert len(self.images) == 90_000
        assert len(self.labels) == 90_000

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            basic_transform = A.Compose(
                [A.Normalize(CINIC_MEAN, CINIC_STD), ToTensorV2()])
            img = basic_transform(image=img)['image']
        return img, label

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    from tqdm import tqdm

    train_dataset, val_dataset, test_dataset = get_cinic10_basic()
    print('Testing load using ImageFolder.')
    for i, j in tqdm(train_dataset):
        pass

    print('Testing load from RAM.')
    train_dataset = Cinic10('~/datasets/CINIC10', mode='train')
    for a, b in tqdm(train_dataset):
        pass

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np



def download_VOC2012(root='data/PascalVOC_Dataset'):
    """Download the Pascal VOC2012 dataset and store the data in the data/"""
    datasets.VOCSegmentation(root, year='2012', image_set='train', download=True)
    datasets.VOCSegmentation(root, year='2012', image_set='val', download=True)                  


class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        self.img_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        self.set_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation')

        with open(os.path.join(self.set_dir, self.image_set + '.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_id + '.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return image, mask

class ToTensorTransform:
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

def get_transforms():
    return Compose([
        ResizeTransform((256, 256)),
        ToTensorTransform()
    ])

def get_VOC2012_dataset(root_dir='data/PascalVOC_Dataset'):
    # Load the train_dataset and val_dataset
    train_dataset = VOCSegmentationDataset(root_dir, image_set='train', transform=get_transforms())
    val_dataset   = VOCSegmentationDataset(root_dir, image_set='val', transform=get_transforms())

    return train_dataset, val_dataset


if __name__ == '__main__':
    # (0) Define the root directory for the Pascal VOC2012 dataset
    root_dir = 'data/PascalVOC_Dataset'
    
    # (1) Download the Pascal VOC2012 dataset
    # download_VOC2012(root=root_dir)

    # (2) For creating an instance of VOC2012 dataset class for semantic segmentation tasks:
    train_dataset, val_dataset = get_VOC2012_dataset(root_dir)
    
    # (3) Get image and mask from the train_dataset
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")


    # TODO: Investigate the image and mask pixel values, data type and how the values are stored in the tensors.
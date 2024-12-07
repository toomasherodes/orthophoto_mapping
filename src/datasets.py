import glob
import numpy as np
import torch
import albumentations as A

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_images(data_root):
    train_images = glob.glob(f'{data_root}/train/input/*')
    train_images.sort()

    train_masks = glob.glob(f'{data_root}/train/target/*')
    train_masks.sort()

    valid_images = glob.glob(f'{data_root}/validation/input/*')
    valid_images.sort()

    valid_masks = glob.glob(f'{data_root}/validation/target/*')
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def normalize():
    transform = A.Compose([
        A.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True
        )
    ])
    return transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.norm_tfms = norm_tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[index]).convert('RGB'))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        return image, mask

def train_transforms(img_size):
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return train_image_transform

def valid_transforms(img_size):
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)
    norm_tfms = normalize()

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_data_loader, valid_data_loader
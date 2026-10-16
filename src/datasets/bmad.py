import os
from pathlib import Path
from typing import *

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
import os
import sys

BMAD_CLASSES = [
    "BraTS2021_slice",
    "camelyon16_256",
    "hist_DIY",
    "OCT2017",
    "RESC",
    "Chest-RSNA"
]

NOMASK_CLASSES = [
    "camelyon16_256",
    "OCT2017",
    "Chest-RSNA"
]

class BMAD(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str,
        input_res: int, 
        split: str, 
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False,
        anom_only=False,
        normal_only=False,
        **kwargs
    ):
        """Dataset for BMAD.
        Args:
            data_root: Root directory of Liver dataset. It should contain the data directories for each class under this directory.
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
            color: rgb or grayscale
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = transform
        self.is_mask = is_mask
        self.cls_label = cls_label
        self.anom_only = anom_only
        self.normal_only = normal_only
        
        assert category in BMAD_CLASSES, f'unknown BMAD object: {category}'
        assert Path(self.data_root).exists(), f"{self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'val' or self.split == 'test'
        
        # Load files from the dataset
        self.img_files = self.get_files()
        self.labels = [0] * len(self.img_files)
        if self.split == 'test':
            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize((input_res, input_res), interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(mask_to_tensor)
                ]
            )

            self.labels = []
            for file in self.img_files:
                if "/good/" in str(file):
                    self.labels.append(0)
                else:
                    self.labels.append(1)
            self.normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.anom_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_classes = len(BMAD_CLASSES)
        self.labels_to_names = {i: name for i, name in enumerate(BMAD_CLASSES)}

    def get_files(self):
        img_files = []
        ext = "*.png" if self.category != "OCT2017" else "*.jpeg"
        
        if self.split == "train":
            img_files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob(ext))
        elif self.split == "val":
            img_files = sorted(Path(os.path.join(self.data_root, self.category, 'val', 'good')).glob(ext))
        elif self.split == "test":
            img_files =  sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob(ext))
            img_files += sorted(Path(os.path.join(self.data_root, self.category, 'test', 'Ungood')).glob(ext))
        return img_files
    
    def __getitem__(self, index):
        inputs = {}

        if self.anom_only:
            img_file = self.img_files[self.anom_indices[index]]
            label = self.labels[self.anom_indices[index]]
        elif self.normal_only:
            img_file = self.img_files[self.normal_indices[index]]
            label = self.labels[self.normal_indices[index]]
        else:
            img_file = self.img_files[index]
            label = self.labels[index]
        
        cls_name_idx = -3 if self.split == 'train' else -4
        cls_name = str(img_file).split(os.path.sep)[cls_name_idx]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        inputs["clsname"] = cls_name
        inputs["clslabel"] = BMAD_CLASSES.index(cls_name)
        inputs["path"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["img"] = sample
            return inputs
        else:
            inputs["img"] = sample
            inputs["label"] = label
            if "/good/" in str(img_file):
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = str(img_file).split(os.path.sep)[-2]
            if self.is_mask:
                # mask_dir =  img_file.parent.parent / 'label' 
                # mask_file = mask_dir / img_file.name
                # if "/good/" in str(img_file) or cls_name in NOMASK_CLASSES:
                #     # If the class does not have a mask or is 'good', create a blank mask
                #     mask = Image.new('L', (self.input_res, self.input_res), 0)
                # else:
                #     with open(mask_file, 'rb') as f:
                #         mask = Image.open(f)
                #         mask = mask.convert('L')
                mask = Image.new('L', (self.input_res, self.input_res), 0)
                mask = self.mask_transform(mask)
                inputs["mask"] = mask
            return inputs
    
    def __len__(self):
        if self.anom_only:
            return len(self.anom_indices)
        elif self.normal_only:
            return len(self.normal_indices)
        else:
            return len(self.img_files)
    
if __name__ == "__main__":

    data_root = "/home/haselab/projects/sakai_ssd/FlowAD/data/bmad"
    category = "BraTS2021_slice"
    input_res = 256
    split = "test"
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor(),
    ])
    dataset = BMAD(data_root, category, input_res, split, transform=transform, is_mask=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Number of samples in {split} split: {len(dataset)}")
    batch = next(iter(dataloader))
    print(f"Batch size: {len(batch['img'])}")
    print(f"Sample class names: {batch['clsname']}")
    print(f"Sample labels: {batch['label'] if 'label' in batch else 'No labels in this split'}")
    
    
    
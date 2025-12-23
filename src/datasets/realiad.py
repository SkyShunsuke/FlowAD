import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

REALIAD_CLASSES = [
    'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
    'fire_hood', 'mint', 'phone_battery', 'plastic_nut', 'plastic_plug',
    'pcb', 'porcelain_doll', 'regulator', 'rolled_strip_base',
    'sim_card_set', 'switch', 'terminalblock', 'toothbrush',
    'toy', 'toy_brick', 'transistor1', 'u_block', 'usb', 'usb_adaptor',
    'vcpill', 'wooden_beads', 'woodstick', 'zipper', 'mounts', 'tape'
] 

NORMAL_PREFIX = 'OK'

class RealIAD(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        meta_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False, 
        **kwargs
    ):
        """Dataset for MVTec AD.
        Args:
            data_root: Root directory of MVTecAD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'hazelnut'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.meta_dir = meta_dir
        self.custom_transforms = transform
        self.is_mask = is_mask
        self.cls_label = cls_label
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'

        # # load files from the dataset
        self.img_files, self.labels_str, self.masks = self.get_files()
        self.labels = []
        for label in self.labels_str:
            if label == NORMAL_PREFIX:
                self.labels.append(0)
            else:
                self.labels.append(1)
        if self.split == 'test':
            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res, interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(mask_to_tensor)
                ]
            )
            self.normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.anom_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_classes = len(REALIAD_CLASSES)
        
    def __getitem__(self, index):
        inputs = {}
        img_file = Path(self.img_files[index])
        label = self.labels[index]
        
        cls_name = str(img_file).split("/")[-5]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsname"] = cls_name
        inputs["clslabel"] = REALIAD_CLASSES.index(cls_name)
        inputs["path"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["img"] = sample
            return inputs
        else:
            inputs["img"] = sample
            inputs["label"] = label
            if self.labels_str[index] == NORMAL_PREFIX:
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = str(img_file).split("/")[-3]
            if self.is_mask:
                if self.labels_str[index] == NORMAL_PREFIX:
                    mask = np.zeros((self.input_res, self.input_res))
                else:
                    mask_file = self.masks[index]
                    with open(mask_file, 'rb') as f:
                        mask = Image.open(f)
                        mask = mask.convert('L')
                    mask = self.mask_transform(mask)
                inputs["mask"] = mask
            return inputs
                
    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        # First load meta file 
        import json
        meta_file = Path(self.meta_dir) / f"{self.category}.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file {meta_file} does not exist.")
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        if self.split == "train":
            train_files = meta_data.get('train', None)
            if not train_files:
                raise ValueError(f"No training files found for category {self.category} in meta file.")
            files = [os.path.join(self.data_root, self.category, self.category, file["image_path"]) for file in train_files]
            labels = [file["anomaly_class"] for file in train_files]
            mask_paths = [os.path.join(self.data_root, self.category, self.category, file["mask_path"]) for file in train_files if file["mask_path"]]
        elif self.split == "test":
            test_files = meta_data.get('test', None)
            if not test_files:
                raise ValueError(f"No test files found for category {self.category} in meta file.")
            files = [os.path.join(self.data_root, self.category, self.category, file["image_path"]) for file in test_files]
            labels = [file["anomaly_class"] for file in test_files]
            mask_paths = []
            for f in test_files:
                if f["mask_path"]:
                    mask_paths.append(os.path.join(self.data_root, self.category, self.category, f["mask_path"]))
                else:
                    mask_paths.append(None)
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train' or 'test'.")
        return files, labels, mask_paths
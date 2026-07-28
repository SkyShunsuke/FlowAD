import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

# breakfast_box  license.txt  readme.txt  splicing_connectors
# juice_bottle   pushpins     screw_bag

LOCO_CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]

LOCO_TYPES = [
    "good",
    "logical_anomalies",
    "structural_anomalies"
]

class MVTecLOCO(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False, 
        **kwargs
    ):
        """Dataset for MVTec LOCO.
        Args:
            data_root: Root directory of MVTecLOCO dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'breakfast_box'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = transform
        self.is_mask = is_mask
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'
        
        # # load files from the dataset
        self.img_files = self.get_files()
        self.labels = [0] * len(self.img_files)
        if self.split == 'test':
            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res, interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(mask_to_tensor)
                ]
            )

            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
        self.num_classes = len(LOCO_CLASSES)
        self.labels_to_names = {i: name for i, name in enumerate(LOCO_CLASSES)}
        self.types_to_names = {i: name for i, name in enumerate(LOCO_TYPES)}
        
    def __getitem__(self, index):
        inputs = {}
        img_file = self.img_files[index]
        label = self.labels[index]
        
        cls_name = str(img_file).split("/")[-4]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsname"] = cls_name
        inputs["clslabel"] = LOCO_CLASSES.index(cls_name)
        inputs["path"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["img"] = sample
            return inputs
        else:
            inputs["img"] = sample
            inputs["label"] = label
            if "good" in str(img_file):
                inputs["anom_type"] = LOCO_TYPES.index("good")
            else:
                inputs["anom_type"] = LOCO_TYPES.index(str(img_file).split("/")[-2])
            if self.is_mask:
                mask_dir =  img_file.parent.parent.parent / 'ground_truth' / img_file.parent.name 
                mask_file = mask_dir / img_file.name.replace('.png', '_mask.png')
                if True: # 'good' == img_file.parent.name:
                    mask = Image.new('L', (self.input_res, self.input_res), 0)
                # else:
                    # with open(mask_file, 'rb') as f:
                    #     mask = Image.open(f)
                    #     mask = mask.convert('L')
                mask = self.mask_transform(mask)
                inputs["mask"] = mask
            return inputs
                
    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            anomalous_img_files = []
            anomalous_dirs = sorted(Path(os.path.join(self.data_root, self.category, 'test')).glob('*'))
            for anomalous_dir in anomalous_dirs:
                if "good" in str(anomalous_dir):
                    continue
                anomalous_img_files += sorted(anomalous_dir.glob('*.png'))
            anomalous_img_files = sorted(anomalous_img_files)
            
            files = normal_img_files + anomalous_img_files
        return files
    
if __name__ == "__main__":
    data_root = "/home/haselab/projects/sakai_ssd/FlowAD/data/mvtec_loco"
    category = "juice_bottle"
    input_res = 256
    split = "test"
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor()
    ])
    dataset = MVTecLOCO(data_root, category, input_res, split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # check whether anomalous samples are included
    anomalous_count = 0
    for batch in dataloader:
        anomalous_count += batch["label"].sum().item()
    print(f"Total samples: {len(dataset)}, Anomalous samples: {anomalous_count}")
    
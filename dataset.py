import os
import torch
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path
from PIL import Image
from typing import Tuple


class COCOTrainImageDataset(Dataset):
    def __init__(self, img_dir: str, annotations_dir: str, max_images: int = None, transform=None):
        self.img_labels = sorted(glob("*.cls", root_dir=annotations_dir))
        if max_images:
            self.img_labels = self.img_labels[:max_images]
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, Path(self.img_labels[idx]).stem + ".jpg")
        labels_path = os.path.join(self.annotations_dir, self.img_labels[idx])
        
        image = Image.open(img_path).convert("RGB")
        
        with open(labels_path) as f: 
            labels = [int(label.strip()) for label in f.readlines()]
        
        if self.transform:
            image = self.transform(image)
        
        labels_tensor = torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)
        return image, labels_tensor


class COCOTestImageDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_list = sorted(glob("*.jpg", root_dir=img_dir))    
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert("RGB")        
        if self.transform:
            image = self.transform(image)
        return image, Path(img_path).stem  # Return filename without extension


class ValidationDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        original_dataset = self.subset.dataset
        original_idx = self.subset.indices[idx]
        
        img_path = os.path.join(original_dataset.img_dir, 
                               Path(original_dataset.img_labels[original_idx]).stem + ".jpg")
        labels_path = os.path.join(original_dataset.annotations_dir, original_dataset.img_labels[original_idx])
        
        image = Image.open(img_path).convert("RGB")
        with open(labels_path) as f: 
            labels = [int(label.strip()) for label in f.readlines()]
        
        if self.transform:
            image = self.transform(image)
        
        labels_tensor = torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)
        return image, labels_tensor
        
    def __len__(self):
        return len(self.subset)

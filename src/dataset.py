import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

# --- CUSTOM TRANSFORMS ---

class ApplyCLAHE(object):
    """
    Áp dụng Contrast Limited Adaptive Histogram Equalization (CLAHE) để làm rõ các biển báo bị tối hoặc bóng đổ
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Chuyển PIL Image -> Numpy array (RGB)
        img_np = np.array(img)
        
        # Chuyển sang không gian màu LAB để chỉ xử lý kênh Lightness (độ sáng) giữ nguyên màu sắc (A, B channels)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE lên kênh L
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l)
        
        # Gộp lại và chuyển về RGB
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(rgb_eq)

# --- DEFINING TRANSFORMS PIPELINE ---

def get_transforms(stage='train', image_size=48):
    """
    Tạo pipeline tiền xử lý ảnh.
    - Resize về 48x48
    - Data Augmentation: Chỉ xoay và dịch chuyển, KHÔNG lật (flip)
    """
    # Các bước chung cho cả Train và Test
    shared_transforms = [
        # Resize về 48x48 cho mô hình ResNet Custom/SimpleNet
        transforms.Resize((image_size, image_size)),
        # Áp dụng CLAHE để cân bằng sáng
        ApplyCLAHE(),
    ]

    if stage == 'train':
        return transforms.Compose([
            *shared_transforms,
            # Augmentation cho tập Train
            transforms.RandomRotation(10), # Xoay nhẹ +/- 10 độ
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Dịch chuyển nhẹ
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Nhiễu màu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation / Test: Chỉ Resize, CLAHE và Normalize
        return transforms.Compose([
            *shared_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
class TransformedSubset(Dataset):
    """Giữ nguyên wrapper này để tách biệt transform giữa Train và Val"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, batch_size=64, image_size=48, val_split=0.2, seed=42, num_workers=2):
    """
    Args:
        seed (int): đảm bảo chia Train/Val giống nhau mọi lần chạy.
    """
    train_dir = os.path.join(data_dir, 'Train')
    
    # 1. Load toàn bộ folder Train
    # Load thô (chưa transform) để split
    full_dataset = datasets.ImageFolder(root=train_dir)
    
    # 2. Tính toán kích thước split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # 3. Random Split với SEED cố định
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # 4. Gán Transform riêng biệt
    train_dataset = TransformedSubset(train_subset, transform=get_transforms('train', image_size))
    val_dataset = TransformedSubset(val_subset, transform=get_transforms('valid', image_size))

    # 5. Tạo Loaders (Chỉ Train và Val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Data Loaded from {train_dir} with Seed {seed}:")
    print(f" - Train images: {len(train_dataset)}")
    print(f" - Val images:   {len(val_dataset)}")
    
    return train_loader, val_loader
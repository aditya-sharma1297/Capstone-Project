import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Creating custom dataset to handle RGB and LiDAR image pairs along with segmentation labels
class RGBLiDARDataset(Dataset):
    def __init__(self, file_list, transform_rgb=None, transform_lidar=None):
        self.file_list = file_list
        self.transform_rgb = transform_rgb
        self.transform_lidar = transform_lidar

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rgb_path, lidar_path, label_path = self.file_list[idx]
        rgb = Image.open(rgb_path).convert('RGB')
        lidar = Image.open(lidar_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_lidar:
            lidar = self.transform_lidar(lidar)

        label = label.resize((rgb.shape[2], rgb.shape[1]), resample=Image.NEAREST)
        label = torch.from_numpy(np.array(label)).long()

        return rgb, lidar, label

"""
tasks/landmark/dataset.py
关键点检测数据集，迁移自 Cardiac_Landmark/data_loader.py + utils.py
"""
import json
import random
from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader

from common.base_dataset import BaseDataset
from .landmark_process import generate_gaussian_heatmaps


# ---------------------------------------------------------------------------
# 数据增强 / 预处理
# ---------------------------------------------------------------------------
def transforms_img_landmark(img_meta, landmarks, args, sec='train'):
    """
    pipeline：Z-score 标准化 → 等比重采样 → 以关键点中心裁剪 160×160 → rotate
    参数 sec='train'/'test'/'val'
    """
    img = img_meta.pixel_array.astype(np.float32)
    origin_spacing = img_meta.PixelSpacing

    # Step-1: normalize + resize to uniform spacing
    img -= img.mean(); img /= img.std()
    resample_size = [int(l * os / args.target_spacing)
                     for l, os in zip(img.shape[-2:], origin_spacing[-2:])]
    tfm = A.Compose(
        [A.Resize(height=resample_size[0], width=resample_size[1])],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    out = tfm(image=img, keypoints=landmarks)
    img, landmarks = out['image'], out['keypoints']

    # Step-2: center-crop ± jitter (仅训练时加抖动)
    cx = int((landmarks[0][0] + landmarks[1][0]) / 2)
    cy = int((landmarks[0][1] + landmarks[1][1]) / 2)
    if sec == 'train':
        cx += random.randint(-15, 15)
        cy += random.randint(-15, 15)

    H_t, W_t = args.target_size
    x_min = max(0, cx - W_t // 2); y_min = max(0, cy - H_t // 2)
    x_max = min(x_min + W_t, img.shape[1]); y_max = min(y_min + H_t, img.shape[0])

    aug_list = [
        A.Crop(x_min, y_min, x_max, y_max),
        A.PadIfNeeded(min_height=H_t, min_width=W_t, p=1),
    ]
    if sec == 'train':
        aug_list.append(A.Rotate(limit=90, p=1))
    aug_list.append(A.ToTensorV2())

    tfm2 = A.Compose(aug_list,
                     keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    out2 = tfm2(image=img, keypoints=landmarks)
    return out2['image'], out2['keypoints']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LandmarkDataset(BaseDataset):
    SUPPORTED_EXT = {'.dcm'}

    def __init__(self, data_path, sce='train', transform=None,
                 sigma=3, debug=False, random_state=42, repeat_factor=1,
                 img_root_path='', **kwargs):
        self.sigma = sigma
        self.img_root_path = img_root_path
        super().__init__(data_path, sce, transform, debug, random_state, repeat_factor)

    def load_samples(self) -> list:
        with open(self.data_path, 'r') as f:
            res = json.load(f)
        return list(res.items())   # [(file_path, landmark), ...]

    @staticmethod
    def load_image(file_path: str):
        ext = ''.join(Path(file_path).suffixes).lower()
        if ext == '.dcm':
            return pydicom.dcmread(file_path)
        raise ValueError(f"Unsupported format: {ext}")

    def __getitem__(self, idx):
        actual_idx = idx % len(self.samples)
        file_path, landmark = self.samples[actual_idx]
        landmark = np.array(landmark)
        img_meta = self.load_image(self.img_root_path + file_path)
        img, landmark = self.transform(img_meta, landmark)
        if self.sce in ('val', 'test'):
            return img, torch.tensor(landmark, dtype=torch.float32)
        H, W = img.shape[-2:]
        heatmap = generate_gaussian_heatmaps((W, H), np.array(landmark), self.sigma)
        return img, torch.from_numpy(heatmap)


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------
class LandmarkDataLoader(DataLoader):
    def __init__(self, data_path, sce='train', transform=None, sigma=3,
                 debug=False, random_state=42, repeat_factor=1,
                 img_root_path='', batch_size=8, shuffle=False,
                 num_workers=8, pin_memory=True, prefetch_factor=None,
                 drop_last=False, **kwargs):
        if sce =='train':
            img_root_path = img_root_path + '/training/'
        elif sce == 'val':
            img_root_path = img_root_path + '/test/'
        elif sce == 'test':
            img_root_path = img_root_path + '/test/'
            
        dataset = LandmarkDataset(
            data_path=data_path, sce=sce, transform=transform, sigma=sigma,
            debug=debug, random_state=random_state, repeat_factor=repeat_factor,
            img_root_path=img_root_path
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor, drop_last=drop_last)

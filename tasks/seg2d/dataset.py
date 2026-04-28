"""
tasks/seg2d/dataset.py
CH4 2D 分割数据集，迁移自 CH4_Segmentation/data_loader.py
"""
import joblib
from pathlib import Path

import pydicom
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from common.base_dataset import BaseDataset


class Seg2DDataset(BaseDataset):
    SUPPORTED_EXT = {'.dcm': 'DICOM', '.nii': 'NIFTI', '.nii.gz': 'NIFTI'}

    def __init__(self, data_path, sce='train', transform=None,
                 debug=False, random_state=42, repeat_factor=1, **kwargs):
        super().__init__(data_path, sce, transform, debug, random_state, repeat_factor)

    def load_samples(self) -> list:
        data = joblib.load(self.data_path)
        if not all(isinstance(item, tuple) and len(item) == 2 for item in data):
            raise ValueError("Expected list of (img_path, label_path) tuples")
        return data   # [(img_path, label_path), ...]

    @staticmethod
    def load_image(file_path: str):
        ext = ''.join(Path(file_path).suffixes).lower()
        if ext == '.dcm':
            return pydicom.dcmread(file_path), False
        elif ext in ('.nii', '.nii.gz'):
            return sitk.ReadImage(file_path), True
        raise ValueError(f"Unsupported format: {ext}")

    def __getitem__(self, idx):
        actual_idx = idx % len(self.samples)
        img_path, label_path = self.samples[actual_idx]
        img_data, is_3d = self.load_image(img_path)
        label_data, _ = self.load_image(label_path)
        img, label, *sizes = self.transform(self.sce, img_data, label_data, is_3d)
        return img, label, img_path, sizes

    def collate_fn(self, batch):
        """把 3D 体积的每张切片作为独立样本拼成 batch。"""
        total = sum(item[0].shape[0] for item in batch)
        sample_img = batch[0][0]
        imgs_out = torch.empty((total, 1, sample_img.shape[1], sample_img.shape[2]))

        if isinstance(batch[0][1], list):
            labels_out = [
                torch.empty((total, batch[0][1][i].shape[1], batch[0][1][i].shape[2]))
                for i in range(len(batch[0][1]))
            ]
        else:
            labels_out = torch.empty((total, batch[0][1].shape[1], batch[0][1].shape[2]))

        file_paths, sizes_list, si = [], [], 0
        for img, labels, fp, sizes in batch:
            D, H, W = img.shape
            imgs_out[si:si+D] = img.reshape(D, 1, H, W)
            if isinstance(labels, list):
                for i, lbl in enumerate(labels):
                    labels_out[i][si:si+D] = lbl
            else:
                labels_out[si:si+D] = labels
            file_paths.append(fp)
            sizes_list.append(sizes)
            si += D
        return imgs_out, labels_out, file_paths, sizes_list


class Seg2DDataLoader(DataLoader):
    def __init__(self, data_path, sce='train', transform=None,
                 debug=False, random_state=42, repeat_factor=1,
                 batch_size=4, shuffle=False, num_workers=8,
                 pin_memory=True, prefetch_factor=2, drop_last=False, **kwargs):
        dataset = Seg2DDataset(data_path, sce, transform,
                               debug, random_state, repeat_factor)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=dataset.collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                         drop_last=drop_last)

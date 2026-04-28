"""
tasks/view/dataset.py
视图分类数据集，迁移自 Cardiac_View/data_loader.py + data_process.py
"""
import random
import joblib
import pydicom
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from common.base_dataset import BaseDataset


class ViewDataset(BaseDataset):
    """
    SAX/CH2/CH4 视图分类数据集。
    label: 0=sax, 1=ch4, 2=ch2（与原项目一致）
    """

    def __init__(self, data_path, sce='train', transform=None,
                 debug=False, random_state=42, repeat_factor=1,
                 downsample=False, **kwargs):
        self.downsample = downsample
        super().__init__(data_path, sce, transform, debug, random_state, repeat_factor)

    def load_samples(self) -> list:
        data = joblib.load(self.data_path)
        sce = self.sce

        if sce == 'train' and self.downsample:
            sax = random.sample(data['sax'], 15000)
            imgs = sax + data['ch4']
            labels = [0] * len(sax) + [1] * len(data['ch4'])
        elif sce == 'debug':
            n = 100
            imgs = data['sax'][:n] + data['ch2'][:n] + data['ch4'][:n]
            labels = [0]*n + [1]*n + [2]*n
        else:
            imgs = (data.get('sax', []) + data.get('ch4', []) + data.get('ch2', []))
            labels = ([0]*len(data.get('sax', [])) +
                      [1]*len(data.get('ch4', [])) +
                      [2]*len(data.get('ch2', [])))
        return list(zip(imgs, labels))

    @staticmethod
    def load_image(file_path):
        if file_path.endswith('.dcm'):
            img = pydicom.dcmread(file_path)
            spacing = img.PixelSpacing
            return img, spacing, False
        elif file_path.endswith(('.nii', '.nii.gz')):
            img = nib.load(file_path)
            spacing = img.header.get_zooms()
            return img, spacing, True
        raise ValueError(f"Unsupported format: {file_path}")

    def __getitem__(self, idx):
        actual_idx = idx % len(self.samples)
        file_path, label = self.samples[actual_idx]
        img_data, origin_spacing, is_3d = self.load_image(file_path)
        img = self.transform(self.sce, img_data, origin_spacing, is_3d)
        return img, label, file_path

    def collate_fn(self, batch):
        """每个 3D 体积拆成 D 张切片，分别作为独立样本。"""
        inputs, labels, file_paths = [], [], []
        for sample, label, file_path in batch:
            D, H, W = sample.shape
            inputs.extend(sample.reshape(D, 1, H, W))
            labels.extend([label] * D)
            file_paths.extend([file_path] * D)
        return torch.stack(inputs), torch.tensor(labels), file_paths


class ViewDataLoader(DataLoader):
    def __init__(self, data_path, sce='train', transform=None,
                 debug=False, random_state=42, repeat_factor=1,
                 batch_size=8, shuffle=False, num_workers=16,
                 pin_memory=True, prefetch_factor=2, downsample=False, **kwargs):
        dataset = ViewDataset(data_path, sce, transform,
                              debug, random_state, repeat_factor, downsample)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=dataset.collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, prefetch_factor=prefetch_factor)

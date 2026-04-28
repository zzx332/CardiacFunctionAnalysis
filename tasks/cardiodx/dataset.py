"""
tasks/cardiodx/dataset.py
CardioDx 数据集：输入 ED+ES 中间3切片，输出二分类标签。
迁移自 CardioDx/data_loader.py
"""
import os
from typing import List, Dict, Any, Optional

import joblib
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path


class CardioDxDataset(Dataset):
    """
    每样本加载 ED/ES 的中间 3 切片（crop by bbox），同时返回二分类标签。
    data PKL 格式：List[{'ED': str, 'ES': str, 'bbox': tuple, 'label': 0|1, ...}]
    """

    def __init__(self, data_path: str, transform=None, random_state: int = 42,
                 infinite: bool = False, debug: bool = False):
        abs_path = os.path.abspath(data_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        self.data = joblib.load(abs_path)
        if debug:
            self.data = self.data[:10]
        self.transform = transform
        self.infinite = infinite
        self.rng = np.random.RandomState(random_state)
        self.file_info = self._preload_file_info()

    def _preload_file_info(self):
        info = []
        for item in self.data:
            ed_nii = nib.load(item['ED'])
            spacing = ed_nii.header.get_zooms()  # (x,y,z)
            info.append({
                'ed_path': item['ED'],
                'es_path': item['ES'],
                'bbox':    item['bbox'],
                'label':   item['label'],
                'spacing': spacing,
            })
        return info

    @staticmethod
    def _load_3slice(file_path: str, bbox) -> np.ndarray:
        nii = nib.load(file_path)
        data = nii.get_fdata().transpose(2, 1, 0)  # (D,H,W)
        mid = data.shape[0] // 2
        slices = [max(0, mid - 2), mid, min(data.shape[0]-1, mid + 2)]
        return data[slices, bbox[2]-10:bbox[3]+10, bbox[4]-10:bbox[5]+10].astype(np.float32)

    def __getitem__(self, idx):
        real_idx = idx % len(self.file_info)
        info = self.file_info[real_idx]
        ed = self._load_3slice(info['ed_path'], info['bbox'])
        es = self._load_3slice(info['es_path'], info['bbox'])

        if self.transform:
            imgs = np.concatenate([ed, es], axis=0)
            imgs = self.transform(imgs, origin_spacing=info['spacing'][1:])
            D = ed.shape[0]
            ed, es = imgs[:D], imgs[D:]

        return {'ed': ed, 'es': es, 'label': info['label'],
                'ed_path': info['ed_path'], 'es_path': info['es_path'], 'index': real_idx}

    def __len__(self):
        return int(1e5) if self.infinite else len(self.file_info)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ed     = torch.from_numpy(np.stack([b['ed']    for b in batch], axis=0)).float()
        es     = torch.from_numpy(np.stack([b['es']    for b in batch], axis=0)).float()
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        return {'ed': ed, 'es': es, 'label': labels,
                'ed_path': [b['ed_path'] for b in batch],
                'es_path': [b['es_path'] for b in batch],
                'index':   torch.tensor([b['index'] for b in batch], dtype=torch.long)}


class CardioDxDataLoader(DataLoader):
    """
    支持 instance / class / progressively_balanced 三种采样策略。
    """

    def __init__(self, data_path: str, transform=None, random_state: int = 42,
                 infinite: bool = False, sampling_strategy: str = 'instance',
                 debug: bool = False, **kwargs):
        sampler = kwargs.pop('sampler', None)
        self._params = dict(data_path=data_path, transform=transform, random_state=random_state,
                            infinite=infinite, sampling_strategy=sampling_strategy,
                            debug=debug, **kwargs)
        ds = CardioDxDataset(data_path, transform, random_state, infinite, debug)
        self.file_info = ds.file_info
        self.length = len(ds)
        self.sampling_strategy = sampling_strategy
        shuffle = kwargs.pop('shuffle', True) and sampling_strategy == 'instance'
        collate = kwargs.pop('collate_fn', ds.collate_fn)
        super().__init__(dataset=ds, collate_fn=collate, shuffle=shuffle,
                         sampler=sampler, **kwargs)

    def _sample_weights(self, step: int = 0, total: int = 1000) -> Optional[torch.Tensor]:
        labels = [f['label'] for f in self.file_info]
        if self.sampling_strategy == 'instance':
            return None
        counts = np.bincount(labels)
        cb_w = 1.0 / counts
        w = np.array([cb_w[l] for l in labels])
        if self.sampling_strategy == 'progressively_balanced':
            alpha = step / total
            ib_w = np.ones(len(labels))
            w = (1 - alpha) * ib_w + alpha * (w / w.sum() * len(labels))
        return torch.FloatTensor(w / w.sum())

    def update_sampler(self, step: int = 0, total: int = 1000) -> 'CardioDxDataLoader':
        weights = self._sample_weights(step, total)
        sampler = WeightedRandomSampler(weights, self.length, replacement=True) if weights is not None else None
        return CardioDxDataLoader(**self._params, sampler=sampler)

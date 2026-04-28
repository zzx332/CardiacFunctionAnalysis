"""
tasks/strain/dataset.py
心肌配准数据集，迁移自 Myocardial_Strain/data_loader.py
输入：ED/ES NIfTI 3D 影像对（ACDC / MnM 数据集）
输出：(moving_img, moving_gt, fixed_img, fixed_gt, directions)
"""
import os
import random
from itertools import chain

import joblib
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from common.base_dataset import BaseDataset


class StrainDataset(Dataset):
    """
    配准任务数据集。
    - 每次随机从 ED→ES 或 ES→ED 方向配准（direction=±1）
    - 仅保留含左心室（label==1）的有效切片
    - collate_fn 将 3D 体→切片展平
    """

    def __init__(self, data_path: str, transform=None, debug: bool = False, **kwargs):
        if not data_path:
            raise ValueError("data_path cannot be empty")
        abs_path = os.path.abspath(data_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        self.imgs_paths = joblib.load(abs_path)
        if debug:
            self.imgs_paths = self.imgs_paths[:10]
        self.transform = transform

    @staticmethod
    def get_ED_ES(path_4d: str):
        """根据 4D 路径定位 ED/ES 帧。支持 ACDC 和 MnM 数据集。"""
        root_dir = os.path.dirname(path_4d)
        subject_name = os.path.basename(root_dir)
        if 'ACDC' in path_4d:
            info = np.loadtxt(root_dir + '/Info.cfg', dtype=str)
            ed_slice, es_slice = int(info[0][1]), int(info[1][1])
            ed_img = root_dir + f'/{subject_name}_frame{ed_slice:02d}.nii.gz'
            ed_gt  = root_dir + f'/{subject_name}_frame{ed_slice:02d}_gt.nii.gz'
            es_img = root_dir + f'/{subject_name}_frame{es_slice:02d}.nii.gz'
            es_gt  = root_dir + f'/{subject_name}_frame{es_slice:02d}_gt.nii.gz'
            source = 'ACDC'
        else:
            ed_img = root_dir + f'/{subject_name}_SA_ED.nii.gz'
            ed_gt  = root_dir + f'/{subject_name}_SA_ED_gt.nii.gz'
            es_img = root_dir + f'/{subject_name}_SA_ES.nii.gz'
            es_gt  = root_dir + f'/{subject_name}_SA_ES_gt.nii.gz'
            source = 'MnM'
        return ed_img, ed_gt, es_img, es_gt, source

    def __getitem__(self, idx):
        path = self.imgs_paths[idx]
        mov_img_p, mov_gt_p, fix_img_p, fix_gt_p, source = self.get_ED_ES(path)
        direction = 1
        if np.random.rand() > 0.5:
            mov_img_p, fix_img_p = fix_img_p, mov_img_p
            mov_gt_p,  fix_gt_p  = fix_gt_p,  mov_gt_p
            direction = -1

        mov_img = nib.load(mov_img_p).get_fdata().squeeze().transpose(2,1,0).astype(np.float32)
        mov_gt  = nib.load(mov_gt_p ).get_fdata().squeeze().transpose(2,1,0).astype(np.int8)
        fix_img = nib.load(fix_img_p).get_fdata().squeeze().transpose(2,1,0).astype(np.float32)
        fix_gt  = nib.load(fix_gt_p ).get_fdata().squeeze().transpose(2,1,0).astype(np.int8)
        spacing = nib.load(mov_img_p).header.get_zooms()[::-1]

        # 标签映射
        if source == 'ACDC':
            for gt in (mov_gt, fix_gt):
                gt[gt == 1] = 0;  gt[gt == 3] = 1
        elif source == 'MnM':
            for gt in (mov_gt, fix_gt): gt[gt == 3] = 0

        # 过滤无效切片（双方都需要有 LV）
        valid = [i for i in range(mov_gt.shape[0])
                 if np.any(mov_gt[i] == 1) and np.any(fix_gt[i] == 1)]
        mov_img, fix_img = mov_img[valid], fix_img[valid]
        mov_gt,  fix_gt  = mov_gt[valid],  fix_gt[valid]

        if self.transform is not None:
            N = mov_img.shape[0]
            paired_img = np.concatenate((mov_img, fix_img), axis=0)
            paired_gt  = np.concatenate((mov_gt,  fix_gt),  axis=0)
            paired_img, paired_gt = self.transform(
                paired_img, paired_gt, spacing[-2:])
            mov_img, fix_img = paired_img[:N], paired_img[N:]
            mov_gt,  fix_gt  = paired_gt[:N],  paired_gt[N:]

        return mov_img, mov_gt, fix_img, fix_gt, [direction] * len(valid)

    def __len__(self):
        return len(self.imgs_paths)

    @staticmethod
    def collate_fn(batch):
        movs, movs_gt, fixs, fixs_gt, directions = zip(*batch)
        movs    = torch.from_numpy(np.concatenate(movs, axis=0))
        movs_gt = torch.from_numpy(np.concatenate(movs_gt, axis=0))
        fixs    = torch.from_numpy(np.concatenate(fixs, axis=0))
        fixs_gt = torch.from_numpy(np.concatenate(fixs_gt, axis=0))
        return movs[:,None], movs_gt, fixs[:,None], fixs_gt, list(chain(*directions))


class RandomBatchSampler(Sampler):
    """每个 iteration 随机采 batch_size 个样本（支持 iterations_per_epoch）。"""
    def __init__(self, data_source, batch_size, iterations_per_epoch):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.iterations = iterations_per_epoch

    def __iter__(self):
        for _ in range(self.iterations):
            yield torch.randint(0, len(self.data_source), (self.batch_size,)).tolist()

    def __len__(self):
        return self.iterations


class StrainDataLoader(DataLoader):
    def __init__(self, data_path, transform=None, debug=False,
                 batch_size=8, iterations_per_epoch=100, num_workers=8,
                 pin_memory=True, prefetch_factor=2, **kwargs):
        dataset = StrainDataset(data_path, transform, debug)
        sampler = RandomBatchSampler(dataset, batch_size, iterations_per_epoch)
        super().__init__(dataset=dataset, batch_sampler=sampler,
                         collate_fn=dataset.collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, prefetch_factor=prefetch_factor)

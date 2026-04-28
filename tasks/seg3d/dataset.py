"""
tasks/seg3d/dataset.py
SAX 3D 心脏分割数据集，迁移自 SAX_Segmentation/my_dataset.py
依赖：torchio, nibabel
"""
import numpy as np
import torch
import torchio as tio
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


def get_split(fold: int, n_splits: int = 5, seed: int = 12345):
    """5-fold split，与原项目保持一致（stratified by group of 20）。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = kf.split(range(20))
    train_idx, test_idx = list(splits)[fold]
    train_keys, test_keys = [], []
    for i in range(5):
        all_keys = np.arange(20 * i + 1, 20 * i + 21)
        train_keys += list(all_keys[train_idx])
        test_keys  += list(all_keys[test_idx])
    return train_keys, test_keys


def get_position_from_txt(txt_path):
    a = np.loadtxt(txt_path, dtype=str)
    return max(int(a[0][7])-20, 0), int(a[0][-1])+20, max(int(a[1][7])-20, 0), int(a[1][-1])+20


class Seg3DDataset(Dataset):
    """
    训练用：每次随机取一张切片返回。
    __len__ 虚拟扩大 10x，相当于 repeat_factor=10。
    """

    def __init__(self, root_dir: str, ids,
                 target_spacing=None, input_patch_size=(352, 352, 1),
                 position_dir=None, transforms=None, label_root=None,
                 num_classes=4):
        super().__init__()
        self.rs = target_spacing
        self.ps = input_patch_size
        self.transforms = transforms
        self.num_classes = num_classes
        self.position_dir = position_dir

        self.data, self.labels, self.positions = [], [], []
        root_dir = Path(root_dir)
        for subid in ids:
            sub_dir = root_dir / f'patient{subid:03d}'
            lb_dir = Path(label_root) if label_root else sub_dir
            info  = np.loadtxt(sub_dir / 'Info.cfg', dtype=str)
            ed, es = int(info[0][1]), int(info[1][1])
            for frame in (ed, es):
                self.data.append(str(sub_dir / f'patient{subid:03d}_frame{frame:02d}.nii.gz'))
                self.labels.append(str(lb_dir / f'patient{subid:03d}_frame{frame:02d}_gt.nii.gz'))
                if position_dir:
                    self.positions.append(
                        get_position_from_txt(f'{position_dir}/patient{subid}_VOIPosition.txt'))
                else:
                    self.positions.append(None)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        tio_img = tio.ScalarImage(self.data[idx])
        img, affine = tio_img.numpy()[0], tio_img.affine
        z_spacing = tio_img.spacing[-1]
        mask = nib.load(self.labels[idx]).get_fdata().astype(np.float32)

        img = img.astype(np.float32)
        img -= img.mean(); img /= img.std()

        # 随机取切片
        slice_id = np.random.choice(img.shape[-1])
        pad_z = int((self.ps[-1] - 1) / 2)
        img  = np.pad(img,  ((0,0),(0,0),(pad_z,pad_z)), mode='edge')
        mask = np.pad(mask, ((0,0),(0,0),(pad_z,pad_z)), mode='edge')
        img  = img[...,  slice_id:slice_id+self.ps[-1]]
        mask = mask[..., slice_id:slice_id+self.ps[-1]]

        if self.positions[idx]:
            p = self.positions[idx]
            img  = img[p[0]:p[1], p[2]:p[3]]
            mask = mask[p[0]:p[1], p[2]:p[3]]

        img  = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        subject = tio.Subject(
            one_image=tio.ScalarImage(tensor=img[None], affine=affine),
            a_segmentation=tio.LabelMap(tensor=mask[None], affine=affine),
        )
        new_tf = []
        if self.rs is not None:
            new_tf.append(tio.Resample((self.rs, self.rs, z_spacing)))
        new_tf.append(tio.CropOrPad(self.ps))
        if self.transforms:
            new_tf.extend(self.transforms)
        new_tf.append(tio.OneHot(num_classes=self.num_classes))
        transformed = tio.Compose(new_tf)(subject)

        img  = transformed.one_image.data[0].transpose(0, 2)          # [d,w,h]
        mask = transformed.a_segmentation.data.transpose(1, 3)         # [C,d,w,h]
        if self.ps[-1] == 3:
            mask = mask[:, 1:2, ...]
        return img, mask

    def __len__(self):
        return len(self.data) * 10

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


class Seg3DTestDataset(Dataset):
    """
    测试用：返回完整体积（H, W, D）及 one-hot 标签（C, H, W, D）。
    直接从 root_dir 扫描 patient* 子目录，按 Info.cfg 中的 ED/ES 组样本。
    """

    def __init__(self, root_dir: str, target_spacing=None,
                 input_patch_size=(192, 192, 1), position_dir=None, label_root=None,
                 num_classes=4):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.label_root = Path(label_root) if label_root else None
        self.target_spacing = target_spacing
        self.patch_size = input_patch_size
        self.position_dir = position_dir
        self.num_classes = num_classes
        self.samples = self._load_test_samples()

    def _load_test_samples(self):
        patient_dirs = sorted(self.root_dir.glob('patient*'))
        samples = []
        for sub_dir in patient_dirs:
            sub_name = sub_dir.name
            if not sub_name.startswith('patient'):
                continue
            sub_id = int(sub_name[-3:])
            lb_dir = self.label_root if self.label_root else sub_dir
            info_cfg = sub_dir / 'Info.cfg'
            if not info_cfg.exists():
                continue
            info = np.loadtxt(info_cfg, dtype=str)
            ed, es = int(info[0][1]), int(info[1][1])
            for frame, phase in [(ed, 'ed'), (es, 'es')]:
                image_path = str(sub_dir / f'patient{sub_id:03d}_frame{frame:02d}.nii.gz')
                label_path = str(lb_dir / f'patient{sub_id:03d}_frame{frame:02d}_gt.nii.gz')
                if not Path(image_path).exists() or not Path(label_path).exists():
                    continue
                samples.append({
                    'case_id': f'patient{sub_id:03d}_{phase}',
                    'image': image_path,
                    'label': label_path,
                    'position': (
                        get_position_from_txt(f'{self.position_dir}/patient{sub_id}_VOIPosition.txt')
                        if self.position_dir else None
                    ),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tio_img = tio.ScalarImage(sample['image'])
        img = tio_img.numpy()[0].astype(np.float32)            # [H, W, D]
        mask = nib.load(sample['label']).get_fdata().astype(np.int64)
        z_spacing = tio_img.spacing[-1]

        img -= img.mean()
        img /= (img.std() + 1e-6)

        pos = sample['position']
        if pos is not None:
            img = img[pos[0]:pos[1], pos[2]:pos[3], :]
            mask = mask[pos[0]:pos[1], pos[2]:pos[3], :]

        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask.astype(np.float32))
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_t[None], affine=tio_img.affine),
            seg=tio.LabelMap(tensor=mask_t[None], affine=tio_img.affine),
        )
        tfms = []
        if self.target_spacing is not None:
            tfms.append(tio.Resample((self.target_spacing, self.target_spacing, z_spacing)))
        transformed = tio.Compose(tfms)(subject) if tfms else subject
        img = transformed.image.data[0].numpy().astype(np.float32)
        mask = transformed.seg.data[0].numpy().astype(np.int64)

        # 对每个切片做 H/W 的 crop or pad，使其与训练输入一致
        target_h, target_w = self.patch_size[0], self.patch_size[1]
        out_img = np.zeros((target_h, target_w, img.shape[-1]), dtype=np.float32)
        out_mask = np.zeros((target_h, target_w, mask.shape[-1]), dtype=np.int64)
        for z in range(img.shape[-1]):
            slice_subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.from_numpy(img[..., z][None, ..., None])),
                seg=tio.LabelMap(tensor=torch.from_numpy(mask[..., z].astype(np.float32)[None, ..., None])),
            )
            resized = tio.CropOrPad((target_h, target_w, 1))(slice_subject)
            out_img[..., z] = resized.image.data[0, :, :, 0].numpy()
            out_mask[..., z] = resized.seg.data[0, :, :, 0].numpy().astype(np.int64)

        onehot = np.eye(self.num_classes, dtype=np.float32)[out_mask.reshape(-1)]
        onehot = onehot.reshape(target_h, target_w, out_mask.shape[-1], self.num_classes).transpose(3, 0, 1, 2)
        return {
            'image': torch.from_numpy(out_img),     # [H, W, D]
            'label_onehot': torch.from_numpy(onehot),  # [C, H, W, D]
            'case_id': sample['case_id'],
        }

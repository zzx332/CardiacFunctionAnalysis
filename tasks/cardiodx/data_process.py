import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import SimpleITK as sitk
import json

class MedicalImageTransforms:
    """
    医学图像数据增强和预处理管道, 目前只支持2D的数据增强
    
    输入必须为(D,H,W)或(H,W)
    """
    def __init__(self,
                 mode: str = 'train',
                 target_size: int = 224,
                 target_spacing: float = 1,
                 augmentation_config: Optional[Dict] = None):

        self.mode = mode
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.aug_config = augmentation_config or {}

        # 存储原始图像信息用于后处理
        self.original_info = {}

        # 创建基础变换管道（不包含Normalize）
        self.dtype = {tv_tensors.Image: torch.float32,
                      tv_tensors.Mask: torch.int64, "others": None}
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> list:
        """构建变换管道（不包含Normalize和Resample）"""
        transforms_list = []

        # 2. 数据增强（仅训练时）
        if self.mode == 'train':
            transforms_list.extend(self._get_augmentation_transforms())

        # 3. 后处理
        transforms_list.extend([
            v2.CenterCrop(self.target_size),
            v2.ToDtype(self.dtype, scale=False),
        ])

        return transforms_list

    def _get_augmentation_transforms(self) -> List:
        """获取数据增强变换"""
        aug_transforms = []
        config = self.aug_config

        # 空间变换
        if config.get('flip_prob', 0) > 0:
            aug_transforms.extend([
                v2.RandomHorizontalFlip(p=config['flip_prob']),
                v2.RandomVerticalFlip(p=config['flip_prob'])
            ])

        # 仿射变换
        affine_transforms = []
        if config.get('rotation_prob', 0) > 0:
            affine_transforms.append(
                v2.RandomRotation(degrees=config.get('rotation_degrees', 180))
            )

        if config.get('scale_prob', 0) > 0:
            affine_transforms.append(
                v2.RandomAffine(degrees=0, scale=config.get(
                    'scale_range', (0.7, 1.4)))
            )

        if affine_transforms:
            aug_transforms.append(
                v2.RandomApply(affine_transforms,
                               p=config.get('rotation_prob', 0.2))
            )

        # 强度变换
        intensity_transforms = []
        if config.get('brightness_prob', 0) > 0:
            intensity_transforms.append(
                v2.ColorJitter(brightness=config.get(
                    'brightness_range', (0.8, 1.2)))
            )

        if config.get('contrast_prob', 0) > 0:
            intensity_transforms.append(
                v2.ColorJitter(contrast=config.get(
                    'contrast_range', (0.8, 1.2)))
            )

        if intensity_transforms:
            aug_transforms.append(v2.RandomApply(intensity_transforms, p=0.5))

        return aug_transforms

    def _calculate_normalization_stats(self, img: np.ndarray) -> Tuple[float, float]:
        """
        计算图像的均值和标准差
        
        return: 3D img: list of mean and std
        
                2D img: float of mean and std
        """
        if img.ndim==3 or img.ndim==2:
            return [float(img.mean())], [float(img.std())]
        else:
            raise ValueError(f"Invalid input shape, expected 3 or 2 dimensions, got {img.shape}")


    def _store_original_info(self, img: np.ndarray, origin_spacing: Tuple[float, ...]) -> None:
        """存储原始图像信息用于后处理"""
        self.original_info = {
            'original_shape': img.shape,
            'original_spacing': origin_spacing,
            'resample_size': self.resample_size,
            'target_size': self.target_size,
            # 'normalization_mean': self.mean,
            # 'normalization_std': self.std
        }

    def __call__(self, img: np.ndarray, origin_spacing: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行变换"""
        assert img.ndim == 3, f"Invalid input shape, got {img.shape}"
        assert len(origin_spacing)>=2, f"Invalid origin_spacing, got {origin_spacing}"
        
        # 1. 计算标准化参数和重采样尺寸
        mean, std = self._calculate_normalization_stats(img)
        self.resample_size = [int(dim * sp / self.target_spacing)
                              for dim, sp in zip(img.shape[-2:], origin_spacing[-2:])]

        # 2. 动态创建包含Normalize的完整变换管道
        full_transforms = [
            v2.ToDtype(self.dtype, scale=False),
            v2.Normalize(mean=mean, std=std),  # 动态添加Normalize
            v2.Resize(self.resample_size),         # 使用计算出的尺寸
        ]

        # 添加数据增强(训练才有)和后处理
        full_transforms.extend(self.transforms)

        # 3. 创建完整的变换管道
        full_pipeline = v2.Compose(full_transforms)

        # 4. 转换为torchvision tensor并应用变换
        img_tensor = tv_tensors.Image(img)

        img_transformed = full_pipeline(img_tensor)

        # 5. 存储原始信息（测试模式需要）
        if self.mode == 'test':
            self._store_original_info(img, origin_spacing)


        return img_transformed


# 使用示例
def create_transforms(args) -> Dict[str, MedicalImageTransforms]:
    """创建训练、验证和测试的变换管道"""
    transforms_dict = {
        'train': MedicalImageTransforms(
            mode='train',
            target_size=args.target_size,
            target_spacing=args.target_spacing,
            augmentation_config=json.loads(args.train_aug_config_cardiodx)
        ),
        'val': MedicalImageTransforms(
            mode='train',  # 使用train模式启用增强
            target_size=args.target_size,
            target_spacing=args.target_spacing,
            augmentation_config=json.loads(args.val_aug_config_cardiodx)  # 轻量级增强
        ),
        'test': MedicalImageTransforms(
            mode='test',
            target_size=args.target_size,
            target_spacing=args.target_spacing,
        )
    }

    return transforms_dict


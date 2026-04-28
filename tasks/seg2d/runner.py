"""tasks/seg2d/runner.py"""
from functools import partial
import os
import torch
import numpy as np
import json
import torch.optim as optim

from torchvision.transforms import v2
import SimpleITK as sitk
from torchvision import tv_tensors
import torch.optim.lr_scheduler as lr_scheduler
from common.utils import PolyLRScheduler
from .dataset import Seg2DDataLoader
from .trainer import Seg2DTrainer


class Seg2DRunner:
    def __init__(self, args):
        self.args = args

    # ─────────────────────────────────────────────
    # 数据预处理（训练 / 验证）
    # ─────────────────────────────────────────────
    def transforms_img(self, sce, img_data, label_data, is_3d=False, deep_supervision=False):
        if is_3d:
            img   = sitk.GetArrayFromImage(img_data).astype(np.float32)
            label = sitk.GetArrayFromImage(label_data).astype(np.uint8)
            origin_spacing = img_data.GetSpacing()[::-1]
        else:
            img   = img_data.pixel_array.astype(np.float32)
            label = label_data.pixel_array.astype(np.uint8)
            origin_spacing = img_data.PixelSpacing

        img   = tv_tensors.Image(img)
        label = tv_tensors.Mask(label)
        mean, std = img.mean(), img.std()

        resample_size = [int(l * os / self.args.target_spacing)
                         for l, os in zip(img.shape[-2:], origin_spacing[-2:])]

        transforms = v2.Compose([
            v2.Normalize(mean=[mean], std=[std]),
            v2.Resize(resample_size),
        ])

        if sce == 'train':
            transforms = v2.Compose([transforms, v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
            if np.random.uniform() < 0.2:
                transforms = v2.Compose([transforms, v2.RandomAffine(degrees=180)])
            if np.random.uniform() < 0.2:
                transforms = v2.Compose([transforms, v2.RandomAffine(degrees=0, scale=(0.7, 1.4))])

        transforms = v2.Compose([transforms, v2.CenterCrop(self.args.target_size_seg2d)])
        new_img, new_label = transforms(img, label)

        if deep_supervision:
            downsampled = [new_label]
            for scale in [2, 4, 8, 16, 32]:
                downsampled.append(v2.Resize(
                    [new_label.shape[-2] // scale, new_label.shape[-1] // scale])(new_label))
            return new_img, downsampled
        return new_img, new_label

    # ─────────────────────────────────────────────
    # 数据预处理（测试，无增强，保留原图尺寸信息）
    # ─────────────────────────────────────────────
    def transforms_img_test(self, sce, img_data, label_data, is_3d=False):
        """返回 (img_tensor, label_tensor, resample_size, origin_size)"""
        if is_3d:
            img   = sitk.GetArrayFromImage(img_data).astype(np.float32)
            label = sitk.GetArrayFromImage(label_data).astype(np.uint8)
            origin_spacing = img_data.GetSpacing()[::-1]
        else:
            img   = img_data.pixel_array.astype(np.float32)
            label = label_data.pixel_array.astype(np.uint8)
            origin_spacing = img_data.PixelSpacing

        img        = tv_tensors.Image(img)
        label      = torch.from_numpy(label)
        mean, std  = img.mean(), img.std()
        origin_size = list(img.shape[-2:])

        resample_size = [int(l * os / self.args.target_spacing)
                         for l, os in zip(origin_size, origin_spacing[-2:])]

        new_img = v2.Compose([v2.Normalize(mean=[mean], std=[std]), v2.Resize(resample_size)])(img)

        target = self.args.target_size
        th, tw = (target[0], target[1]) if isinstance(target, (list, tuple)) else (target, target)
        if resample_size[0] < th or resample_size[1] < tw:
            new_img = v2.CenterCrop([th, tw])(new_img)

        return new_img, label, resample_size, origin_size

    def inverse_transforms_img(self, pred, sizes):
        """将预测结果从推理空间还原到原图尺寸。"""
        resample_size, origin_size = sizes
        target = self.args.target_size
        th, tw = (target[0], target[1]) if isinstance(target, (list, tuple)) else (target, target)
        if resample_size[0] < th or resample_size[1] < tw:
            pred = v2.CenterCrop(resample_size)(pred)
        pred = v2.Resize(list(origin_size))(pred)
        return pred

    # ─────────────────────────────────────────────
    # 构建模型（训练 + 测试共用）
    # ─────────────────────────────────────────────
    def _build_model(self):
        from .models.unet import UNet
        network_config = json.loads(self.args.network_config)
        model = UNet(
            in_channels=getattr(self.args, 'in_channels', 1),
            num_classes=getattr(self.args, 'num_classes', 4),
            **network_config,
        )
        return model, network_config

    # ─────────────────────────────────────────────
    # 训练入口
    # ─────────────────────────────────────────────
    def run(self):
        args = self.args
        model, network_config = self._build_model()
        losses_config = json.loads(args.losses_config)

        train_loader = Seg2DDataLoader(
            data_path=args.train_data_path, sce='train',
            transform=partial(self.transforms_img, deep_supervision=args.deep_supervision),
            debug=args.debug, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            repeat_factor=getattr(args, 'repeat_factor', 1),
        )
        eval_loader = None
        if getattr(args, 'val_data_path', None):
            eval_loader = Seg2DDataLoader(
                data_path=args.val_data_path, sce='val',
                transform=self.transforms_img, debug=args.debug,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            )
        
        pg = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(pg, args.lr, weight_decay=args.weight_decay,
                                            momentum=0.99, nesterov=True)
        else:
            raise ValueError("optimizer not support")

        sch_name = getattr(args, 'scheduler', 'cosine')
        import math
        if sch_name == 'cosine':
            lambda_cosine = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)
        elif sch_name == 'warm_cosine':
            warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
                math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        elif sch_name == 'poly':
            scheduler = PolyLRScheduler(optimizer, args.lr, args.epochs)
        else:
            raise ValueError("scheduler not support")

        trainer = Seg2DTrainer(
            model=model, args=args,
            train_loader=train_loader, eval_loader=eval_loader,
            losses_config=losses_config, network_config=network_config,
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()

    # ─────────────────────────────────────────────
    # 测试入口（滑窗推理 + 可选 TTA）
    # ─────────────────────────────────────────────
    def test(self):
        """
        滑窗推理测试，对应 CH4_Segmentation/test.py 的逻辑。

        必要参数：
          --test_data_path   测试集 pkl 路径
          --checkpoint       checkpoint 文件路径（.pt）

        可选参数（均有默认值）：
          --roi_size         滑窗大小，默认同 target_size_seg2d
          --sw_batch_size    滑窗并行 batch，默认 4
          --overlap          滑窗重叠率，默认 0.5
          --tta              是否 TTA，默认 True
          --save_path        预测结果保存目录
        """
        args = self.args
        model, _ = self._build_model()

        # 加载 checkpoint
        ckpt_path = getattr(args, 'checkpoint', None)
        if ckpt_path is None:
            raise ValueError("测试模式需要指定 --checkpoint <path>")
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")

        test_loader = Seg2DDataLoader(
            data_path=args.test_data_path, sce='test',
            transform=self.transforms_img_test,
            debug=args.debug, batch_size=1,
            shuffle=False, num_workers=args.num_workers,
        )

        target        = args.target_size
        roi_size      = getattr(args, 'roi_size',       target if isinstance(target, list) else [target, target])
        sw_batch_size = getattr(args, 'sw_batch_size',  4)
        overlap       = getattr(args, 'overlap',         0.5)
        tta           = getattr(args, 'tta',             True)
        save_path     = getattr(args, 'save_path',       None)

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # 仅用于推理，不需要 optimizer / train_loader
        trainer = Seg2DTrainer(
            model=model, args=args,
            train_loader=None, eval_loader=None,
            losses_config={}, network_config={},
            optimizers=(None, None),
        )

        metrics = trainer.predict_with_sliding_window(
            test_loader,
            self.inverse_transforms_img,
            roi_size, sw_batch_size, overlap, tta,
            save_path=save_path,
        )
        print("Test metrics:", metrics)
        return metrics

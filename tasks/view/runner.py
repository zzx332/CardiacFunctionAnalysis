"""tasks/view/runner.py"""
import json

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2

from .dataset import ViewDataLoader
from .trainer import ViewTrainer


class ViewRunner:
    def __init__(self, args):
        self.args = args

    def transforms_img(self, sce, img_data, origin_spacing, is_3d=False):
        if is_3d:
            img = img_data.get_fdata().astype(np.float32)  # [D, W, H]
        else:
            img = img_data.pixel_array.astype(np.float32)  # [W, H]

        mean = img.mean()
        std = img.std()
        if std == 0:
            std = 1.0

        resample_size = [int(l * os / self.args.target_spacing) for l, os in zip(img.shape[:2], origin_spacing[:2])]

        transforms = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[mean], std=[std]),
            v2.Resize(resample_size),
            v2.CenterCrop(self.args.target_size)
        ])

        new_img = transforms(img)

        if sce == 'train':
            transforms = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=90)
            ])
            new_img = transforms(new_img)
        elif sce == 'tta':
            transforms = v2.Compose([v2.RandomHorizontalFlip(p=1)])
            img_s2 = transforms(new_img)
            new_img = torch.concat([new_img, img_s2])

            transforms = v2.Compose([v2.RandomVerticalFlip(p=1)])
            img_s2 = transforms(new_img)
            new_img = torch.concat([new_img, img_s2])

        return new_img

    def run(self):
        args = self.args

        # ── 动态导入模型 ─────────────────────────────────────
        arc = getattr(args, 'arch', 'efficientnet')
        if arc == 'resnet':
            from .models.resnet import resnet50
            model = resnet50(ori_c=args.in_channels, num_classes=args.num_classes)
        else:
            from .models.efficientnet import efficientnetv2_s
            model = efficientnetv2_s(in_c=args.in_channels, num_classes=args.num_classes)

        # ── Data ─────────────────────────────────────────────
        train_loader = ViewDataLoader(
            data_path=args.train_data_path, sce='train',
            transform=self.transforms_img, debug=args.debug,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            downsample=getattr(args, 'downsample', False),
        )
        eval_loader = None
        if getattr(args, 'val_data_path', None):
            eval_loader = ViewDataLoader(
                data_path=args.val_data_path, sce='val',
                transform=self.transforms_img, debug=args.debug,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            )

        # ── Optimizer ────────────────────────────────────────
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_lr, gamma=args.lr_decay)

        trainer = ViewTrainer(
            model=model, args=args,
            train_loader=train_loader, eval_loader=eval_loader,
            losses_config=json.loads(args.losses_config),
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()

    # ─────────────────────────────────────────────
    # 测试入口（分类推理 + 可选 TTA）
    # ─────────────────────────────────────────────
    def test(self):
        """
        分类测试，对齐 Cardiac_View/test.py 的推理流程。

        必要参数：
          --test_data_path   测试集 pkl 路径
          --checkpoint       checkpoint 文件路径（.pt）

        可选参数：
          --tta              是否开启 TTA（默认 False）
        """
        args = self.args
        # ── 动态导入模型 ─────────────────────────────────────
        arc = getattr(args, 'arch', 'efficientnet')
        if arc == 'resnet':
            from .models.resnet import resnet50
            model = resnet50(ori_c=args.in_channels, num_classes=args.num_classes)
        else:
            from .models.efficientnet import efficientnetv2_s
            model = efficientnetv2_s(in_c=args.in_channels, num_classes=args.num_classes)

        # 加载 checkpoint
        ckpt_path = getattr(args, 'checkpoint', None)
        if ckpt_path is None:
            raise ValueError("测试模式需要指定 --checkpoint <path>")
        if not getattr(args, 'test_data_path', None):
            raise ValueError("测试模式需要指定 --test_data_path <path>")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")

        sce = 'tta' if getattr(args, 'tta', False) else 'test'
        test_loader = ViewDataLoader(
            data_path=args.test_data_path, sce=sce,
            transform=self.transforms_img,
            debug=args.debug, batch_size=1,
            shuffle=False, num_workers=args.num_workers,
            downsample=getattr(args, 'downsample', False),
        )

        # 仅用于推理，不需要 optimizer / train_loader
        trainer = ViewTrainer(
            model=model, args=args,
            train_loader=None, eval_loader=None,
            losses_config={},
            optimizers=(None, None),
        )

        metrics = trainer.predict_tta(test_loader)
        print("Test metrics:", metrics)
        return metrics

"""tasks/view/runner.py"""
import torch
import numpy as np
import json
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
        
        resample_size = [int(l * os / self.args.target_spacing) for l, os in zip(img.shape[:2], origin_spacing[:2])]
        
        transforms = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[mean], std=[std]),
            v2.Resize(resample_size),
            v2.CenterCrop(self.args.target_size)
        ])
        
        if sce == 'train':
            transforms = v2.Compose([
                transforms,
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=90)
            ])
        
        new_img = transforms(img)
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

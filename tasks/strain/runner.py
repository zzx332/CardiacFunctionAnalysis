"""
tasks/strain/runner.py
StrainRunner：整合 VxmDense + StrainDataset + StrainTrainer
"""
import math
from functools import partial

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torchvision import tv_tensors

from common.utils import PolyLRScheduler
from .dataset import StrainDataLoader
from .trainer import StrainTrainer


def _transforms_img_and_mask(img, mask, origin_spacing, args):
    """img/mask: [2D, H, W] numpy arrays → [2D, H, W] tensors（含数据增强）"""
    img  = tv_tensors.Image(img)
    mask = tv_tensors.Mask(mask)
    resample_size = [int(l * os / args.target_spacing)
                     for l, os in zip(img.shape[-2:], origin_spacing[-2:])]
    t = v2.Compose([
        v2.Normalize(mean=[img.mean()], std=[img.std()]),
        v2.Resize(resample_size),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.RandomAffine(degrees=20),
    ])
    img, mask = t(img), t(mask)
    # 按 centroid 裁剪
    from common.utils import center_by_centroid  # 可选，若未导入则用 CenterCrop
    try:
        img, mask = center_by_centroid(img, mask, output_size=tuple(args.target_size))
    except Exception:
        img  = v2.CenterCrop(args.target_size)(img)
        mask = v2.CenterCrop(args.target_size)(mask)
    return img.numpy(), mask.numpy()


class StrainRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args

        # ── Data ──────────────────────────────────────────────
        train_loader = StrainDataLoader(
            data_path=args.train_data_path,
            transform=partial(_transforms_img_and_mask, args=args),
            debug=args.debug,
            batch_size=args.batch_size,
            iterations_per_epoch=getattr(args, 'iterations_per_epoch', 100),
            num_workers=args.num_workers,
        )

        # ── Model（VxmDense） ───────────────────────────────────
        from .models.networks import VxmDense
        enc_nf = getattr(args, 'enc', None)
        if enc_nf is None:
            enc_nf = [16, 32, 32, 32]
        dec_nf = getattr(args, 'dec', None)
        if dec_nf is None:
            dec_nf = [32, 32, 32, 32, 32, 16, 16]
        model = VxmDense(
            inshape=tuple(args.target_size),
            nb_unet_features=[enc_nf, dec_nf],
            bidir=getattr(args, 'bidir', False),
            int_steps=getattr(args, 'int_steps', 7),
            int_downsize=getattr(args, 'int_downsize', 2),
            unet_half_res=getattr(args, 'half_res', False),
        )

        # ── Optimizer / Scheduler ─────────────────────────────
        pg = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == 'AdamW':
            optimizer = optim.AdamW(pg, lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(pg, args.lr, momentum=0.99, nesterov=True)
        else:
            optimizer = optim.Adam(pg, lr=args.lr)

        if getattr(args, 'scheduler', 'cosine') == 'poly':
            scheduler = PolyLRScheduler(optimizer, args.lr, args.epochs)
        elif getattr(args, 'scheduler', 'cosine') == 'constant':
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
        else:  # cosine default
            lrf = getattr(args, 'lrf', 0.01)
            scheduler = lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - lrf) + lrf
            )

        # ── Trainer ───────────────────────────────────────────
        trainer = StrainTrainer(
            model=model, args=args,
            train_loader=train_loader,
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()

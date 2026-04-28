"""tasks/seg3d/runner.py"""
import torch
import torchio as tio
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from common.losses import DiceLoss, MyLoss
from .dataset import Seg3DDataset, get_split
from .trainer import Seg3DTrainer


class Seg3DRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        fold = getattr(args, 'fold', 0)
        train_keys, val_keys = get_split(fold)

        # ── Data augmentation ────────────────────────────────
        aug = [
            tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
            tio.RandomAffine(scales=(0.7,1.3,0.7,1.3,1,1),
                             degrees=(0,0,90), p=0.5),
            tio.RandomElasticDeformation(
                num_control_points=(10,10,7),
                max_displacement=(10,10,0), locked_borders=2, p=0.5),
        ]

        train_ds = Seg3DDataset(
            root_dir=args.dataset_root, ids=train_keys,
            target_spacing=args.target_spacing,
            input_patch_size=args.patch_size,
            position_dir=getattr(args, 'position_dir', None),
            transforms=aug,
            label_root=getattr(args, 'label_root', None),
            num_classes=args.num_classes,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True,
                                  pin_memory=True, prefetch_factor=2)

        val_loader = None
        if getattr(args, 'val', True):
            val_ds = Seg3DDataset(
                root_dir=args.dataset_root, ids=val_keys,
                target_spacing=args.target_spacing,
                input_patch_size=args.patch_size,
                position_dir=getattr(args, 'position_dir', None),
                label_root=getattr(args, 'label_root', None),
                num_classes=args.num_classes,
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=False,
                                    pin_memory=True)

        # ── Model ────────────────────────────────────────────
        arch = getattr(args, 'arch', 'unet')
        if arch == 'network':
            from .models.network import U_Net
            model = U_Net(getattr(args, 'img_c', 1), args.num_classes,
                          base_c=getattr(args, 'base_c', 48))
        else:
            from .models.unet import UNet
            model = UNet(getattr(args, 'img_c', 1), args.num_classes,
                         base_c=getattr(args, 'base_c', 48),
                         drop_rate=getattr(args, 'drop_rate', None))

        # ── Optimizer ────────────────────────────────────────
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=getattr(args, 'weight_decay', 1e-4))
        scheduler = StepLR(optimizer, step_size=getattr(args, 'step_lr', 2),
                           gamma=getattr(args, 'lr_decay', 0.985))

        trainer = Seg3DTrainer(
            model=model, args=args,
            train_loader=train_loader, eval_loader=val_loader,
            losses_config={'dice_loss': {'batch_dice': True, 'weight': 1.0}},
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()

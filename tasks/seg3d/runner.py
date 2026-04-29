"""tasks/seg3d/runner.py"""
import torch
import torchio as tio
import json
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .dataset import Seg3DDataset, Seg3DTestDataset, get_split
from .trainer import Seg3DTrainer


class Seg3DRunner:
    def __init__(self, args):
        self.args = args

    def _build_model(self):
        args = self.args
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
        return model

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
        model = self._build_model()

        # ── Optimizer ────────────────────────────────────────
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=getattr(args, 'weight_decay', 1e-4))
        scheduler = StepLR(optimizer, step_size=getattr(args, 'step_lr', 2),
                           gamma=getattr(args, 'lr_decay', 0.985))
        losses_config = args.losses_config
        if isinstance(args.losses_config, str):
            losses_config = json.loads(args.losses_config)
        trainer = Seg3DTrainer(
            model=model, args=args,
            train_loader=train_loader, eval_loader=val_loader,
            losses_config=losses_config,
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()

    def test(self):
        args = self.args
        if not getattr(args, 'checkpoint', None):
            raise ValueError("测试模式需要指定 --checkpoint <path>")
        if not getattr(args, 'dataset_root', None):
            raise ValueError("测试模式需要指定 --dataset_root <path>")

        test_ds = Seg3DTestDataset(
            root_dir=args.dataset_root,
            target_spacing=args.target_spacing,
            input_patch_size=args.patch_size,
            position_dir=getattr(args, 'position_dir', None),
            label_root=getattr(args, 'label_root', None),
            num_classes=args.num_classes,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        model = self._build_model()
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")

        trainer = Seg3DTrainer(
            model=model, args=args,
            train_loader=None, eval_loader=None,
            losses_config={},
            optimizers=(None, None),
        )
        return_predictions = bool(getattr(args, 'return_predictions', False))
        result = trainer.predict_test(test_loader, return_predictions=return_predictions)
        if return_predictions:
            metrics, pred_outputs = result
            print("Test metrics:", metrics)
            print(f"Returned predictions for {len(pred_outputs)} cases")
            return metrics, pred_outputs

        print("Test metrics:", result)
        return result

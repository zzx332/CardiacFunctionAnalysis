"""
tasks/strain/trainer.py
心肌配准 Trainer：NCC/MSE + Smooth + Dice + RadialDirection 联合损失
迁移自 Myocardial_Strain/trainer.py
"""
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.config import LoggerSingleton


class StrainTrainer:
    """
    配准任务训练器（不继承 BaseTrainer，因为损失计算方式特殊）。
    """

    def __init__(self, model, args, train_loader, eval_loader=None,
                 image_loss='ncc', smooth_weight=1.0, optimizers=(None, None)):
        self.logger = LoggerSingleton()
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.args = args
        self.optimizer = optimizers[0]
        self.scheduler = optimizers[1]

        # 配准专用损失（从 tasks/strain/models/ 导入）
        from .models.losses import NCC, MSE, Grad, MultiClassBatchDiceLoss, RadialDirectionLoss
        if image_loss == 'ncc':
            self.sim_loss = NCC().loss
        elif image_loss == 'mse':
            self.sim_loss = MSE().loss
        else:
            raise ValueError(f"image_loss must be 'ncc' or 'mse', got {image_loss}")

        self.smooth_loss = Grad('l2', loss_mult=getattr(args, 'int_downsize', 2)).loss
        self.dice_loss   = MultiClassBatchDiceLoss(
            torch.tensor([0, 0.6, 1.4]), include_background=False, num_classes=3)
        self.radial_loss = RadialDirectionLoss()
        self.smooth_weight = smooth_weight

    def train_one_epoch(self, epoch, tb_writer):
        self.model.train()
        mean_loss = torch.zeros(1).to(self.device)
        self.optimizer.zero_grad()

        for it, data in enumerate(tqdm(self.train_loader, dynamic_ncols=True, file=sys.stdout)):
            moving, moving_gt, fixed, fixed_gt, directions = data
            moving    = moving.to(self.device,    non_blocking=True)
            moving_gt = moving_gt.to(self.device, non_blocking=True)
            fixed     = fixed.to(self.device,     non_blocking=True)
            fixed_gt  = fixed_gt.to(self.device,  non_blocking=True)

            moved, flow, flow_fullsize = self.model(moving, fixed)
            moved_gt = self.model.transformer(moving_gt[:, None].float(), flow_fullsize, mode='nearest')

            loss_sim     = self.sim_loss(fixed, moved)
            loss_smooth  = self.smooth_loss(0, flow)
            loss_dice    = self.dice_loss(moved_gt.squeeze(), fixed_gt)
            loss_radial  = self.radial_loss(flow_fullsize, fixed_gt == 2, directions)

            total_loss = loss_sim + self.smooth_weight * loss_smooth + loss_dice + loss_radial
            total_loss.backward()
            if getattr(self.args, 'clip_grad_norm', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
            self.optimizer.step()
            self.optimizer.zero_grad()

            mean_loss = (mean_loss * it + total_loss.detach()) / (it + 1)
            tb_writer.add_scalars('Train/Losses', {
                'sim': loss_sim.item(), 'smooth': loss_smooth.item(),
                'dice': loss_dice.item(), 'radial': loss_radial.item(),
            }, it + epoch * len(self.train_loader))
            tb_writer.add_scalar('Train/TotalLoss', mean_loss.item(),
                                 it + epoch * len(self.train_loader))
            tb_writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'],
                                 it + epoch * len(self.train_loader))
        return mean_loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)   # VxmDense 有自己的 save()
        self.logger.info(f"Saved: {path}")

    def train(self):
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)

        for epoch in range(self.args.epochs):
            self.logger.info(f"===== EPOCH {epoch} =====")
            self.train_one_epoch(epoch, tb_writer)
            self.scheduler.step()

            if (epoch + 1) % self.args.save_epoch == 0 or epoch == self.args.epochs - 1:
                ckpt = os.path.join(self.args.model_output_path, f'checkpoint-{epoch}.pt')
                self.save(ckpt)

        return self.model

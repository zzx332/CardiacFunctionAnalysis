"""
tasks/seg3d/trainer.py
3D SAX 分割 Trainer，接近原 SAX_Segmentation/train.py 风格，
但套入 BaseTrainer 骨架并支持 TensorBoard。
"""
import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.base_trainer import BaseTrainer


class Seg3DTrainer(BaseTrainer):
    """
    3D SAX segmentation trainer.
    loss_fn 在 runner 里直接传入（原 config.py 的 loss_func）。
    metric: 平均 Dice（这里简化为 training loss 本身作排名依据，
            更完整的验证参见原 test.py）
    """

    def is_better(self, new_metric, best_metric):
        # loss 越小越好
        return new_metric < best_metric

    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        model.train()
        epoch_loss = 0.0

        for it, (data, seg) in enumerate(tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            # seg shape from dataset: [B, C, d, w, h], take middle slice channel
            seg = seg[:, :, 0]          # [B, C, w, h]
            data, seg = data.to(device), seg.to(device)
            output = model(data)

            total_loss, _ = self.loss_fn(output, seg)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.detach().item()
            tb_writer.add_scalar('Train/Loss', total_loss.item(),
                                 it + epoch * len(data_loader))
            tb_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'],
                                 it + epoch * len(data_loader))

        avg = epoch_loss / len(data_loader)
        return avg

    @torch.no_grad()
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        """
        轻量 eval：以验证集 loss 作为排名依据。
        （完整 3D 滑窗推理建议用独立 test.py，与原项目保持一致）
        """
        self.logger.info(f'Evaluating {sce}...')
        model.eval()
        total_loss = 0.0

        for data, seg in tqdm(data_loader, dynamic_ncols=True, file=sys.stdout):
            seg = seg[:, :, 0]
            data, seg = data.to(self.device), seg.to(self.device)
            output = model(data)
            loss, _ = self.loss_fn(output, seg)
            total_loss += loss.item()

        avg = total_loss / max(len(data_loader), 1)
        self.logger.info(f'[{sce}] avg loss = {avg:.4f}')
        tb_writer.add_scalar(f'{sce}/Loss', avg, epoch)
        # is_better 越小越好，返回负值与 BaseTrainer 逻辑统一
        return -avg

    @torch.no_grad()
    def predict_test(self, data_loader, return_predictions: bool = False):
        self.model.eval()
        dataset_root = getattr(self.args, 'dataset_root', '')
        self.logger.info(f'*********** predict on {dataset_root}')
        test_data_name = os.path.basename(os.path.normpath(dataset_root)) or "seg3d"
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)

        num_classes = getattr(self.args, 'num_classes', 4)
        dsc_all = []
        pred_outputs = {}
        for batch in tqdm(data_loader, desc="predict...", dynamic_ncols=True, file=sys.stdout):
            img = batch['image'][0].to(self.device)          # [H, W, D]
            label = batch['label_onehot'][0].cpu().numpy()   # [C, H, W, D]
            case_id = batch.get('case_id', ['unknown_case'])[0]
            h, w, d = img.shape
            ps_d = max(int(getattr(self.args, 'patch_size', (192, 192, 1))[2]), 1)
            pad_d = (ps_d - 1) // 2
            img_pad = torch.nn.functional.pad(img, (pad_d, pad_d), mode='replicate')

            prob = np.zeros((num_classes, h, w, d), dtype=np.float32)
            for z in range(d):
                patch = img_pad[:, :, z:z+ps_d].permute(2, 0, 1).unsqueeze(0)  # [1, d, H, W]
                pred = self.model(patch.to(self.device))                         # [1, C, H, W]
                prob[:, :, :, z] = pred.squeeze(0).detach().cpu().numpy()

            pred_cls = np.argmax(prob, axis=0)  # [H, W, D]
            if return_predictions:
                # 返回预处理空间下的预测标签图（H, W, D）
                pred_outputs[case_id] = pred_cls.astype(np.int16)
            pred_onehot = np.eye(num_classes, dtype=np.float32)[pred_cls.reshape(-1)] \
                .reshape(h, w, d, num_classes).transpose(3, 0, 1, 2)

            dice_per_case = []
            for cls in range(1, num_classes):
                inter = 2 * np.sum(pred_onehot[cls] * label[cls])
                denom = np.sum(pred_onehot[cls]) + np.sum(label[cls]) + 1e-6
                dice_per_case.append(float(inter / denom))
            dsc_all.append(dice_per_case)

        dsc_all = np.asarray(dsc_all, dtype=np.float32)
        mean_per_class = dsc_all.mean(axis=0) if len(dsc_all) else np.zeros((num_classes - 1,), dtype=np.float32)
        mean_dice = float(mean_per_class.mean()) if len(mean_per_class) else 0.0

        class_name_map = {1: 'RVC', 2: 'LVM', 3: 'LVC'}
        metrics = {'mean_dice': mean_dice, 'num_cases': int(len(dsc_all))}
        for i, v in enumerate(mean_per_class, start=1):
            name = class_name_map.get(i, f'class_{i}')
            metrics[f'dice_{name}'] = float(v)

        os.makedirs(self.args.model_output_path, exist_ok=True)
        report_path = os.path.join(self.args.model_output_path, f'{test_data_name}_test_results.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Seg3D Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"checkpoint: {getattr(self.args, 'checkpoint', '')}\n")
            f.write(f"num_cases: {metrics['num_cases']}\n")
            f.write(f"mean_dice: {metrics['mean_dice']:.6f}\n")
            for i in range(1, num_classes):
                name = class_name_map.get(i, f'class_{i}')
                f.write(f"dice_{name}: {metrics[f'dice_{name}']:.6f}\n")

        tb_writer.add_scalar(f'Test{test_data_name}/MeanDice', metrics['mean_dice'], 0)
        for i in range(1, num_classes):
            name = class_name_map.get(i, f'class_{i}')
            tb_writer.add_scalar(f'Test{test_data_name}/Dice_{name}', metrics[f'dice_{name}'], 0)
        self.logger.info("Test metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if k != 'num_cases']))
        self.logger.info(f"Saved test report: {report_path}")
        if return_predictions:
            return metrics, pred_outputs
        return metrics

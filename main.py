"""
main.py  —  CardiacAI 统一入口
用法：
  python main.py --task landmark  [--debug] [--epochs N] ...
  python main.py --task view      ...
  python main.py --task seg2d     ...
  python main.py --task seg3d     ...
"""
import argparse
import os
import logging
import torch

# 导入注册表（同时触发各子任务的 register_task 调用）
import tasks
from tasks import TASK_REGISTRY
from common.config import LoggerSingleton

import warnings
warnings.filterwarnings("ignore", message="The transform `ToTensor\(\)` is deprecated")
# ─────────────────────────────────────────────────────────
# 公共参数（每个任务都需要）
# ─────────────────────────────────────────────────────────
def build_common_args():
    p = argparse.ArgumentParser(
        description='CardiacAI — 统一心脏分析训练框架',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--task', required=True,
                   choices=['landmark', 'view', 'seg2d', 'seg3d', 'strain', 'cardiodx'],
                   help='要运行的任务')
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                   help='train=训练, test=推理评估')

    # 路径
    p.add_argument('--train_data_path', type=str, default='')
    p.add_argument('--val_data_path',   type=str, default='')
    p.add_argument('--test_data_path',  type=str, default='')
    # 测试专用
    p.add_argument('--checkpoint',    type=str,   default=None,
                   help='测试时加载的 checkpoint 路径')
    p.add_argument('--save_path',     type=str,   default=None,
                   help='推理结果保存目录')
    p.add_argument('--sw_batch_size', type=int,   default=4,
                   help='滑窗推理并行 batch')
    p.add_argument('--overlap',       type=float, default=0.5,
                   help='滑窗重叠率')
    p.add_argument('--tta',           action='store_true',
                   help='测试时数据增强（翻转 TTA）')
    p.add_argument('--img_root_path',   type=str, default='',
                   help='DICOM 图像根目录（landmark 任务）')
    p.add_argument('--model_output_path',      type=str, default='./checkpoints')
    p.add_argument('--tensorboard_output_path', type=str, default='./runs')
    p.add_argument('--log_file',        type=str, default=None)

    # 训练超参
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lr_decay',     type=float, default=0.985,
                   help='StepLR gamma（view/seg3d 任务）')
    p.add_argument('--step_lr',      type=int,   default=2,
                   help='StepLR step size')
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--random_seed',  type=int,   default=42)
    p.add_argument('--save_epoch',   type=int,   default=50)
    p.add_argument('--clip_grad_norm', action='store_true')
    p.add_argument('--earlystop',    action='store_true')
    p.add_argument('--earlystop_patient', type=int, default=20)
    p.add_argument('--debug',        action='store_true')
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--target_size', type=int, nargs='+', default=[320, 256])
    p.add_argument('--target_spacing', type=float, default=1.0)
    p.add_argument('--num_classes',    type=int,   default=1)
    p.add_argument('--in_channels',    type=int,   default=1)
    p.add_argument("--losses_config", type=str, default='{"focal_loss":{"weight":1, "gamma":2}}',
                        help="Loss functions and their weights, e.g., 'cross_entropy:0.5 mse:0.5'")
    # ── landmark 专用 ──────────────────────────────────────
    p.add_argument('--sigma',          type=int,   default=3)
    # p.add_argument('--target_size',    type=int,   nargs=2, default=[160, 160])

    p.add_argument('--base_channel',   type=int,   default=32)
    p.add_argument('--num_joints',     type=int,   default=2)
    p.add_argument('--head',           type=str,   default='srp',
                   choices=['simple', 'srp'])
    p.add_argument('--repeat_factor',  type=int,   default=1)
    # ── view 专用 ────────────────────────────────────────
    p.add_argument('--arch',           type=str,   default='efficientnetv2_s',
                   choices=['efficientnetv2_s', 'resnet', 'unet', 'network'])
    p.add_argument('--downsample',     action='store_true')
    # ── seg2d 专用 ────────────────────────────────────────
    p.add_argument('--img_c',          type=int,   default=1)
    p.add_argument('--deep_supervision', action='store_true')
    p.add_argument("--network_config", type=str, default='{"bilinear": true, "base_c":48}',
                        help="Network Config")
    p.add_argument('--warm_up_epochs', type=int, default=5)

    # ── seg3d 专用 ─────────────────────────────────────
    p.add_argument('--fold',           type=int,   default=0)
    p.add_argument('--dataset_root',   type=str,   default='')
    p.add_argument('--label_root',     type=str,   default=None)
    p.add_argument('--position_dir',   type=str,   default=None)
    p.add_argument('--patch_size',     type=int,   nargs=3, default=[192, 192, 1])
    p.add_argument('--drop_rate',      type=float, default=None)

    # ── strain 专用（VoxelMorph 配准）─────────────────
    p.add_argument('--image_loss',   type=str,   default='ncc',
                   choices=['ncc', 'mse'])
    p.add_argument('--smooth_weight', type=float, default=1.0)
    p.add_argument('--int_steps',    type=int,   default=7)
    p.add_argument('--int_downsize', type=int,   default=2)
    p.add_argument('--bidir',        action='store_true')
    p.add_argument('--half_res',     action='store_true')
    p.add_argument('--enc', type=int, nargs='+', default=None)
    p.add_argument('--dec', type=int, nargs='+', default=None)
    p.add_argument('--iterations_per_epoch', type=int, default=100)

    # ── cardiodx 专用 ─────────────────────────────────
    p.add_argument('--sampling_strategy', type=str, default='progressively_balanced',
                   choices=['instance', 'class', 'progressively_balanced'])
    p.add_argument('--total_steps', type=int, default=None,
                   help='Total training steps (cardiodx). Defaults to epochs*100.')
    p.add_argument('--iter_per_epoch', type=int, default=None)
    p.add_argument('--lrf',          type=float, default=0.01)
    p.add_argument('--optimizer',    type=str,   default='AdamW',
                   choices=['Adam', 'AdamW', 'SGD'])
    p.add_argument('--scheduler',    type=str,   default='cosine',
                   choices=['cosine', 'poly', 'constant', 'warm_cosine'])
    p.add_argument("--train_aug_config_cardiodx", type=str, default='{"flip_prob": 0.5, "rotation_prob": 0.5, "scale_prob": 0, "brightness_prob": 0, "contrast_prob": 0}',
                        help="Train augmentation config")
    p.add_argument("--val_aug_config_cardiodx", type=str, default='{"flip_prob": 0.5}',
                        help="Validation augmentation config")

    return p


def main():
    parser = build_common_args()
    args = parser.parse_args()

    # 输出目录
    os.makedirs(args.model_output_path, exist_ok=True)
    os.makedirs(args.tensorboard_output_path, exist_ok=True)

    # 统一 logger
    log_file = args.log_file or os.path.join(args.model_output_path, 'train.log')
    logger = LoggerSingleton(log_file=log_file)
    logger.info(f"Task      : {args.task}")
    logger.info(f"Device    : {args.device}")
    logger.info(f"Epochs    : {args.epochs}")

    # patch_size 需要是 tuple
    args.patch_size = tuple(args.patch_size)
    args.target_size = tuple(args.target_size)

    # 查找并运行对应任务
    if args.task not in TASK_REGISTRY:
        raise ValueError(f"Task '{args.task}' not in registry: {list(TASK_REGISTRY.keys())}")

    RunnerClass = TASK_REGISTRY[args.task]
    runner = RunnerClass(args)

    if args.mode == 'test':
        if not hasattr(runner, 'test'):
            raise NotImplementedError(
                f"Task '{args.task}' 尚未实现 test() 方法。"
                f"当前支持测试的任务：seg2d")
        runner.test()
    else:
        runner.run()


if __name__ == '__main__':
    main()

#!/bin/bash
# run/seg3d.sh
# python main.py \
#   --task seg3d \
#   --dataset_root /home/zzx/data/ACDC/training \
#   --position_dir /home/zzx/data/ACDC/Position \
#   --num_classes 4 \
#   --base_c 48 \
#   --arch unet \
#   --target_spacing 1.25 \
#   --patch_size 192 192 1 \
#   --fold 0 \
#   --model_output_path ./checkpoints/seg3d \
#   --tensorboard_output_path ./runs/seg3d \
#   --epochs 300 \
#   --batch_size 10 \
#   --lr 3e-4 \
#   --lr_decay 0.985 \
#   --step_lr 2 \
#   --num_workers 16 \
#   --save_epoch 300 \
#   --losses_config='{"multiclass_batch_dice_loss_seg3d":{"weight":1}}'
  # --checkpoint ./checkpoints/seg3d/checkpoint-299.pt \
python main.py --task seg3d --mode test \
  --dataset_root /home/zzx/data/ACDC/training \
  --checkpoint ./checkpoints/seg3d/best_checkpoint.pt \
  --position_dir /home/zzx/data/ACDC/Position \
  --num_classes 4 \
  --base_c 48 \
  --arch unet \
  --target_spacing 1.25 \
  --patch_size 192 192 1 \
  --model_output_path ./checkpoints/seg3d \
  --tensorboard_output_path ./runs/seg3d_test \
  --batch_size 1 \
  --num_workers 8

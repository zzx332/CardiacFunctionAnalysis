#!/bin/bash
# run/seg3d.sh
python main.py \
  --task seg3d \
  --dataset_root /media/lxx/data/data/Heart/ACDC/training \
  --position_dir /media/lxx/data/data/Heart/ACDC/Position \
  --num_classes 4 \
  --base_c 48 \
  --arch unet \
  --target_spacing 1.25 \  
  --patch_size 192 192 1 \
  --fold 0 \
  --model_output_path ./checkpoints/seg3d \
  --tensorboard_output_path ./runs/seg3d \
  --epochs 300 \
  --batch_size 10 \
  --lr 3e-4 \
  --lr_decay 0.985 \
  --step_lr 2 \
  --num_workers 16 \
  --save_epoch 300

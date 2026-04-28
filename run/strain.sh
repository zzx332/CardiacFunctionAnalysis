#!/bin/bash
# run/strain.sh — 心肌配准（VoxelMorph）
python main.py \
  --task strain \
  --train_data_path ../Myocardial_Strain/dataset/train_file_paths.pkl \
  --model_output_path ./checkpoints/strain \
  --tensorboard_output_path ./runs/strain \
  --target_size 128 128 \
  --image_loss mse \
  --smooth_weight 0.01 \
  --epochs 300 \
  --iterations_per_epoch 100 \
  --batch_size 8 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 2e-4 \
  --lrf 0.01 \
  --save_epoch 50 \
  --clip_grad_norm \

#!/bin/bash
# run/landmark.sh
# python main.py \
#   --task landmark \
#   --img_root_path /home/zzx/data/Landmarks \
#   --train_data_path ./tasks/landmark/dataset/training_rv_inserts.json \
#   --val_data_path ./tasks/landmark/dataset/val_rv_inserts.json \
#   --model_output_path ./checkpoints/landmark \
#   --tensorboard_output_path ./runs/landmark \
#   --epochs 300 \
#   --batch_size 8 \
#   --lr 3e-4 \
#   --head srp \
#   --num_joints 2 \
#   --sigma 3 \
#   --target_spacing 1.25 \
#   --target_size 160 160 \
#   --num_workers 8 \
#   --save_epoch 50 \
#   --clip_grad_norm

python main.py --task landmark --mode test \
  --img_root_path /home/zzx/data/Landmarks \
  --test_data_path ./tasks/landmark/dataset/test_rv_inserts.json \
  --checkpoint ./checkpoints/landmark/best_checkpoint.pt \
  --model_output_path ./checkpoints/landmark \
  --tensorboard_output_path ./runs/landmark_test \
  --batch_size 8 \
  --head srp \
  --num_joints 2 \
  --sigma 3 \
  --target_spacing 1.25 \
  --target_size 160 160 \
  --num_workers 8
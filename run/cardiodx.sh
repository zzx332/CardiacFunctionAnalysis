#!/bin/bash
# run/cardiodx.sh — 心脏疾病诊断二分类
# python main.py \
#   --task cardiodx \
#   --train_data_path ../CardioDx/dataset/train_dataset.pkl \
#   --val_data_path   ../CardioDx/dataset/val_dataset.pkl \
#   --model_output_path ./checkpoints/cardiodx_instance \
#   --tensorboard_output_path ./runs/cardiodx_instance \
#   --arch resnet \
#   --in_channels 3 \
#   --num_classes 1 \
#   --target_size 352 \
#   --target_spacing 1.25 \
#   --sampling_strategy instance \
#   --iter_per_epoch 200 \
#   --total_steps 20000 \
#   --batch_size 16 \
#   --optimizer Adam \
#   --scheduler constant \
#   --lr 5.21e-4 \
#   --lrf 0.05 \
#   --weight_decay 1e-4 \
#   --num_workers 8 \
#   --losses_config='{"ce_loss":{"weight":1,"class_weights":[1]}}'

python main.py --task cardiodx --mode test \
  --test_data_path ./tasks/cardiodx/dataset/test_dataset.pkl \
  --checkpoint ./checkpoints/cardiodx_instance/best.pt \
  --model_output_path ./checkpoints/cardiodx_instance \
  --tensorboard_output_path ./runs/cardiodx_instance_test \
  --arch resnet \
  --in_channels 3 \
  --num_classes 1 \
  --target_size 352 \
  --target_spacing 1.25 \
  --batch_size 16 \
  --num_workers 8
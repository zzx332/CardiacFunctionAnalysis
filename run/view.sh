#!/bin/bash
# run/view.sh
python main.py \
  --task view \
  --train_data_path ../Cardiac_View/dataset/SADSB_train.pkl \
  --val_data_path   ../Cardiac_View/dataset/SADSB_val.pkl \
  --arch efficientnetv2_s \
  --model_output_path ./checkpoints/view \
  --tensorboard_output_path ./runs/view \
  --target_size 352 \
  --num_classes 3 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 16 \
  --save_epoch 20 \
  --earlystop \
  --earlystop_patient 5 \
  --losses_config='{"focal_loss":{"weight":1, "gamma":2}}' \
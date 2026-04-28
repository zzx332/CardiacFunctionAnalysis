#!/bin/bash
# run/seg2d.sh
python main.py \
  --task seg2d \
  --train_data_path ../CH4_Segmentation/dataset/MnMs2_train.pkl \
  --val_data_path ../CH4_Segmentation/dataset/MnMs2_val.pkl \
  --num_classes 4 \
  --model_output_path ./checkpoints/seg2d \
  --tensorboard_output_path ./runs/seg2d \
  --epochs 300 \
  --batch_size 16 \
  --lr 0.01 \
  --lrf 0.001\
  --scheduler poly \
  --optimizer SGD \
  --repeat_factor 10 \
  --num_workers 8 \
  --save_epoch 100 \
  --target_size 320 256 \
  --deep_supervision \
  --network_config '{"layers": [32, 64, 128, 256, 512, 512, 512],
                        "deep_supervision": true, 
                        "nonlinearity":"LeakyReLU", "dropout": 0}' \
  --losses_config='{"multiclass_batch_dice_loss":{"weight":1, "include_background":false},
                "ce_loss":{"weight":1}}' \

# python main.py --task seg2d --mode test \
#   --test_data_path ../CH4_Segmentation/dataset/MnMs2_test.pkl \
#   --checkpoint ./checkpoints/seg2d/checkpoint-299.pt \
#   --num_classes 6 \
#   --network_config '{"layers":[32,64,128,256,512,512,512],"deep_supervision":true,"nonlinearity":"ReLU","dropout":0}' \
#   --target_size 320 256 \
#   --sw_batch_size 4 \
#   --overlap 0.5 \
#   --tta \
#   --save_path ./results/seg2d

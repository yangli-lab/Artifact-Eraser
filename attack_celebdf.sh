#!/bin/bash

DATASET=celebdf
CLASSIFIER=efficient
ATTACK_METHOD=fgsm
EPOCH=50
FRAME=14
ALPHA=0.0002
NAME=attack_latent_nc_${DATASET}_${CLASSIFIER}_${ATTACK_METHOD}yes_${EPOCH}_${FRAME}_${ALPHA}
CUDA_VISIBLE_DEVICES=4 python3 ./attack/attack_celebdf.py \
--dataset ${DATASET} \
--stylegan_size 1024 \
--encoder_type Encoder4Editing \
--device cuda:0 \
--start_from_latent_avg True \
--classifier_ckpt classifier_path \
--classifier ${CLASSIFIER} \
--attack_method ${ATTACK_METHOD} \
--mask full \
--beta 0.02 \
--alpha ${ALPHA} \
--epoch ${EPOCH} \
--frames ${FRAME} \
--loss_type mse_loss \
--lpips_lambda 0 \
--lpips_type alex \
--checkpoint_path /path/to/e4e_model \
--id_lambda 0 \
--mse_lambda 0 \
--dataset_dir data_dir \
--label_file label_for_dfdc_dataset \
--video_level False \
--attack_num 50 \
--log_dir log_path/${NAME} \
2>&1 | tee log_path/${NAME}.log \
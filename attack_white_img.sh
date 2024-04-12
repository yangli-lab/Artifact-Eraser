#!/bin/bash

DATASET=ffpp
CLASSIFIER=efficient
ATTACK_METHOD=mifgsm
EPOCH=8
FRAME=14
ALPHA=2
NAME=comtest_${DATASET}_${CLASSIFIER}_${ATTACK_METHOD}_${EPOCH}_${FRAME}_${ALPHA}
CUDA_VISIBLE_DEVICES=4 python3 ./attack/attack_white_img.py \
--stylegan_size 1024 \
--encoder_type Encoder4Editing \
--device cuda:0 \
--start_from_latent_avg True \
--classifier_ckpt classifier_path \
--beta 25 \
--alpha ${ALPHA} \
--dataset ${DATASET} \
--classifier  ${CLASSIFIER} \
--attack_method ${ATTACK_METHOD} \
--epoch ${EPOCH} \
--frames ${FRAME} \
--loss_type mse_loss \
--checkpoint_path /path/to/e4e_model \
--lpips_lambda 0 \
--lpips_type alex \
--id_lambda 0 \
--mse_lambda 0 \
--dataset_dir dataset_path \
--label_file label_file_for_dfdc_dataset \
--video_level False \
--log_dir log_dir/${NAME} \
2>&1 | tee log_dir/${NAME}.log
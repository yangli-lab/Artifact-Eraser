#!/bin/bash

DATASET=ffpp
CLASSIFIER=efficient
ATTACK_METHOD=fgsm
EPOCH=2
FRAME=16
ALPHA=2
BETA=5
KAPA=0.2
NAME=attack_img_mask_${DATASET}_${CLASSIFIER}_${ATTACK_METHOD}_${FRAME}_${ALPHA}
CUDA_VISIBLE_DEVICES=7 python3 ./attack/attack_mask_img.py \
--stylegan_size 1024 \
--encoder_type Encoder4Editing \
--device cuda:0 \
--start_from_latent_avg True \
--classifier_ckpt classifier_path \
--beta ${BETA} \
--alpha ${ALPHA} \
--kapa ${KAPA} \
--dataset ${DATASET} \
--classifier  ${CLASSIFIER} \
--attack_method ${ATTACK_METHOD} \
--epoch ${EPOCH} \
--checkpoint_path /path/to/e4e_model \
--frames ${FRAME} \
--loss_type mse_loss \
--lpips_lambda 0 \
--lpips_type alex \
--id_lambda 0 \
--mse_lambda 0 \
--dataset_dir data_dir \
--label_file label_file_for_dfdc_dataset \
--video_level True \
--log_dir log_path/${NAME} \
2>&1 | tee log_path/${NAME}.log \
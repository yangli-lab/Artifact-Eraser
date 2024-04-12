#!/bin/bash
DATASET=ffpp
CLASSIFIER=efficient
ATTACK_METHOD=fgsm
EPOCH=1000
FRAME=14
BETA=0.05
ALPHA=0.0008
NAME=weights_mask_${DATASET}_${CLASSIFIER}_${ATTACK_METHOD}_${EPOCH}_${FRAME}_${BETA}_${ALPHA}
CUDA_VISIBLE_DEVICES=4 python3 ./attack/attack.py \
--dataset ${DATASET} \
--stylegan_size 1024 \
--encoder_type Encoder4Editing \
--device cuda:0 \
--start_from_latent_avg true \
--classifier_ckpt /hd4/liyang/classifier_for_attack_to_video_detection/classifier_exps/efficient_celebdf_nosoftmax/snap/snapshot_2000.pth \
--classifier  ${CLASSIFIER} \
--attack_method ${ATTACK_METHOD} \
--alpha ${ALPHA} \
--kapa 0.2 \
--epoch ${EPOCH} \
--frames ${FRAME} \
--loss_type mse_loss \
--checkpoint_path /path/to/e4e_model \
--lpips_lambda 0 \
--lpips_type alex \
--id_lambda 0 \
--mse_lambda 0 \
--dataset_dir /hd6/guanweinan/Data/celeb-DF-v2s-face/Celeb-synthesis \
--label_file /hd5/liyang/attack_to_video_detection/dfdc_deepfake_challenge/folds_margin2.csv \
--attack_num 500 \
--log_dir /hd7/liyang/attack_to_video_detection/attack_exp_2/${NAME} \
2>&1 | tee /hd7/liyang/attack_to_video_detection/attack_exp_2/${NAME}.log \
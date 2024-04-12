<div align="center">
    <div>
    <a href="arxiv.index"><img src="https://img.shields.io/badge/Arxiv-2401:08276-red"/></a>
    <!-- <a href=> -->
    </div>

    <h1> Counterfactual Explanations for Face Forgery Detection Via Adversarial Removal of Artifacts </h1>

</div>
<br>
Training code for the paper [Counterfactual Explanations for Face Forgery Detection Via Adversarial Removal of Artifacts]().

## Getting Started

### Installation

- clone the repository:
```
git clone 
cd
```


```
conda create -n AR python=3.7
conda activate AR
pip install -r requirements.txt
```

### Pretrained Models
Please download the pretrained models from the following links and save them in the 'Save_Folder'

|    Path  | Save_Folder | Description 
| :------- | :---------- | :----------
| [E4E Encoder](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)    | ./pretrained_models/    | Encoder4editing model pretrained by [omertov](https://github.com/omertov/encoder4editing).
| [StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)    | ./pretrained_models/     | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
| [IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)    | ./pretrained_models/    | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID loss during training.
| [MAT](https://drive.google.com/file/d/1lYyUe99Goh1YCilt1IOiD9oMO6ig8j1o/view)    | ./classifiers/multiple_attention/pretrained/    | Pretrained deepfake detection model, MAT, by [yoctta](https://github.com/yoctta/multiple-attention).
| [RECCE](https://github.com/VISION-SJTU/RECCE/issues/2#issuecomment-1221564190)    | ./classifiers/RECCE/pretrained_models/ |  Pretrained deepfake detection model, RECCE, by [VISION-SJTU](https://github.com/VISION-SJTU/RECCE).



## Training
<bar>

### Training the e4e encoder
The training code of [E4E encoder](https://arxiv.org/abs/2102.02766) is borrowed from their [official repository](https://github.com/omertov/encoder4editing).
To train the e4e encoder, make sure the paths to the required models, and prepare your dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/train_finetune.py --dataset_type ffhq_encode \
--exp_dir ./exp/celebdf_decoder \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 0 \
--progressive_start 0 \
--train_encoder 0 \
--train_decoder 1 \
--lpips_lambda 1.0 \
--id_lambda 0.1 \
--l2_lambda 1.0 \
--val_interval 1000 \
--max_steps 1000 \
--image_interval 100 \
--board_interval 100 \
--stylegan_size 1024 \
--save_interval 500 \
--checkpoint_path /path/to/pretrained/e4e_model \
--workers 4 \
--batch_size 8 \
--test_batch_size 2 \
--test_workers 4 
```

### Training Classifier(Optional)
You can use your pretrained deepfake classifier model, or train with the provided code, please refer to the `classifiers` folder.
Let's take training `MAT` as an example
```bash
CUDA_VISIBLE_DEVICES=0 python3 ./classifiers/train_mat.py \
--dataset_name ffpp \
--root_dir ffpp_path \
--batch_size 32 \
--read_method frame_by_frame \
--lr 1e-3 \
--momentum 0.9 \
--weight_decay 1e-6 \
--max_iteration 1200 \
--max_warmup_iteration 0 \
--epochs 1000 \
--num_workers 2 \
--snapshot_frequency 100 \
--snapshot_template snapshot \
--exp_root save_folder \
```

### Artifact Eraser Attack
Make sure you have specified a pretrained e4e model by `--checkpoint_path` and pretrained deepfake classifier by `--classifier_ckpt`, and run the code bellow. You can specify the attack strength and attack times by setting parameters `ALPHA` and `EPOCH`.
```bash
DATASET=celebdf
CLASSIFIER=efficient
ATTACK_METHOD=fgsm
EPOCH=50
FRAME=14
ALPHA=0.0002
NAME=attack_latent_nc_${DATASET}_${CLASSIFIER}_${ATTACK_METHOD}_${EPOCH}_${FRAME}_${ALPHA}
CUDA_VISIBLE_DEVICES=0 python3 ./attack/attack_celebdf.py \
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
--id_lambda 0 \
--mse_lambda 0 \
--dataset_dir data_dir \
--label_file label_for_dfdc_dataset \
--video_level False \
--attack_num 50 \
--log_dir log_path/${NAME} \
2>&1 | tee log_path/${NAME}.log \
```

## Acknowledgments
<bar>

This code borrows heavily from [e4e_encoder](https://github.com/omertov/encoder4editing)


## Citation
If you find our work interesting, please feel free to cite our paper:
```bibtex
put the citation here
```

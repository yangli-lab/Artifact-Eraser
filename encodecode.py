import sys
import os
from data_transform.transform_config import get_transform
from models.encoders.psp_encoders import Encoder4Editing, BackboneEncoderUsingLastLayerIntoW, GradualStyleEncoder
from models.stylegan2.model import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from PIL import Image
import numpy as np

MODEL_PATH_IR_SE50 = 'pretrained_models/model_ir_se50.pth'
def get_keys(ckpt, name):
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    state_dict_filt = {k[len(name) + 1:]: v for k, v in ckpt.items() if k[: len(name)] == name}
    return state_dict_filt


class EncodeDecode(nn.Module):
    def __init__(self, opts):# opts.stylegan_size, opts.encoder_type
        super(EncodeDecode, self).__init__()
        self.opts = opts
        self.encoder = self.get_encoder(
            50, mode='ir_se', encoder_type=opts.encoder_type)
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier = 2)
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    def move_model(self, device: str):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        # self.face_pool = self.face_pool.to(device)
        self.latent_avg = self.latent_avg.to(device)

    def get_encoder(self, num_layers, mode='ir', encoder_type='GradualStyleBlock'):#opts.stylegan_size
        if self.opts == None:
            self.opts.stylegan_size = 128
        if encoder_type == 'Encoder4Editing':
            encoder = Encoder4Editing(num_layers, mode, self.opts)
        elif encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = BackboneEncoderUsingLastLayerIntoW(num_layers, mode, self.opts)
        elif encoder_type == 'GradualStyleBlock':
            encoder = GradualStyleEncoder(num_layers, mode, self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(encoder_type))
        return encoder
    
    def load_weights(self):#opts.checkpoint_path, opts.pretrained_model, opts.stylegan_weights
        if self.opts.checkpoint_path:
            ckpt = torch.load(self.opts.checkpoint_path, map_location = 'cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict = True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict = True)
            self.load_latent_avg(ckpt)
        else:
            print('loading encoder weights from ir_se50')
            encoder_ckpt = torch.load(MODEL_PATH_IR_SE50)
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('loading decoder weight from pretrained')
            decoder_ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(decoder_ckpt['g_ema'], strict=False)


    def forward(self, x, input_code = False, latent_mask = None, return_latents = False,
            resize = True, inject_latent = None, alpha = None, randomize_noise = True, return_codes = False):#opts.start_from_latent_avg
        if input_code:
            codes = x    
        else:
            codes = self.encoder(x)
            #
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0
        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
            input_is_latent = input_is_latent,
            randomize_noise = randomize_noise,
            return_latents = return_latents)
        if resize:
            images = self.face_pool(images)
        if return_latents:
            return images, result_latent
        if return_codes:
            return images, codes
        else:
            return images


    def load_latent_avg(self, ckpt, repeat=None):#device, start_from_latent_avg
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            with torch.no_grad():
                print('here')
                self.latent_avg = self.decoder.mean_latent(1000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', default = False, help = 'if use pretrained model')
    parser.add_argument('--im_path', default = '', help = 'test image path')
    parser.add_argument('--im_save_path', default = '', help = 'path to save the result')
    parser.add_argument('--stylegan_size', default = 1024, type = int, help = '')
    parser.add_argument('--encoder_type', default = 'Encoder4Editing', help = 'encoder type')
    parser.add_argument('--stylegan_weights', default = 'pretrained_models/', type=str, help = 'stylegan model path')
    parser.add_argument('--start_from_latent_avg', action = 'store_true', help = 'Whether to add average latent vector to generate codes from encoder.')
    parser.add_argument('--device', default = 'cuda:0', help = 'GPU choose')
    args = parser.parse_args()
    e4e_model = EncodeDecode(args).to(args.device)
    e4e_model.eval()
    transform_dict = get_transform()
    test_transform = transform_dict['test']
    image = Image.open(args.im_path)
    image = image.convert('RGB')
    image = test_transform(image)
    with torch.no_grad():
        final_input = image.unsqueeze(dim = 0).to(args.device).float()
        print(final_input.shape)
        x = final_input
        result_img, latents = e4e_model(x, return_latents = True)
        print(result_img)

    result_img = result_img[0].cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    result_img = ((result_img + 1) / 2)
    result_img[result_img < 0] = 0
    result_img[result_img > 1] = 1
    result_img = result_img * 255
    result_img = Image.fromarray(result_img.astype('uint8'))
    result_img.save(args.im_save_path)
    
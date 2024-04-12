import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import os
import sys
import shutil
sys.path.append(".")
sys.path.append("..")
sys.path.append("./classifiers/multiple_attention")
sys.path.append("./classifiers/RECCE")
sys.path.append("./classifiers/I3D")

from argparse import Namespace
import numpy as np
import cv2
import time
from argparse import ArgumentParser
import random

from criteria.lpips.lpips import LPIPS
from criteria import id_loss, moco_loss
from encodecode import EncodeDecode
from classifiers.slowfast import slowfast50, load_state
from dataset.DataSet.dataset_celebdf import CELEB_DF
from dataset.DataSet.dataset_dfdc import DFDC
from dataset.DataSet.dataset_ffpp import FFPP
from classifiers.train_xception import xception
from classifiers.train_efficient import load_pretrained_efficient
from classifiers.multiple_attention.mat_models.MAT import MAT
from classifiers.RECCE.recce_model.network import Recce
from classifiers.I3D.model import  load_pretrained_I3D_RES
from classifiers.multiple_attention.config import train_config
from data_transform.transform_config import get_transform
from utils.common import tensor2im
from glob import glob
from PIL import Image

# set seed, reproduce the result
seed = 36
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

#opts:stylegan_size, encoder_type, checkpoint_path, stylegan_weights, device, start_from_latent_avg, epsilon, alpha
class Attack_Video(nn.Module):
    def __init__(self, opts):
        """
        classifier_ckpt:
        e4e_ckpt:
        attack_method:
        alpha:
        beta:
        epoch:
        device:
        """
        super(Attack_Video, self).__init__()
        if opts.classifier == 'xception':
            self.classifier = xception(num_classes = 2, pretrained = 'imagenet').to(opts.device)
            ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(ckpt, strict = True)
        elif opts.classifier == 'efficient':
            self.classifier = load_pretrained_efficient(encoder_name = "tf_efficientnet_b7_ns", classes = 2, device = opts.device)
            ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(ckpt, strict = True)
        elif opts.classifier == 'slowfast':
            self.classifier = slowfast50(num_classes = 2, alpha = 4, beta = 0.125, tau = 16).to(opts.device)
            ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(ckpt, strict = True)
        elif opts.classifier == 'mat':
            name = 'a1_b5_b2'
            config = train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer='b2',
            ckpt = 'checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
            self.classifier =  MAT(**config.net_config).to(opts.device)
            stat_dict = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(stat_dict, strict = False)
        elif opts.classifier == 'recce':
            self.classifier = Recce(num_classes = 1).to(opts.device)
            self.classifier.fc = nn.Linear(2048, 2)
            self.classifier.to(opts.device)
            weights = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(weights)
        elif opts.classifier == 'I3D':
            self.classifier = load_pretrained_I3D_RES(device = opts.device)
            state_dict = torch.load(opts.classifier_ckpt, map_location = 'cpu')
            self.classifier.load_state_dict(state_dict, strict = True)

        self.attack_method = opts.attack_method
        self.alpha = opts.alpha
        self.beta = opts.beta
        self.decay = 0.99
        self.np_index = 0

        self.e4e = EncodeDecode(opts).to(opts.device)
        
        self.device = opts.device
        self.opts = opts
        # TODO: lpips loss and other id loss
        # opts.lpips_lambda, opts.lpips_type
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type = opts.lpips_type).to(self.device).eval()
        # opts.id_lambda
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        # opts.mse_lambda
        if self.opts.mse_lambda > 0:
            self.mse_loss = nn.MSELoss().to(self.device).eval()
        # loss func
        self.loss_fn = self.select_lossfn(opts.loss_type)
    
    def from_ims_to_latent(self, ims):
        return self.e4e(ims, return_codes = True)

    def from_ims_to_latent_directly(self, ims):
        """compute the codes directly"""
        codes = self.e4e.encoder(ims)
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.e4e.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.e4e.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes
    
    def from_latent_to_ims(self, codes, resize = True):
        """
        reconstruct ims from latents
        no need to return latents
        """
        images, result_latent = self.e4e.decoder([codes], input_is_latent = True,
                                        randomize_noise = True,
                                        return_latents = False
                                        )
        if resize:
            images = self.e4e.face_pool(images)
        return images

    def attack(self, x, label):
        """
        x:
        label:
        """
        if self.attack_method == 'fgsm':
            attack_result = self.fgsm_attack(x, label)
            return attack_result

        elif self.attack_method == 'mifgsm':
            attack_result = self.mifgsm_attack(x, label)
            return attack_result
            
        elif self.attack_method == 'pgd':
            attack_result = self.pgd_attack(x, label)
            return attack_result

        else:
            raise ValueError("the value of attack_method must be in ['fgd', 'fgsm', 'mifgsm']")
    

    def select_lossfn(self, loss_type: str):
        if loss_type == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'mse_loss':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be in ['cross_entropy', 'mse_loss']")
        return loss_fn

    # input：x shape [faces, frames, channels, h, w]
    def mifgsm_attack(self, x, label):
        ims = x.squeeze(dim = 0).clone().detach().to(self.device)
        if self.opts.video_level:
            target = label.clone().detach().to(self.device).float()
        else:
            target = label.clone().detach().to(self.device).float()
            target = target.repeat(ims.shape[0])
        with torch.no_grad():
            latent_advs = self.from_ims_to_latent_directly(ims)
        # momentumn
        momentumn = torch.zeros_like(latent_advs).detach().to(self.device)

        # latent_advs = self.from_ims_to_latent_directly(ims)
        ori_latent = latent_advs.clone().detach()
        latent_advs.requires_grad = True
        ims_adv = self.from_latent_to_ims(latent_advs)
        channel = latent_advs.shape[1]
        if self.opts.mask == 'full':
            mask = [a for a in range(0, 18)]
        elif self.opts.mask == 'mid':
            mask = [a for a in range(6, 12)]
        elif self.opts.mask == 'first':
            mask = [a for a in range(0, 6)]
        elif self.opts.mask == 'last':
            mask = [a for a in range(12, 18)]
        mask = torch.Tensor(mask).long()

        for i in range(self.opts.epoch):
            # [N, C, H, W]
            # N, C, H, W = ims_adv.shape
            #TODO:send to classifier
            # check data shape
            loss_value = 0.0
            if self.opts.lpips_lambda > 0:
                loss_lpips = self.lpips_loss(ims_adv, ims)
                print('loss_lpips:', loss_lpips)
                loss_value += loss_lpips * self.opts.lpips_lambda

            if self.opts.mse_lambda > 0:
                loss_l2 = self.mse_loss(ims_adv, ims)
                print('loss_l2:', loss_l2)
                loss_value += loss_l2 * self.opts.mse_lambda
            
            if self.opts.id_lambda > 0:
                loss_id, sim_improvement, id_logs = self.id_loss(ims_adv, ims, ims)
                print('loss id:', loss_id)
                loss_value += loss_id * self.opts.id_lambda

            if self.opts.video_level:
                ims_adv = ims_adv.unsqueeze(dim = 0)
                ims_adv = ims_adv.permute(0, 2, 1, 3, 4)
            softmax = nn.Softmax(dim = 1)
            score = self.classifier(ims_adv)
            print(score)
            if self.opts.classifier == 'recce':
                decide_score = torch.sigmoid(score)
            else:
                classifier_result = softmax(score)
                decide_score = classifier_result[:, 1]
            # calculate loss
            loss_value += self.loss_fn(decide_score, target)
            print('total loss:', loss_value)
            # [1, channel, n, h, w]
            grads = autograd.grad(loss_value, latent_advs, retain_graph = False, create_graph = False)[0]
            # grads = grads / torch.mean(torch.abs(grads), dim = (1, 2), keepdim = True)
            grads = grads + momentumn * self.decay
            momentumn = grads
            # beta is the bound
            # subscribe is the gradient increase direction
            # alpha: each change magnitude
            adv_latent = latent_advs.detach().clone()
            adv_latent[:, mask, :] = latent_advs[:, mask, :] - self.alpha * grads[:, mask, :].sign() # use sign or not?
            latent_advs = adv_latent
            # beta: the change bound each time
            # delta = torch.clamp(adv_latent - latent_adv, -self.beta, self.beta)
            # latent_advs = adv_latent.detach()
            # print(latent_advs.requires_grad)
            ims_adv = self.from_latent_to_ims(latent_advs)
        
        return ims_adv.detach()

    def pgd_attack(self, x, labels):
        eps_for_division = 1e-16
        ims = x.squeeze(dim = 0).clone().detach()
        batch_size = ims.shape[0]
        if self.opts.video_level:
            target = labels.clone().detach().float()
        else:
            target = labels.clone().detach().float()
            target = target.repeat(batch_size)
        print(target.detach().cpu().numpy())
        softmax = nn.Softmax(dim = 1)
        # get latent
        with torch.no_grad():
            latent_adv = self.from_ims_to_latent_directly(ims)
        ori_latent = latent_adv.clone().detach()
        latent_adv.requires_grad = True
        #TODO: channel select?
        ims_adv = self.from_latent_to_ims(latent_adv)
        for i in range(self.opts.epoch):
            loss_value = 0.0
            if self.opts.lpips_lambda > 0:
                loss_lpips = self.lpips_loss(ims_adv, ims)
                print('loss_lpips: ', loss_lpips)
                loss_value += loss_lpips * self.opts.lpips_lambda

            if self.opts.mse_lambda > 0:
                loss_l2 = self.mse_loss(ims_adv, ims)
                print('loss_l2: ', loss_l2)
                loss_value += loss_l2 * self.opts.mse_lambda
            
            if self.opts.id_lambda > 0:
                loss_id, sim_improvement, id_logs = self.id_loss(ims_adv, ims, ims)
                print('loss_id: ', loss_id)
                loss_value += loss_id * self.opts.id_lambda
            if self.opts.video_level:
                ims_adv = ims_adv.unsqueeze(dim = 0)
                ims_adv = ims_adv.permute(0, 2, 1, 3, 4)
            score = self.classifier(ims_adv)
            print(score)
            classifier_result = softmax(score)
            loss_value += self.loss_fn(classifier_result[:, 1], target)
            print('total loss: ', loss_value)
            grad = autograd.grad(loss_value, latent_adv, retain_graph = False, create_graph = False)[0]
            # the total delta contains in the search epsilon ball 
            # 1, alpha single step alpha-base ball
            # beta total beta-base ball for total perturbation bound

            grad_norms = torch.norm(grad.view(batch_size, -1), p = 2, dim = 1) + eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1)
            # the importance of each grad
            adv_latent = latent_adv.detach() - self.alpha * grad
            delta = adv_latent - ori_latent
            # print(delta.view(batch_size, -1).is_contiguous())
            delta_norms = torch.norm(delta.reshape(batch_size, -1), p = 2, dim = 1)
            # rescale the delta to the epsilon bound
            factor = self.beta / delta_norms # eps the epsilon-base ball
            delta = delta * factor.view(batch_size, 1, 1)
            latent_adv = ori_latent + delta
            latent_adv.requires_grad = True
            ims_adv = self.from_latent_to_ims(latent_adv)

            # each step within the epsilon search ball
            # 2, alpha alpha-base ball
            # beta modification bound
            # grad_norms = torch.norm(grad.view(batch_size, -1), p = float('inf'), dim = 1) + eps_for_division

            # grad_norms = torch.norm(grad.view(batch_size, -1), p = 2, dim = 1) + eps_for_division
            # grad = grad / grad_norms.view(batch_size, 1, 1)
            # adv_latent = latent_adv.detach() - self.alpha * grad
            # delta = adv_latent - ori_latent
            # delta = torch.clamp(delta, -self.beta, self.beta)
            # latent_adv = ori_latent + delta
            # latent_adv.requires_grad = True
            # ims_adv = self.from_latent_to_ims(latent_adv)
        return ims_adv.detach()


    def attack_one_video(self, video, label):
        """
            video_frames: 输入的视频序列
        """
        # attack on the video
        # return the inversion attack pics
        attack_result = self.attack(video, label)
        # TODO: save all the pic in one file
        if self.opts.video_level:
            eval_data = attack_result.detach().unsqueeze(dim = 0)
            eval_data = eval_data.permute(0, 2, 1, 3, 4)
        else:
            eval_data = attack_result.detach()
        # eval
        with torch.no_grad():
            eval_result = self.classifier(eval_data)
        score = eval_result.cpu().detach()
        print(score)
        return attack_result

def tensor2npimg(var):
    # var: [3, H, W]
    var = var.detach().cpu().permute(1, 2, 0).numpy()
    var = (var + 1) / 2
    var[var > 1] = 1
    var[var < 0] = 0
    var *= 255
    var = var.astype('uint8')
    return var


def log_pics(save_dir, origin: torch.Tensor, modified: torch.Tensor):
    """
    both should have shape: [N, C, H, W]
    """
    assert origin.shape[0] == modified.shape[0], 'the modified_pics must have the same frames with the origin_pics'
    origin_pics = [tensor2npimg(origin[i]) for i in range(origin.shape[0])]
    modified_pics = [tensor2npimg(modified[i]) for i in range(modified.shape[0])]
    scale_h = origin.shape[2]
    scale_w = origin.shape[3]
    w = origin.shape[0]
    h = 2
    full_im = np.zeros((h * scale_h, w * scale_w, 3), dtype = np.uint8)
    for index in range(w):
        full_im[0: scale_h, index * scale_w: (index + 1) * scale_w, :] = origin_pics[index]
        full_im[scale_h : 2 * scale_h, index * scale_w: (index + 1) * scale_w, :] = modified_pics[index]
    print(full_im.shape)
    full_im = cv2.cvtColor(full_im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_dir, full_im)
        

def main():
    parser = ArgumentParser('attack with white box')
    parser.add_argument('--dataset', default = 'celebdf', type = str, help = 'attack dataset')
    parser.add_argument('--stylegan_size', default = 1024, type = int, help = 'para of stylegan')
    parser.add_argument('--encoder_type', default = 'Encoder4Editing', type = str, help = 'using encoder and decoder type')
    parser.add_argument('--follow_checkpoint_path', default = [], type = str, nargs = '+', help = 'encoder4editing checkpoint')
    parser.add_argument('--device', default = 'cuda:0', type = str, help = 'devices')
    parser.add_argument('--start_from_latent_avg', default = True, type = bool, help = 'para of stylegan')
    parser.add_argument('--classifier_ckpt', default = '', required = True, type = str, help = 'classifier type, modified by inner choose of dataset')
    parser.add_argument('--classifier', default = 'xception', type = str, help = 'classifier type')
    parser.add_argument('--attack_method', default = 'fpg', type = str, help = 'attack method in [fgd, bim]')
    parser.add_argument('--beta', default = 0.01, type = float, help = 'para for attack')
    parser.add_argument('--alpha', default = 0.0002, type = float, help = 'para for attack')
    parser.add_argument('--epoch', default = 100, type = int, help = 'attack loops')
    parser.add_argument('--frames', default = 12, type = int, help = 'data frames')
    parser.add_argument('--loss_type', default = 'mse_loss', type = str, help = 'loss type')
    parser.add_argument('--lpips_lambda', default = 0, type = float, help = 'lpips loss weight')
    parser.add_argument('--lpips_type', default = 'alex', type = str, help = 'lpips type')
    parser.add_argument('--id_lambda', default = 0, type = float, help = 'id loss weight')
    parser.add_argument('--mse_lambda', default = 0, type = float, help = 'mse loss weight')
    parser.add_argument('--checkpoint_path', default = '', type = str, help = 'pretrained e4e model path')
    parser.add_argument('--dataset_dir', default = '', type = str, help = 'attack dataset path, modified inner')
    parser.add_argument('--label_file', default = '', type = str, help = 'label file for ffpp dataset')
    parser.add_argument('--video_level', default = False, type = bool, help = 'use video classifier, modified inner')
    parser.add_argument('--log_dir', default = None, type = str, help = 'log_dir')
    parser.add_argument('--attack_num', default = 500, type = int, help = 'waiting to be attack video num')
    parser.add_argument('--mask', default = 'full', type = str, help = 'which mask')

    
    opts = parser.parse_args()

    if opts.classifier == 'slowfast':
        opts.video_level = True
    if opts.classifier == 'I3D':
        opts.video_level = True
    elif opts.classifier == 'xception':
        opts.video_level = False
    elif opts.classifier == 'efficient':
        opts.video_level = False
    elif opts.classifier == 'mat':
        opts.video_level = False
    elif opts.classifier == 'recce':
        opts.video_level = False


    log_folder = opts.log_dir

    follow_what_attack =[] 

    print(opts)

    attack = Attack_Video(opts)
    
    if opts.dataset == 'celebdf':
        dataset = CELEB_DF(root_dir = opts.dataset_dir, 
                                mix_real_fake = False,
                                transform = train_transform, 
                                frames_count = opts.frames, 
                                stride = 1, 
                                read_method = 'frame_by_frame', 
                                train_target = 'inversion_attack',
                                train_test = 'test')
    elif opts.dataset == 'dfdc':
        dataset = DFDC(opts.dataset_dir, 
                    opts.label_file, 
                    train_transform, 
                    frames_count = opts.frames,
                    read_method = 'frame_by_frame',
                    train_target = 'inversion_attack',
                    train_test = 'test')
    elif opts.dataset == 'ffpp':
        dataset = FFPP(root_dir = opts.dataset_dir, 
                    mix_real_fake = False,
                    transform = train_transform, 
                    frames_count = opts.frames, 
                    stride = 1, 
                    read_method = 'frame_by_frame', 
                    train_target = 'inversion_attack',
                    train_test = 'test')

    data_num = len(dataset)
    print(f'attack dataset len: {data_num}')
    celebdf_dataloader = DataLoader(dataset, batch_size = 14, 
                                    shuffle = False,
                                    num_workers = 1,
                                    drop_last = True)
    for i in range(len(opts.follow_checkpoint_path) + 1):
        if i != 0:
            ckpt = torch.load(opts.follow_checkpoint_path[i - 1], map_location = 'cpu')
            attack.e4e.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict = True)
            attack.e4e.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict = True)
            attack.e4e.load_latent_avg(ckpt)
            opts.log_dir = log_folder + follow_what_attack[i - 1]
        if not os.path.exists(opts.log_dir):
            os.mkdir(opts.log_dir)
            print(f"save all the logs to {opts.log_dir}")
        shutil.copyfile('./attack/attack_celebdf.py', os.path.join(opts.log_dir, 'attack_celebdf.py'))
        shutil.copyfile('./attack_celebdf.sh', os.path.join(opts.log_dir, 'attack_celebdf.sh'))
        attack.e4e.eval()
        attack.classifier.eval()
        total_time = 0
        index = 0
        for idx, data in enumerate(celebdf_dataloader):
            if index > opts.attack_num:
                break
            video, label = data
            imgs, labels = data
            video = imgs.unsqueeze(0)
            label = torch.Tensor([0])
            index += 1
            print('pic index:', index)
            tic = time.time()
            if opts.classifier in ['efficient'] and opts.frames > 14:
                    first_batch = attack.attack_one_video(video[:, :int(opts.frames / 2)].to(opts.device), label.to(opts.device).float())
                    second_batch = attack.attack_one_video(video[:, int(opts.frames / 2):].to(opts.device), label.to(opts.device).float())
                    attack_result = torch.vstack((first_batch, second_batch))
            else:
                attack_result = attack.attack_one_video(video.to(opts.device), label.to(opts.device).float())
            toc = time.time()
            total_time = total_time + (toc - tic)
            video = video.squeeze(0)
            log_path = os.path.join(opts.log_dir, str(index) + '.png')
            log_pics(save_dir = log_path, origin = video, modified = attack_result)
        print(f"==> each attack take average: {total_time / data_num}")

if __name__ == "__main__":
    main()
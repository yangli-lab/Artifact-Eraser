import torch
import torch.nn as nn
import numpy as np
import cv2
from argparse import Namespace
from PIL import Image
from argparse import ArgumentParser

import time
import os
import sys
import random
import shutil
sys.path.append(".")
sys.path.append("..")
sys.path.append("./classifiers/multiple_attention")
sys.path.append("./classifiers/RECCE")
sys.path.append("./classifiers/I3D")

from encodecode import EncodeDecode
from classifiers.slowfast import slowfast50, load_state
from classifiers.train_xception import xception
from classifiers.train_efficient import load_pretrained_efficient
from classifiers.multiple_attention.mat_models.MAT import MAT
from classifiers.RECCE.recce_model.network import Recce
from classifiers.multiple_attention.config import train_config
from classifiers.I3D.model import  load_pretrained_I3D_RES
from data_transform.transform_config import get_transform
from eval_metric import folder_metric_calandsave
from torchvision import transforms


import pandas as pd

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
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def read_ims(im_path, transform, size = (256, 256)):
    try:
        im = cv2.imread(im_path)
    except Exception as e:
        print(e)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h = int(im.shape[0] / size[0])
    w = int(im.shape[1] / size[1])
    im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
    im_attack = np.zeros_like(im_origin)
    im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
    im_attack = im[size[0]: h * size[0], :im.shape[1], :im.shape[2]]
    ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
    ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
    ims_origin = ims_origin.transpose(1, 0, 2, 3)
    ims_attack = ims_attack.transpose(1, 0, 2, 3)
    oris = []
    atts = []
    for i in range(ims_origin.shape[0]):
        ori = ims_origin[i, :, :, :]
        oris.append(transform(Image.fromarray(ori.astype('uint8'))))
        att = Image.fromarray(ims_attack[i, :, :, :].astype('uint8'))
        atts.append(transform(att))
    return torch.stack(oris, dim = 0).unsqueeze(dim = 0), torch.stack(atts, dim = 0).unsqueeze(dim = 0)

def eval(classifier, x, device = 'cuda:0'):
    classifier.eval()
    softmax = nn.Softmax(dim = -1)
    with torch.no_grad():
        pred = softmax(classifier(x.to(device)))
    return pred

def eval_recce(classifier, x, device = 'cuda:0'):
    classifier.eval()
    with torch.no_grad():
        pred = classifier(x.to(device))
        pred = torch.sigmoid(pred)
    return pred

def tensor2npimg(var):
    # var: [3, H, W]
    var = var.detach().cpu().permute(1, 2, 0).numpy()
    var = (var + 1) / 2
    var[var > 1] = 1
    var[var < 0] = 0
    var *= 255
    var = var.astype('uint8')
    return var


def log_pics(save_dir, all_ims: torch.Tensor, group: int):#, origin: torch.Tensor, modified: torch.Tensor):
    """
    both should have shape: [N, C, H, W]
    """
    h_num = group
    w_num = int(all_ims.shape[0] / h_num)
    n, c, h, w = all_ims.shape
    split_ims = np.zeros((n, h, w, c))
    for index in range(h_num):
        for i in range(index * w_num, (index + 1) * w_num):
            split_ims[i] = tensor2npimg(all_ims[i])
    h = all_ims.shape[2]
    w = all_ims.shape[3]
    full_im = np.zeros((h * h_num, w * w_num, 3), dtype = np.uint8)
    for index in range(h_num):
        full_im[h * index: h * (index + 1)] = split_ims[index * w_num: (index + 1) * w_num].transpose(1, 0, 2, 3).reshape(h, w * w_num, -1)
    full_im = cv2.cvtColor(full_im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_dir, full_im)


def main(data_folder: str, save_success_attack_folder:str , metric: list, transfer_model: str, transfer_dataset: str, classifier_ckpt, checkpoint_path):
    opts = dict()
    opts['data_folder'] = data_folder
    opts['classifier_ckpt'] = classifier_ckpt
    opts['device'] = 'cuda:0'
    opts['save_pic'] = True
    opts['save_success_attack_folder'] = save_success_attack_folder
    if 'celebdf' in data_folder:
        opts['dataset'] = 'celebdf'
    if 'dfdc' in data_folder:
        opts['dataset'] = 'dfdc'
    if 'ffpp' in data_folder:
        opts['dataset'] = 'ffpp'
    if transfer_dataset != None:
        opts['dataset'] = transfer_dataset
    if 'xception' in data_folder:
        opts['classifier'] = 'xception' #'xception' #'slowfast50' #'efficient'
    if 'efficient' in data_folder:
        opts['classifier'] = 'efficient' #'xception' #'slowfast50' #'efficient'
    if 'slowfast' in data_folder:
        opts['classifier'] = 'slowfast' #'xception' #'slowfast50' #'efficient'
    if 'mat' in data_folder:
        opts['classifier'] = 'mat'
    if 'recce' in data_folder:
        opts['classifier'] = 'recce'
    if 'I3D' in data_folder:
        opts['classifier'] = 'I3D'
    if transfer_model != None:
        opts['classifier'] = transfer_model

    opts.checkpoint_path = checkpoint_path

    if 'latent' in data_folder:
        opts['save_inversion'] = True
    elif 'img' in data_folder:
        opts['save_inversion'] = False
    # opts['save_inversion'] = False

    # stylegan 
    opts['stylegan_size'] = 1024
    opts['encoder_type'] = 'Encoder4Editing'
    opts['start_from_latent_avg'] = True

    opts['metrics'] = metric

    opts = Namespace(**opts)

    if opts.save_inversion:
        e4e = EncodeDecode(opts).to(opts.device)
        e4e.eval()
    
    folder = os.path.basename(opts.data_folder)
    print(folder)
    opts.save_success_attack_folder = os.path.join(opts.save_success_attack_folder, folder)
    # data
    im_names = os.listdir(opts.data_folder)
    im_paths = sorted([os.path.join(opts.data_folder, im_name) for im_name in im_names])
    # classifier
    if opts.classifier == 'slowfast':
        classifier = slowfast50(num_classes = 2, alpha = 4, beta = 0.125, tau = 16).to(opts.device)
        ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(ckpt, strict = True)
        classifier.eval()
    elif opts.classifier == 'I3D':
        classifier = load_pretrained_I3D_RES(device = opts.device)
        state_dict = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(state_dict, strict = True)
    elif opts.classifier == 'xception':
        classifier = xception(num_classes = 2, pretrained = 'imagenet').to(opts.device)
        ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(ckpt, strict = True)
        classifier.eval()
    elif opts.classifier == 'efficient':
        classifier = load_pretrained_efficient(encoder_name = "tf_efficientnet_b7_ns", classes = 2, device = opts.device)
        ckpt = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(ckpt, strict = True)
        classifier.eval()
    elif opts.classifier == 'mat':
        name = 'a1_b5_b2'
        config = train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer='b2',
        ckpt = 'checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
        classifier =  MAT(**config.net_config).to(opts.device)
        stat_dict = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(stat_dict, strict = False)
    elif opts.classifier == 'recce':
        classifier = Recce(num_classes = 1)
        classifier.fc = nn.Linear(2048, 2)
        classifier.to(opts.device)
        weights = torch.load(opts.classifier_ckpt, map_location = 'cpu')
        classifier.load_state_dict(weights)

    # target label, vary with the folder you choose
    gt_acc = 0
    in_acc = 0
    attack_acc = 0
    suc_attack = 0
    num_count = 0
    if os.path.exists(opts.save_success_attack_folder) and opts.save_pic:
        shutil.rmtree(opts.save_success_attack_folder)
    if opts.save_pic:
        os.makedirs(opts.save_success_attack_folder)
    tic = time.time()
    for index, im_path in enumerate(im_paths):
        if opts.save_inversion:
            e4e = EncodeDecode(opts).to(opts.device)
            e4e.eval()
        if '.png' in im_path:
            ims_origin, ims_attack = read_ims(im_path, transform = train_transform, size = (256, 256))
        else:
            continue
        ims_inversion = torch.zeros_like(ims_origin)
        if opts.save_inversion:
            with torch.no_grad():
                ims_inversion = e4e(ims_origin.squeeze().to(opts.device)).unsqueeze(0).cpu()
        else:
            ims_inversion = ims_origin.detach()

        if opts.classifier in ['slowfast', 'I3D']:
            ims_origin = ims_origin.permute(0, 2, 1, 3, 4)
            ims_inversion = ims_inversion.permute(0, 2, 1, 3, 4)
            ims_attack = ims_attack.permute(0, 2, 1, 3, 4)
        elif opts.classifier in ['xception', 'efficient', 'mat', 'recce']:
            ims_origin = ims_origin.squeeze(0)
            ims_inversion = ims_inversion.squeeze(0)
            ims_attack = ims_attack.squeeze(0)
        num_count += ims_attack.shape[0]
        origin_score = np.zeros((ims_origin.shape[0], 2))
        inversion_score = np.zeros_like((origin_score))
        attack_score = np.zeros_like((origin_score))
        origin_score = eval(classifier, ims_origin, device = opts.device).detach().cpu().numpy()
        inversion_score = eval(classifier, ims_inversion, device = opts.device).detach().cpu().numpy()
        attack_score = eval(classifier, ims_attack, device = opts.device).detach().cpu().numpy()
        mask = np.zeros(origin_score.shape[0]).astype('bool')
        if opts.classifier in ['recce', 'mat', 'xception', 'efficient', 'slowfast', 'I3D']:
            gt_acc += sum(origin_score[:, 1] > 0.5)
            in_acc += sum(inversion_score[:, 1] > 0.5)
            attack_acc += sum(attack_score[:, 1] < 0.5)
            suc_attack += sum((origin_score[:, 1] > 0.5) * (attack_score[:, 1] < 0.5) * (inversion_score[:, 1] > 0.5))
            mask[((origin_score[:, 1] > 0.5) * (attack_score[:, 1] < 0.5) * (inversion_score[:, 1] > 0.5))] = 1
            save_ori = ims_origin[mask]
            save_in = ims_inversion[mask]
            save_adv = ims_attack[mask]
        else:
            gt_acc += sum(origin_score > 0.5)
            in_acc += sum(inversion_score > 0.5)
            attack_acc += sum(attack_score < 0.5)
            suc_attack += sum((origin_score > 0.5) * (attack_score < 0.5) * (inversion_score > 0.5))
            final_score = (origin_score > 0.5) * (attack_score < 0.5) * (inversion_score > 0.5)
            for i in range(final_score.shape[0]):
                if final_score[i] == True:
                    mask[i] = 1
            save_ori = ims_origin[mask]
            save_in = ims_inversion[mask]
            save_adv = ims_attack[mask]
        if sum(mask) == 0:
            continue
        if opts.classifier in ['slowfast', 'I3D']:
            save_ori = save_ori.squeeze(0).permute(1, 0, 2, 3)
            save_in = save_in.squeeze(0).permute(1, 0, 2, 3)
            save_adv = save_adv.squeeze(0).permute(1, 0, 2, 3)
        if opts.save_pic == True:
            if opts.save_inversion == True:
                all_ims = torch.vstack((save_ori, save_in, save_adv))
                group = 3
            else:
                all_ims = torch.vstack((save_ori, save_adv))
                group = 2
            log_pics(os.path.join(opts.save_success_attack_folder, os.path.basename(im_path)), all_ims, group)

    toc = time.time()
    pd_dict = dict()
    pd_dict = {'attack_acc: ': [attack_acc / num_count], 
                'gt_acc: ': [gt_acc / num_count],
                'in_acc:': [in_acc / num_count],
                'suc_attack: ': [suc_attack / gt_acc],
                'use_time: ': [toc - tic]
                }
    df = pd.DataFrame(pd_dict)
    if opts.save_pic:
        df.to_csv(os.path.join(opts.save_success_attack_folder, 'accuracy.csv'))
    print(f'==> num_count is: {num_count}')
    print(f'==> attack_acc is: {attack_acc / num_count}')
    print(f'==> gt_acc is: {gt_acc / num_count}')
    print(f'==> in_acc is: {in_acc / num_count}')
    print(f'==> suc_attack is: {suc_attack / in_acc}')
    print(f'==> use time: {toc - tic}')

  # boring
    if opts.save_pic:
        if len(opts.metrics) != 0:
            for metric in opts.metrics:
                folder_metric_calandsave(os.path.dirname(opts.save_success_attack_folder), opts.save_success_attack_folder, metric)
    print('Done!')
    if opts.save_inversion:
        del e4e
    del classifier


def cal_res(im_folder, size = (256, 256)):
    im_names = os.listdir(im_folder)
    im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    for im_path in im_paths:
        try:
            print(im_path)
            im = cv2.imread(im_path)
            h = int(im.shape[0] / size[0])
            w = int(im.shape[1] / size[1])
            res = np.zeros((size[0], size[1] * w, im.shape[2]))
            res = im[0: size[0], ...] - im[size[0]: size[0] * h, ...]
            print(os.path.join(im_folder, os.path.basename(im_path)[:-4] + '_res.png'))
            cv2.imwrite(os.path.join(im_folder, os.path.basename(im_path)[:-4] + '_res.png'), res)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = ArgumentParser('eval attack success rate')
    parser.add_argument('--transfer_model', default = None, type = str, help = 'transfer_model')
    parser.add_argument('--transfer_dataset', default = None, type = str, help = 'transfer_dataset')
    parser.add_argument('--save_success_attack_folder', default = None, type = str, help = 'folder to save the success attack examples')
    parser.add_argument('--data_folder', default = None, type = str, help = 'attacked results of artifact eraser')
    parser.add_argument('--classifier_ckpt', default = None, type = str, help = 'checkpoint path to classifier')
    parser.add_argument('--checkpoint_path', default = None, type = str, help = 'checkpoint path to e4e model')

    opts = parser.parse_args()

    transfer_model = opts.transfer_model
    transfer_dataset = opts.transfer_dataset
    save_success_attack_folder = opts.save_success_attack_folder
    metric = []
    main(data_folder, save_success_attack_folder, metric, transfer_model, transfer_dataset, classifier_ckpt, checkpoint_path)
    print(f'Done {data_folder}')
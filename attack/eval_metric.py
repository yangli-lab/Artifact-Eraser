from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
# refer to skimage
from PIL import Image

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics import UniversalImageQualityIndex
# from torchmetrics import TotalVariation
# from torchmetrics.functional import total_variation
from torchmetrics import SpectralAngleMapper
# refer to https://torchmetrics.readthedocs.io/en/stable/image/spectral_angle_mapper.html

import cv2
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('.')
sys.path.append('..')

import torch
from torchvision import transforms

from criteria.lpips.lpips import LPIPS
from criteria import id_loss

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def read_ims(im_path: str, size: tuple = (256, 256), read_which: str = 'ina'):
    assert read_which in ['ora', 'orin', 'ina'], "there is no other read im method"
    if read_which == 'ora':
        return read_ims_ora(im_path, size)
    elif read_which == 'orin':
        return read_ims_orin(im_path, size)
    elif read_which == 'ina':
        return read_ims_ina(im_path, size)
    

def read_ims_ora(im_path: str, size: tuple = (256, 256)):
    try:
        im = cv2.imread(im_path)
    except Exception as e:
        print(e)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h = int(im.shape[0] / size[0])
    w = int(im.shape[1] / size[1])
    if h == 2:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_attack = np.zeros_like(im_origin)
        im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
        im_attack = im[size[0]: h * size[0], :im.shape[1], :im.shape[2]]
        ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
        ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_origin = ims_origin.transpose(1, 0, 2, 3)
        ims_attack = ims_attack.transpose(1, 0, 2, 3)

        return ims_origin, ims_attack
    elif h == 3:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_attack = np.zeros_like(im_origin)
        im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
        im_attack = im[2 * size[0]: 3 * size[0], :im.shape[1], :im.shape[2]]
        ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
        ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_origin = ims_origin.transpose(1, 0, 2, 3)
        ims_attack = ims_attack.transpose(1, 0, 2, 3)

        return ims_origin, ims_attack

def read_ims_orin(im_path: str, size: tuple = (256, 256)):
    try:
        im = cv2.imread(im_path)
    except Exception as e:
        print(e)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h = int(im.shape[0] / size[0])
    w = int(im.shape[1] / size[1])
    if h == 2:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_attack = np.zeros_like(im_origin)
        im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
        im_attack = im[size[0]: h * size[0], :im.shape[1], :im.shape[2]]
        ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
        ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_origin = ims_origin.transpose(1, 0, 2, 3)
        ims_attack = ims_attack.transpose(1, 0, 2, 3)

        return ims_origin, ims_attack
    elif h == 3:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_inversion = np.zeros_like(im_origin)
        im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
        im_inversion = im[size[0]: 2 * size[0], :im.shape[1], :im.shape[2]]
        ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
        ims_inversion = im_inversion.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_origin = ims_origin.transpose(1, 0, 2, 3)
        ims_inversion = ims_inversion.transpose(1, 0, 2, 3)

        return ims_origin, ims_inversion

def read_ims_ina(im_path: str, size: tuple = (256, 256)):
    # print(im_path)
    try:
        im = cv2.imread(im_path)
    except Exception as e:
        print(e)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h = int(im.shape[0] / size[0])
    w = int(im.shape[1] / size[1])
    if h == 2:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_attack = np.zeros_like(im_origin)
        im_origin = im[0: size[0], :im.shape[1], :im.shape[2]]
        im_attack = im[size[0]: h * size[0], :im.shape[1], :im.shape[2]]
        ims_origin = im_origin.reshape(size[0], w, size[1], im.shape[2])
        ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_origin = ims_origin.transpose(1, 0, 2, 3)
        ims_attack = ims_attack.transpose(1, 0, 2, 3)

        return ims_origin, ims_attack
    elif h == 3:
        im_origin = np.zeros((int(im.shape[0] / h), im.shape[1], im.shape[2]))
        im_attack = np.zeros_like(im_origin)
        im_inversion = im[size[0]: 2 * size[0], :im.shape[1], :im.shape[2]]
        im_attack = im[2 * size[0]: 3 * size[0], :im.shape[1], :im.shape[2]]
        ims_inversion = im_inversion.reshape(size[0], w, size[1], im.shape[2])
        ims_attack = im_attack.reshape(size[0], w, size[1], im.shape[2])
        # n, h, w, c
        ims_inversion = ims_inversion.transpose(1, 0, 2, 3)
        ims_attack = ims_attack.transpose(1, 0, 2, 3)

        return ims_inversion, ims_attack

def GAN_METRIC(im_folder: str, im_size: tuple = (256, 256), metric: str = 'ics'):
    """
    inception score
    frechet inception score
    kernel inception score
    total variation
    """
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    img_num = 5000
    eval_im_ori = np.zeros((img_num, 3, 256, 256), dtype = np.uint8)
    eval_im_adv = np.zeros((img_num, 3, 256, 256), dtype = np.uint8)
    count_read_num = 0
    metric_dict = dict()
    for idx, full_im_path in enumerate(full_im_paths):
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size)
        else:
            continue
        ims_origin = ims_origin.transpose(0, 3, 1, 2)
        ims_adv = ims_adv.transpose(0, 3, 1, 2)
        batch_size = ims_origin.shape[0]
        eval_im_ori[count_read_num: min(img_num, count_read_num + batch_size), ...] = ims_origin[: min(img_num, count_read_num + batch_size) - count_read_num, ...]
        eval_im_adv[count_read_num: min(img_num, count_read_num + batch_size), ...] = ims_adv[: min(img_num, count_read_num + batch_size) - count_read_num, ...]
        count_read_num += batch_size
        if count_read_num >= img_num:
            break
    if count_read_num < img_num:
        eval_im_ori = eval_im_ori[:count_read_num]
        eval_im_adv = eval_im_adv[:count_read_num]
    eval_im_ori = torch.from_numpy(eval_im_ori)
    eval_im_adv = torch.from_numpy(eval_im_adv)
    if metric == 'ics':
        ics = InceptionScore()
        ics.update(eval_im_adv)
        ics_score = ics.compute()
        metric_dict[im_folder] = [ics_score]
        ics.update(eval_im_ori)
        ics_score = ics.compute()
        metric_dict[im_folder].append(ics_score)
    elif metric == 'fid':
        fid = FrechetInceptionDistance(feature=64)
        fid.update(eval_im_ori, real = True)
        fid.update(eval_im_adv, real = False)
        fid_score = fid.compute()
        metric_dict[im_folder] = fid_score
    elif metric == 'kid':
        kid = KernelInceptionDistance(subset_size = 50)
        kid.update(eval_im_ori, real = True)
        kid.update(eval_im_adv, real = False)
        kid_score = kid.compute()
        metric_dict[im_folder] = kid_score
    elif metric == 'tv':
        tv = TotalVariation()
        tv_score = tv(eval_im_adv)
        metric_dict[im_folder] = tv_score
        tv_score = tv(eval_im_ori)
        metric_dict[im_folder].append(tv_score)
    return metric_dict

def OBJECTIVE_METRIC(im_folder: str, im_size: tuple = (256, 256), metric: str = 'uqi'):
    """
    universal image quality index
    spectral angle mapper
    """
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    metric_dict = dict()
    for idx, full_im_path in enumerate(full_im_paths):
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size)
        else:
            continue
        batch_size = ims_origin.shape[0]
        if metric == 'uqi':
            uqi = UniversalImageQualityIndex()
            uqi_score = uqi(ims_adv, ims_origin)
            metric_dict[full_im_path] = [uqi_score]
        elif metric == 'sam':
            sam = SpectralAngleMapper()
            sam_score = sam(ims_adv, ims_origin)
            metric_dict[full_im_path] = [sam_score]
    return metric_dict


def METRIC(im_folder: str, im_size: tuple = (256, 256), metric: str = 'ssim'):
    """
    structural similarity index measure
    peak signal-to-noise ratio
    mean-square error
    """
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    metric_dict = dict()
    for idx, full_im_path in enumerate(full_im_paths):
        # if idx > 3:
            # break
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size)
        else:
            continue
        metric_values = np.zeros(ims_origin.shape[0])
        for pic_index in range(ims_origin.shape[0]):
            if metric == 'ssim':
                metric_values[pic_index] = ssim(ims_origin[pic_index], ims_adv[pic_index], multichannel = True)
            elif metric == 'psnr':
                metric_values[pic_index] = psnr(ims_origin[pic_index], ims_adv[pic_index])
            elif metric == 'mse':
                metric_values[pic_index] = mse(ims_origin[pic_index], ims_adv[pic_index])
        metric_dict[full_im_path] = metric_values
    return metric_dict

def LPIPS_METRIC(im_folder: str, lpips_type: str = 'alex', im_size: tuple = (256, 256), device: str = 'cuda:0'):
    """
    learned perceptual image patch similarity
    """
    lpips_loss = LPIPS(net_type = lpips_type).to(device).eval()
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    lpips_dict = dict()
    for idx, full_im_path in enumerate(full_im_paths):
        # if idx > 3:
        #     break
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size)
        else:
            continue
        oris = []
        atts = []
        for i in range(ims_origin.shape[0]):
            ori = ims_origin[i, :, :, :]
            oris.append(transform(Image.fromarray(ori.astype('uint8'))))
            att = Image.fromarray(ims_adv[i, :, :, :].astype('uint8'))
            atts.append(transform(att))
        oris = torch.stack(oris, dim = 0).to(device)
        atts = torch.stack(atts, dim = 0).to(device)
        lpips_dict[full_im_path] = lpips_loss(oris, atts).detach().cpu().numpy()
    return lpips_dict

def ID_METRIC(im_folder: str, im_size: tuple = (256, 256), device: str = 'cuda:0'):
    """
    Arcface identity loss
    """
    ID_loss = id_loss.IDLoss().to(device).eval()
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    id_dict = dict()
    for idx, full_im_path in enumerate(full_im_paths):
        # if idx > 3:
        #     break
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size)
        else:
            continue
        oris = []
        atts = []
        for i in range(ims_origin.shape[0]):
            ori = ims_origin[i, :, :, :]
            oris.append(transform(Image.fromarray(ori.astype('uint8'))))
            att = Image.fromarray(ims_adv[i, :, :, :].astype('uint8'))
            atts.append(transform(att))
        oris = torch.stack(oris, dim = 0).to(device)
        atts = torch.stack(atts, dim = 0).to(device)
        loss_id, sim_improvement, id_logs = ID_loss(atts, oris, oris)
        id_dict[full_im_path] = loss_id.detach().cpu().numpy()
    return id_dict


def folder_metric_calandsave(metric_log_folder: str, im_folder: str, metric: str):
    if metric in ['psnr', 'mse', 'ssim']:
        dict_value = METRIC(im_folder, im_size = (256, 256), metric = metric)
    elif metric in ['lpips']:
        dict_value = LPIPS_METRIC(im_folder, 'alex', im_size = (256, 256), device = 'cuda:0')
    elif metric in ['id']:
        dict_value = ID_METRIC(im_folder, im_size = (256, 256), device = 'cuda:0')
    elif metric in ['ics', 'fid', 'kid']:
        dict_value = GAN_METRIC(im_folder, im_size = (256, 256), metric = metric)
    elif metric in ['uqi', 'sam']:
        dict_value = OBJECTIVE_METRIC(im_folder, im_size = (256, 256), metric = metric)
    additive_score = 0
    count = 0
    if metric in ['psnr', 'mse', 'ssim', 'id', 'lpips', 'uqi', 'sam']:
        rearrange_dict = {'im_path': [], 'each_value': [], 'average': [], 'final_score': []}
        for im_path in dict_value.keys():
            avg = np.mean(dict_value[im_path])
            additive_score += np.sum(dict_value[im_path])
            if metric in ['id', 'lpips']:
                count += 1
            else:
                count += len(dict_value[im_path])
            rearrange_dict['im_path'].append(im_path)
            rearrange_dict['each_value'].append(dict_value[im_path])
            rearrange_dict['average'].append(avg)
            rearrange_dict['final_score'].append(additive_score / count)
        df = pd.DataFrame(rearrange_dict)
        save_dir = os.path.join(metric_log_folder, f"{im_folder.split('/')[-1]}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(os.path.join(save_dir, f'ina_{metric}.csv'))
    elif metric in ['ics', 'fid', 'kid']:
        rearrange_dict = {'im_path': [], 'value': []}
        rearrange_dict['im_path'] = dict_value.keys()
        rearrange_dict['value'] = dict_value.values()
        df = pd.DataFrame(rearrange_dict)
        save_dir = os.path.join(metric_log_folder, f"{im_folder.split('/')[-1]}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(os.path.join(save_dir, f'ina_{metric}.csv'))

    print(f'Done: {metric}, folder: {im_folder}')

if __name__ == "__main__":
    # for inversion
    fa_folder = ''
    # metric = ['mse', 'psnr', 'ssim', 'fid', 'kid', 'id', 'lpips']
    metric = ['ssim']
    final_folders = [os.path.join(fa_folder, folder) for folder in os.listdir(fa_folder)]
    for sub_folder in final_folders:
        for me in metric:
            metric_log_folder = os.path.dirname(sub_folder)
            im_folder = sub_folder
            folder_metric_calandsave(metric_log_folder, im_folder, me)
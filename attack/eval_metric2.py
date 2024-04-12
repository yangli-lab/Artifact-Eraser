from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage import img_as_float
# refer to skimage
from PIL import Image
import sys
sys.path.append(".")

from noise_eval.ESNLE import noise_estimate
from noise_eval.NLE import noiseLevelEstimation
# import pyiqac

import time
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
nr_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def TV(x: np.ndarray):
    """
    x: [N, C, H, W]
    """
    n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x = x * 255
    count_h = (h - 1) * w * c
    count_w = h * (w - 1) * c
    h_tv = np.square(x[:, :, 1:, :] - x[:, :, : (h - 1), :]).sum()
    w_tv = np.square(x[:, :, :, 1:] - x[:, :, :, :(w - 1)]).sum()
    tv_value = (h_tv / count_h + w_tv / count_w) / n
    return tv_value

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
                metric_values[pic_index] = ssim(ims_origin[pic_index], ims_adv[pic_index], channel_axis = True)
            elif metric == 'psnr':
                metric_values[pic_index] = psnr(ims_origin[pic_index], ims_adv[pic_index])
            elif metric == 'mse':
                metric_values[pic_index] = mse(ims_origin[pic_index], ims_adv[pic_index])
        metric_dict[full_im_path] = metric_values
    return metric_dict

def NOISE_METRIC(im_folder:str, im_size: tuple = (256, 256), metric: str = 'nle'):
    """
    Additive Noise Evaluation
    """
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    metric_dict_or, metric_dict_ad = dict(), dict()
    for idx, full_im_path in enumerate(full_im_paths):
        print(idx)
        # if idx > 3:
            # break
        if full_im_path[-4:] == '.png':
            im_read = cv2.imread(full_im_path)
            ims_origin = im_read[:256, :, :]
            ims_adv = im_read[-256:, :, :]
        else:
            continue
        if metric == 'nle':
            if idx > 80:
                break
            im_gray = cv2.cvtColor(ims_origin, cv2.COLOR_BGR2GRAY)
            metric_values_or = noiseLevelEstimation(im_gray, patchSize = 7, confidenceLevel = 1 - 1e-6, numIteration = 3)
            metric_values_or = np.array([metric_values_or])
            im_gray = cv2.cvtColor(ims_adv, cv2.COLOR_BGR2GRAY)
            metric_values_ad = noiseLevelEstimation(im_gray, patchSize = 7, confidenceLevel = 1 - 1e-6, numIteration = 3)
            metric_values_ad = np.array([metric_values_ad])
        elif metric == 'esnle':
            if idx > 200:
                break
            ims_origin = img_as_float(ims_origin)
            ims_adv = img_as_float(ims_adv)
            folds = 2
            each_lince = int(ims_origin.shape[1] / folds)
            im_split_or = [ims_origin[:, i * each_lince: (i + 1) * each_lince, :] for i in range(folds)]
            im_split_ad = [ims_adv[:, i * each_lince: (i + 1) * each_lince, :] for i in range(folds)]
            metric_values_or = np.zeros(folds)
            metric_values_ad = np.zeros_like(metric_values_or)
            for i in range(folds):
                metric_values_or[i] = noise_estimate(im_split_or[i], 8) * 255
                metric_values_ad[i] = noise_estimate(im_split_ad[i], 8) * 255
            # ims_origin = img_as_float(ims_origin)
            # metric_values_or = noise_estimate(ims_origin, 8) * 255
            # metric_values_or = np.array([metric_values_or])
            # ims_adv = img_as_float(ims_adv)
            # metric_values_ad = noise_estimate(ims_adv, 8) * 255
            # metric_values_ad = np.array([metric_values_ad])
        metric_dict_or[full_im_path] = metric_values_or
        metric_dict_ad[full_im_path] = metric_values_ad
    return (metric_dict_or, metric_dict_ad)


def NR_METRIC(im_folder: str, im_size: tuple = (256, 256), metric: str = 'nima'):
    """
    NO reference distortion evaluation
    """
    im_names = os.listdir(im_folder)
    full_im_paths = [os.path.join(im_folder, im_name) for im_name in im_names]
    metric_dict_or, metric_dict_ad = dict(), dict()
    for idx, full_im_path in enumerate(full_im_paths):
        print(idx)
        # if idx > 3:
            # break
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size, read_which = 'ora')
        else:
            continue
        oris = []
        atts = []
        for i in range(ims_origin.shape[0]):
            ori = ims_origin[i, :, :, :]
            oris.append(nr_transform(Image.fromarray(ori.astype('uint8'))))
            att = Image.fromarray(ims_adv[i, :, :, :].astype('uint8'))
            atts.append(nr_transform(att))
        oris = torch.stack(oris, dim = 0).to(device)
        atts = torch.stack(atts, dim = 0).to(device)
        metric_values_or = np.zeros(len(oris))
        metric_values_ad = np.zeros(len(oris))
        if metric == 'brisque':
            metric_values_or = brisque_metric(oris).cpu().numpy().flatten()
            metric_values_ad = brisque_metric(atts).cpu().numpy().flatten()
        elif metric == 'nima':
            metric_values_or = nima_metric(oris).cpu().numpy().flatten()
            metric_values_ad = nima_metric(atts).cpu().numpy().flatten()
        elif metric == 'niqe':
            if idx > 200:
                break
            metric_values_or = niqe_metric(oris).cpu().numpy().flatten()
            metric_values_ad = niqe_metric(atts).cpu().numpy().flatten()
        elif metric == 'tv':
            metric_values_or = np.array([TV(oris.detach().cpu().numpy())])
            metric_values_ad = np.array([TV(atts.detach().cpu().numpy())])
        metric_dict_or[full_im_path] = metric_values_or
        metric_dict_ad[full_im_path] = metric_values_ad
    return (metric_dict_or, metric_dict_ad)

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
            ims_origin, ims_adv = read_ims(full_im_path, im_size, 'ina')
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
    count = 0
    for idx, full_im_path in enumerate(full_im_paths):
        # if idx > 3:
        #     break
        if full_im_path[-4:] == '.png':
            ims_origin, ims_adv = read_ims(full_im_path, im_size, 'ora')
        else:
            continue
        oris = []
        atts = []
        id_count = np.zeros(ims_origin.shape[0])
        for i in range(ims_origin.shape[0]):
            ori = ims_origin[i, :, :, :]
            oris.append(transform(Image.fromarray(ori.astype('uint8'))))
            att = Image.fromarray(ims_adv[i, :, :, :].astype('uint8'))
            atts.append(transform(att))
        oris = torch.stack(oris, dim = 0).to(device)
        atts = torch.stack(atts, dim = 0).to(device)
        loss_id, sim_improvement, id_logs = ID_loss(atts, oris, oris)
        logs = id_logs
        for i, log in enumerate(logs):
            if log["diff_input"] > 0.75:
                id_count[i] = 1
        id_dict[full_im_path] = id_count
    return id_dict


def folder_metric_calandsave(metric_log_folder: str, im_folder: str, metric: str):
    if metric in ['psnr', 'mse', 'ssim']:
        dict_value = METRIC(im_folder, im_size = (256, 256), metric = metric)
    elif metric in ['lpips']:
        dict_value = LPIPS_METRIC(im_folder, 'alex', im_size = (256, 256), device = 'cuda:0')
    elif metric in ['id']:
        dict_value = ID_METRIC(im_folder, im_size = (256, 256), device = 'cuda:0')

    if metric in ['psnr', 'mse', 'ssim', 'id', 'lpips']:
        additive_score = 0
        count = 0
        rearrange_dict = {'im_path': [], 'each_value': [], 'average': [], 'final_score': []}
        for im_path in dict_value.keys():
            avg = np.mean(dict_value[im_path])
            additive_score += np.sum(dict_value[im_path])
            if metric in ['lpips']:
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
        df.to_csv(os.path.join(save_dir, f'ina_{metric}_2.csv'))
    elif metric in ['brisque', 'niqe', 'nima', 'nle', 'esnle', 'tv']:
        count1, count2 = 0, 0
        additive_score1, additive_score2 = 0, 0
        rearrange_dict1 = {'im_path': [], 'each_value': [], 'average': [], 'final_score': []}
        rearrange_dict2 = {'im_path': [], 'each_value': [], 'average': [], 'final_score': []}
        if metric in ['brisque', 'niqe', 'nima', 'tv']:
            dict_value1, dict_value2 = NR_METRIC(im_folder, im_size = (256, 256), metric = metric)
        elif metric in ['nle', 'esnle']:
            st = time.time()
            dict_value1, dict_value2 = NOISE_METRIC(im_folder, im_size = (256, 256), metric = metric)
            print('each take:', time.time() - st)
        for im_path in dict_value1.keys():
            avg1 = np.mean(dict_value1[im_path])
            avg2 = np.mean(dict_value2[im_path])
            additive_score1 += np.sum(dict_value1[im_path])
            additive_score2 += np.sum(dict_value2[im_path])
            count1 += dict_value1[im_path].shape[-1]
            count2 += dict_value2[im_path].shape[-1]
            rearrange_dict1['im_path'].append(im_path)
            rearrange_dict1['each_value'].append(dict_value1[im_path])
            rearrange_dict1['average'].append(avg1)
            rearrange_dict1['final_score'].append(additive_score1 / count1)
            rearrange_dict2['im_path'].append(im_path)
            rearrange_dict2['each_value'].append(dict_value2[im_path])
            rearrange_dict2['average'].append(avg2)
            rearrange_dict2['final_score'].append(additive_score2 / count2)
        df1 = pd.DataFrame(rearrange_dict1)
        df2 = pd.DataFrame(rearrange_dict2)
        save_dir = os.path.join(metric_log_folder, f"{im_folder.split('/')[-1]}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df1.to_csv(os.path.join(save_dir, f'or_{metric}_2.csv'))
        df2.to_csv(os.path.join(save_dir, f'at_{metric}_2.csv'))

    print(f'Done: {metric}, folder: {im_folder}')

if __name__ == "__main__":
    # for inversion
    metric = ['id', 'lpips', 'esnle', 'nle']
    eval_folder = [""]
    for me in metric:
        for sub_final_folders in [eval_folder]:
            for sub_folder in sub_final_folders:
                print(sub_folder)
                metric_log_folder = os.path.dirname(sub_folder)
                im_folder = sub_folder
                folder_metric_calandsave(metric_log_folder, im_folder, me)
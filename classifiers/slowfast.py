# slowfast models
# modified from 
"""
https://github.com/Siyu-C/RobustForensics/blob/master/inference/ensemble-align-face2-frame-all-rob-policy-flip.py
"""
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np

import os
import sys
sys.path.append('..')
sys.path.append('.')

from dataset.DataSet.dataset_celebdf import CELEB_DF
from data_transform.transform_config import get_transform


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_bn = nn.BatchNorm3d(planes * 4)
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = self.downsample_bn(res)

        out = out + res
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, shortcut_type='B',
                 dropout=0.5, alpha=8, beta=0.125, tau=16, zero_init_residual=False):
        """
        alpha, beta, tau三个参数作用是什么
        """
        super(SlowFast, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        '''Fast Network'''
        self.fast_inplanes = int(64 * beta)
        fast_inplanes = self.fast_inplanes
        self.fast_conv1 = nn.Conv3d(3, fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                                    bias=False)
        self.fast_bn1 = nn.BatchNorm3d(int(64 * beta))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res1 = self._make_layer_fast(
            block, int(64 * beta), layers[0], shortcut_type, head_conv=3)
        self.fast_res2 = self._make_layer_fast(
            block, int(128 * beta), layers[1], shortcut_type, stride=2, head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, int(256 * beta), layers[2], shortcut_type, stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, int(512 * beta), layers[3], shortcut_type, stride=2, head_conv=3)

        '''Slow Network'''
        self.slow_inplanes = 64
        slow_inplanes = self.slow_inplanes
        self.slow_conv1 = nn.Conv3d(3, slow_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                    bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res1 = self._make_layer_slow(
            block, 64, layers[0], shortcut_type, head_conv=1)
        self.slow_res2 = self._make_layer_slow(
            block, 128, layers[1], shortcut_type, stride=2, head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 256, layers[2], shortcut_type, stride=2, head_conv=3)  # Here we add non-degenerate t-conv
        self.slow_res4 = self._make_layer_slow(
            block, 512, layers[3], shortcut_type, stride=2, head_conv=3)  # Here we add non-degenerate t-conv

        '''Lateral Connections'''
        self.Tconv1 = nn.Conv3d(int(64 * beta), int(128 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv2 = nn.Conv3d(int(256 * beta), int(512 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv3 = nn.Conv3d(int(512 * beta), int(1024 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv4 = nn.Conv3d(int(1024 * beta), int(2048 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)

        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes + self.slow_inplanes, num_classes)

        for m in self.modules():
            # initialize parameters
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, input):
        fast, Tc = self.FastPath(input)
        slow_stride = self.alpha
        slow = self.SlowPath(input[:, :, ::slow_stride, :, :], Tc)

        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input, Tc):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, Tc[0]], dim=1)
        x = self.slow_res1(x)
        x = torch.cat([x, Tc[1]], dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, Tc[2]], dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, Tc[3]], dim=1)
        x = self.slow_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        x = self.fast_maxpool(x)
        Tc1 = self.Tconv1(x)
        x = self.fast_res1(x)
        Tc2 = self.Tconv2(x)
        x = self.fast_res2(x)
        Tc3 = self.Tconv3(x)
        x = self.fast_res3(x)
        Tc4 = self.Tconv4(x)
        x = self.fast_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x, [Tc1, Tc2, Tc3, Tc4]

    def _make_layer_fast(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.fast_inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, stride, stride),
                        bias=False))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2 != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, stride, stride),
                        bias=False))

        layers = []
        layers.append(block(self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2, planes, stride, downsample,
                            head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)


def slowfast50(**kwargs):
    """Constructs a SlowFast-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def load_state(path, model, cuda=True):
    def map_func(storage, location):
        return storage.cuda()

    if not cuda:
        map_func = torch.device('cpu')
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        have_load = set(pretrain(model, checkpoint['state_dict'], cuda))
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - have_load
        for k in missing_keys:
            # print('caution: missing keys from checkpoint {}: {}'.format(path, k))
            pass
    else:
        print("=> no checkpoint found at '{}'".format(path))

def pretrain(model, state_dict, cuda):
    own_state = model.state_dict()
    have_load = []
    for name, param in state_dict.items():
        # remove "module." prefix
        name = name.replace(name.split('.')[0] + '.', '')
        if name in own_state:
            have_load.append(name)
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
    return have_load

def predict_batch(img_faces, model, device, softmax_func = nn.Softmax(dim = 1), isRGB = True):
    """
    img_faces, should be: [faces, frames, H, W, C], value range are [0, 255]
    """
    # not here, but use the transfomer in torchvision
    # the parameters are used in kinetics
    sf_mean = torch.from_numpy(
        np.array([110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).to(device)
    sf_std = torch.from_numpy(
        np.array([38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).to(device)

    face_num, face_frames, face_size = img_faces.shape[0:2]
    img_faces = np.float32(img_faces)
    img_faces = torch.from_numpy(img_faces).to(device)
    # normalize
    img_faces = img_faces / 255

    assert face_size == 256, 'the input size of the face must be [256, 256]'

    sf_image_faces = img_faces.detach().clone()
    # [faces, C, frames, H, W]
    sf_image_faces = sf_image_faces.permute([0, 4, 1, 2, 3])
    # channel 
    if isRGB:
        # TODO: don't need this
        sf_image_faces = sf_image_faces[:, [2, 1, 0], :, :, :]
    sf_image_faces = (sf_image_faces - sf_mean) / sf_std
    try:
        cls_result = model(sf_image_faces)
        final_pred = softmax_func(cls_result)
        # but what we actual need is the final_pred of the model
        sf_out = final_pred[:, 1].cpu().numpy() # 第0维度，是对各帧的检测结果
        sf_max, sf_min, sf_avg = np.max(sf_out), np.min(sf_out), np.mean(sf_out)
        if sf_max > 0.9:
            sf_score = sf_max
        elif len(np.where(sf_out > 0.6)[0]) == sf_out.shape[0]:
            sf_score = sf_max
        elif len(np.where(sf_out < 0.4)[0]) == sf_out.shape[0]:
            sf_score = sf_min
        else:
            sf_score = sf_avg
    except Exception as e:
        print(e)
        sf_score = -1
    return sf_score


if __name__ == "__main__":
    pretrained_model_path = '/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/classifiers/pretrained_models/sf_trainval_52000.pth.tar'
    dataset_dir = '/hd6/guanweinan/Data/celeb-DF-v2-face/Celeb-real/'
    device = torch.device('cuda:0')
    cuda = False # or False
    # initial slowfast model
    slowfast_model = slowfast50(num_classes = 2, alpha = 4, beta = 0.125, tau = 16).to(device)
    load_state(path = pretrained_model_path, model = slowfast_model, cuda = cuda)
    input_tensor = torch.randn((1, 3, 16, 256, 256)).to(device)
    slowfast_model.eval()
    while True:
        pred = slowfast_model(input_tensor)
        print(pred)
    # if cuda:
        # model.cuda()
    # slowfast_model.to(device)
    slowfast_model.eval()
    # print('Done!')
    # read raw video? or what
    # dataloader
    transform_dict = get_transform()
    transform = transform_dict['test']
    celebdf_dataset = Celeb_DF(root_dir = dataset_dir, transform = transform, frames_count = 16, stride = 1, read_method = 'frame_by_frame', train_target = 'inversion_attack')
    celebdf_dataloader = DataLoader(celebdf_dataset, batch_size = 1, shuffle = False, num_workers = 1, drop_last = True)
    for idx, data in enumerate(celebdf_dataloader):
        videos, label = data
        # just skip or something else when no data read
        if len(videos.shape) != 5:
            pass
        else:
            videos = videos.permute(0, 2, 1, 3, 4)
            # [faces, frames, channel, h, w]
            # print(videos.shape) # ? is shape right?
            # [batch_size, 2], seperately represent true or fake, which the original user use the [:, 1] as the value to judge the class of video
            predict = slowfast_model(videos.to(device))
            softmax_func = nn.Softmax(dim = 1)
            predict = softmax_func(predict)
            print(predict.shape)
            print(label, predict[:, 1])
            print('-'*20)
    # # if so, let's do eval
    #     predict_batch(videos, slowfast_model, device, softmax_func = nn.Softmax(dim = 1), isRGB = True)
    # # preprocess



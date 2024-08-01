"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import

import os
import sys
from argparse import ArgumentParser
import math
import numpy as np
import random
import shutil
sys.path.append('.')
sys.path.append('..')

from dataset.DataSet.dataset_dfdc import DFDC
from dataset.DataSet.dataset_ffpp import FFPP
from dataset.DataSet.dataset_celebdf import CELEB_DF


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

__all__ = ['xception']

# set seed, so many
# seed = 42
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'local': '',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
            # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        # assert num_classes == settings['num_classes'], \
            # "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=1000)
        # model.load_state_dict(model_zoo.load_url(settings['url']))
        ckpt = torch.load(settings['local'], map_location = 'cpu')
        model.load_state_dict(ckpt, strict = True)

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = nn.Linear(2048, num_classes)
    del model.fc
    return model

def train_detector(model, device, transform, opts):
    # opts.dataset_name, root_dir, label_file, batch_size, 
    # read_method, lr, momentum, weight_decay, max_iteration
    # max_warmup_iteration, epochs, num_workers, snapshot_frequency
    # snapshot_template, exp_root
    if opts.dataset_name == "dfdc":
        dataset = DFDC(opts.root_dir, opts.label_file, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'classifier', train_test = 'train')
    elif opts.dataset_name == "celebdf":
        dataset = CELEB_DF(opts.root_dir, True, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'classifier', train_test = 'train')
    elif opts.dataset_name == "ffpp":
        dataset = FFPP(opts.root_dir, True, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'classifier', train_test = 'train')
    
    snapshot_root = os.path.join(opts.exp_root, 'snap')
    os.makedirs(snapshot_root)
    log_root = os.path.join(opts.exp_root, 'logs')
    os.makedirs(log_root)

    shutil.copyfile('train_xception.py', os.path.join(opts.exp_root, 'train_xception.py'))
    shutil.copyfile('train_xception.sh', os.path.join(opts.exp_root, 'train_xception.sh'))


    writer = SummaryWriter(log_root)

    print('Train xception dataset size: {}'.format(len(dataset)))

    model.train()

    iteration = 0
    warmup_optimizer = torch.optim.SGD(model.last_linear.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    full_optimizer = torch.optim.SGD(model.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    full_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(full_optimizer, lambda iteration: (opts.max_iteration - iteration) / opts.max_iteration)

    if iteration < opts.max_warmup_iteration:
        print('Start {} warmup iterations'.format(opts.max_warmup_iteration))
        model.eval()
        model.last_linear.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.last_linear.parameters():
            param.requires_grad = True
        optimizer = warmup_optimizer
    else:
        print('Start without warmup')
        model.train()
        optimizer = full_optimizer

    max_lr = max(param_group["lr"] for param_group in full_optimizer.param_groups)
    writer.add_scalar('train/max_lr', max_lr, iteration)

    for epoch in range(opts.epochs):
        print('Epoch {} is in progress'.format(epoch))
        dataset_loader = DataLoader(dataset, batch_size = opts.batch_size, shuffle = True, num_workers = opts.num_workers, drop_last = True)
        for i, data in enumerate(dataset_loader):
            video_sequence, label = data
            iteration += 1
            pred = model(video_sequence.to(device))
            # TODO: what should label be
            # pred = F.softmax(pred, dim = -1)
            loss = F.binary_cross_entropy_with_logits(pred, label.to(device))
            print(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration > opts.max_warmup_iteration:
                full_lr_scheduler.step()
                max_lr = max(param_group["lr"] for param_group in full_optimizer.param_groups)
                writer.add_scalar('train/max_lr', max_lr, iteration)
            
            writer.add_scalar('train/loss', loss.item(), iteration)
            
            if iteration == opts.max_warmup_iteration:
                print('Stop warmup iterations')
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = full_optimizer
            
            if iteration % opts.snapshot_frequency == 0:
                snapshot_name = opts.snapshot_template + '_{}.pth'.format(iteration)
                snapshot_path = os.path.join(snapshot_root, snapshot_name)
                print('saving snapshot {} to {}'.format(iteration, snapshot_path))
                torch.save(model.state_dict(), snapshot_path)
            
            if iteration > opts.max_iteration:
                print('Stop training')
                return


if __name__ == "__main__":
    device = 'cuda:0'
    parser = ArgumentParser('train xception parameters')
    parser.add_argument('--dataset_name', default = 'dfdc', type = str, help = 'training dataset')
    parser.add_argument('--root_dir', default = './', type = str, help = 'dir for dataset')
    parser.add_argument('--label_file', default = './', type = str, help = 'dir of label file for dfdc dataset')
    parser.add_argument('--batch_size', default = 16, type = int, help = 'batch size of training data')
    parser.add_argument('--read_method', default = 'frame_by_frame', type = str, help = 'method for reading the video frames')
    parser.add_argument('--lr', default = 0.05, type = float, help = 'learning rate for optimize')
    parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum for optimize')
    parser.add_argument('--weight_decay', default = 1e-4, type = float, help = 'weight decay for optimize')
    parser.add_argument('--max_iteration', default = 100000, type = int, help = 'max iteration for training')
    parser.add_argument('--max_warmup_iteration', default = 100, type = int, help = 'the iteration before ending up warmup')
    parser.add_argument('--epochs', default = 1000, type = int, help = 'training epochs')
    parser.add_argument('--num_workers', default = 1, type = int, help = 'number of workers for data loading')
    parser.add_argument('--snapshot_frequency', default = 1000, type = int, help = 'each snapshot_frequency iteration for saving checkpoint')
    parser.add_argument('--snapshot_template', default = 'snapshot', type = str, help = 'checkpoint first name')
    parser.add_argument('--exp_root', default = './', type = str, help = 'experiment dir')
    
    opts = parser.parse_args()

    model = xception(num_classes = 2, pretrained = 'imagenet').to(device)
    # train from scratch
    # model = Xception(num_classes = 2).to(device)
    # model.last_linear = nn.Linear(2048, 2).to(device)
    transform = train_transform
    train_detector(model, device, transform, opts)
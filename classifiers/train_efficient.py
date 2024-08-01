from functools import partial
import re
import random
import os
import numpy as np
from argparse import ArgumentParser
import shutil
import sys
sys.path.append('.')
sys.path.append('..')

from dataset.DataSet.dataset_dfdc import DFDC
from dataset.DataSet.dataset_ffpp import FFPP
from dataset.DataSet.dataset_celebdf import CELEB_DF

import torch
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

encoder_params = {
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
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

def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv


class DeepFakeClassifierSRM(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.srm_conv = setup_srm_layer(3)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        noise = self.srm_conv(x)
        x = self.encoder.forward_features(noise)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




class DeepFakeClassifierGWAP(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = GlobalWeightedAvgPool2d(encoder_params[encoder]["features"])
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_pretrained_efficient(path, encoder_name, classes, device):
    model = DeepFakeClassifier(encoder = encoder_name)
    if path != None:
        print("load state dict {}".format(path))
        checkpoint = torch.load(path, map_location = "cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        del checkpoint
    model.fc = Linear(encoder_params[encoder_name]["features"], classes)
    return model.to(device)

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
    if not os.path.exists(snapshot_root):
        os.makedirs(snapshot_root)
    log_root = os.path.join(opts.exp_root, 'logs')
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    shutil.copyfile('train_efficient.py', os.path.join(opts.exp_root, 'train_efficient.py'))
    shutil.copyfile('train_efficient.sh', os.path.join(opts.exp_root, 'train_efficient.sh'))


    writer = SummaryWriter(log_root)

    print('Train efficient: dataset size: {}'.format(len(dataset)))

    model.train()

    iteration = 0
    warmup_optimizer = torch.optim.SGD(model.fc.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    full_optimizer = torch.optim.SGD(model.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    # TODO: may need to modify scheduler
    full_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(full_optimizer, lambda iteration: (opts.max_iteration - iteration) / opts.max_iteration)

    if iteration < opts.max_warmup_iteration:
        print('Start {} warmup iterations'.format(opts.max_warmup_iteration))
        model.eval()
        model.fc.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
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
    device = "cuda:0"
    parser = ArgumentParser('train efficient parameters')
    parser.add_argument('--pretrained_path', default = '', type = str, help = 'path to pretrained efficient net checkpoint')
    parser.add_argument('--local_pretrained_path', default = None, type = str, help = 'path to pretrained efficient net checkpoint')
    parser.add_argument('--dataset_name', default = 'dfdc', type = str, help = 'training dataset')
    parser.add_argument('--root_dir', default = './', type = str, help = 'dir for dataset')
    parser.add_argument('--label_file', default = './', type = str, help = 'dir of label file for dfdc dataset')
    parser.add_argument('--batch_size', default = 16, type = int, help = 'batch size of training data')
    parser.add_argument('--read_method', default = 'frame_by_frame', type = str, help = 'method for reading the video frames')
    parser.add_argument('--lr', default = 0.01, type = float, help = 'learning rate for optimize')
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

    model = load_pretrained_efficient(opts.pretrained_path, encoder_name = "tf_efficientnet_b7_ns", classes = 2, device = device)
    # encoder_name = "tf_efficientnet_b7_ns"
    # model = DeepFakeClassifier(encoder = encoder_name)
    # model.fc = Linear(encoder_params[encoder_name]["features"], 2)
    # model.to(device)
    if opts.local_pretrained_path != None:
        ckpt = torch.load(opts.local_pretrained_path, map_location = 'cpu')
        model.load_state_dict(ckpt, strict = True)

    train_detector(model, device, train_transform, opts)
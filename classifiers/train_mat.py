import os
import sys
import shutil
sys.path.append('.')
sys.path.append('..')
sys.path.append('multiple_attention')

from mat_models.MAT import MAT
from config import train_config
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import re
import numpy as np
import argparse


from dataset.DataSet.dataset_ffpp import FFPP
from dataset.DataSet.dataset_celebdf import CELEB_DF
from dataset.DataSet.dataset_dfdc import DFDC

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def load_model(name):
    with open('runs/%s/config.pkl'%name,'rb') as f:
        config=pickle.load(f)
    net= MAT(**config.net_config)
    return config,net

def find_best_ckpt(name,last=False):
    if last:
        return len(os.listdir('checkpoints/%s'%name))-1
    with open('runs/%s/train.log'%name) as f:
        lines=f.readlines()[1::2]
    accs=[float(re.search('acc\\:(.*)\\,',a).groups()[0]) for a in lines]
    best=accs.index(max(accs))
    return best

def acc_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    acc=np.mean((preds>=thres)==labels)
    return thres,acc


def load_pretrained_mat():
    # model
    name='a1_b5_b2'
    config=train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer='b2',
        ckpt='checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
    device = 'cuda:0'
    net= MAT(**config.net_config)
    ckpt_path = ''
    stat_dict = torch.load(ckpt_path, map_location = 'cpu')['state_dict']
    net.load_state_dict(stat_dict, strict = False)
    return net.to(device)


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

    shutil.copyfile('train_mat.py', os.path.join(opts.exp_root, 'train_mat.py'))
    shutil.copyfile('train_mat.sh', os.path.join(opts.exp_root, 'train_mat.sh'))


    writer = SummaryWriter(log_root)

    print('Train efficient: dataset size: {}'.format(len(dataset)))

    model.train()

    iteration = 0
    warmup_optimizer = torch.optim.SGD(model.ensemble_classifier_fc[2].parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    full_optimizer = torch.optim.SGD(model.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    # TODO: may need to modify scheduler
    full_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(full_optimizer, lambda iteration: (opts.max_iteration - iteration) / opts.max_iteration)

    if iteration < opts.max_warmup_iteration:
        print('Start {} warmup iterations'.format(opts.max_warmup_iteration))
        model.eval()
        model.ensemble_classifier_fc[2].train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.ensemble_classifier_fc[2].parameters():
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train recce code")
    parser.add_argument('--device', '-d', type=str,
                        default="cuda:0",
                        help="Specify the device to load the model. Default: 'cpu'.")
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

    model = load_pretrained_mat()
    train_detector(model, opts.device, train_transform, opts)

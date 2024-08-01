import os
import sys
import shutil
sys.path.append('.')
sys.path.append('..')
sys.path.append('I3D')

from I3D import resnet
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
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


def load_pretrained_I3D_RES(device):
    model_depth = 200
    n_classes = 1039
    n_input_channels = 3
    resnet_shortcut = 'B'
    conv1_t_size = 7
    conv1_t_stride = 1
    no_max_pool = False
    resnet_widen_factor = 1.0
    model = resnet.generate_model(model_depth = model_depth,
                                  n_classes = n_classes,
                                  n_input_channels = n_input_channels,
                                  shortcut_type =  resnet_shortcut,
                                  conv1_t_size = conv1_t_size,
                                  conv1_t_stride = conv1_t_stride,
                                  no_max_pool = no_max_pool,
                                  widen_factor = resnet_widen_factor)
    pretrain_path = "/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/classifiers/pretrained_models/r3d200_KM_200ep.pth"
    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path, map_location='cpu')

    model.load_state_dict(pretrain['state_dict'])
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)



def train_detector(model, device, transform, opts):
    # opts.dataset_name, root_dir, label_file, batch_size, 
    # read_method, lr, momentum, weight_decay, max_iteration
    # max_warmup_iteration, epochs, num_workers, snapshot_frequency
    # snapshot_template, exp_root
    if opts.dataset_name == "dfdc":
        dataset = DFDC(opts.root_dir, opts.label_file, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'inversion_attack', train_test = 'train')
    elif opts.dataset_name == "celebdf":
        dataset = CELEB_DF(opts.root_dir, True, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'inversion_attack', train_test = 'train')
    elif opts.dataset_name == "ffpp":
        dataset = FFPP(opts.root_dir, True, transform, 
                    frames_count = opts.batch_size, read_method = opts.read_method,
                    train_target = 'inversion_attack', train_test = 'train')
    
    snapshot_root = os.path.join(opts.exp_root, 'snap')
    if not os.path.exists(snapshot_root):
        os.makedirs(snapshot_root)
    log_root = os.path.join(opts.exp_root, 'logs')
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    shutil.copyfile('train_I3D.py', os.path.join(opts.exp_root, 'train_I3D.py'))
    shutil.copyfile('train_I3D.sh', os.path.join(opts.exp_root, 'train_I3D.sh'))


    writer = SummaryWriter(log_root)

    print('Train slowfast dataset size: {}'.format(len(dataset)))

    model.train()

    iteration = 0
    warmup_optimizer = torch.optim.SGD(model.fc.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
    full_optimizer = torch.optim.SGD(model.parameters(), opts.lr, 
                                    momentum = opts.momentum, weight_decay = opts.weight_decay, nesterov = True)
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
        dataset_loader = DataLoader(dataset, batch_size = 12, shuffle = True, num_workers = opts.num_workers, drop_last = True)
        for i, data in enumerate(dataset_loader):
            video_sequence, label = data
            if video_sequence.dim() != 5:
                continue
            iteration += 1
            label = label[:, 0]
            video_sequence = video_sequence.to(device)
            video_sequence = video_sequence.permute(0, 2, 1, 3, 4)
            pred = model(video_sequence)
            # pred = F.softmax(pred, dim = -1)
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

    model = load_pretrained_I3D_RES(opts.device)
    train_detector(model, opts.device, train_transform, opts)

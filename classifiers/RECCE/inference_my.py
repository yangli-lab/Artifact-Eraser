import cv2
import torch
import random
import argparse
from glob import glob
from os.path import join
from recce_model.network import Recce
from recce_model.common import freeze_weights
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
import os
from PIL import Image
import numpy as np
sys.path.append('..')
sys.path.append('../..')
from dataset.DataSet.dataset_ffpp import FFPP

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# fix random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description="This code helps you use a trained model to "
                                             "do inference.")
parser.add_argument("--weight", "-w",
                    type=str,
                    default='/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/classifiers/RECCE/pretrained_model/model_params_ffpp_c23.pickle',
                    help="Specify the path to the model weight (the state dict file). "
                         "Do not use this argument when '--bin' is set.")
parser.add_argument("--bin", "-b",
                    type=str,
                    default=None,
                    help="Specify the path to the model bin which ends up with '.bin' "
                         "(which is generated by the trainer of this project). "
                         "Do not use this argument when '--weight' is set.")
parser.add_argument("--image", "-i",
                    type=str,
                    default=None,
                    help="Specify the path to the input image. "
                         "Do not use this argument when '--image_folder' is set.")
parser.add_argument("--image_folder", "-f",
                    type=str,
                    default=None,
                    help="Specify the directory to evaluate all the images. "
                         "Do not use this argument when '--image' is set.")
parser.add_argument('--device', '-d', type=str,
                    default="cuda:0",
                    help="Specify the device to load the model. Default: 'cpu'.")
parser.add_argument('--image_size', '-s', type=int,
                    default=299,
                    help="Specify the spatial size of the input image(s). Default: 299.")
parser.add_argument('--visualize', '-v', action="store_true",
                    default=False, help='Visualize images.')


def preprocess(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    compose = Compose([Resize(height=args.image_size, width=args.image_size),
                       Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                       ToTensorV2()])
    img = compose(image=img)['image'].unsqueeze(0)
    return img


def prepare_data():
    paths = list()
    images = list()
    # check the console arguments
    if args.image and args.image_folder:
        raise ValueError("Only one of '--image' or '--image_folder' can be set.")
    elif args.image:
        images.append(preprocess(args.image))
        paths.append(args.image)
    elif args.image_folder:
        image_paths = glob(join(args.image_folder, "*.jpg"))
        image_paths.extend(glob(join(args.image_folder, "*.png")))
        for _ in image_paths:
            images.append(preprocess(_))
            paths.append(_)
    else:
        raise ValueError("Neither of '--image' nor '--image_folder' is set. Please specify either "
                         "one of these two arguments to load input image(s) properly.")
    return paths, images


def inference(model, images, paths, device):
    for img, pt in zip(images, paths):
        img = img.to(device)
        prediction = model(img)
        prediction = torch.sigmoid(prediction).cpu()
        fake = True if prediction >= 0.5 else False
        print(f"path: {pt} \t\t| fake probability: {prediction.item():.4f} \t| "
              f"prediction: {'fake' if fake else 'real'}")
        if args.visualize:
            cvimg = cv2.imread(pt)
            cvimg = cv2.putText(cvimg, f'p: {prediction.item():.2f}, ' + f"{'fake' if fake else 'real'}",
                                (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255) if fake else (255, 0, 0), 2)
            cv2.imshow("image", cvimg)
            cv2.waitKey(0)
            cv2.destroyWindow("image")


def main():
    print("Arguments:\n", args, end="\n\n")
    # set device
    device = torch.device(args.device)
    # load model
    model = eval("Recce")(num_classes=1)
    # check the console arguments
    if args.weight and args.bin:
        raise ValueError("Only one of '--weight' or '--bin' can be set.")
    elif args.weight:
        weights = torch.load(args.weight, map_location="cpu")
    elif args.bin:
        weights = torch.load(args.bin, map_location="cpu")["model"]
    else:
        raise ValueError("Neither of '--weight' nor '--bin' is set. Please specify either "
                         "one of these two arguments to load model's weight properly.")
    model.load_state_dict(weights)
    model = model.to(device)
    freeze_weights(model)
    model.eval()

    paths, images = prepare_data()
    print("Inference:")
    inference(model, images=images, paths=paths, device=device)


def local_verify():
    
    # hyperparameter
    print("Arguments:\n", args, end="\n\n")
    
    # model
    # set device
    device = torch.device(args.device)
    # load model
    model = eval("Recce")(num_classes=1)
    # check the console arguments
    if args.weight and args.bin:
        raise ValueError("Only one of '--weight' or '--bin' can be set.")
    elif args.weight:
        weights = torch.load(args.weight, map_location="cpu")
    elif args.bin:
        weights = torch.load(args.bin, map_location="cpu")["model"]
    else:
        raise ValueError("Neither of '--weight' nor '--bin' is set. Please specify either "
                         "one of these two arguments to load model's weight properly.")
    model.load_state_dict(weights)
    model = model.to(device)
    freeze_weights(model)
    model.eval()

    # dataset
    dataset = FFPP(root_dir = '/hd6/guanweinan/Data/FF++_MaskFace/', 
                    mix_real_fake = True,
                    transform = train_transform, 
                    frames_count = 2, 
                    stride = 1, 
                    read_method = 'frame_by_frame', 
                    train_target = 'inversion_attack',
                    train_test = 'train')
    dataload = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True)
    acc = 0
    total = 0
    for i, (imgs, labels) in enumerate(dataload):
        imgs = imgs.to(device)
        with torch.no_grad():
            # if labels[0, 0, 0] == 1:
            #     continue
            logits=model(imgs.squeeze(0).to(device))
            pred = torch.sigmoid(logits).cpu()
            print(pred)
            print(labels)
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            acc += sum(pred == torch.argmax(labels))
            total += pred.shape[0]
    print(acc)
    print(total)
    print(acc / total)

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
    im_attack = im[(h - 1) * size[0]: (h) * size[0], :im.shape[1], :im.shape[2]]
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


def local_verify_imgs():
    
    # hyperparameter
    print("Arguments:\n", args, end="\n\n")
    
    # model
    # set device
    device = torch.device(args.device)
    # load model
    model = eval("Recce")(num_classes=1)
    # check the console arguments
    if args.weight and args.bin:
        raise ValueError("Only one of '--weight' or '--bin' can be set.")
    elif args.weight:
        weights = torch.load(args.weight, map_location="cpu")
    elif args.bin:
        weights = torch.load(args.bin, map_location="cpu")["model"]
    else:
        raise ValueError("Neither of '--weight' nor '--bin' is set. Please specify either "
                         "one of these two arguments to load model's weight properly.")
    model.load_state_dict(weights)
    model = model.to(device)
    freeze_weights(model)
    model.eval()

    # dataset
    data_path = '/hd3/liyang/attack_exp_2_suc/attack_latent_nc_ffpp_efficient_fgsmyes_100_14'
    print(os.path.basename(data_path))
    file_names = os.listdir(data_path)
    acc_origin = 0
    acc_attack = 0
    total = 0
    for file_name in file_names:
        im_path = os.path.join(data_path, file_name)
        if im_path.endswith('.png'):
            ims_origin, ims_attack = read_ims(im_path, train_transform, size = (256, 256))
        else:
            continue
        ims_origin = ims_origin.squeeze(0)
        ims_attack = ims_attack.squeeze(0)
        with torch.no_grad():
            imgs_origin = ims_origin.to(device)
            imgs_attack = ims_attack.to(device)
            logits_origin = model(imgs_origin)
            logits_attack = model(imgs_attack)
            pred_origin = torch.sigmoid(logits_origin).cpu()
            pred_attack = torch.sigmoid(logits_attack).cpu()
            pred_origin[pred_origin > 0.5] = 1
            pred_origin[pred_origin <= 0.5] = 0
            pred_attack[pred_attack > 0.5] = 1
            pred_attack[pred_attack <= 0.5] = 0
            acc_origin += sum(pred_origin == 1)
            acc_attack += sum(pred_attack == 1)
            total += pred_origin.shape[0]
    print(acc_origin / total)
    print(acc_attack / total)

if __name__ == '__main__':
    args = parser.parse_args()
    # main()
    # local_verify()
    local_verify_imgs()
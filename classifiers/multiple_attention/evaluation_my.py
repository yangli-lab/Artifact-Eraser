from datasets.data import FF_dataset,deeperforensics_dataset,dfdc_dataset#,Celeb_test
from datasets.dataset import DeepfakeDataset
from mat_models.MAT import MAT
from config import train_config
import pickle
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import re
import os
from  sklearn.metrics import roc_auc_score as AUC
import numpy as np
from copy import deepcopy
import sys
from PIL import Image
import cv2

# sys.path.append('.')
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


def test_eval(net,setting,testset):
    test_dataset=DeepfakeDataset(phase='test',**setting)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=setting['imgs_per_video'],shuffle=False,pin_memory=True,num_workers=8)
    for i, (X, y) in enumerate(test_loader):
        testset[i].append([])
        if -1 in y:
            testset[i].append(0.5)
            continue
        X = X.to('cuda',non_blocking=True)
        with torch.no_grad():
            for x in torch.split(X,20):
                logits=net(x)
                pred=torch.nn.functional.softmax(logits,dim=1)[:,1]
                testset[i][-1]+=pred.cpu().numpy().tolist()
        testset[i].append(np.mean(testset[i][-1]))
        

def test_metric(testset):
        frame_labels=[]
        frame_preds=[]
        video_labels=[]
        video_preds=[]
        for i in testset:
            frame_preds+=i[2]
            frame_labels+=[i[1]]*len(i[2])
            video_preds.append(i[3])
            video_labels.append(i[1])
        video_thres,video_acc=acc_eval(video_labels,video_preds)
        frame_thres,frame_acc=acc_eval(frame_labels,frame_preds)
        video_auc=AUC(video_labels,video_preds)
        frame_auc=AUC(frame_labels,frame_preds)
        rs={'video_acc':video_acc,'video_threshold':video_thres,'video_auc':video_auc,'frame_acc':frame_acc,'frame_threshold':frame_thres,'frame_auc':frame_auc}
        return rs

def dfdc_metric(testset):
    rs=test_metric(testset)
    video_preds=[]
    video_labels=[]
    for i in testset:
        video_preds.append(i[3])
        video_labels.append(i[1])
    video_preds=torch.tensor(video_preds).cuda()
    video_labels=torch.tensor(video_labels).cuda()
    video_preds=torch.stack([1-video_preds,video_preds],dim=1)
    rs['logloss']=torch.nn.functional.cross_entropy(video_preds,video_labels).item()
    return rs

def ff_metrics(testset):
    result=dict()
    temp_set=dict()
    for k,j in enumerate(['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']):
        d=testset[k*140:(k+1)*140]
        temp_set[j]=d

    for i in ['Deepfakes','NeuralTextures','FaceSwap','Face2Face','all']:
        if i!='all':
            rs=test_metric(temp_set[i]+temp_set['Origin'])
        else:
            rs=test_metric(testset) 
        result[i]=rs
    return result

def all_eval(name,ckpt=None,test_sets=['ff-all','celeb','deeper']):
    config,net=load_model(name)
    setting=config.val_dataset
    codec=setting['datalabel'].split('-')[2]
    setting['min_frames']=100
    setting['frame_interval']=5
    setting['imgs_per_video']=20
    setting['datalabel']='ff-all-%s'%codec
    list_of_files = os.listdir('checkpoints/%s'%name)
    list_of_files=list(map(lambda x:int(x[5:-4]),list_of_files))
    if ckpt is None:
        ckpt=find_best_ckpt(name)
    if ckpt<0:
        ckpt=max(list_of_files)+1+ckpt
    
    state_dict=torch.load('checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    net.load_state_dict(state_dict,strict=False)
    os.makedirs('evaluations/%s'%name,exist_ok=True)
    net.eval()
    net.cuda()
    result=dict()
    if 'ff-all' in test_sets:
        testset=[]
        for i in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']:
            testset+=FF_dataset(i,codec,'test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/ff-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['ff']=ff_metrics(testset)
    if 'deeper' in test_sets:
        setting['datalabel']='deeper-'+codec
        testset=deeperforensics_dataset('test')+FF_dataset('Origin',codec,'test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/deeper-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['deeper']=test_metric(testset)
    if 'celeb' in test_sets:
        setting['datalabel']='celeb'
        setting['min_frames']=100
        setting['frame_interval']=5
        setting['imgs_per_video']=20
        testset=deepcopy(Celeb_test)
        test_eval(net,setting,testset)
        with open('evaluations/%s/celeb-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['celeb']=test_metric(testset)
    if 'dfdc' in test_sets:
        setting['datalabel']='dfdc'
        setting['min_frames']=100
        setting['frame_interval']=5
        setting['imgs_per_video']=20
        testset=dfdc_dataset('test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/dfdc-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['dfdc']=dfdc_metric(testset)
    with open('evaluations/%s/metrics-%s.json'%(name,ckpt),'w') as f:
        json.dump(result,f)

def eval_meancorr(name,ckpt=None):
    config,net=load_model(name)
    setting=config.val_dataset
    codec=setting['datalabel'].split('-')[2]
    setting['frame_interval']=5
    setting['imgs_per_video']=60
    setting['datalabel']='ff-all-%s'%codec
    if ckpt is None:
        ckpt=find_best_ckpt(name)
    if ckpt<0:
        ckpt=len(os.listdir('checkpoints/%s'%name))+ckpt
    state_dict=torch.load('checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    net.load_state_dict(state_dict,strict=False)
    net.eval()
    net.cuda()
    testset=[]
    test_dataset=DeepfakeDataset(phase='test',**setting)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=30,shuffle=False,pin_memory=True,num_workers=8)
    count=0
    mc_count=0
    for i, (X, y) in enumerate(test_loader):
        x = X.to('cuda',non_blocking=True)
        with torch.no_grad():
            count+=x.shape[0]
            layers = net.net(x)
            raw_attentions = layers[config.attention_layer]
            attention_maps=net.attentions(raw_attentions).flatten(-2)
            #print(attention_maps.shape)
            srs=torch.norm(attention_maps,dim=2)
            #print(srs.shape)
            for a in range(0,config.num_attentions-1):
                for b in range(a+1,config.num_attentions):
                    mc_count+=torch.sum(torch.sum(attention_maps[:,a,:]*attention_maps[:,b,:],dim=-1)/(srs[:,a]*srs[:,b]))
    return mc_count/(config.num_attentions-1)/config.num_attentions*2/count

                


def merge(g):
    if type(g[0])==float:
        return np.mean(g)
    else:
        c=dict()
        for i in g[0].keys():
            c[i]=merge([u[i] for u in g])
    return c


def gather_metrics(name,fl=None):
    path='evaluations/%s/'%name
    l=os.listdir(path)
    l=[path+i for i in l if i.startswith('metrics-')]
    if fl:
        l=list(filter(fl,l))
    g=[]
    for i in l:
        with open(i) as f:
            g.append(json.load(f))
    return merge(g)
        
def load_and_eval(config):
    # model
    device = 'cuda:0'
    ckpt_path = '/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/classifiers/multiple_attention/pretrained/ff_c23.pth'
    net= MAT(**config.net_config)
    stat_dict = torch.load(ckpt_path, map_location = 'cpu')['state_dict']
    net.load_state_dict(stat_dict, strict = False)
    net.to(device)
    net.eval()
    # dataset
    dataset = FFPP(root_dir = '/hd6/guanweinan/Data/FF++_MaskFace/', 
                    mix_real_fake = True,
                    transform = train_transform, 
                    frames_count = 4, 
                    stride = 1, 
                    read_method = 'frame_by_frame', 
                    train_target = 'inversion_attack',
                    train_test = 'train')
    dataload = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True)
    acc = 0
    total = 0
    for i, (imgs, labels) in enumerate(dataload):
        # if labels[0, 0, 1] == 0:
        #     continue
        imgs = imgs.to(device, non_blocking=True).squeeze(0)
        with torch.no_grad():
            logits=net(imgs)
            pred = torch.nn.functional.softmax(logits,dim=1).cpu()
            print(pred)
            print(labels)
            acc += sum(torch.argmax(pred, dim = 1) == torch.argmax(labels.squeeze(0), dim = 1))
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


def load_and_eval_img(config):
    # model
    device = 'cuda:0'
    ckpt_path = '/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/classifiers/multiple_attention/pretrained/ff_c23.pth'
    net= MAT(**config.net_config)
    stat_dict = torch.load(ckpt_path, map_location = 'cpu')['state_dict']
    net.load_state_dict(stat_dict, strict = False)
    net.to(device)
    net.eval()
    # dataset
    data_path = '/hd5/liyang/attack_to_video_detection/models/inversion/encoder4editing/attack_img_exp_suc/attack_img_mask_ffpp_xception_fgsm_2_16_5'
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
            logits_origin = net(imgs_origin)
            logits_attack = net(imgs_attack)
            pred_origin = torch.nn.functional.softmax(logits_origin, dim = 1).cpu()
            pred_attack = torch.nn.functional.softmax(logits_attack, dim = 1).cpu()
            acc_origin += sum(torch.argmax(pred_origin, dim = 1) == 1)
            acc_attack += sum(torch.argmax(pred_attack, dim = 1) == 1)
            total += pred_origin.shape[0]
    print(acc_origin / total)
    print(acc_attack / total)


if __name__=="__main__":
    # config = train_config(name = 'a', recipes = ['ff-c40', 'efficientnet-b5'])
    # pretrain
    # feature = 2 # [2,3,4]
    # feature_layer='b%s'%feature
    # name='EFB4_ALL_c23_trunc_%s'%feature_layer
    # Config=train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer=feature_layer)
    # aexp
    name='a1_b5_b2'
    config=train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer='b2',
        ckpt='checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
    # load_and_eval(config)
    load_and_eval_img(config)
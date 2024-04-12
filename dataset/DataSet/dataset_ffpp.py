import torch

import numpy as np
import random
from PIL import Image
import pandas as pd
import os
from data_transform.transform_config import get_transform, DFDC_Transform

def read_train_test_split(csv_loc: str, dataset_name: str, train_test: str):
    read_file_name = os.path.join(csv_loc, dataset_name + '_' + train_test + '.csv')
    df = pd.read_csv(read_file_name)
    folders = df["folder"].values.tolist()
    return folders

# all good

# now I have found the new directory that store all the FF++ data

class FFPP():
    def __init__(self, root_dir, mix_real_fake = True, transform = None, frames_count = 32, stride = 1, read_method = 'frame_by_frame', train_target = 'e4e', train_test = 'test'):
        """
        train_test = ['gt_train', 'test']
        """
        inner_folder = 'c23'
        self.video_folders = []
        self.root_dir = root_dir
        if train_target in ['e4e']:
            if mix_real_fake:
                manipulated = ''
                mani_subs = ['Deepfakes', 'Face2Face', 'FaceSwap', "NeuralTextures"]
                for mani_sub in mani_subs:
                    video_folders = os.listdir(os.path.join(manipulated, mani_sub, inner_folder))
                    for video_folder in video_folders:
                        self.video_folders.append(os.path.join(manipulated, mani_sub, inner_folder, video_folder)) 
                origin = ''
                video_folders = os.listdir(os.path.join(origin, inner_folder))
                for video_folder in video_folders:
                    self.video_folders.append(os.path.join(origin, inner_folder, video_folder))
            else:
                video_folders = os.listdir(self.root_dir)
                for video_folder in video_folders:
                    self.video_folders.append(os.path.join(self.root_dir, video_folder))
        # train test split
        elif train_target in ['classifier', 'inversion_attack']:
            train_ori_folders = []
            folders = read_train_test_split('./dataset/DataSet/train_test', 'ffpp_full', train_test)
            for folder in folders:
                if 'original_sequences' in folder and train_test == 'train':
                    train_ori_folders.append(folder)
                    train_ori_folders.append(folder)
                    train_ori_folders.append(folder)
            folders = folders + train_ori_folders
            random.shuffle(folders)
            self.video_folders = folders
        self.frames_count = frames_count
        assert read_method in ['frame_by_frame', 'frame_by_frame_with_stride', 'random', 'start_to_end'], "read_method must in ['frame_by_frame', 'frame_by_frame_with_stride', 'random', 'start_to_end']"
        self.read_method = read_method
        assert train_target in ['e4e', 'inversion_attack', 'classifier'], "train_target must in ['e4e', 'inversion_attack']"
        self.train_target = train_target
        self.stride = stride
        self.transform = transform
        if transform is None:
            if self.train_target == 'e4e':
                transform_dict = get_transform()
                self.transform = transform_dict[train_test]
            elif self.train_target == 'inversion_attack':
                # modify
                transform_dict = DFDC_Transform()
                self.transform = transform_dict[train_test]


    def __len__(self):
        return len(self.video_folders)
        

    def __getitem__(self, idx):
        folder_path = self.video_folders[idx]
        # construct label
        if 'origin' in folder_path:
            self.label = 0
        elif 'manipulated' in folder_path:
            self.label = 1
        pic_names = os.listdir(folder_path)
        pic_names.remove('info.pkl')
        pic_names_sorted = sorted(pic_names)
        if self.train_target == 'classifier':
            indice = np.random.randint(0, len(pic_names), 1)
            image_dir = pic_names[indice[0]]
            try:
                # print(os.path.join(folder_path, image_dir))
                im = Image.open(os.path.join(folder_path, image_dir))
            except Exception as e:
                print(e)
            # preprocess
            # if used for e4e
            im.convert('RGB')
            im = self.transform(im)
            if self.label == 0:
                self.label = torch.Tensor([1, 0])
            else:
                self.label = torch.Tensor([0, 1])
            return im, self.label
        if self.train_target == 'e4e':
            indice = np.random.randint(0, len(pic_names), 1)
            image_dir = pic_names[indice[0]]
            try:
                im = Image.open(os.path.join(folder_path, image_dir))
            except Exception as e:
                print(e)
            # preprocess
            # if used for e4e
            if self.label == 0:
                self.label = torch.Tensor([1, 0])
            else:
                self.label = torch.Tensor([0, 1])
            im.convert('RGB')
            im = self.transform(im)
            from_im = im.clone()
            to_im = im.clone()
            return from_im, to_im
        elif self.train_target == 'inversion_attack':
            if self.frames_count > len(pic_names_sorted):
                print('too many frames to read for {}'.format(os.path.join(folder_path)))
                print('return None data, do something please')
                return torch.ones([1, 1]), torch.Tensor([0])
            frames = []
            labels = []
            if self.read_method == 'frame_by_frame':
                start_index = np.random.randint(0, len(pic_names_sorted) - self.frames_count + 1)
                frame_index_sorted = np.arange(start_index, start_index + self.frames_count)
        
            elif self.read_method == 'frame_by_frame_with_stride':
                actual_frame_num = self.frames_count * self.stride
                start_index = np.random.randint(0, len(pic_names_sorted) - actual_frame_num + 1)
                frame_index_sorted = np.arange(start_index, start_index + actual_frame_num, self.stride)

            elif self.read_method == 'random':
                all_frame_indexs = np.arange(0, len(pic_names_sorted))
                frame_index = np.random.choice(all_frame_indexs, size = self.frames_count, replace = False)
                index_sorted = np.argsort(frame_index, kind = 'mergesort')
                frame_index_sorted = frame_index[index_sorted]

            elif self.read_method == 'start_to_end':
                if len(pic_names_sorted) % self.frames_count == 0:
                    seq = len(pic_names_sorted) / self.frames_count
                    frame_index_sorted = np.arange(0, len(pic_names_sorted), step = int(seq), dtype = np.int)
                else:
                    remain = len(pic_names_sorted) % self.frames_count
                    seq = (len(pic_names_sorted) - int(remain)) / self.frames_count
                    frame_index_sorted = np.arange(0, len(pic_names_sorted) - int(remain), step = int(seq), dtype = np.int)
                    offset = np.random.randint(0, int(seq + 1))
                    frame_index_sorted = np.add(frame_index_sorted, offset)

            assert frame_index_sorted.shape[0] == self.frames_count, 'the selected total frames {} different from frame you want to read {}'.format(len(frame_index_sorted.shape[0], self.frames_count))
        
            for idx in range(self.frames_count):
                im = Image.open(os.path.join(folder_path, pic_names_sorted[frame_index_sorted[idx]]))
                # preprocess
                im.convert('RGB')
                im = self.transform(im)
                frames.append(im)
                # if used for e4e
                labels.append(self.label)
                # TODO:finish code for inversion_attack dataset read
            videos = torch.stack([frame for frame in frames], dim = 0)
            all_labels = torch.Tensor(labels).long().unsqueeze(1)
            self.labels = torch.zeros(all_labels.shape[0], 2).scatter_(1, all_labels, 1)
            # print('{}'.format(os.path.join(folder_path)))
            return videos, self.labels

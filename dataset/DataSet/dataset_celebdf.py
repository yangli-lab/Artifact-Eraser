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

class CELEB_DF():
    def __init__(self, root_dir, mix_real_fake = True, transform = None, frames_count = 32, stride = 1, read_method = 'frame_by_frame', train_target = 'e4e', train_test = 'test'):
        """
        train_test = ['gt_train', 'test']
        """
        self.root_dir = root_dir
        self.video_folders = []
        if train_target in ['e4e']:
            if mix_real_fake:
                folders = os.listdir(root_dir)
                if 'YouTube-real' in folders:
                    folders.remove('YouTube-real')
                for folder in folders:
                    video_folders = os.listdir(os.path.join(self.root_dir, folder))
                    for video_folder in video_folders:
                        self.video_folders.append(os.path.join(self.root_dir, folder, video_folder))
            else:
                video_folders = os.listdir(self.root_dir)
                for video_folder in video_folders:
                    self.video_folders.append(os.path.join(self.root_dir, video_folder))
        elif train_target in ['classifier', 'inversion_attack']:
            split_folders = read_train_test_split('./dataset/DataSet/train_test', 'celebdf', train_test)
            self.video_folders = split_folders
        not_good_set = []
        for remove_video in not_good_set:
            if remove_video in self.video_folders:
                self.video_folders.remove(remove_video)
        self.frames_count = frames_count
        assert read_method in ['frame_by_frame', 'frame_by_frame_with_stride', 'random', 'start_to_end'], "read_method must in ['frame_by_frame', 'frame_by_frame_with_stride', 'random', 'start_to_end']"
        self.read_method = read_method
        assert train_target in ['e4e', 'inversion_attack', 'classifier'], "train_target must in ['e4e', 'inversion_attack']"
        self.train_target = train_target
        self.stride = stride
        self.transform = transform
        if transform is None:
            if self.train_target in ['e4e', 'classifier']:
                transform_dict = get_transform()
                self.transform = transform_dict[train_test]
            elif self.train_target == 'inversion_attack':
                # TODO: modify
                transform_dict = DFDC_Transform()
                self.transform = transform_dict[train_test]

    def __len__(self):
        return len(self.video_folders)
        

    def __getitem__(self, idx):
        folder_path = self.video_folders[idx]
        # construct label
        if 'real' in folder_path:
            self.label = 0
        elif 'synthesis' in folder_path:
            self.label = 1
        pic_names = os.listdir(folder_path)
        pic_names_sorted = sorted(pic_names)
        if self.train_target == 'classifier':
            indice = np.random.randint(0, len(pic_names), 1)
            image_dir = pic_names[indice[0]]
            try:
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
        elif self.train_target == 'e4e':
            indice = np.random.randint(0, len(pic_names), 1)
            image_dir = pic_names[indice[0]]
            try:
                im = Image.open(os.path.join(folder_path, image_dir))
            except Exception as e:
                print(e)
            # preprocess
            # if used for e4e
            im.convert('RGB')
            im = self.transform(im)
            from_im = im.clone()
            to_im = im.clone()
            return from_im, to_im
        elif self.train_target == 'inversion_attack':
            if self.frames_count > len(pic_names_sorted):
                print('too many frames to read for {}'.format(os.path.join(folder_path)))
                print('return None data, do something please')
                return torch.ones([1, 1]), self.label
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
                frame_index = np.random.choice(all_frame_indexs, size = frames_count, replace = False)
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
                    # random offset
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
            return videos, self.labels
import torch

import numpy as np
import random
from PIL import Image
import os
import cv2
import pandas as pd

import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('../..')

from data_transform.face_align import alignment

from data_transform.transform_config import get_transform, DFDC_Transform


def get_label(csv_file):
    df = pd.read_csv(csv_file)
    label_dict = {}
    for i in range(len(df)):
        video_name = df['video'][i]
        file_name = df['file'][i]
        label = df['label'][i]
        if video_name not in label_dict.keys():
            label_dict[video_name] = {file_name: label}
        else:
            label_dict[video_name][file_name] = label
    return label_dict

def read_train_test_split(csv_loc: str, dataset_name: str, train_test: str):
    read_file_name = os.path.join(csv_loc, dataset_name + '_' + train_test + '.csv')
    df = pd.read_csv(read_file_name)
    folders = df["folder"].values.tolist()
    return folders

class DFDC():
    def __init__(self, root_dir, label_file, transform = None, stride = 1, frames_count = 32, read_method = 'frame_by_frame', train_target = 'e4e', train_test = 'test'):
        self.root_dir = root_dir
        # TODO: get the label of data
        self.label_dict = get_label(label_file)
        # maybe we need a pre-defined file to tell the label of the video
        if train_target in ['e4e']:
            self.video_folders = os.listdir(self.root_dir)
        # train test split
        elif train_target in ['classifier', 'inversion_attack']:
            folders = read_train_test_split('./dataset/DataSet/train_test', 'dfdc', train_test)
            self.video_folders = folders
        not_good_set = []
        for remove_video in not_good_set:
            if remove_video in self.video_folders:
                self.video_folders.remove(remove_video)
        self.frames_count = frames_count
        assert read_method in ['frame_by_frame', 'random', 'start_to_end'], "read_method must in ['frame_by_frame', 'random', 'start_to_end']"
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
                transform_dict = DFDC_Transform()
                self.transform = transform_dict[train_test]

    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        # self.label = torch.Tensor([self.label_dict[video_folder]])
        labels = self.label_dict[video_folder]
        self.label = torch.Tensor([0])
        folder_path = os.path.join(self.root_dir, video_folder)
        pic_names = os.listdir(folder_path)
        person_pic_nums = list()
        pic_names_sorted = list()
        persons = set()
        pic_indexs = set()
        for i in range(len(pic_names)):
            out = pic_names[i][:-4].split('_')
            persons.add(int(out[1]))
            pic_indexs.add(int(out[0]))
        persons = sorted(persons)
        person_num = len(persons)
        pic_indexs = sorted(pic_indexs)
        person_pic_nums = [0] * len(persons)
        for i in range(len(pic_indexs)):
            for j in range(len(persons)):
                pic_name = str(pic_indexs[i]) + '_' + str(persons[j]) + '.png'
                if pic_name in pic_names:
                    pic_names_sorted.append(pic_name)
                    person_pic_nums[j] += 1
        max_picnum = max(person_pic_nums)
        for i, picnum in enumerate(person_pic_nums):
            if picnum != max_picnum:
                persons.remove(i)
        if len(persons) != person_num:
            pic_names_sorted = list()
            for i in range(len(pic_indexs)):
                for j in persons:
                    pic_name = str(pic_indexs[i]) + '_' + str(j) + '.png'
                    if pic_name in pic_names:
                        pic_names_sorted.append(pic_name)
                
        if self.train_target == 'classifier':
            indice = np.random.randint(0, len(pic_names_sorted), 1)
            image_dir = pic_names_sorted[indice[0]]
            try:
                im = Image.open(os.path.join(folder_path, image_dir))
            except Exception as e:
                print(e)
            # preprocess
            # if used for e4e
            im.convert('RGB')
            im = self.transform(im)
            label = labels[image_dir]
            if label == 0:
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
            #TODO: get label for the video
            if self.frames_count > len(pic_names_sorted) / len(persons):
                print('too many frames to read for {}'.format(os.path.join(folder_path)))
                print('return None data, do something please!')
                return torch.Tensor([1, 1]), self.label
            frames = []
            all_labels = []
            if self.read_method == 'frame_by_frame':
                start_index = np.random.randint(0, len(pic_names_sorted) - self.frames_count * len(persons) + 1)
                frame_index_sorted = np.arange(start_index, start_index + self.frames_count * len(persons), len(persons))
        
            elif self.read_method == 'frame_by_frame_with_stride':
                actual_frame_num = self.frames_count * self.stride
                start_index = np.random.randint(0, len(pic_names_sorted) - actual_frame_num * len(persons) + 1)
                frame_index_sorted = np.arange(start_index, start_index + actual_frame_num * len(persons), self.stride * len(persons))

            elif self.read_method == 'random':
                all_frame_indexs = np.arange(0, len(pic_names_sorted) / len(persons))
                frame_index = np.random.choice(all_frame_indexs, size = self.frames_count, replace = False)
                index_sorted = np.argsort(frame_index, kind = 'mergesort')
                frame_index_sorted = frame_index[index_sorted] * len(persons)

            elif self.read_method == 'start_to_end':
                if (len(pic_names_sorted) / len(persons)) % self.frames_count == 0:
                    seq = (len(pic_names_sorted) / len(persons)) / self.frames_count
                    frame_index_sorted = np.arange(0, len(pic_names_sorted), step = int(seq) * len(persons), dtype = np.int)
                else:
                    remain = (len(pic_names_sorted) / len(persons)) % self.frames_count
                    seq = ((len(pic_names_sorted) / len(persons)) - int(remain)) / self.frames_count
                    frame_index_sorted = np.arange(0, (len(pic_names_sorted) / len(persons)) - int(remain), step = int(seq) * len(persons), dtype = np.int)
                    # random offset
                    offset = np.random.randint(0, int(remain + 1))
                    frame_index_sorted = np.add(frame_index_sorted, offset * len(persons))

            assert frame_index_sorted.shape[0] == self.frames_count, 'the selected total frames {} different from frame you want to read {}'.format(len(frame_index_sorted.shape[0], self.frames_count))
        
            for idx in range(self.frames_count):
                im = Image.open(os.path.join(folder_path, pic_names_sorted[frame_index_sorted[idx]]))
                # preprocess
                im.convert('RGB')
                im = self.transform(im)
                frames.append(im)
                label = labels[pic_names_sorted[frame_index_sorted[idx]]]
                all_labels.append(label)
            videos = torch.stack([frame for frame in frames], dim = 0)
            all_labels = torch.Tensor(all_labels).long().unsqueeze(1)
            self.label = torch.zeros(all_labels.shape[0], 2).scatter_(1, all_labels, 1)

            # videos with shape [n, c, h, w]
            return videos, self.label



if __name__ == "__main__":

    num_workers = 4
    im_folder = ''
    # eval_multiprocess(num_workers, im_folder)
import os
import pandas as pd
import numpy as np
from random import shuffle


def split_ffpp(dataset_name: str, save_folder: str, testing_ratio: float = 0.2, spe_sub = None):
    total_folders = []
    inner_folder = 'c23'
    manipulated = ''
    mani_subs = ['Deepfakes', 'Face2Face', 'FaceSwap', "NeuralTextures"]
    for mani_sub in mani_subs:
        if spe_sub == None:
            video_folders = os.listdir(os.path.join(manipulated, mani_sub, inner_folder))
            for video_folder in video_folders:
                total_folders.append(os.path.join(manipulated, mani_sub, inner_folder, video_folder))
        elif mani_sub == spe_sub:
            video_folders = os.listdir(os.path.join(manipulated, mani_sub, inner_folder))
            for video_folder in video_folders:
                total_folders.append(os.path.join(manipulated, mani_sub, inner_folder, video_folder))
    origin = ''
    video_folders = os.listdir(os.path.join(origin, inner_folder))
    for video_folder in video_folders:
        total_folders.append(os.path.join(origin, inner_folder, video_folder))
    total_len = len(total_folders)
    shuffle(total_folders)
    test_folders = total_folders[:int(total_len * testing_ratio)]
    train_folders = total_folders[int(total_len * testing_ratio):]
    column = ['folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    try:
        df_train = pd.DataFrame(columns = column, data = train_folders)
        if spe_sub == None:
            df_train.to_csv(os.path.join(save_folder, dataset_name + '_full_train.csv'))
        else:
            df_train.to_csv(os.path.join(save_folder, dataset_name + '_' + spe_sub + '_train.csv'))
        df_test = pd.DataFrame(columns = column, data = test_folders)
        if spe_sub == None:
            df_test.to_csv(os.path.join(save_folder, dataset_name + '_full_test.csv'))
        else:
            df_test.to_csv(os.path.join(save_folder, dataset_name + '_' + spe_sub + '_test.csv'))
    except Exception as e:
        print(e)
    print(f'Done: {dataset_name}')

def train_test_split(dataset_name: str, dataset_path:str, save_folder: str, testing_ratio: float = 0.2):
    test_folders = []
    train_folders = []
    if dataset_name == 'dfdc':
        folders = os.listdir(dataset_path)
        total_len = len(folders)
        test_index = np.random.permutation(total_len)
        for i in range(len(test_index)):
            if i < int(total_len * testing_ratio):
                test_folders.append(folders[test_index[i]])
            else:
                train_folders.append(folders[test_index[i]])
    elif dataset_name == 'celebdf':
        total_folders = []
        sub_folders = os.listdir(dataset_path)
        if 'YouTube-real' in sub_folders:
            sub_folders.remove('YouTube-real')
        for folder in sub_folders:
            video_folders = os.listdir(os.path.join(dataset_path, folder))
            for video_folder in video_folders:
                total_folders.append(os.path.join(dataset_path, folder, video_folder))
        total_len = len(total_folders)
        test_index = np.random.permutation(total_len)
        for i in range(len(test_index)):
            if i < int(total_len * testing_ratio):
                test_folders.append(total_folders[test_index[i]])
            else:
                train_folders.append(total_folders[test_index[i]])
    elif dataset_name == 'ffpp':
        total_folders = []
        inner_folder = 'face-img'
        manipulated = ''
        mani_subs = ['Deepfakes', 'FaceSwap']
        for mani_sub in mani_subs:
            video_folders = os.listdir(os.path.join(manipulated, mani_sub, inner_folder))
            for video_folder in video_folders:
                total_folders.append(os.path.join(manipulated, mani_sub, inner_folder, video_folder)) 
        origin = ''
        video_folders = os.listdir(os.path.join(origin, inner_folder))
        for video_folder in video_folders:
            total_folders.append(os.path.join(origin, inner_folder, video_folder))
        total_len = len(total_folders)
        test_index = np.random.permutation(total_len)
        for i in range(len(test_index)):
            if i < int(total_len * testing_ratio):
                test_folders.append(total_folders[test_index[i]])
            else:
                train_folders.append(total_folders[test_index[i]])
    column = ['folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    try:
        df_train = pd.DataFrame(columns = column, data = train_folders)
        df_train.to_csv(os.path.join(save_folder, dataset_name + '_train.csv'))
        df_test = pd.DataFrame(columns = column, data = test_folders)
        df_test.to_csv(os.path.join(save_folder, dataset_name + '_test.csv'))
    except Exception as e:
        print(e)
    print(f'Done: {dataset_name}')

def read_train_test_split(csv_loc: str, dataset_name: str, train_test: str):
    read_file_name = os.path.join(csv_loc, dataset_name + '_' + train_test + '.csv')
    df = pd.read_csv(read_file_name)
    folders = df["folder"].values.tolist()
    return folders

def split_test_val(csv_loc: str):
    df = pd.read_csv(csv_loc)
    

if __name__ == "__main__":
    dataset_name = 'ffpp'
    dataset_path = ''
    testing_ratio = 0.2
    save_folder = ''
    split_ffpp(dataset_name = dataset_name, save_folder = save_folder, testing_ratio = testing_ratio, spe_sub = 'Deepfakes')
    split_ffpp(dataset_name = dataset_name, save_folder = save_folder, testing_ratio = testing_ratio, spe_sub = 'Face2Face')
    split_ffpp(dataset_name = dataset_name, save_folder = save_folder, testing_ratio = testing_ratio, spe_sub = 'FaceSwap')
    split_ffpp(dataset_name = dataset_name, save_folder = save_folder, testing_ratio = testing_ratio, spe_sub = 'NeuralTextures')




from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random
import os

import cv2
import albumentations as A
from glob import glob
# from albumentations.pytorch import ToTensor
import torchvision.transforms as transforms

# for meta
class Mini_MetaDataset(Dataset):
    def __init__(self, args, support, query, mode='train'):
        self.args = args
        self.s_tasks = support
        self.q_tasks = query
        self.mode = mode
        self.resize = self.args.resize
        self.data_path_prefix = '/data1/wangjingtao/workplace/python/data/meta-oct/classic/mini-imagenet/images'

        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((84, 84)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

    def __len__(self):
        return len(self.s_tasks)

    def __getitem__(self, idx):
        s_task = self.s_tasks[idx]
        q_task = self.q_tasks[idx]
        
        s_img_list, s_lab_list = [], []
        for each_path in s_task:

            image = self.transform(os.path.join(self.data_path_prefix,each_path[0]))
            label = each_path[1]

            s_img_list.append(image)
            s_lab_list.append(label)

        s_img_tensor = torch.stack(s_img_list)  # 沿指定维度拼接,会多一维 [shot, channel, height, width]

        q_img_list, q_lab_list = [], []
        for each_path in q_task:

            image = self.transform(os.path.join(self.data_path_prefix,each_path[0]))  
            label = each_path[1]

            q_img_list.append(image)
            q_lab_list.append(label)
        q_img_tensor = torch.stack(q_img_list)
        
        return s_img_tensor, torch.LongTensor(s_lab_list), q_img_tensor,  torch.LongTensor(q_lab_list)

class MiniDataset(Dataset):
    def __init__(self, args, data_list, mode='train'):
        self.args = args
        self.img = data_list
        self.resize = self.args.resize

        self.data_path_prefix = '/data1/wangjingtao/workplace/python/data/meta-oct/classic/mini-imagenet/images'

        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((84, 84)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.transform(os.path.join(self.data_path_prefix, self.img[idx][0]))
        label =  self.img[idx][1]
        
        return image, label
    


if __name__ == '__main__':
    trainframe = pd.read_csv("train_data.csv")
    train_classes = np.unique(trainframe["ID"])
    train_classes = list(train_classes)
    
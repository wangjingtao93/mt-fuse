
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



class BasicDataset(Dataset):
    def __init__(self, fileroots, resize= 256, mode='train'):
        self.tasks = fileroots
        self.mode = mode
        
        self.resize = resize

        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])


    def __len__(self):
        return len(self.tasks)


    def __getitem__(self, idx):
        task = self.tasks[idx]

        img_list, lab_list = [], []
        for each_path in task:

            image = self.transform(each_path[0])  
            label = each_path[1]

            img_list.append(image)
            lab_list.append(label)

        img_tensor = torch.stack(img_list)  # 沿指定维度拼接,会多一维 [shot, channel, height, width]
        
        # lab_tensor = torch.stack(lab_list)  # [shot, label] # 要考虑以下

        # 很关键的一步，重新生成标签
        lab_relative = np.zeros(len(lab_list))
        unique_c = np.unique(lab_list)
        for idx, l in enumerate(unique_c):
            lab_relative[lab_list == l] =idx

        lab_tensor = torch.Tensor(lab_relative) # 并不需要拼接

        return [img_tensor, lab_tensor]
    
class DLdataset(Dataset):
    def __init__(self, data_list, resize=256, mode='train', relative_path=''):
        self.img = data_list
        self.resize = resize
        self.relavtive_path = relative_path
        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.transform(os.path.join(self.relavtive_path, self.img[idx][0]))
        label =  self.img[idx][1]
        
        return image, label

if __name__ == '__main__':
    trainframe = pd.read_csv("../TL/data/crop_forge/train_data.csv")
    train_classes = np.unique(trainframe["ID"])
    train_classes = list(train_classes)
    
    # train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    # num_classes = 2 # 每个task包含的类别
    # num_instances = 5 # 每个task，shot 和query. 即每个类别选取的样本量
    # for each_task in range(10):  # num_train_task 训练任务的总数
    #     task = Task(train_classes, num_classes, num_instances, trainframe)
    #     train_support_fileroots_alltask.append(task.train_support_roots)
    #     train_query_fileroots_alltask.append(task.train_query_roots)
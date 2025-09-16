

import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    def __init__(self, n_way):
        super(Learner, self).__init__()
        
        # 定义卷积层和池化层的网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        
        # 定义全连接层
        self.fc = nn.Linear(32 * 5 * 5, n_way)

    def forward(self, x):
        # 第一层卷积 -> BN -> ReLU -> 池化
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 第二层卷积 -> BN -> ReLU -> 池化
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 第三层卷积 -> BN -> ReLU -> 池化
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 第四层卷积 -> BN -> ReLU -> 池化
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        
        # 拉平并通过全连接层
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x

# # 测试网络结构
# n_way = 5  # 假设有5个类别
# model = Learner(n_way)
# sample_input = torch.randn(1, 3, 84, 84)  # 输入为3通道，84x84大小
# output = model(sample_input)
# print("输出尺寸:", output.shape)  # 应输出 [1, 5]

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)  # 保留batch size以用于view操作
        return x.view(batch_size, -1)

class Conv84(nn.Module):
    def __init__(self, nc):
        super(Conv84, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 添加了padding，保持尺寸
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 42x42
            nn.Conv2d(64, 64, 3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 21x21
            nn.Conv2d(64, 64, 3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 10x10
        )
        self.flatten = Flatten()
        self.fc = nn.Linear(64 * 10 * 10, nc)  # 根据新的输出尺寸调整

    def forward(self, x):
        x = self.network(x)
        # print("Before Flatten:", x.shape)
        x = self.flatten(x)
        # print("After Flatten:", x.shape)
        x = self.fc(x)
        return x


# # 检查网络结构
# model = Conv84(5)
# sample_input = torch.randn(1, 3, 84, 84)  # 3通道，84x84的输入
# output = model(sample_input)
# print("输出尺寸:", output.shape)

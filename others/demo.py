import time
import os
import common.utils as utils
# #Read fime name
# FileName='/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/maml/tmp/resnet18/2023-07-10-22-11-58/meta_potential_model_for_meta_val_2023-07-10-22-11-58.pth'

# #print file creation time
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(os.stat(FileName).st_ctime)))

# #print file modified time
# print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(os.stat(FileName).st_mtime)))


# import random

# def random_partition(input_list, partition_sizes):
#     # 检查划分是否合法
#     total_elements = len(input_list)
#     if sum(partition_sizes) != total_elements:
#         raise ValueError("划分大小之和必须等于输入列表的长度")

#     # 随机打乱输入列表
#     random.shuffle(input_list)

#     # 进行划分
#     partitions = []
#     start = 0
#     for size in partition_sizes:
#         end = start + size
#         partition = input_list[start:end]
#         partitions.append(partition)
#         start = end

#     return partitions

# # 示例用法
# input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # partition_sizes = [3, 2, 4]

# # result = random_partition(input_list, partition_sizes)
# # print(result)
# utils.set_seed(1)

# random.shuffle(input_list)
# print(input_list)

# random.shuffle(input_list)
# print(input_list)

# def replace_elements_with_dict_values(string_list, mapping_dict):
#     result_list = [mapping_dict.get(item, item) for item in string_list]
#     return result_list

# # 示例用法
# string_list = ['apple', 'banana', 'cherry', 'apple', 'date']
# mapping_dict = {'apple': 1, 'banana': 2, 'cherry': 3}

# result = replace_elements_with_dict_values(string_list, mapping_dict)
# print(result)


# import torch

# # 创建一个 PyTorch 张量
# x = torch.randn(3, 4)

# # 获取张量的形状
# shape = x.shape
# print("形状:", shape)

# # 或者使用 size() 方法
# size = x.size()
# print("大小:", size)

import timm

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
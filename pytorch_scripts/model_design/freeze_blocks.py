
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import timm
import torch
import model.transformer.vit_model as vit
from model.mtb.mtb_model import create_mtb 

import torch.nn as nn
# net1 = nn.Sequential(nn.Conv1d(20,10,3),
#                     nn.ReLU(True),
#                     nn.Linear(10,5))
# print(net1)
# print(net1.modules)
# print(net1[2].weight)
# # 遍历所有参数
# for param in net1.parameters():
#     print(param)
# net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)

def func_1():

    net = vit.vit_base_patch16_224(num_classes=4)
    print(net.blocks[0].attn.qkv.weight.requires_grad)
    for param in net.parameters():
        param.requires_grad = False

    net.train() # 并不会改变requires_grad属性
    print(net.blocks[0].attn.qkv.weight.requires_grad)

    for i in [11]:
        net.blocks[i]
        for param in net.blocks[i].parameters():
            param.requires_grad = True

    print(net.blocks[0].attn.qkv.weight.requires_grad)
    print(net.blocks[11].attn.qkv.weight.requires_grad)

    # 经过torch.load，requires_grad会重新置为True
    state_dict = torch.save(net.state_dict(), 'tmp.pth')
    net_2 = vit.vit_base_patch16_224(num_classes=4)
    net_2.load_state_dict(torch.load('tmp.pth'))
    print('++++++++')
    print(net_2.blocks[0].attn.qkv.weight.requires_grad)
    print(net_2.blocks[11].attn.qkv.weight.requires_grad)
    print(net_2.blocks[9].attn.qkv.weight.requires_grad)
    print(net.blocks[9].attn.qkv.weight.requires_grad)

# print net.blocks.0.attn.qkv.weight.requires_grad

def func_2():
    net_3 = create_mtb(num_classes=4, depth=6)
    print(net_3)

    for param in net_3.parameters():
        param.requires_grad = False
    for param in net_3.meta_learner.parameters():
        param.requires_grad = True

    print(net_3.blocks[0].attn.qkv.weight.requires_grad)
    print(net_3.blocks[1].attn.qkv.weight.requires_grad)

    print(net_3.meta_learner.fc1.weight.requires_grad)
    print(net_3.meta_learner.head.weight.requires_grad)


def func_3():
    # 加载预训练之后会是继续冻结状态吗？
    net_3 = create_mtb(num_classes=4, depth=6)
    net_3.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imaml/result_20230908_sub10/mtb/2023-09-28-11-59-14/meta_epoch_0.pth'))

    print(net_3.blocks[0].attn.qkv.weight.requires_grad)
    print(net_3.blocks[1].attn.qkv.weight.requires_grad)

    print(net_3.meta_learner.fc1.weight.requires_grad)
    print(net_3.meta_learner.head.weight.requires_grad)


func_2()
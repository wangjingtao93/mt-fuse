# mtb_res, 保存固定层的参数
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import timm
import torch
import model.transformer.vit_model as vit
import model.mtb.mtb_res_bak as mtb_res_bak
import torchvision.models
import model.resnet
import torch.nn as nn
# 基本vit是12层
vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
# net_2 = vit.vit_base_patch16_224(num_classes=4, depth=6)

resnet18 = torchvision.models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 4)
# torch.save(resnet18.state_dict(), 'resnet18.pth')
# tmp_net = model.resnet.ResNet18(4)
# tmp_net.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/resnet18.pth'))

mtb_res_net = mtb_res_bak.create_mtb_res(num_classes=4, depth=3)


# 定义一个函数，用于复制一个模型的参数到另一个模型
def copy_trans_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()


    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)
        else:
            print(name)

def copy_resnet_params(src_model, dest_model):
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if 'layer' in name:
           name = 'resdiual_' + name

        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)
        else:
            print(name)



copy_trans_params(vit_net, mtb_res_net)

# copy_resnet_params(resnet18, mtb_res_net)

torch.save(mtb_res_net.state_dict(),'mtb_res_without_res.pth')
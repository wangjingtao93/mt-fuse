# transformer, 保存固定层的参数
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import timm
import torch
import model.transformer.vit_model as vit
import model.mtb.mtb_model as mtb

# 基本vit是12层
net_1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
# net_2 = vit.vit_base_patch16_224(num_classes=4, depth=6)

net_2 = mtb.create_6b_mfc(num_classes=4, depth=6)





# 定义一个函数，用于复制一个模型的参数到另一个模型
def copy_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    # i = 1
    # for name, params in src_state_dict.items():
    #     print(i, ': ', name)
    #     i += 1
    # i = 1
    # for name, params in dest_state_dict.items():
    #     print(i, ': ', name)
    #     i += 1
        
    # 更新目标模型的状态字典，将源模型的参数复制过去

    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)
        else:
            print(name)


copy_params(net_1, net_2)

# 6个transform_encoder, 用fc做中间连接层
torch.save(net_2.state_dict(), "mtb_6b_mfc_tmp.pth")




# print(net_1)
# print(net_2)
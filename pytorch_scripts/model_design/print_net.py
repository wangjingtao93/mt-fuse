# transformer
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import timm
import torch
import model.transformer.vit_model as vit
from model.mtb.mtb_res_bak import create_mtb_res

# net_1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
net_1 = create_mtb_res(num_classes=4, depth=3)

print(net_1)
net_state_dict = net_1.state_dict()
i = 1
for name, params in net_state_dict.items():
    print(i, ': ', name)
    i += 1

exit()

print(len(list(net_1.parameters())))
print(list(net_1.parameters())[151])

torch.save(net_1.state_dict(), 'tmp.pth')



net_2 = vit.vit_base_patch16_224(num_classes=4)

params_state_dict = torch.load('tmp.pth')
net_2.load_state_dict(params_state_dict)


# 定义一个函数，用于复制一个模型的参数到另一个模型
def copy_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()
    
    # 更新目标模型的状态字典，将源模型的参数复制过去
    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)
        else:
            print(name)


copy_params(net_1, net_2)




# print(net_1)
# print(net_2)
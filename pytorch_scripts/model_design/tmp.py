import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import timm
import torch
from model.transformer.vit_model import VisionTransformer

# print(timm.list_models())

# 创建 vit_base_r50_s16_224 模型
# model = timm.create_model("vit_base_patch16_224", num_classes=2, pretrained=False)
# model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
# model = VisionTransformer(num_classes=2,embed_dim=768, num_heads=12)
model = VisionTransformer(num_classes=2,embed_dim=384, num_heads=6)

# with open('model_strue.txt', 'w') as f:
#     f.write(str(model))



model_init = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)

state_dict1  = model_init.state_dict()
state_dict2 = model.state_dict()

for name, param in state_dict1.items():
    if name in state_dict2:
        state_dict2[name].copy_(param)
    else:
        print(name)
        print('xxxxxxx')
# print(model)

print('finish++++++++++++++++++++++')


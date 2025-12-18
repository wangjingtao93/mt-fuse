import timm
from model.mt_fuse.mt_fuse_model import MT_Fuse_Model
import torch

def mt_fuse_net_load(args):
    model = MT_Fuse_Model(num_classes=args.num_classes,embed_dim=384, num_heads=6, is_fuse=args.is_fuse)

    # # 使用timm初始化方式
    model_init = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)

    # model_init.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/tmp/thyroid_fuse_micro/maml/mt_fuse_model/pretrain_False/2025-08-22-15-19-59/save_meta_pth/meta_epoch_22.pth'))

    # state_dict1  = model_init.state_dict()
    # state_dict2 = model.state_dict()

    # for name, param in state_dict1.items():
    #     if name in state_dict2 and name != 'pos_embed':
    #         state_dict2[name].copy_(param)
    #     else:
    #         print(name)

    return model
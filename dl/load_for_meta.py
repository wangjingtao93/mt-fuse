import torch
import os
from model.mt_fuse.mt_fuse_model import MT_Fuse_Model

def load_for_meta(args, meta_epoch):
    # meta的测试
    # 先采用命名规则吧
    path = os.path.join(args.store_dir,'save_meta_pth', f'meta_epoch_{meta_epoch}.pth')

    if args.net == 'mt_fuse_model':
        mt_fuse_model = MT_Fuse_Model(num_classes=args.num_classes,embed_dim=384, num_heads=6, is_fuse=args.is_fuse)
        state_dict1  = torch.load(path)
        state_dict =  mt_fuse_model.state_dict()
        for name, param in state_dict1.items():
            if name == 'pos_embed' and args.is_fuse:
                print(name)
                print('using mt_fuse')
            elif name in state_dict:
                state_dict[name].copy_(param)
            else:
                print(name, ' not in src_state_dict')
    else:
        state_dict = torch.load(path)

    return state_dict
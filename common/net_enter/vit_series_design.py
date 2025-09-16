import timm
from timm.models.layers import trunc_normal_
import torch
import model.timm_register_models
import model.retf_found.models_vit as models_vit
from model.retf_found.util.pos_embed import interpolate_pos_embed
from model.transformer.vit_model import VisionTransformer

def vit_load(args):
    if args.net == 'vit_base_patch16_224':
        model = get_vit_base_patch16_224(args, depth=12)

    elif args.net == 'vit_base_patch16_224_depth_6':

        model = get_vit_base_patch16_224(args, depth=6)
        print('using transformer with pretrain depth_6')

    elif args.net == 'vit_base_patch16_224_depth_3':
        print('using transformer with pretrain depth_3')

    elif args.net == 'vit_tiny_patch16_224':
        model = timm.create_model(args.net, pretrained=args.is_load_imagenet, num_classes=args.num_classes)
        if args.is_load_zk:
            model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result_20231010_sub10/pre_train/zk/dl/vit_tiny_patch16_224/2023-10-23-17-48-54/meta_epoch/taskid_0/best_model_for_valset_0.pth'))
            print('using **vit_tiny_patch16_224** with zk pretrain')

    elif args.net == 'vit_small_patch16_224':
        # model = timm.create_model(args.net, pretrained=args.is_load_imagenet, num_classes=args.num_classes)
        # if args.is_load_imagenet:
        #     model.load_state_dict(torch.load(glob.glob(os.path.join(imagenet_pre_path, args.net, '*'))[0]))
        #     print(f'using **{args.net} depth={args.trans_depth}**  imagenet Pretrain')
        model = VisionTransformer(num_classes=args.num_classes,embed_dim=384, num_heads=6)
    elif args.net == 'Conformer_tiny_patch16':
        model = timm.create_model(args.net, pretrained=args.is_load_imagenet, num_classes=args.num_classes)
        print('using **Conformer_tiny_patch16** with pretrain')

    elif args.net == 'retfound':
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=args.num_classes,
            drop_path_rate=0.2,
            global_pool=False,
        )

        # load RETFound weights
        # chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_cfp_weights.pth'
        chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_oct_weights.pth'
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    return model


def get_vit_base_patch16_224(args, depth):
    # model = vit_base_patch16_224(num_classes=args.num_classes, depth=depth)
    if args.is_load_imagenet:
        model =  timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        # vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        # vit_net = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.num_classes)
        # vit_net.load_state_dict(torch.load(glob.glob(os.path.join(imagenet_pre_path, args.net, '*'))[0]))
        # copy_trans_params(vit_net, model)
        # model.load_state_dict(torch.load(glob.glob(os.path.join(imagenet_pre_path, args.net, '*'))[0]))
        print(f'Using **[vit] {depth}** load **[imagenet]***')

    elif args.is_load_zk:
        if depth == 12:
            load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224/2023-10-16-19-49-04/best_model.pth'
        elif depth == 6:
            load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224_depth_6/2023-10-17-10-31-27/best_model.pth'
        elif depth == 3:
            load_path == '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/imagenet_zk/vit_base_patch16_224_depth_3/2023-10-16-19-22-26/best_model.pth'

        model.load_state_dict(torch.load(load_path))
        print(f'Using **[resnet18] {depth}** load **[zk]***')
    elif args.is_load_imagenet_zk:
        if depth == 12:
            load_path = os.path.join(args.project_path, 'result_20231010_sub10/pre_train/zk/dl/vit_base_patch16_224/2023-10-20-15-15-45/meta_epoch/taskid_0/best_model_for_valset_0.pth')
        elif depth == 6:
            load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224_depth_6/2023-10-17-10-31-27/best_model.pth'
        elif depth == 3:
            load_path == '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/imagenet_zk/vit_base_patch16_224_depth_3/2023-10-16-19-22-26/best_model.pth'

        model.load_state_dict(torch.load(load_path))
        print(f'Using **[resnet18] {depth}** load **[imagenet & zk]***')

    else:
        # model =  timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.num_classes)
        model = VisionTransformer(num_classes=args.num_classes,embed_dim=768, num_heads=12)

    return model
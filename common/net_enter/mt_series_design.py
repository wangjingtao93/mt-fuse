from model.mtb.mtb_model import create_mtb, create_6b_mfc
from model.mtb.mtb_res_bak import create_mtb_res
import timm
import model.timm_register_models
import torch
import torchvision
import glob

def mt_load(args):
    if args.net == 'mtb':
        model = get_mtb()
        print('using mtb with pretrain')

    elif args.net == 'mtb_6b_mfc':
        model == get_mtb_6b_mfc()
        print('using mtb_6b_mfc with pretrain')

    elif args.net == 'mtb_res':
        model = get_mtb_res_net()
        print('using **mtb_res** with pretrain')

    elif args.net == 'mt_tiny_model_lymph':
        model = timm.create_model(args.net, pretrained=args.is_load_imagenet, num_classes=args.num_classes)
        print(f'using **{args.net} depth={args.trans_depth}** No Pretrain')

    elif args.net == 'mt_small_model_lymph':
        model = timm.create_model(args.net, pretrained=False, num_classes=args.num_classes, depth=args.trans_depth)
        if args.is_load_imagenet:
            # mt_lymph_pro_online()
            model = mt_lymph_pro_location(model)
            print(f'using **{args.net} depth={args.trans_depth}** imagenet Pretrain')

def get_mtb(args):
    model = create_mtb(num_classes=args.num_classes, depth=6)
    # model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_pth/mtb.pth'))
    if args.is_load_imagenet:
        vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        copy_trans_params(vit_net, model)



def get_mtb_6b_mfc(args):
    model = create_6b_mfc(num_classes=args.num_classes, depth=6)
    # model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/mtb_6b_mfc_tmp.pth'))
    if args.is_load_imagenet:
        vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        copy_trans_params(vit_net, model)

    return model

def get_mtb_res_net(args):
    model = create_mtb_res(num_classes=args.num_classes, depth=3)

    if args.is_load_zk:
        model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/mtb_res/2023-10-17-11-27-36/best_model.pth'))

    if args.is_load_imagenet:
        resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 4)

        vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)

        copy_trans_params(vit_net, model)
        copy_resnet_params(resnet18, model)
    return model


def copy_trans_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)


def copy_resnet_params(src_model, dest_model):
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if 'layer' in name:
           name = 'resdiual_' + name

        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)

def mt_lymph_pro_location(model):
    path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20240219/lymph/imagenet/dl/mt_small_model_lymph/2024-02-20-21-58-08/meta_epoch/taskid_0/best_model_for_valset_0.pth'
    src_state_dict = torch.load(path)

    dest_state_dict = model.state_dict()

    for name, param in src_state_dict.items():
        if 'meta_cnn_fc' not in name and 'meta_trans_fc' not in name and 'meta_fc' not in name:
            dest_state_dict[name].copy_(param)
        # else:
        #     print('moudle_name: ', name)
    return model

def mt_lymph_pro_online(args, model):
    # 加载transblock预训练参数
    imagenet_pre_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20231010_sub10/pre_train/imagenet'

    src_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=args.num_classes, depth=args.trans_depth)
    src_model.load_state_dict(torch.load(glob.glob(os.path.join(imagenet_pre_path, 'vit_small_patch16_224', '*'))[0]))

    src_state_dict = src_model.state_dict()
    dest_state_dict = model.state_dict()

    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)

    # 到底需不需要这一步呢？dest_state_dict是潜拷贝，貌似不需要再load了
    model.load_state_dict(dest_state_dict)

    return model

def meta_found_pre(model):
    for name, param in model.named_parameters():
        if 'meta' in name:
            param.requires_grad = True
        else:
            param.requires_grad  = False

    return model
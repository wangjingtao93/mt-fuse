
import torch
import torch.nn as nn
import torchvision.models
import os

def resnet_load(args):
    if args.net == 'resnet18':
        if args.is_load_imagenet:
                model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet18(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,  args.num_classes)

        if args.is_load_zk:
            load_path = os.path.join(args.project_path,'result_20231010_sub10/zk/dl/resnet18/2023-10-16-19-44-22/best_model.pth')
            model.load_state_dict(torch.load(load_path))
            print("Using ****[resnet18]**** load [zk]")
        elif args.is_load_imagenet_zk:
            load_path = os.path.join(args.project_path, 'result_20231010_sub10/pre_train/zk/dl/resnet18/2023-10-20-15-15-55/meta_epoch/taskid_0/best_model_for_valset_0.pth')
            model.load_state_dict(torch.load(load_path))

            print("Using ****[resnet18]**** load [imagenet & zk]")

        elif args.is_load_st_sub:

            model.load_state_dict(del_fc(args.num_classes))

            print("Using ****[resnet18]**** load []")
    elif args.net == 'resnet34':
        if args.is_load_imagenet:
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet34(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,  args.num_classes)

    elif args.net == 'resnet50':
        if args.is_load_imagenet:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet50(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,  args.num_classes)

    elif args.net == 'resnet101':
        if args.is_load_imagenet:
            model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet101(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,  args.num_classes)

    return model

# 去除预训练模型的全连接层
def del_fc(num_classes):
    model =  torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,  16)
    model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication-20250703/result_20231010_sub10/pre_train/st_sub_pretrain/dl/resnet18/2023-11-14-15-33-07/meta_epoch/taskid_0/best_model_for_valset_0.pth'))

    del model.fc
    model.add_module('fc',nn.Linear(num_ftrs,num_classes))
    return model.state_dict()

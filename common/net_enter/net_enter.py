import glob

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models

import timm

import os
import model.timm_register_models
from model.meta_found.meta_found import Meta_Found

from common.net_enter.resnet_design import resnet_load
from common.net_enter.easy_net_design import easy_net_load
from common.net_enter.vit_series_design import vit_load
from  common.net_enter.mt_series_design import mt_load
from  common.net_enter.mt_fuse_net_design import mt_fuse_net_load

from model.more_models import *


def net_enter(args):

    resnet_ls = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    easy_net_ls = ['vgg11', 'vgg13','vgg16','vgg19', 'squeezenet1_0', 'densenet121', 'convnet_4']
    vit_ls = ['vit_base_patch16_224', 'vit_base_patch16_224_depth_6', 'vit_base_patch16_224_depth_3', 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'Conformer_tiny_patch16', 'retfound']

    mt_ls = ['mtb','mtb_6b_mfc','mtb_res','mt_tiny_model_lymph', 'mt_small_model_lymph']

    if args.net == 'alexnet':
        model = alexnet()

    elif args.net in easy_net_ls:
        model = easy_net_load(args)

    elif args.net  in resnet_ls :
        # model = torchvision.models.resnet18(pretrained=args.is_load_imagenet)
        model = resnet_load(args)

    elif args.net in vit_ls:
        model = vit_load(args)

    elif args.net in mt_ls:
        model = mt_load(args)

    elif args.net == 'meta_found':
        model = Meta_Found(args.num_classes)
        # meta_found_pre()

    elif args.net == 'mt_fuse_model':
        model = mt_fuse_net_load(args)
    else:
        raise ValueError('No implmentation model')


    print(f'++++++++++++++using {args.net}--------------')

    return model






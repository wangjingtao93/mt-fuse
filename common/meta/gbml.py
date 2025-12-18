import numpy as np
import glob
import torch
import torch.nn as nn
import os
import torchvision.models
import timm
from model.transformer.vit_model import vit_base_patch16_224 
from model.mtb.mtb_model import create_mtb 
from model.mtb.mtb_res_bak import create_mtb_res
from model.ConvNet import Fourlayers
from model.more_models.conv_84 import Learner, Conv84
from model.meta_found import meta_found
from model.meta_found.meta_found import Meta_Found
from model.mt_fuse.mt_fuse_model import MT_Fuse_Model
# 暂时未找到这俩
# from net.convnet import ConvNet
# from net.resnet import ResNet

# https://github.com/sungyubkim/GBML/blob/master/main.py
class GBML:
    '''
    Gradient-Based Meta-Learning
    '''
    def __init__(self, args):
        self.args = args
        # self.batch_size = self.args.batch_size
        self.imagenet_pre_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result/result_20231010_sub10/pre_train/imagenet'

        return None

    def _init_net(self):
        if self.args.net == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=self.args.is_meta_load_imagenet)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, self.args.n_way)
        elif self.args.net == 'resnet34':
            self.network = torchvision.models.resnet34(pretrained=self.args.is_meta_load_imagenet)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, self.args.n_way)
        elif self.args.net == 'resnet50':
            self.network = torchvision.models.resnet50(pretrained=self.args.is_meta_load_imagenet)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, self.args.n_way)
        elif self.args.net == 'resnet101':
            self.network = torchvision.models.resnet101(pretrained=self.args.is_meta_load_imagenet)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, self.args.n_way)

        elif self.args.net == 'vit_base_patch16_224':
            self.network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.args.num_classes)
            # self.network = vit_base_patch16_224(num_classes=self.args.n_way)
            # if self.args.is_meta_load_imagenet:
            #     self.network.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result/result_20231010_sub10/pre_train/imagenet/vit_base_patch16_224/vit_base_patch16_224.pth'))
            print('using transformer with pretrain')
            # self.freeze_blocks()

        elif self.args.net == 'vit_tiny_patch16_224':
            self.network = timm.create_model('vit_tiny_patch16_224', pretrained=self.args.is_meta_load_imagenet, num_classes=self.args.num_classes)
            if self.args.is_load_zk:
                self.network.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result_20231010_sub10/pre_train/zk/dl/vit_tiny_patch16_224/2023-10-23-17-48-54/meta_epoch/taskid_0/best_model_for_valset_0.pth'))
                print('using **vit_tiny_patch16_224** with zk pretrain')

        elif self.args.net == 'vit_base_patch16_224_depth_6':
            self.network = vit_base_patch16_224(num_classes=self.args.n_way, depth=6)
            self.network.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_pth/vit_base_patch16_224_depth_6.pth'))
            print('using transformer with pretrain depth_6') 
        elif self.args.net == 'mtb':
            self.network = create_mtb(num_classes=self.args.n_way, depth=6)
            self.network.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_pth/mtb.pth'))
            print('using mtb with pretrain')

            self.freeze_blocks_mtb()

        elif self.args.net == 'mtb_res':
            self.network = create_mtb_res(num_classes=self.args.n_way)
            self.network.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/dl/mtb_res/2023-10-17-11-27-36/best_model.pth'))
            print('using mtb_res with pretrain')

            self.freeze_blocks_mtb()

        elif self.args.net == 'vit_small_patch16_224':
            self.network = timm.create_model(self.args.net, pretrained=self.args.is_meta_load_imagenet, num_classes=self.args.num_classes)

        elif self.args.net == 'mt_small_model_lymph':
            self.network = timm.create_model(self.args.net, pretrained=False, num_classes=self.args.num_classes, depth=self.args.trans_depth)
            # self.mt_lymph_pro_online()
            self.mt_lymph_pro_location()

        elif self.args.net == 'convnet_4':
            self.network = timm.create_model(self.args.net, num_classes=self.args.num_classes)
            print('using convnet')

        elif self.args.net == 'conv_84':
            self.network = Learner(self.args.num_classes)
            print('using conv_84')

        elif self.args.net == 'meta_found':
            self.network = Meta_Found(self.args.num_classes)
            self.meta_found_pre()
            print('using meta_found')

        elif self.args.net == 'mt_fuse_model':
            self.network = MT_Fuse_Model(num_classes=self.args.num_classes,embed_dim=384, num_heads=6, is_fuse=False)
            # 使用timm初始化方式
            # model_init = timm.create_model('vit_small_patch16_224', pretrained=self.args.is_meta_load_imagenet, num_classes=2)

            # 直接加载
            if self.args.is_meta_load_imagenet:
                model_init = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
                model_init.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/checkpoint/vit/vit_small_patch16_224.pth'))
            else:
                 model_init = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
                 
            state_dict1  = model_init.state_dict()
            state_dict2 =  self.network.state_dict()
            for name, param in state_dict1.items():
                if name == 'pos_embed' and self.args.is_fuse:
                    print(name)
                    print('using mt_fuse')
                elif name in state_dict2:
                    state_dict2[name].copy_(param)
                else:
                    print(name, ' not in src_state_dict')


        self.network.train()
        self.network.cuda()
        return None

    def _init_opt(self):
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None

    # 换成自己的
    def unpack_batch_bak(self, batch):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()
        return train_inputs, train_targets, test_inputs, test_targets
    def unpack_batch(self, batch):
        device = torch.device('cuda')
        train_inputs, train_targets = batch[0]
        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device, dtype=torch.long)

        test_inputs, test_targets = batch[1]
        test_inputs = test_inputs.to(device=device)
        test_targets = test_targets.to(device=device, dtype=torch.long)

        return train_inputs, train_targets, test_inputs, test_targets

    def inner_loop(self):
        raise NotImplementedError

    def outer_loop(self):
        raise NotImplementedError

    def lr_sched(self):
        self.lr_scheduler.step()
        return None

    def load(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.encoder.load_state_dict(torch.load(path))

    def save(self,filename):
        path = os.path.join(self.args.result_path, self.args.alg, filename)
        torch.save(self.network.state_dict(), path)

    def freeze_blocks(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for i in [11]:
            for param in self.network.blocks[i].parameters():
                param.requires_grad = True

    def freeze_blocks_mtb(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.meta_learner.parameters():
            param.requires_grad = True

    def freeze_blocks_mtb_res(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.meta_learner.parameters():
            param.requires_grad = True

    def mt_lymph_pro_online(self):

        # 加载transblock预训练参数
        src_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=self.args.num_classes, depth=self.args.trans_depth)
        src_model.load_state_dict(torch.load(glob.glob(os.path.join(self.imagenet_pre_path, 'vit_small_patch16_224', '*'))[0]))

        src_state_dict = src_model.state_dict()
        dest_state_dict = self.network.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

        # 到底需不需要这一步呢？dest_state_dict是潜拷贝，貌似不需要再load了
        self.network.load_state_dict(dest_state_dict)


        # lock trns_block参数
        for name, param in self.network.named_parameters():
            if 'meta' in name:
                param.requires_grad = True
            else:
                param.requires_grad  = False

    def mt_lymph_pro_location(self):
        if self.args.is_meta_load_imagenet:
            path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20240219/lymph/imagenet/dl/mt_small_model_lymph/2024-02-20-21-58-08/meta_epoch/taskid_0/best_model_for_valset_0.pth'
            src_state_dict = torch.load(path)

            dest_state_dict = self.network.state_dict()

            for name, param in src_state_dict.items():
                if 'meta_cnn_fc' not in name and 'meta_trans_fc' not in name and 'meta_fc' not in name:
                    dest_state_dict[name].copy_(param)
                else:
                    print('moudle_name: ', name)

            # 到底需不需要这一步呢？dest_state_dict是潜拷贝，貌似不需要再load了
            self.network.load_state_dict(dest_state_dict)
        if self.args.is_lock_notmeta:
        # lock trns_block参数
            for name, param in self.network.named_parameters():
                if 'meta' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad  = False

    # 冻结参数，可以减少显存使用
    def meta_found_pre(self):
        for name, param in self.network.named_parameters():
            if 'meta' in name:
                param.requires_grad = True
            else:
                param.requires_grad  = False
def copy_trans_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)




import os
from torch.utils.data import  DataLoader
import time

from common.originize_df import originze_df_four
from common.lymph.originize_ly_df import ly_originze_df,originze_source_data_pre
import common.lymph.originize_thyroid_fuse as ori_tf
from common.lymph.dataloader_ly import LY_MetaDataset, LY_dataset, TF_Dataset
from common.dataloader_oct import OCTDataset,OCT_MetaDataset
from common.imagenet.originize_imagenet import originze_imagenet
from common.mini_imagenet.mini_imagenet import Mini_MetaDataset, MiniDataset
from common.build_tasks import Train_task,Meta_Test_task
from common.dataloader import *


def oct_data(args):
# 组织数据
    df_meta_train, df_meta_test, df_test_dc = originze_df_four(args)

    # 测试集，最终测试任务
    # query set相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for task_id in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test)
        train_set = OCTDataset(args, test_task.support_roots, mode='train')
        val_set = OCTDataset(args, test_task.query_roots, mode='val')
        test_set = OCTDataset(args, test_task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        test_support_fileroots_all_task.append(train_loader)
        test_query_fileroots_alltask.append(val_loader)
        final_test_alltask.append(test_loader)

    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]


    # 训练集，元训练任务
    meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask = [], []
    for task_id in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_meta_train,task_id, mode='train')
        meta_train_support_fileroots_alltask.append(task.support_roots)
        meta_train_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，meta_train_task S 和Q
    meta_train = DataLoader(OCT_MetaDataset(args, meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)


    # meta_test 任务
    # 元测试测试任务
    meta_test_support_fileroots_alltask, meta_test_query_fileroots_alltask = [], []
    for task_id in range(args.n_val_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_meta_test,task_id, mode='oct_meta_test')
        meta_test_support_fileroots_alltask.append(task.support_roots)
        meta_test_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，meta_train_task S 和Q
    meta_test = DataLoader(OCT_MetaDataset(args, meta_test_support_fileroots_alltask, meta_test_query_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)

    return test_data_ls, meta_train, meta_test

def lymph_data(args):
    # 组织数据
    df_train, df_test_dc = ly_originze_df(args)


    # 测试集，测试任务
    # query set相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for task_id in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        # test_query_fileroots_alltask.append(test_task.query_roots)
        # test_support_fileroots_all_task.append(test_task.support_roots)
        # # final_test_alltask.append(test_task.test_roots)
        # test_support_fileroots_all_task.append(LY_dataset(args, test_task.support_roots, mode='train'))
        # test_query_fileroots_alltask.append(LY_dataset(args, test_task.query_roots, mode='val'))
        # final_test_alltask.append(LY_dataset(args, test_task.test_roots, mode='test'))

        train_set = LY_dataset(args, test_task.support_roots, mode='train')
        val_set = LY_dataset(args, test_task.query_roots, mode='val')
        test_set = LY_dataset(args, test_task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        test_support_fileroots_all_task.append(train_loader)
        test_query_fileroots_alltask.append(val_loader)
        final_test_alltask.append(test_loader)
        
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]


    # 构建训练任务
    # 训练集，训练任务
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    for task_id in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_train,task_id, mode='train')
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，train_task S 和Q   
    train_support_loader = DataLoader(LY_MetaDataset(args, train_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(LY_MetaDataset(args, train_query_fileroots_alltask), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    train_data_ls = train_support_loader+train_query_loader

    return test_data_ls, train_data_ls

def source_data_pre(args):

    df_train, df_test_dc = originze_source_data_pre()

    # 测试集，测试任务
    # query相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for task_id in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    train_data_ls = []

    return test_data_ls, train_data_ls


# lymph
def imagenet_exe(args):
    df_train, df_test_dc = originze_source_data_pre()

    # 测试集，测试任务
    # query相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for task_id in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    train_data_ls = []

    return test_data_ls, train_data_ls


# mini-imagenet
def mini_imagenet_enter(args):

    df_meta_train = pd.read_csv(os.path.join(args.project_path, 'data/mini-imagenet/train.csv'))

    df_meta_test = pd.read_csv(os.path.join(args.project_path, 'data/mini-imagenet/test.csv'))

    # 测试集，最终测试任务
    # query set相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试

    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]


    # 训练集，元训练任务
    meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask = [], []
    for task_id in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_meta_train,task_id, mode='train')
        meta_train_support_fileroots_alltask.append(task.support_roots)
        meta_train_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，meta_train_task S 和Q   
    meta_train = DataLoader(Mini_MetaDataset(args, meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)

  

    # meta_test 任务
    # 训练集，元训练任务
    meta_test_support_fileroots_alltask, meta_test_query_fileroots_alltask = [], []
    for task_id in range(args.n_val_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_meta_test,task_id, mode='meta_test')
        meta_test_support_fileroots_alltask.append(task.support_roots)
        meta_test_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，meta_train_task S 和Q   
    meta_test = DataLoader(Mini_MetaDataset(args, meta_test_support_fileroots_alltask, meta_test_query_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
 


    return test_data_ls, meta_train, meta_test


def thyroid_fuse_enter(args):
    # 组织数据
    df_train, df_test_dc = ori_tf.originze_df(args)

    # 测试集，测试任务
    # query set相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = {}, {}
    final_test_alltask = {}#用于最终测试
    for key, df_test in df_test_dc.items(): 
        test_task = Meta_Test_task(args, df_test, test_class='lymph')

        train_set = TF_Dataset(args, test_task.support_roots, mode='train')
        val_set = TF_Dataset(args, test_task.query_roots, mode='val')
        test_set = TF_Dataset(args, test_task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        test_support_fileroots_all_task[key] = train_loader
        test_query_fileroots_alltask[key] = val_loader
        final_test_alltask[key] = test_loader

    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]


    # 构建训练任务
    # 训练集，训练任务
    meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask = [], []
    for task_id in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_train,task_id, mode='train')
        meta_train_support_fileroots_alltask.append(task.support_roots)
        meta_train_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，meta_train_task S 和Q
    meta_train = DataLoader(LY_MetaDataset(args, meta_train_support_fileroots_alltask, meta_train_query_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)

    return test_data_ls, meta_train

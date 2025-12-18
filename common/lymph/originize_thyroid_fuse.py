import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
import common.utils as utils


def add_classID(pdframe, dict_labels):

    label_list = []
    for class_name in pdframe['Class']:
        if class_name in dict_labels:
            label_list.append(dict_labels[class_name])
        else:
            label_list.append('others')

    pdframe['Label'] = label_list
    return pdframe

# 构造多个测试任务
def originze_df(args):

    fuse_data_dir_ls = ['data/lymph/thyroid_fuse', 'data/lymph/camelyon17_fuse', 'data/lymph/camelyon16_fuse',\
                        'data/lymph/camelyon_fuse']
    if args.datatype == 'thyroid_fuse_micro':
        target_classes = ['micro', 'normal']
        label_dict = {'micro':1, "normal":0}
        fuse_data_dir =fuse_data_dir_ls[0]
    elif args.datatype == 'thyroid_fuse_macro':
        target_classes = ['macro', 'normal']
        label_dict = {'macro':1, "normal":0}
        fuse_data_dir =fuse_data_dir_ls[0]
    elif args.datatype == 'thyroid_fuse_ipbs':
        target_classes = ['ipbs', 'normal']
        label_dict = {'ipbs':1, "normal":0, 'micro':1}
        fuse_data_dir =fuse_data_dir_ls[0]
    elif args.datatype == 'thyroid_fuse_itcs':
        target_classes = ['itcs', 'normal']
        label_dict = {'itcs':1, "normal":0, 'micro':1}
        fuse_data_dir =fuse_data_dir_ls[0]
    elif args.datatype == 'camelyon17_fuse':
        target_classes = ['micro', 'normal']
        label_dict = {'micro':1, "normal":0}
        fuse_data_dir =fuse_data_dir_ls[1]
    elif args.datatype == 'camelyon16_fuse':
        target_classes = ['micro', 'normal']
        label_dict = {'micro':1, "normal":0}
        fuse_data_dir =fuse_data_dir_ls[2]
    elif args.datatype == 'camelyon_fuse':
        target_classes = ['micro', 'normal']
        label_dict = {'micro':1, "normal":0}
        fuse_data_dir =fuse_data_dir_ls[3]

    test_d = {}
    for i in args.idx_fold:
        df_meta_test_S = pd.read_csv(os.path.join(args.project_path, f'{fuse_data_dir}/multi_tasks/{target_classes[0]}/t_{i}_s.csv'))
        df_meta_test_Q = pd.read_csv(os.path.join(args.project_path, f'{fuse_data_dir}/multi_tasks/{target_classes[0]}/t_{i}_q.csv'))
        df_final_test = pd.read_csv(os.path.join(args.project_path, f'{fuse_data_dir}/multi_tasks/{target_classes[0]}/final_test.csv'))

        test_S = add_classID(df_meta_test_S, label_dict)
        test_Q = add_classID(df_meta_test_Q, label_dict)
        final_test = add_classID(df_final_test, label_dict)

        # 测试任务需要是一个列表，因为不止一个任务
        test_d[i] = [test_S, test_Q, final_test]


    df_meta_train = pd.read_csv(os.path.join(args.project_path, 'data/lymph/source_data/tissue_and_source.csv'))
    source_label_dict = {'ryzh': 0, 'liver': 1, 'gzfb': 2, 'xwzhblyb': 3, 'rg': 4, 'parathyroid': 5, 'wnm-wt': 6, 'hwj': 7, 'xy': 8, 'zqgsp': 9, 'sgs': 10, 'colon': 11, 'gyx': 12, 'pgbysp': 13, 'wbm-ym': 14, 'xwzh': 15, 'nong': 16, 'bp': 17, 'zf': 18, 'xg': 19, 'hs': 20, 'smoothmuscle': 21, 'sxgsz': 22, 'adrenalcortex': 23, 'brain': 24, 'intestinalvillus': 25, 'lung': 26, 'mammarygland': 27, 'pancreas': 28, 'prostate': 29, 'smallintestinalgland': 30, 'spleen': 31, 'thyroid': 32, 'tonsil': 33}

    df_meta_train = add_classID(df_meta_train, source_label_dict)

    return df_meta_train, test_d




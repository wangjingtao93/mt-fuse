import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import numpy as np
import torch
import time
from collections import OrderedDict
from tqdm import tqdm
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from maml.maml_alg import MAML
import common.utils as utils
import dl.dl_func as dlf

def meta_test_enter(args, meta_train_data, meta_test_data, final_test_data):

    # store_dir_meta_test =str(os.path.join(project_path,args.save_path, args.alg.lower(), args.net, time_name, 'meta_test'))
    store_dir_meta_test = args.store_dir
    # utils.mkdir(store_dir_meta_test)
    store_meta_pth = str(os.path.join(args.store_dir, 'save_meta_pth'))
    utils.mkdir(store_meta_pth)


    # 创建一个记录meta_test的query set的acc
    metric_dir_meta_test_acc = os.path.join(store_dir_meta_test, 'metric_meta_test_acc' + '.csv')
    with open(str(metric_dir_meta_test_acc), 'w') as f:
        fields = ['Meta_Epoch', 'meta_loss'] + list(range(0, args.n_inner_meta_test, 1)) + ['best_inner_acc','auc', 'precision','recall','f1','sensi', 'spec','best_inner', 'best_meta_epoch', 'best_acc']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)


    # 创建一个记录meta_test的query set的sens
    metric_dir_meta_test_sens = os.path.join(store_dir_meta_test, 'metric_meta_test_sens' + '.csv')
    with open(str(metric_dir_meta_test_sens), 'w') as f:
        fields = ['Meta_Epoch', 'meta_loss'] + list(range(0, args.n_inner_meta_test, 1)) + ['acc', 'auc', 'precision','recall','f1','best_inner_sens', 'spec', 'best_inner', 'best_meta_epoch', 'best_sens']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 创建一个记录meta_test的query set的cm和report
    metric_dir_meta_test_cm = os.path.join(store_dir_meta_test, 'metric_meta_test_cm' + '.txt')

    # 创建一个记录final_test的test set的acc
    metric_dir_final_test = os.path.join(store_dir_meta_test, 'metric_final_test' + '.csv')
    with open(str(metric_dir_final_test), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(0, args.n_epoch, 1)) + ['best_val_acc', 'best_val_epoch', 'best_q_acc', 'best_meta_epoch', 'best_q_acc_1', 'best_meta_epoch_1', 'final_test_acc']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    model = MAML(args)

    # for meta test
    best_meta_test_acc = 0.0
    best_meta_epoch_for_meta_test_acc = 0
    best_meta_test_sens = 0.0
    best_meta_epoch_for_meta_test_sens = 0

    for meta_epoch in range(1):
        res = run_epoch(meta_epoch, args, model, meta_train_data,  meta_test_data, final_test_data)

        is_best_meta_test_acc = res['meta_test_best_acc'] > best_meta_test_acc
        best_meta_test_acc = max(res['meta_test_best_acc'], best_meta_test_acc)
        if is_best_meta_test_acc:
            best_meta_epoch_for_meta_test_acc = meta_epoch

        is_best_meta_test_sens = res['meta_test_best_sens'] > best_meta_test_sens
        best_meta_test_sens = max(res['meta_test_best_sens'], best_meta_test_sens)
        if is_best_meta_test_sens:
            best_meta_epoch_for_meta_test_sens = meta_epoch

        with open(str(metric_dir_meta_test_acc), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            values = [res['meta_test_best_acc'], res['meta_test_auc'],res['meta_test_report_acc'].loc['weighted avg', 'precision'], res['meta_test_report_acc'].loc['weighted avg', 'recall'], res['meta_test_report_acc'].loc['weighted avg', 'f1-score'], res['meta_test_inner_sens'][res['meta_test_best_inner_acc']], res['meta_test_report_acc'].loc['weighted avg', 'specificity']]
            data_row = [res['meta_epoch'], res['meta_test_loss']] + res['meta_test_inner_acc'] + values + [res['meta_test_best_inner_acc'],best_meta_epoch_for_meta_test_acc, best_meta_test_acc]
            csv_write.writerow(data_row)

        with open(str(metric_dir_meta_test_sens), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            values = [res['meta_test_inner_acc'][res['meta_test_best_inner_sens']], res['meta_test_auc_sens'], res['meta_test_report_sens'].loc['weighted avg', 'precision'], res['meta_test_report_sens'].loc['weighted avg', 'recall'], res['meta_test_report_sens'].loc['weighted avg', 'f1-score'], res['meta_test_best_sens'], res['meta_test_report_sens'].loc['weighted avg', 'specificity']]
            data_row = [res['meta_epoch'], res['meta_test_loss']] + res['meta_test_inner_sens'] + values + [res['meta_test_best_inner_sens'],best_meta_epoch_for_meta_test_sens, best_meta_test_sens]
            csv_write.writerow(data_row)

        with open(metric_dir_meta_test_cm, 'a+') as file:
            file.write(f"Meta_epoch: {meta_epoch}, Best acc inner: {res['meta_test_best_inner_acc']}, Best sens inner: {res['meta_test_best_inner_sens']}\n")
            file.write("Acc Confusion Matrix:\n")
            file.write(np.array2string(res['meta_test_cm_acc'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("Acc Classification Report:\n")
            file.write(res['meta_test_report_acc'].to_string())
            meta_test_auc = res['meta_test_auc']
            file.write(f'\nauc: {meta_test_auc}')

            file.write("\n\nSens Confusion Matrix:\n")
            file.write(np.array2string(res['meta_test_cm_sens'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("Sens Classification Report:\n")
            file.write(res['meta_test_report_sens'].to_string())
            meta_test_auc_sens = res['meta_test_auc_sens']
            file.write(f'\nauc: {meta_test_auc_sens}')

            file.write("\n+++++++++++++++++++++++++++\n")

def run_epoch(meta_epoch, args, model, meta_train_loader, meta_test_loader, test_ls):
    res = OrderedDict()
    print('meta_epoch {}'.format(meta_epoch))

    res_meta_test = model.val(meta_test_loader, meta_epoch)

    res['meta_epoch'] = meta_epoch
    res['meta_test_inner_acc'] = res_meta_test[0]
    res['meta_test_inner_sens'] = res_meta_test[1]
    res['meta_test_loss'] = res_meta_test[2]
    res['meta_test_cm_acc'] = res_meta_test[3]
    res['meta_test_report_acc'] = res_meta_test[4]
    res['meta_test_cm_sens'] = res_meta_test[5]
    res['meta_test_report_sens'] = res_meta_test[6]
    res['meta_test_best_acc'] = res_meta_test[7]
    res['meta_test_best_inner_acc'] =  res_meta_test[8]
    res['meta_test_best_sens'] = res_meta_test[9]
    res['meta_test_best_inner_sens'] =  res_meta_test[10]
    res['meta_test_auc'] = res_meta_test[11]
    res['meta_test_auc_sens'] = res_meta_test[12]

    return res
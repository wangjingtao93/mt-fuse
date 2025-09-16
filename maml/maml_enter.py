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

def maml_enter(args, meta_train_data, meta_test_data, final_test_data):

    # store_dir_meta_test =str(os.path.join(project_path,args.save_path, args.alg.lower(), args.net, time_name, 'meta_test'))
    store_dir_meta_test = args.store_dir
    # utils.mkdir(store_dir_meta_test)
    store_meta_pth = str(os.path.join(args.store_dir, 'save_meta_pth'))
    utils.mkdir(store_meta_pth)

    # 创建一个记录meta_train, 用于观察
    metric_dir_meta_train = os.path.join(args.store_dir, 'metric_meta_train' + '.csv')
    with open(str(metric_dir_meta_train), 'w') as f:
        fields = ['Meta_Epoch', 'Loss', 'ACC', 'Lr']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

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

    # final test acc
    best_pred_for_final_acc_1 = 0.0 #最终测试中，测试任务 querry set上最好的值，使用所有任务，相同epoch的均值
    best_meta_epoch_for_final_acc_1 = 0
    # 另外一种计算方式， 使用所有任务，每个任务最好的val_acc
    best_pred_for_final_acc_2 = 0.0
    best_meta_epoch_for_final_acc_2 = 0

    tensor_writer = SummaryWriter('{}/tensorboard_log/meta_train'.format(args.store_dir))

    for meta_epoch in range(args.n_meta_epoch):
        res = run_epoch(meta_epoch, args, model, meta_train_data,  meta_test_data, final_test_data)
        final_val_values = res['final_test_val_values']
        final_ave_tasks = res['final_ave_tasks']

        # final_test 使用所有任务，相同epoch的均值 acc
        best_final_val_value =  max(final_val_values)
        is_best_for_final =best_final_val_value > best_pred_for_final_acc_1
        best_pred_for_final_acc_1 = max(best_pred_for_final_acc_1, best_final_val_value)
        if is_best_for_final :
            best_meta_epoch_for_final_acc_1 = meta_epoch

        # final_test 使用所有任务，最好的val_acc
        is_best_2 = final_ave_tasks[-2] > best_pred_for_final_acc_2
        best_pred_for_final_acc_2 = max(final_ave_tasks[-2], best_pred_for_final_acc_2)
        if is_best_2:
            best_meta_epoch_for_final_acc_2 = meta_epoch

        with open(str(metric_dir_final_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')

            best_val_acc = max(final_val_values)
            best_index = final_val_values.index(best_val_acc)

            data_row = [meta_epoch] + final_val_values + [best_val_acc, best_index+1,best_pred_for_final_acc_1,best_meta_epoch_for_final_acc_1, best_pred_for_final_acc_2,best_meta_epoch_for_final_acc_2, final_ave_tasks[1]]
            csv_write.writerow(data_row)

        current_outer_lr =  model.outer_optimizer.state_dict()['param_groups'][0]['lr']
        with open(str(metric_dir_meta_train), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [res['meta_epoch'], res['meta_train_loss'], res['meta_train_acc'], current_outer_lr]
            csv_write.writerow(data_row)

        if args.is_meta_test:
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

        tensor_writer.add_scalar('meta_train_loss', res['meta_train_loss'], meta_epoch)
        tensor_writer.add_scalar('meta_train_acc',  res['meta_train_acc'], meta_epoch)
        tensor_writer.add_scalar('meta_learn_rate', current_outer_lr, meta_epoch)


        # torch.cuda.empty_cache()
        # 默认不执行
        if args.lr_sched:
            model.lr_sched()

def run_epoch(meta_epoch, args, model, meta_train_loader, meta_test_loader, test_ls):
    res = OrderedDict()
    print('meta_epoch {}'.format(meta_epoch))

    ave_acc_meta_train_tasks, acc_meta_train_last_task, meta_train_loss= model.train(meta_train_loader, meta_epoch)

    res['meta_epoch'] = meta_epoch
    res['meta_train_acc'] = ave_acc_meta_train_tasks
    res['meta_train_loss'] = meta_train_loss

    meta_test_loader = []
    if len(meta_test_loader) != 0:
        res_meta_test = model.val(meta_test_loader, meta_epoch)
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

    # 保存每一个meta_epoch的元参数
    meta_epoch_model = os.path.join(args.store_dir, 'save_meta_pth',f'meta_epoch_{meta_epoch}.pth')
    torch.save(model.network.state_dict(), meta_epoch_model)

    # test_ls[0] = []
    if len(test_ls[0]) == 0:
        final_test_val_values = [0,0,0,0,0]
        final_ave_tasks=[0,0,0,0,0]
        final_test_val_values_sens = [0,0,0,0,0]
        final_ave_tasks_sens=[0,0,0,0,0]
    else:
        final_test_val_values, final_ave_tasks,  final_test_val_values_sens, final_ave_tasks_sens= dlf.trainer(args, test_ls[0], test_ls[1], test_ls[2], meta_epoch)

    res['final_test_val_values'] = final_test_val_values
    res['final_ave_tasks'] = final_ave_tasks
    res['final_test_val_values_sens'] = final_test_val_values_sens
    res['final_ave_tasks_sens'] =final_ave_tasks_sens

    return res
import os
import csv
from copy import deepcopy
from torch.utils.data import  DataLoader
from copy import deepcopy

from common.dl_comm import dl_comm
from common.dataloader import *
from common.lymph.dataloader_ly import LY_dataset
import common.utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
from dl.load_for_meta import load_for_meta




from torch.utils.tensorboard import SummaryWriter

# 测试任务包含S Q 和final test需要将S和Q重新组batch,S作为train data, Q 作为val data,
def trainer(args, sppport_all_task, query_all_task, final_test_task, meta_epoch):

    # 创建一个csv文件记录测试任务的指标
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)

    metric_dir = os.path.join(dir_meta_epoch,'metric_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx', 'epoch', 'loss','acc', 'auc', 'precision','recall','f1','ka', 'sensi', 'spec', 'best_acc', 'best_epoch','best_recall', 'best_epoch_recall']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 创建一个txt文件记录混淆矩阵
    utils.mkdir( os.path.join(dir_meta_epoch,'cm'))
    test_cm_res_dir = os.path.join(dir_meta_epoch,'cm', 'cm_test_' + str(meta_epoch) + '.txt')
    val_cm_res_dir = os.path.join(dir_meta_epoch,'cm', 'cm_val_' + str(meta_epoch) + '.txt')

    task_num = len(sppport_all_task) # 多构建几个任务，结果会更准确
    # 记录所有测试任务每个epoch的val 的acc
    val_acc_all_epoch = [0] * args.n_epoch
    # val_acc_all_task = [0] * task_num
    # test_acc_all_epoch= []
    # 记录所有测试任务每个epoch的val 的sens(仅目标类的)
    val_sens_all_epoch = [0] * args.n_epoch
    # val_sens_all_task = [0] * task_num

    val_values_all_task = {}
    test_values_all_task = {}

    val_values_all_task_sens = {}
    test_values_all_task_sens = {}

    for task_idx in sppport_all_task.keys():
        # 创建 网络 对象
        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()
        if args.alg == 'dl' and args.load != '':
            dl_ob.model.load_state_dict(torch.load(args.load))
        elif args.alg == 'reptile' or args.alg =='maml' or args.alg == 'imaml':
            dl_ob.model.load_state_dict(load_for_meta(args,meta_epoch))


        # writer = SummaryWriter('{}/tensorboard_log/{}_meta_epoch/{}_task_id'.format(args.store_dir,meta_epoch, task_idx))

        train_loader = sppport_all_task[task_idx]
        val_loader = query_all_task[task_idx]
        test_loader = final_test_task[task_idx]

        # for val acc
        best_val = 0.0
        best_val_epoch = 0
        best_final_val_state_dict = deepcopy(dl_ob.model.state_dict()) # 不用deepcopy是万万不行的，否则会随dl_ob.model发生变化
        best_cm_val = []

        # for val sens
        best_val_sens = 0.0
        best_val_epoch_sens = 0
        best_final_val_state_dict_sens = deepcopy(dl_ob.model.state_dict()) # 不用deepcopy是万万不行的，否则会随dl_ob.model发生变化
        best_cm_val_sens = []

        # 注意训练过程中，对test_loder进行测试，会影响最后模型的精度，因为random 的原因
        # for test
        # best_test = 0.0
        # best_epoch_for_test = 1
        for epoch in range(args.n_epoch):
            dl_ob.adjust_learning_rate(epoch)
            print('task_idx--------: ', task_idx)
            train_loss = dl_ob.train(train_loader, epoch)
            res_val = dl_ob.val(val_loader)

            # val_acc_all_epoch.append(val_value[1])
            val_acc_all_epoch[epoch] = val_acc_all_epoch[epoch] + res_val['acc']
            val_sens_all_epoch[epoch] = val_sens_all_epoch[epoch] + res_val['sens']

            is_val_set_best = res_val['acc'] > best_val
            best_val = max(best_val, res_val['acc'])
            if is_val_set_best:
                best_val_epoch = epoch
                val_values_all_task[task_idx] = [res_val['loss'], res_val['acc'], res_val['auc'], res_val['prec'], res_val['recall'], res_val['f1'],res_val['sens'], res_val['spec']] + [best_val, best_val_epoch]
                best_final_val_state_dict = deepcopy(dl_ob.model.state_dict())
                best_cm_val = [res_val['cm'], res_val['report']]
                if args.is_save_val_net:
                    taskpth_store_dir = os.path.join(dir_meta_epoch, f'taskid_{task_idx}')
                    utils.mkdir(taskpth_store_dir)
                    torch.save(best_final_val_state_dict,os.path.join(taskpth_store_dir,f'best_model_for_valset_{meta_epoch}.pth' ))

            # 根据特定类别的sens，罕见病类别[0,1,2]
            spec_cls_sens= []
            if len(spec_cls_sens) != 0:
                is_val_set_best_sens = res_val['sens'] > best_val_sens
                best_val_sens = max(best_val_sens, res_val['sens'])
                if is_val_set_best_sens:
                    best_val_epoch_sens = epoch
                    val_values_all_task_sens[task_idx] = [res_val['loss'], res_val['acc'], res_val['auc'], res_val['prec'], res_val['recall'], res_val['f1'],res_val['sens'], res_val['spec']] + [best_val_sens, best_val_epoch_sens]
                    best_final_val_state_dict_sens = deepcopy(dl_ob.model.state_dict())
                    best_cm_val_sens = [res_val['cm'], res_val['report']]
                    if args.is_save_val_net_sens:
                        taskpth_store_dir = os.path.join(dir_meta_epoch, f'taskid_{task_idx}')
                        utils.mkdir(taskpth_store_dir)
                        torch.save(best_final_val_state_dict,os.path.join(taskpth_store_dir,f'best_model_for_valset_sens_{meta_epoch}.pth' ))
            else:
                val_values_all_task_sens[task_idx] = [0]

            with open(str(metric_dir), 'a+') as f:
                csv_write = csv.writer(f, delimiter=',')
                data_row = [task_idx, epoch] + [res_val['loss'], res_val['acc'], res_val['auc'], res_val['prec'], res_val['recall'], res_val['f1'],res_val['sens'], res_val['spec']]
                data_row.append(best_val)
                data_row.append(best_val_epoch)
                data_row.append(best_val_sens)
                data_row.append(best_val_epoch_sens)
                csv_write.writerow(data_row)

        # val_acc_all_task[task_idx] = best_val

        dl_ob.model.load_state_dict(best_final_val_state_dict)
        res_test = dl_ob.val(test_loader)
        test_values_all_task[task_idx] = [res_test['loss'], res_test['acc'], res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'], res_test['ka'], res_test['sens'], res_test['spec']] + [best_val, best_val_epoch]

        # 根据特定类别的sens，罕见病类别[0,1,2]
        spec_cls_sens= []
        if len(spec_cls_sens) != 0:
            dl_ob.model.load_state_dict(best_final_val_state_dict_sens)
            res_test_sens = dl_ob.val(test_loader)
            test_values_all_task_sens[task_idx] = [res_test_sens['loss'], res_test_sens['acc'], res_test_sens['auc'], res_test_sens['prec'], res_test_sens['recall'], res_test_sens['f1'],res_test_sens['sens'], res_test_sens['spec']] + [best_val_sens, best_val_epoch_sens]

        else:
            test_values_all_task_sens[task_idx] = [0]

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = ['finaltest', ' '] + [res_test['loss'], res_test['acc'],  res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'],res_test['sens'], res_test['spec']]
            data_row.append(best_val)
            data_row.append(best_val_epoch)
            csv_write.writerow(data_row)

            spec_cls_sens= []
            if len(spec_cls_sens) != 0:
                data_row = ['finaltest_sens', ' '] + [res_test_sens['loss'], res_test_sens['acc'],  res_test_sens['auc'], res_test_sens['prec'], res_test_sens['recall'], res_test_sens['f1'],res_test_sens['sens'], res_test_sens['spec']]
                data_row.append(best_val_sens)
                data_row.append(best_val_epoch_sens)
                csv_write.writerow(data_row)

        with open(test_cm_res_dir, 'a+') as file:
            file.write(f"Task_ID: {task_idx}, best_acc_epoch: {best_val_epoch}, best_sens_epoch:{best_val_epoch_sens}\n")
            file.write("Acc Confusion Matrix:\n")
            file.write(np.array2string(res_test['cm'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("ACC Classification Report:\n")
            file.write(res_test['report'].to_string())

            # 根据特定类别的sens，罕见病类别[0,1,2]
            spec_cls_sens= []
            if len(spec_cls_sens) != 0:
                file.write("\n\nSens Confusion Matrix:\n")
                file.write(np.array2string(res_test_sens['cm'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
                file.write("Sens Classification Report:\n")
                file.write(res_test_sens['report'].to_string())

            file.write("\n+++++++++++++++++++++++++++\n")

        with open(val_cm_res_dir, 'a+') as file:
            file.write(f"Task_ID: {task_idx}, best_acc_epoch: {best_val_epoch}, best_sens_epoch:{best_val_epoch_sens}\n")
            file.write("Acc Confusion Matrix:\n")
            file.write(np.array2string(best_cm_val[0], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("Acc Classification Report:\n")
            file.write(best_cm_val[1].to_string())

            # 根据特定类别的sens，罕见病类别[0,1,2]
            spec_cls_sens= []
            if len(spec_cls_sens) != 0:
                file.write("\n\nSens Confusion Matrix:\n")
                file.write(np.array2string(best_cm_val[0], separator=', ') + "\n\n")  # 将矩阵转换为字符串
                file.write("Sens Classification Report:\n")
                file.write(best_cm_val_sens[1].to_string())

            file.write("\n+++++++++++++++++++++++++++\n")

        # 保存y_true和y_score, fpr和tpr
        y_score_dir = os.path.join(dir_meta_epoch,'cm')
        save_model_output(res_test['y_true'], res_test['y_score'], y_score_dir,task_idx, meta_epoch)
        save_fpr_tpr(res_test['fpr'], res_test['tpr'],y_score_dir, task_idx, meta_epoch)


    all_tasks_val_ave = np.around(np.mean(list(val_values_all_task.values()), axis=0), 4).tolist()
    all_tasks_val_ave_sens = np.around(np.mean(list(val_values_all_task_sens.values()), axis=0), 4).tolist()
    all_tasks_ave = np.around(np.mean(list(test_values_all_task.values()), axis=0), 4).tolist()
    all_tasks_ave_sens = np.around(np.mean(list(test_values_all_task_sens.values()), axis=0), 4).tolist()
    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        csv_write.writerow(['ACC_Summary'])
        csv_write.writerow(['val_set'])
        for task_idx, task_valuse_ls in val_values_all_task.items():
            data_row = [task_idx, ' '] + task_valuse_ls
            csv_write.writerow(data_row)
        csv_write.writerow(['Final_ave_val', ' '] + all_tasks_val_ave)

        csv_write.writerow(['test_set'])
        for task_idx, task_valuse_ls in  test_values_all_task.items():
            data_row = [task_idx, ' '] + task_valuse_ls
            csv_write.writerow(data_row)
        csv_write.writerow(['Final_ave_test', ' '] + all_tasks_ave)

        spec_cls_sens = []
        if len(spec_cls_sens) != 0:
            csv_write.writerow(['Sens_Summary'])
            csv_write.writerow(['val_set'])
            for task_idx, task_valuse_ls in  val_values_all_task_sens.items():
                data_row = [task_idx, ' '] + task_valuse_ls
                csv_write.writerow(data_row)
            csv_write.writerow(['Final_ave_val', ' '] + all_tasks_val_ave_sens)

            csv_write.writerow(['test_set'])
            for task_idx, task_valuse_ls in  test_values_all_task_sens.items():
                data_row = [task_idx, ' '] + task_valuse_ls
                csv_write.writerow(data_row)
            csv_write.writerow(['Final_ave_test', ' '] + all_tasks_ave_sens)


    return [round(item / task_num, 4) for item in val_acc_all_epoch], all_tasks_ave,  [round(item / task_num, 4) for item in val_sens_all_epoch], all_tasks_ave_sens




def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def save_model_output(y_true, y_pred_score, store_dir,task_idx, meta_epoch):
    """
    保存AUC计算所需数据到CSV文件

    参数:
    y_true - 真实标签数组 (n_samples,)
    y_pred_score - 预测分数 (n_samples, n_classes)
    filename - 保存的文件名（带.csv扩展名）
    """
    filename = os.path.join(store_dir, f'y_score_{meta_epoch}_{task_idx}.csv')
    # 创建DataFrame
    data_dict = {'true_label': y_true}
    n_classes = y_pred_score.shape[1]

    # 添加每个类别的预测概率列
    for i in range(n_classes):
        data_dict[f'prob_class_{i}'] = y_pred_score[:, i]

    # 创建DataFrame并保存
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False)
    # print(f"数据已保存至 {os.path.abspath(filename)}")
    # print(f"文件包含 {len(df)} 个样本和 {n_classes} 个类别的概率")

def load_model_output(filename):
    """
    从CSV文件加载AUC计算数据

    参数:
    filename - CSV文件名

    返回:
    y_true, y_pred_score
    """
    df = pd.read_csv(filename)

    # 提取真实标签
    y_true = df['true_label'].values

    # 提取所有概率列
    prob_columns = [col for col in df.columns if col.startswith('prob_class_')]
    y_pred_score = df[prob_columns].values

    print(f"从 {filename} 成功加载数据")
    print(f"样本数: {len(y_true)}, 类别数: {y_pred_score.shape[1]}")

    return y_true, y_pred_score



def save_fpr_tpr(fpr, tpr, store_dir, task_idx, meta_epoch):
    """
    计算微平均ROC曲线并保存FPR和TPR到CSV文件
    """


    filename = os.path.join(store_dir, f"fpr_tpr_{meta_epoch}_{task_idx}.csv")

    # 创建DataFrame
    roc_df = pd.DataFrame({
        'False_Positive_Rate': np.round(fpr, 6),
        'True_Positive_Rate': np.round(tpr,6)
    })

    # # 添加AUC值到文件头
    # auc_micro = np.trapz(tpr_micro, fpr_micro)  # 计算AUC

    # 保存到CSV
    roc_df.to_csv(filename, index=False)

    # print(f"微平均ROC曲线数据已保存至: {os.path.abspath(filename)}")
    # print(f"包含 {len(roc_df)} 个点，AUC = {auc_micro:.4f}")

    return roc_df
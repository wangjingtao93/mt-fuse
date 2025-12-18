# 用于生成csv文件，记录实验结果
import csv
import os
import numpy as np
import pandas as pd

# 全局设置要需要记录的信息
fields = ['task_idx', 'epoch', 'loss','acc', 'auc', 'precision','recall','f1','ka',
                   'sensi', 'spec', 'best_acc', 'best_epoch']

def ceate_files(path):
    with open(str(path), 'w') as f:

        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)



def write_result(metric_dir, res, task_idx, epoch, best_val, best_val_epoch):
    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        data_row = [task_idx, epoch] + [res['loss'], res['acc'], res['auc'], res['prec'], res['recall'],
                                            res['f1'], res['ka'], res['sens'], res['spec']]
        data_row.append(best_val)
        data_row.append(best_val_epoch)
        csv_write.writerow(data_row)


def write_result_sum(metric_dir, val_values_all_task, test_values_all_task):
    all_tasks_val_ave = np.around(np.mean(list(val_values_all_task.values()), axis=0), 4).tolist()
    all_tasks_ave = np.around(np.mean(list(test_values_all_task.values()), axis=0), 4).tolist()


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

    return all_tasks_ave


def write_cm(cm_dir, task_idx, best_val_epoch, cm, report):
    with open(cm_dir, 'a+') as file:
        file.write(f"Task_ID: {task_idx}, best_acc_epoch: {best_val_epoch}\n")
        file.write("Acc Confusion Matrix:\n")
        file.write(np.array2string(cm, separator=', ') + "\n\n")  # 将矩阵转换为字符串
        file.write("ACC Classification Report:\n")
        file.write(report.to_string())

        file.write("\n+++++++++++++++++++++++++++\n")


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
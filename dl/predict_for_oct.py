import torch
import os
import cv2
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report
from collections import OrderedDict
from sklearn.preprocessing import label_binarize

from collections import Counter

from common.dl_comm import dl_comm
import common.utils as utils
from common.eval import classification_report_with_specificity as cr


def predict(args, loader_ls, meta_epoch):
    # 创建一个csv文件记录测试任务的指标
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)

    metric_dir = os.path.join(dir_meta_epoch,'metric_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx','acc', 'auc', 'precision','recall','f1','ka','sensi', 'spec']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 创建一个txt文件记录混淆矩阵
    test_cm_res_dir = os.path.join(dir_meta_epoch, 'cm_test_' + str(meta_epoch) + '.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    task_num = len(loader_ls)
    all_tasks_value = []

    for task_idx in range(task_num):


        test_loader = loader_ls[task_idx]

        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))

        res_test = dl_ob.val(test_loader)

        all_tasks_value.append([res_test['acc'],  res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'],res_test['ka'], res_test['sens'], res_test['spec']])

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [task_idx] + [res_test['acc'],  res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'],res_test['ka'], res_test['sens'], res_test['spec']]
            csv_write.writerow(data_row)

        with open(test_cm_res_dir, 'a+') as file:
            file.write(f"Task_ID: {task_idx}, best_acc_epoch: {0}, best_sens_epoch:{0}\n")
            file.write("Acc Confusion Matrix:\n")
            file.write(np.array2string(res_test['cm'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("ACC Classification Report:\n")
            file.write(res_test['report'].to_string())
            file.write("\n+++++++++++++++++++++++++++\n")

        drow_roc(args, res_test, dir_meta_epoch, task_idx)

    arr = np.array(all_tasks_value)
    # 计算列平均值
    column_means = np.round(np.mean(arr, axis=0), 5)

    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        data_row = ['ave'] + list(column_means)
        csv_write.writerow(data_row)



def drow_roc(args, res_test, store_dir, task_idx):
    roc_dir = os.path.join(store_dir, f'taskid_{task_idx}')
    utils.mkdir(roc_dir)
    ave_path = os.path.join(roc_dir, 'ave_roc.png')
    cls_path = os.path.join(roc_dir, 'cls_roc.png')

    n_classes = args.num_classes
    # class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    class_names = ['VRL', 'PIC', 'RP', 'Normal', 'Common']
    y_true = res_test['y_true']
    y_pred_score = res_test['y_score']

    # 将真实标签二值化为one-hot编码
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # ========================= 第一张图：每个类别的ROC曲线 =========================
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 存储每个类别的fpr, tpr和auc
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_pred_score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2,
                label=f'Class {class_names[i]} (AUC = {roc_auc_dict[i]:.5f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    plt.savefig(cls_path, dpi=500)

    # ========================= 第二张图：宏观平均ROC曲线 =========================
    # # 计算宏观平均ROC曲线（对所有类别平均）
    # all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))

    # # 对每个类别在统一FPR点上插值TPR
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # # 平均TPR
    # mean_tpr /= n_classes

    # # 计算宏观AUC
    # macro_auc = auc(all_fpr, mean_tpr)
    # micro_auc = roc_auc_score(y_true_bin, y_pred_score, average='micro')
    ave_auc = res_test['auc']

    plt.figure(figsize=(10, 8))
    # plt.plot(all_fpr, mean_tpr, color='darkorange', lw=3,
    #         label=f'Macro-average ROC (AUC = {macro_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # 添加微平均曲线（可选）
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_score.ravel())
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=3,
            label=f'AUC = {ave_auc:.4f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(ave_path, dpi=500)

    save_auc_data(res_test['y_true'], res_test['y_score'], roc_dir)
    save_micro_roc_curve(fpr_micro, tpr_micro, roc_dir)


def save_auc_data(y_true, y_pred_score, store_dir):
    """
    保存AUC计算所需数据到CSV文件
    
    参数:
    y_true - 真实标签数组 (n_samples,)
    y_pred_score - 预测分数 (n_samples, n_classes)
    filename - 保存的文件名（带.csv扩展名）
    """
    filename = os.path.join(store_dir, 'y_score.csv')
    # 创建DataFrame
    data_dict = {'true_label': y_true}
    n_classes = y_pred_score.shape[1]

    # 添加每个类别的预测概率列
    for i in range(n_classes):
        data_dict[f'prob_class_{i}'] = y_pred_score[:, i]

    # 创建DataFrame并保存
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False)
    print(f"数据已保存至 {os.path.abspath(filename)}")
    print(f"文件包含 {len(df)} 个样本和 {n_classes} 个类别的概率")


def load_auc_data(filename):
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


# 事例用法
def test():
    n_samples = 100
    n_classes = 5
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred_score = np.random.rand(n_samples, n_classes)
    y_pred_score = y_pred_score / y_pred_score.sum(axis=1, keepdims=True)  # 归一化

    # 保存数据
    save_auc_data(y_true, y_pred_score, "auc_data.csv")

    # 加载数据
    loaded_y_true, loaded_y_pred_score = load_auc_data("auc_data.csv")

    # 验证数据一致性
    print("\n数据验证:")
    print("真实标签一致:", np.array_equal(y_true, loaded_y_true))
    print("预测概率一致:", np.allclose(y_pred_score, loaded_y_pred_score))



# def save_micro_roc_curve(y_true, y_pred_score, filename="micro_roc_curve.csv"):
def save_micro_roc_curve(fpr_micro, tpr_micro, store_dir):
    """
    计算微平均ROC曲线并保存FPR和TPR到CSV文件

    参数:
    y_true - 真实标签数组 (n_samples,)
    y_pred_score - 预测分数 (n_samples, n_classes)
    filename - 保存的文件名 (默认: micro_roc_curve.csv)
    """
    # # 将标签二值化为one-hot格式
    # n_classes = y_pred_score.shape[1]
    # y_true_bin = np.zeros((len(y_true), n_classes))
    # for i, label in enumerate(y_true):
    #     y_true_bin[i, label] = 1

    # # 计算微平均ROC曲线
    # fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_score.ravel())

    filename = os.path.join(store_dir, "micro_roc_curve.csv")

    # 创建DataFrame
    roc_df = pd.DataFrame({
        'False_Positive_Rate': np.round(fpr_micro, 6),
        'True_Positive_Rate': np.round(tpr_micro,6)
    })

    # # 添加AUC值到文件头
    # auc_micro = np.trapz(tpr_micro, fpr_micro)  # 计算AUC

    # 保存到CSV
    roc_df.to_csv(filename, index=False)

    # print(f"微平均ROC曲线数据已保存至: {os.path.abspath(filename)}")
    # print(f"包含 {len(roc_df)} 个点，AUC = {auc_micro:.4f}")

    return roc_df

def plot_micro_roc_from_csv(filename, title="Micro-average ROC Curve"):
    """
    从CSV文件加载微平均ROC曲线并绘制

    参数:
    filename - CSV文件名
    title - 图表标题 (可选)
    """
    # 加载数据
    roc_df = pd.read_csv(filename)

    # 提取数据
    fpr = roc_df['False_Positive_Rate']
    tpr = roc_df['True_Positive_Rate']

    # 计算AUC
    auc_score = np.trapz(tpr, fpr)

    # 绘制曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Micro-average ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return auc_score


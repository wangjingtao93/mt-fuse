import numpy as np
import torch.nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd

from sklearn.preprocessing import label_binarize

def accuracy(output, target, topk=(1,)): #(1,)定义元组中有且仅有一个元素
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) #输出两个tensor分别为值，和索引
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = [] #记录top1,top2,,,topk准确率
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def acc_1(output, target,truelabel=None):
    batch_size=target.size(0)
    _, pred = torch.max(output.data, 1)
    correct = torch.sum(pred == target)
    return correct/batch_size

def allevluate(logits,labels):
    _, pred = torch.max(logits.data, 1)
    acc = accuracy_score(pred, labels)
    fpr,tpr,thresholds=roc_curve(labels,logits[:,1],drop_intermediate=False)
    f1=f1_score(labels,pred)
    try:
        auc=roc_auc_score(labels,logits[:,1])
    except  ValueError:
        auc = None
        pass
    tn, fp, fn, tp = confusion_matrix(labels,pred).ravel()
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)

    return [acc,auc,sensitivity,specificity,f1,fpr,tpr,tn, fp, fn, tp]

def acc_2(logits,labels):
    _, pred = torch.max(logits.data, 1)
    acc = accuracy_score(pred, labels)

    return acc



# 多分类
def cal_metrics():
    # 示例真实标签和预测结果
    true_labels = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
    print("true label",true_labels)
    # 生成随机数据作为概率值，实际应用中需要替换为模型的预测概率值
    model_output = torch.randn(len(true_labels), 3)
    print("model output",model_output)


    # # 做成多个batch
    # true_labels_ls.append(true_labels)
    # true_labels_ls.append(true_labels)
    # true_labels = np.concatenate(true_labels_ls)
    # print("true label",true_labels)


    # model_output_ls =  []
    # model_output_ls.append(model_output)
    # model_output_ls.append(model_output)
    # model_output = torch.concatenate(model_output_ls)
    # print("model output",model_output)


    # 获得最大类别的index
    _, predicted_labels = torch.max(model_output, 1)
    print("predicted label",predicted_labels)


    # 准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)
    # 混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)
    # 分类报告
    class_report = classification_report(true_labels, predicted_labels,digits=4)
    print("Classification Report:\n", class_report)
    # 精确率
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print("Precision:", precision)
    # 召回率
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print("Recall:", recall)
    # F1 分数
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("F1 Score:", f1)
    # ROC AUC
    # 计算ROC需要将模型输出概率归一化
    prob_new = torch.nn.functional.softmax(model_output, dim=1)

    print(prob_new)
    roc_auc = roc_auc_score(true_labels, prob_new, average='weighted', multi_class='ovo')
    print("ROC AUC Score:", roc_auc)



    cm = confusion_matrix(true_labels, predicted_labels)

    print("混淆矩阵:")
    print(cm)

    recall_labels =  recall_score(true_labels, predicted_labels, labels=[0,1], average='weighted')
    print('+++', recall_labels)

    with open("classification_results.txt", "w") as file:
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=', ') + "\n\n")  # 将矩阵转换为字符串
        file.write("Classification Report:\n")
        file.write(class_report)

    report_with_spec = classification_report_with_specificity(true_labels, predicted_labels,cm=cm)

    a = report_with_spec.loc['weighted avg', 'specificity']

    print(report_with_spec)

def classification_report_with_specificity(y_true, y_pred, cm=None, target_names=None):
    if cm is None:
        # 获取 confusion matrix
        cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]  # 类别数
    if target_names is None:
        target_names = []
        for i in range(n_classes):
            target_names.append(f'c{i}')
    # 生成基本的 classification_report
    report = classification_report(y_true, y_pred, target_names=target_names,digits=5, output_dict=True)


    # 初始化特异性字典
    specificity = {}

    # 计算每个类的特异性
    for i in range(n_classes):
        # 计算 TN 和 FP
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]

        # 计算特异性
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity[target_names[i]] = spec

    # 将特异性加入报告的每一行
    for key in report.keys():
        if key in target_names:
            report[key]['specificity'] = specificity[key]

    # 计算 macro avg 和 weighted avg 的特异性
    macro_specificity = np.mean(list(specificity.values()))
    weighted_specificity = np.sum([specificity[key] * report[key]['support'] for key in target_names]) / np.sum([report[key]['support'] for key in target_names])

    report['macro avg']['specificity'] = macro_specificity
    report['weighted avg']['specificity'] = weighted_specificity

    # 将输出转换为 DataFrame，使其易于打印和查看
    df_report = pd.DataFrame(report).transpose()

    # 调整列顺序和格式
    df_report['support'] = df_report['support'].astype(int)  # 保留支持数的整数位
    column_order = ['precision', 'recall', 'f1-score', 'specificity', 'support']
    if 'accuracy' in df_report.index:
        accuracy_row = df_report.loc['accuracy'].iloc[0]
        df_report = df_report[column_order]
        df_report.loc['accuracy'] = [' ', ' ', ' ', accuracy_row,' ' ]
    else:
        df_report = df_report[column_order]

    # 保留四位小数
    df_report.iloc[:, :-1] = df_report.iloc[:, :-1].applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    return df_report

if __name__ == '__main__':
    cal_metrics()


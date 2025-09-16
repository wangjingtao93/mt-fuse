import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, precision_recall_fscore_support

# 定义真实标签和预测标签
# 这里假设第一行代表真实标签，第二行代表预测标签
# 根据你提供的混淆矩阵，我们需要将混淆矩阵转换为真实标签和预测标签的列表

# 混淆矩阵
conf_mat = np.array(
[[10,  0,  2,  0,  8],
 [ 0, 19,  0,  1,  0],
 [ 0,  0, 20,  0,  0],
 [ 0,  0,  0, 20,  0],
 [ 0,  0,  0,  0, 24]]

 )

# 获取类别数
num_classes = conf_mat.shape[0]

# 初始化真实标签和预测标签的列表
y_true = []
y_pred = []

# 遍历混淆矩阵，填充真实标签和预测标签
for i in range(num_classes):
    for j in range(num_classes):
        count = conf_mat[i, j]
        y_true.extend([i] * count)  # 真实标签
        y_pred.extend([j] * count)  # 预测标签

# 将列表转换为 NumPy 数组（可选）
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 计算准确率（Accuracy）
accuracy = accuracy_score(y_true, y_pred)

# 计算加权Kappa系数
kappa = cohen_kappa_score(y_true, y_pred)

# 计算加权召回率（Weighted Recall）和加权F1分数
# 使用 'weighted' 平均方式
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted', zero_division=0
)

# 计算每个类别的特异性（Specificity）
# 特异性 = TN / (TN + FP)
# 需要先计算混淆矩阵的补集，即 TN 矩阵
specificity_per_class = []
for i in range(num_classes):
    # 计算真正例（True Negatives）
    tn = np.sum(conf_mat) - np.sum(conf_mat) + conf_mat[i, i]
    # 更准确的计算方式是总和所有不属于当前类别的真实负例和预测负例
    # TN = sum(conf_mat[:, j] for j != i) + sum(conf_mat[i, k] for k != i) - conf_mat[i, i]
    # 但是更简单的方法是使用 sklearn 的 confusion_matrix 和一些数学运算
    # 另一种方法是手动计算每个类别的 TN
    
    # 手动计算 TN: 对于每个类别i，TN是所有非i类样本中被正确预测为非i类的数量
    # 即，对每个类别i，TN = sum(conf_mat[j, k] for j in range(num_classes) if j != i for k in range(num_classes) if k != i)
    # 但这可能过于复杂。另一种方法是使用混淆矩阵的总和减去相关的 TP, FP, FN
    # 更简单的方法是使用 multilabel_confusion_matrix，但对于多分类问题，可能需要自定义计算
    
    # 使用另一种方法计算 TN:
    # TN for class i is the sum of all elements not in row i and not in column i
    # 具体来说，TN_i = total samples - (TP_i + FP_i + FN_i)
    # 但需要分别计算 FP_i 和 FN_i
    
    # 更准确的方法是逐个计算
    # 初始化 TN_i
    tn_i = 0
    for j in range(num_classes):
        if j != i:
            tn_i += conf_mat[j, :].sum() - conf_mat[j, i]  # 所有非i类样本中，预测不为i的数量
            tn_i += conf_mat[i, :].sum() - conf_mat[i, i]  # 所有i类样本中，真实不为i的数量（这部分其实已经是FP和FN的一部分）
    # 这种方法并不准确，因此推荐使用 sklearn 的 multilabel_confusion_matrix 或其他方法
    
    # 更好的方法是使用 sklearn 的 confusion_matrix 和自定义计算
    # 使用 sklearn.metrics.multilabel_confusion_matrix 需要将多分类转换为多个二分类，这里采用另一种方式
    
    # 另一种方法：TN_i = sum(conf_mat[:, j] for j != i) - conf_mat[i, j] （不正确）
    
    # 最可靠的方法是手动计算：
    # TN_i = 总样本数 - (TP_i + FP_i + FN_i)
    # 其中 TP_i = conf_mat[i, i]
    # FP_i = sum(conf_mat[:, i]) - conf_mat[i, i]
    # FN_i = sum(conf_mat[i, :]) - conf_mat[i, i]
    tp_i = conf_mat[i, i]
    fp_i = conf_mat[:, i].sum() - tp_i
    fn_i = conf_mat[i, :].sum() - tp_i
    tn_i = conf_mat.sum() - (tp_i + fp_i + fn_i)
    
    if (tn_i + fp_i) == 0:
        specificity_i = 1.0  # 如果没有FP和TN，特异性定义为1
    else:
        specificity_i = tn_i / (tn_i + fp_i)
    specificity_per_class.append(specificity_i)

# 计算加权特异性（Weighted Specificity）
# 这里简单地使用权重与每个类别的样本数成比例
class_counts = conf_mat.diagonal()  # 每个类别的真实样本数
total_samples = conf_mat.sum()
weights = class_counts / total_samples if total_samples != 0 else np.ones(num_classes)

weighted_specificity = np.sum(np.array(specificity_per_class) * weights)

# 输出结果
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"加权召回率 (Weighted Recall): {recall:.4f}")
print(f"加权F1分数 (Weighted F1 Score): {f1:.4f}")
print(f"加权特异性 (Weighted Specificity): {weighted_specificity:.4f}")
print(f"加权Kappa系数 (Weighted Kappa): {kappa:.4f}")

# # 如果需要每个类别的特异性，可以打印出来
# for i, spec in enumerate(specificity_per_class):
#     print(f"类别 {i} 的特异性 (Specificity): {spec:.4f}")
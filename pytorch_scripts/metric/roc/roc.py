import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from collections import Counter

def plot_weighted_multiclass_roc(y_true, y_pred_prob, class_names):
    """
    绘制多分类ROC曲线（使用加权平均AUC）
    
    参数:
    y_true - 真实标签 (n_samples,)
    y_pred_prob - 预测概率矩阵 (n_samples, n_classes)
    class_names - 类别名称列表
    """
    n_classes = len(class_names)
    
    # 将真实标签二值化为one-hot编码
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # ========================= 第一张图：每个类别的ROC曲线 =========================
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:n_classes]
    
    # 存储每个类别的fpr, tpr和auc
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    # 计算每个类别的样本权重
    class_counts = Counter(y_true)
    total_samples = len(y_true)
    class_weights = {cls: count/total_samples for cls, count in class_counts.items()}
    
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2,
                 label=f'Class {class_names[i]} (AUC = {roc_auc_dict[i]:.2f}, Weight = {class_weights[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # ========================= 第二张图：加权平均ROC曲线 =========================
    # 计算所有FPR点
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    
    # 对每个类别在统一FPR点上插值TPR
    weighted_tpr = np.zeros_like(all_fpr)
    
    # 计算加权平均AUC
    weighted_auc = 0.0
    
    for i in range(n_classes):
        # 插值TPR
        interp_tpr = np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        interp_tpr[0] = 0.0  # 确保从0开始
        
        # 累加权重的TPR
        weighted_tpr += interp_tpr * class_weights[i]
        
        # 累加加权AUC
        weighted_auc += roc_auc_dict[i] * class_weights[i]
    
    # 计算微平均AUC
    micro_auc = roc_auc_score(y_true_bin, y_pred_prob, average='micro')
    
    # 计算宏平均AUC
    macro_auc = roc_auc_score(y_true_bin, y_pred_prob, average='macro')
    
    plt.figure(figsize=(10, 8))
    
    # 绘制加权平均ROC曲线
    plt.plot(all_fpr, weighted_tpr, color='darkorange', lw=3,
             label=f'Weighted-average ROC (AUC = {weighted_auc:.2f})')
    
    # 绘制微平均ROC曲线
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=3,
             label=f'Micro-average ROC (AUC = {micro_auc:.2f})')
    
    # 绘制宏平均ROC曲线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Weighted, Micro, and Macro-average ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 打印AUC值比较
    print(f"Micro-average AUC: {micro_auc:.4f}")
    print(f"Macro-average AUC: {macro_auc:.4f}")
    print(f"Weighted-average AUC: {weighted_auc:.4f}")

# ========================= 示例用法（带类别不平衡） =========================
if __name__ == "__main__":
    # 模拟数据（实际使用时替换为你的数据）
    n_samples = 1000
    n_classes = 5
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    # 创建类别不平衡的数据
    # 类别分布：0:40%, 1:30%, 2:15%, 3:10%, 4:5%
    class_distribution = [0.4, 0.3, 0.15, 0.1, 0.05]
    y_true = np.random.choice(np.arange(n_classes), n_samples, p=class_distribution)
    
    # 生成随机预测概率（模拟模型输出）
    y_pred_prob = np.random.rand(n_samples, n_classes)
    # 使预测概率偏向真实类别（模拟实际模型）
    for i in range(n_samples):
        y_pred_prob[i, y_true[i]] += 0.5  # 增加真实类别的概率
    y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)  # 归一化为概率
    
    # 绘制ROC曲线
    plot_weighted_multiclass_roc(y_true, y_pred_prob, class_names)
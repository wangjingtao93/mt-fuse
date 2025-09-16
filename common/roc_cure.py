import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from scipy import interp
import pandas as pd

def plot_weighted_ovo_roc(y_true, y_pred_prob, class_names=None):
    """
    绘制Weighted OVO AUC对应的ROC曲线
    
    参数:
    y_true - 真实标签数组 (n_samples,)
    y_pred_prob - 预测概率矩阵 (n_samples, n_classes)
    class_names - 类别名称列表 (可选)
    """
    n_classes = y_pred_prob.shape[1]
    classes = np.unique(y_true)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # 计算每个类别对的ROC曲线
    pair_roc_auc = {}
    pair_fpr = {}
    pair_tpr = {}
    weights = {}
    
    # 获取所有类别对组合
    class_pairs = list(combinations(classes, 2))
    
    # 计算每个类别对的ROC和权重
    for (i, j) in class_pairs:
        # 提取属于这对类别的样本
        idx = np.logical_or(y_true == i, y_true == j)
        y_true_pair = y_true[idx]
        y_pred_pair = y_pred_prob[idx]
        
        # 创建二分类标签：i类为1，j类为0
        y_binary = (y_true_pair == i).astype(int)
        
        # 使用i类的概率作为正类概率
        prob_i = y_pred_pair[:, i]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_binary, prob_i)
        roc_auc = auc(fpr, tpr)
        
        # 存储结果
        pair_roc_auc[(i, j)] = roc_auc
        pair_fpr[(i, j)] = fpr
        pair_tpr[(i, j)] = tpr
        
        # 计算权重（i类和j类的样本数之和）
        weight = np.sum(y_true == i) + np.sum(y_true == j)
        weights[(i, j)] = weight
    
    # 计算总权重
    total_weight = sum(weights.values())
    
    # 创建统一的FPR点
    all_fpr = np.linspace(0, 1, 10000)
    
    # 初始化平均TPR
    mean_tpr = np.zeros_like(all_fpr)
    
    # 对每个类别对进行插值并加权平均
    for (i, j) in class_pairs:
        # 在统一的FPR点上插值TPR
        interp_tpr = interp(all_fpr, pair_fpr[(i, j)], pair_tpr[(i, j)])
        interp_tpr[0] = 0.0  # 确保从0开始
        
        # 加权累加
        mean_tpr += interp_tpr * weights[(i, j)]
    
    # 加权平均
    mean_tpr /= total_weight
    
    # 确保曲线以(1,1)结束
    mean_tpr[-1] = 1.0
    
    # 计算加权AUC
    weighted_auc = auc(all_fpr, mean_tpr)
    
    # 绘制曲线
    plt.figure(figsize=(10, 8))
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2,
             label=f'Weighted OVO ROC (AUC = {weighted_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Weighted One-vs-One ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('tmp.png', dpi=500)
    
    
    # 返回曲线数据
    return all_fpr, mean_tpr, weighted_auc

def load_auc_data(filename):
    """
    从CSV文件加载AUC计算数据
    
    参数:
    filename - CSV文件名

    返回:
    y_true, y_pred_prob
    """
    df = pd.read_csv(filename)

    # 提取真实标签
    y_true = df['true_label'].values

    # 提取所有概率列
    prob_columns = [col for col in df.columns if col.startswith('prob_class_')]
    y_pred_prob = df[prob_columns].values

    print(f"从 {filename} 成功加载数据")
    print(f"样本数: {len(y_true)}, 类别数: {y_pred_prob.shape[1]}")

    return y_true, y_pred_prob

def save_weighted_ovo_roc_csv(fpr, tpr, filename="weighted_ovo_roc.csv"):
    """
    保存Weighted OVO ROC曲线数据到CSV文件
    
    参数:
    fpr - FPR值数组
    tpr - TPR值数组
    filename - 保存的文件名
    """
    import pandas as pd
    import os
    
    # 创建DataFrame
    df = pd.DataFrame({
        'False_Positive_Rate': fpr,
        'True_Positive_Rate': tpr
    })
    
    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f"加权OVO ROC曲线数据已保存至: {os.path.abspath(filename)}")
    print(f"包含 {len(df)} 个点")

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
# ===================== 示例用法 =====================
if __name__ == "__main__":
    # # 创建不平衡数据集
    # np.random.seed(42)
    # n_samples = 1000
    # class_distribution = [0.5, 0.3, 0.2]  # 三个类别的分布
    # n_classes = len(class_distribution)
    
    # # 生成真实标签
    # y_true = np.random.choice(range(n_classes), n_samples, p=class_distribution)
    
    # # 生成预测概率（使预测偏向真实类别）
    # y_pred_prob = np.random.rand(n_samples, n_classes)
    # for i in range(n_samples):
    #     y_pred_prob[i, y_true[i]] += 0.5
    # y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
    

    filename='/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20241031/oct/predict/maml/meta_found/pretrain_True/2025-07-09-15-45-52/meta_epoch/taskid_1/y_score.csv'
    y_true, y_pred_logits = load_auc_data(filename)
    y_pred_prob = softmax(y_pred_logits)
    # 计算并绘制加权OVO ROC曲线
    fpr, tpr, weighted_auc = plot_weighted_ovo_roc(y_true, y_pred_prob)
    
    # 保存曲线数据
    save_weighted_ovo_roc_csv(fpr, tpr, "weighted_ovo_roc_data.csv")
    
    # 验证AUC值
    from sklearn.metrics import roc_auc_score
    
    sklearn_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average='weighted')
    print(f"手动计算的加权OVO AUC: {weighted_auc:.6f}")
    print(f"Scikit-learn计算的加权OVO AUC: {sklearn_auc:.6f}")
    print(f"差异: {abs(weighted_auc - sklearn_auc):.6f}")


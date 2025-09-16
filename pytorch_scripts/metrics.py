import numpy as np

def calculate_accuracy(confusion_matrix):
    """计算准确率"""
    correct = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix)
    return correct / total

def calculate_weighted_kappa(confusion_matrix):
    """计算加权Kappa系数（使用线性权重）"""
    n_classes = confusion_matrix.shape[0]
    total = np.sum(confusion_matrix)
    
    # 计算权重矩阵（线性权重）
    weights = 1 - np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]) / (n_classes - 1)
    
    # 计算观察一致性（加权）
    observed = np.sum(confusion_matrix * weights) / total
    
    # 计算期望一致性（加权）
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    expected = np.sum((row_sums[:, None] * col_sums[None, :]) * weights) / (total ** 2)
    
    # 计算加权Kappa
    kappa = (observed - expected) / (1 - expected)
    return kappa

# 示例混淆矩阵（5x5）
confusion_matrix = np.array([
    [10,  0,  0,  0, 10],
    [ 0, 11,  0,  3,  6],
    [ 0,  0, 20,  0,  0],
    [ 0,  0,  1,  9, 10],
    [ 0,  0,  0,  0, 24]
])

# 计算指标
accuracy = calculate_accuracy(confusion_matrix)
weighted_kappa = calculate_weighted_kappa(confusion_matrix)

# 输出结果
print(f"混淆矩阵:\n{confusion_matrix}")
print(f"\n准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"加权Kappa系数: {weighted_kappa:.4f}")

# 结果解读
print("\n结果解读:")
if weighted_kappa > 0.8:
    print("Kappa > 0.8: 模型一致性非常好")
elif weighted_kappa > 0.6:
    print("Kappa > 0.6: 模型一致性良好")
elif weighted_kappa > 0.4:
    print("Kappa > 0.4: 模型一致性一般")
else:
    print("Kappa ≤ 0.4: 模型一致性较差")

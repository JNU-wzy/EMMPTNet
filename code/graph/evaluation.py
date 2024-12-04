import numpy as np
from scipy import stats
import torch

# 皮尔逊相关系数 (PCC)
def pearson_correlation_coefficient(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    return correlation_matrix[0, 1]


# 斯皮尔曼等级相关系数 (SPCC)
def spearman_rank_correlation_coefficient(y_true, y_pred):
    return stats.spearmanr(y_true, y_pred).correlation


# 均方根误差 (RMSE)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    # 确保y_true和y_pred是PyTorch张量
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))



# 平均绝对误差 (MAE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

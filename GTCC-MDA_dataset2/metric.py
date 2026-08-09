try:
    from graphormer_dataset1.utils import *
except ModuleNotFoundError:
    from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(predict_score.flatten()))))  # set只保留唯一值，并从小到大排序
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]  # 抽取999个作为阈值
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # 将predict_score复制hresholds_num（999）次
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)  # 正确预测为正样本的数量（真阳率）
    FP = predict_score_matrix.sum(axis=1) - TP  # 错误预测为正样本的数量  求和表示所有正样本个数
    FN = real_score.sum() - TP  # 错误预测为负样本的数量
    TN = len(real_score.T) - TP - FP - FN  # 正确预测为负样本的数量（真阴率）

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [round(float(accuracy), 4), round(float(precision), 4), round(float(recall), 4), round(float(f1_score), 4)]


#  interaction_matrix原邻接矩阵4536个1  predict_matrix预测邻接矩阵  train_matrix 去掉测试集的训练矩阵4536-907=3629个1
def get_metrics_fast(real_score, predict_score, threshold_steps=1000):
    """Compute threshold metrics without allocating a thresholds-by-N matrix."""
    labels = np.asarray(real_score).reshape(-1).astype(np.int64)
    scores = np.asarray(predict_score).reshape(-1)

    unique_scores = np.unique(scores)
    threshold_indices = (
        len(unique_scores) * np.arange(1, threshold_steps) / threshold_steps
    ).astype(np.int64)
    thresholds = unique_scores[threshold_indices]

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    prefix_positive = np.concatenate(([0], np.cumsum(sorted_labels)))

    # Scores at and above a threshold are predicted positive.
    cut_positions = np.searchsorted(
        sorted_scores, thresholds, side='left')
    false_negative = prefix_positive[cut_positions]
    true_negative = cut_positions - false_negative

    total_positive = labels.sum()
    total_negative = labels.size - total_positive
    true_positive = total_positive - false_negative
    false_positive = total_negative - true_negative

    recall_list = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive, dtype=float),
        where=(true_positive + false_negative) != 0,
    )
    precision_list = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=(true_positive + false_positive) != 0,
    )
    f1_score_list = np.divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        out=np.zeros_like(true_positive, dtype=float),
        where=(2 * true_positive + false_positive + false_negative) != 0,
    )
    accuracy_list = (
        true_positive + true_negative
    ) / labels.size

    max_index = np.argmax(f1_score_list)
    return [
        round(float(accuracy_list[max_index]), 4),
        round(float(precision_list[max_index]), 4),
        round(float(recall_list[max_index]), 4),
        round(float(f1_score_list[max_index]), 4),
    ]


def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_neg_edge_index).to(device)  # 训练集中正样本标签
    return val_scores.cpu().numpy(), val_labels.cpu().numpy(), get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy())

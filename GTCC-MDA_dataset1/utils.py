import numpy as np
import torch
import pandas as pd
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
# from scipy import interp
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import _LRScheduler
import math
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# device = torch.device('cpu')


def rescale_nonzero_edge_weights(adjacency):
    """Keep relative SNF strengths while making the mean retained edge 1."""
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    nonzero_mask = adjacency != 0
    if not nonzero_mask.any():
        return adjacency
    mean_weight = adjacency[nonzero_mask].mean()
    if not torch.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("SNF non-zero edge weights must have a positive mean")
    scaled = adjacency.clone()
    scaled[nonzero_mask] = scaled[nonzero_mask] / mean_weight
    return scaled


def normalize_adjacency(adjacency, add_self_loops=True):
    """Return D^(-1/2) (A + I) D^(-1/2) for stable propagation."""
    adjacency = adjacency.to(dtype=torch.float32)
    if add_self_loops:
        adjacency = adjacency + torch.eye(
            adjacency.size(0), dtype=adjacency.dtype,
            device=adjacency.device)
    degree = adjacency.sum(dim=1)
    inverse_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

def constructNet(association_matrix):  # 构造网络G，分块形式，对角线分块为0, 初始化特征矩阵
    # 获取矩阵的形状
    n, m = association_matrix.shape
    # 构造零矩阵
    meta_matrix = torch.zeros((n, n), dtype=torch.int8)
    drug_matrix = torch.zeros((m, m), dtype=torch.int8)
    # 将矩阵按照规则进行组合
    mat1 = torch.cat((meta_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), drug_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0

def constructHNet(association_matrix, meta_matrix, drug_matrix):
    mat1 = torch.cat((meta_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.T, drug_matrix), dim=1)
    adj = torch.cat((mat1, mat2), dim=0)
    # # 保存结果
    # result = pd.DataFrame(adj)
    # result.to_excel('../output/adj.xlsx', index=False)  # index=False设置不生成序号列
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(seed, n_components, k_folds=5, dataset_dir=None):
    dataset_dir = (
        PROJECT_ROOT / 'Dataset1'
        if dataset_dir is None
        else Path(dataset_dir)
    )
    Adj = pd.read_csv(dataset_dir / 'association_matrix.csv', header=0)
    count_ones = np.count_nonzero(Adj == 1)
    print("元素为1的个数：", count_ones)

    Drug_simi_Net = pd.read_csv(
        dataset_dir / 'Drug_SNF_MutualKNN_Network.csv', header=0)
    # 统计数组中值为 1 的元素个数
    count_ones_drug = np.count_nonzero(Drug_simi_Net)
    print("Drug_adj 中值为 1 的元素个数:", count_ones_drug)
    Drug_adj = rescale_nonzero_edge_weights(Drug_simi_Net.values)

    Meta_simi_Net = pd.read_csv(
        dataset_dir / 'Metabolite_SNF_MutualKNN_Network.csv', header=0)
    # 统计数组中值为 1 的元素个数
    count_ones_meta = np.count_nonzero(Meta_simi_Net)
    print("Meta_adj 中值为 1 的元素个数:", count_ones_meta)
    Meta_adj = rescale_nonzero_edge_weights(Meta_simi_Net.values)

    Drug_MESH2vec = pd.read_csv(dataset_dir / 'Drug_mol2vec.csv', header=0)
    Meta_mol2vec = pd.read_csv(
        dataset_dir / 'Metabolite_mol2vec.csv', header=0)


    # Both mol2vec tables share the same original feature semantics. Fit a
    # single scaler and PCA so the two node types use one coordinate system.
    all_features = np.vstack([
        Meta_mol2vec.values,
        Drug_MESH2vec.values,
    ])
    all_features = StandardScaler().fit_transform(all_features)
    all_pca_features = PCA(
        n_components=n_components,
        random_state=seed,
    ).fit_transform(all_features)

    num_metabolites = len(Meta_mol2vec)
    PCA_metabolite_feature = all_pca_features[:num_metabolites]
    PCA_drug_feature = all_pca_features[num_metabolites:]

    Drug_feature = torch.FloatTensor(PCA_drug_feature).to(device)
    Meta_feature = torch.FloatTensor(PCA_metabolite_feature).to(device)
    feature = torch.cat((Meta_feature, Drug_feature), dim=0).to(device)

    # 训练，验证，测试的样本
    index_matrix = np.asmatrix(np.where(Adj == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)  # random.seed(): 设定随机种子，使得random.shuffle随机打乱的顺序一致
    random.shuffle(random_index)  # random.shuffle将random_index列表中的元素打乱顺序

    CV_size = int(association_nam / k_folds)  # 每折的个数
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()  # %取余
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]  # 将余下的元素加到最后一折里面
    random_index = temp

    return Adj, Drug_adj, Meta_adj, feature, random_index, k_folds


def laplacian_positional_encoding(adj, pe_dim):
    # 计算度矩阵 D 和归一化矩阵 N
    D = torch.diag(torch.sum(adj, dim=1))  # 度矩阵D
    N = torch.diag(torch.pow(torch.sum(adj, dim=1).clamp(min=1), -0.5))  # 归一化矩阵N
    # 计算拉普拉斯矩阵 L
    L = torch.eye(
        adj.shape[0], dtype=adj.dtype, device=adj.device
    ) - N @ adj @ N
    # 使用 PyTorch 的 eigs 函数计算特征值和特征向量
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal = EigVal.real  # 获取特征值的实部
    EigVec = EigVec.real  # 获取特征向量的实部
    # 对特征值进行排序，并提取除了第一个特征向量之外的特征向量
    sorted_indices = EigVal.argsort()  # 对特征值从小到大进行排序，返回排序后索引
    EigVal_sorted = EigVal[sorted_indices]
    EigVec_sorted = EigVec[:, sorted_indices]  # 对EigVec按列进行排序
    # 提取特征向量，去除第一个列向量，得到位置编码特征
    lap_pos_enc = (EigVec_sorted[:, 1:pe_dim + 1]).float()
    return lap_pos_enc


def re_features(adj, features, K):
    """Build [N, K+1, D] hop features with normalized propagation."""
    if K < 0:
        raise ValueError("K must be non-negative")
    normalized_adj = normalize_adjacency(adj, add_self_loops=True)
    propagated = features.to(dtype=torch.float32)
    nodes_features = [propagated]
    for _ in range(K):
        propagated = normalized_adj @ propagated
        nodes_features.append(propagated)
    return torch.stack(nodes_features, dim=1)


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        # PyTorch 2.6+ removed the ``verbose`` argument from LRScheduler.
        # Keyword-only last_epoch is compatible with both old and new releases.
        super(PolynomialDecayLR, self).__init__(
            optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.warmup_updates > 0 and self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels



def get_tensor(features, edge_index):
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data

def plot_auc_curves(fprs, tprs, auc, directory, name):
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9,
             label=r'Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(output_dir / f'{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label=r'Mean AUPR: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(output_dir / f'{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

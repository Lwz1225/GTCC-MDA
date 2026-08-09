"""使用 SNF 融合药物/代谢物相似性，并构建加权 Mutual-KNN 网络。

矩阵计算使用 NumPy，CSV 读取、保存和统计使用 pandas。
输入和输出 CSV 的第一行是实体 ID，不包含额外的行索引列。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = np.finfo(np.float64).eps


def read_similarity(path: Path) -> tuple[list[str], np.ndarray]:
    """用 pandas 读取相似性矩阵，并返回列标签和 NumPy 数组。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    labels = frame.columns.astype(str).tolist()
    matrix = frame.to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path} must contain a square matrix; got {matrix.shape}")
    if matrix.shape[1] != len(labels):
        raise ValueError(
            f"{path} has {len(labels)} column labels but matrix shape is {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    if matrix.min() < 0 or matrix.max() > 1:
        raise ValueError(f"{path} contains values outside [0, 1]")
    return labels, matrix


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    """用 pandas 保存矩阵；index=False 避免生成多余的序号列。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, columns=labels).to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.16g",
    )


def matrix_statistics(name: str, matrix: np.ndarray) -> None:
    """打印非对角、非零元素的分布，便于检查数值尺度和稀疏度。"""
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    nonzero = off_diagonal[off_diagonal != 0]
    density = nonzero.size / off_diagonal.size

    if nonzero.size == 0:
        print(f"  {name}: 非对角线没有非零元素")
        return

    quantiles = pd.Series(nonzero).quantile([0.25, 0.50, 0.75])
    print(
        f"  {name}: 非零密度={density:.4%}, "
        f"最小值={nonzero.min():.6g}, "
        f"中位数={quantiles.loc[0.50]:.6g}, "
        f"均值={nonzero.mean():.6g}, "
        f"最大值={nonzero.max():.6g}"
    )


def _transition_matrix(similarity: np.ndarray) -> np.ndarray:
    """构造 SNF 状态矩阵：对角线为 0.5，其余权重每行合计约为 0.5。"""
    w = similarity.copy()
    np.fill_diagonal(w, 0.0)
    row_sum = w.sum(axis=1)
    p = w / np.maximum(2.0 * row_sum[:, None], EPS)  # 行归一化，使得除对角线元素之外的其他元素之和为1/2
    np.fill_diagonal(p, 0.5) # 对角线元素置为0.5，这样每一行的之和就为1了
    return p


def _knn_transition(similarity: np.ndarray, k: int) -> np.ndarray:
    """构造 SNF 扩散使用的局部 KNN 行随机矩阵。"""
    n = similarity.shape[0]
    k = min(k, n - 1)
    w = similarity.copy()
    np.fill_diagonal(w, -np.inf)
    neighbors = np.argpartition(w, n - k, axis=1)[:, -k:] # 获取每一行最大的K个元素
    local = np.zeros_like(similarity)
    rows = np.arange(n)[:, None]
    local[rows, neighbors] = similarity[rows, neighbors]
    return local / np.maximum(local.sum(axis=1, keepdims=True), EPS)


def snf(similarities: list[np.ndarray], k: int = 20, iterations: int = 20) -> np.ndarray:
    """使用 Similarity Network Fusion 融合两个或多个相似性矩阵。"""
    if len(similarities) < 2:
        raise ValueError("SNF requires at least two similarity matrices")
    shape = similarities[0].shape
    if any(matrix.shape != shape for matrix in similarities):
        raise ValueError("All similarity matrices for one entity must have equal shape")
    if not 1 <= k < shape[0]:
        raise ValueError(f"k must satisfy 1 <= k < {shape[0]}; got {k}")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    # 只做对称化以消除浮点误差，不对输入做 Min-Max 缩放。
    views = [(matrix + matrix.T) / 2.0 for matrix in similarities]
    states = [_transition_matrix(matrix) for matrix in views] # 调用_transition_matrix 的函数，将相似性矩阵归一化为行和为1的概率转移矩阵
    local_states = [_knn_transition(matrix, k) for matrix in views] # 调用 _knn_transition 函数，对每个矩阵只保留前K个最相似的邻居，然后归一化为转移矩阵。
    view_count = len(states)

    for _ in range(iterations):
        total = np.sum(states, axis=0)
        updated = []
        for index in range(view_count):
            other_mean = (total - states[index]) / (view_count - 1)
            propagated = local_states[index] @ other_mean @ local_states[index].T
            updated.append(_transition_matrix(propagated))
        states = updated

    fused = np.mean(states, axis=0)
    fused = (fused + fused.T) / 2.0
    # SNF 状态是概率尺度，因此节点数较多时非对角值通常在 1/n 量级。
    # 将对角线恢复为 1；非对角元素保留连续权重和相对大小。
    np.fill_diagonal(fused, 1.0)
    return fused


def weighted_mutual_knn(similarity: np.ndarray, k: int = 20) -> np.ndarray:
    """构建加权 Mutual-KNN 网络，并为孤立节点补一条最强正权重边。"""
    n = similarity.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < {n}; got {k}")

    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbors = np.argpartition(scores, n - k, axis=1)[:, -k:]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.arange(n)[:, None], neighbors] = True
    mutual = directed & directed.T # 只有当两个节点互相将对方选为近邻时，边才保留连接 mij=dij*dji

    graph = np.where(mutual, similarity, 0.0)
    graph = (graph + graph.T) / 2.0
    np.fill_diagonal(graph, 0.0)

    # 严格 Mutual-KNN 可能产生孤立节点，因此为孤立节点补上最强连接。
    isolated = np.flatnonzero(np.count_nonzero(graph, axis=1) == 0)
    for node in isolated:
        neighbor = int(np.argmax(scores[node]))
        weight = max(float(similarity[node, neighbor]), float(similarity[neighbor, node]))
        if weight > 0:
            graph[node, neighbor] = weight
            graph[neighbor, node] = weight
    return graph


def process_entity(
    dataset_dir: Path,
    entity: str,
    gaussian_name: str,
    fingerprint_name: str,
    snf_k: int,
    iterations: int,
    graph_k: int,
) -> None:
    gaussian_labels, gaussian = read_similarity(dataset_dir / gaussian_name)
    fingerprint_labels, fingerprint = read_similarity(dataset_dir / fingerprint_name)
    if gaussian_labels != fingerprint_labels:
        raise ValueError(f"{entity} matrix column labels are not in the same order")
    if gaussian.shape != fingerprint.shape:
        raise ValueError(
            f"{entity} input shapes differ: {gaussian.shape} and {fingerprint.shape}"
        )

    fused = snf([gaussian, fingerprint], k=snf_k, iterations=iterations)
    graph = weighted_mutual_knn(fused, k=graph_k)
    fused_path = dataset_dir / f"{entity}_SNF_Similarity.csv"
    graph_path = dataset_dir / f"{entity}_SNF_MutualKNN_Network.csv"
    write_matrix(fused_path, gaussian_labels, fused)
    write_matrix(graph_path, gaussian_labels, graph)

    edge_count = int(np.count_nonzero(np.triu(graph, k=1)))
    isolated_count = int(np.sum(np.count_nonzero(graph, axis=1) == 0))
    print(
        f"{entity}: nodes={fused.shape[0]}, undirected_edges={edge_count}, "
        f"isolated_nodes={isolated_count}"
    )
    matrix_statistics("SNF 融合矩阵", fused)
    matrix_statistics("Mutual-KNN 网络", graph)
    print(f"  fused:  {fused_path}")
    print(f"  network: {graph_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("../Dataset2"))
    parser.add_argument("--snf-k", type=int, default=20, help="SNF local neighbors")
    parser.add_argument("--iterations", type=int, default=10, help="SNF iterations")# 10
    parser.add_argument("--graph-k", type=int, default=30, help="mutual-KNN neighbors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print('args', args)
    process_entity(
        args.dataset_dir,
        "Drug",
        "Drug_Gaussian_Simi.csv",
        "Drug_Fingerprint_sim.csv",
        args.snf_k,
        args.iterations,
        args.graph_k,
    )
    process_entity(
        args.dataset_dir,
        "Metabolite",
        "Metabolite_Gaussian_Simi.csv",
        "Metabolite_Fingerprint_sim.csv",
        args.snf_k,
        args.iterations,
        args.graph_k,
    )


if __name__ == "__main__":
    main()

# Indices of Cluster Validity
import numpy as np
import math
from utility import norm_distances, extract_labels, division_by_zero
import numpy as np
from scipy.spatial.distance import pdist, cdist

def align_labels(y_true, y_pred):
    """
    Map nhãn dự đoán từ thuật toán phân cụm về đúng nhãn thực tế
    dựa trên độ phủ lớn nhất (Hungarian algorithm).
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64).flatten()
    
    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    # Tìm cách ghép cặp (row, col) sao cho tổng các phần tử trên đường chéo là lớn nhất
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Tạo mảng nhãn mới đã được map
    aligned_labels = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[y_pred == j] = i
        
    return aligned_labels

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / np.sum(cm)


def clustering_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    new_pred = np.zeros_like(y_pred)
    for r, c in zip(row_ind, col_ind):
        new_pred[y_pred == c] = r
        
    from sklearn.metrics import f1_score
    return f1_score(y_true, new_pred, average="weighted")


# DI +
def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    C = len(np.unique(labels))
    cluster_points = [data[labels == i] for i in range(C)]
    cluster_centers = np.array([np.mean(points, axis=0) for points in cluster_points])
    # Tính khoảng cách nhỏ nhất giữa các tâm cụm
    min_cluster_distance = np.inf
    from itertools import combinations
    for i, j in combinations(range(C), 2):
        dist = norm_distances(cluster_centers[i], cluster_centers[j])
        min_cluster_distance = min(min_cluster_distance, dist)
    # Tính đường kính lớn nhất của các cụm
    max_cluster_diameter = 0
    for points in cluster_points:
        if len(points) > 1:  # Cụm phải có ít nhất 2 điểm để tính đường kính
            distances = norm_distances(points[:, np.newaxis], points, axis=2)
            cluster_diameter = np.max(distances)
            max_cluster_diameter = max(max_cluster_diameter, cluster_diameter)
    # Tránh chia cho 0
    if max_cluster_diameter == 0:
        return np.inf
    return min_cluster_distance / max_cluster_diameter

def dunn(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter).
    
    - Inter-cluster distance: khoảng cách nhỏ nhất giữa 2 điểm thuộc 2 cluster khác nhau
    - Intra-cluster diameter: khoảng cách lớn nhất giữa 2 điểm trong cùng 1 cluster
    
    Dùng pdist (O(n²) nhưng không tạo tensor 3D) → RAM-friendly.
    
    Parameters:
        X: (n, p) — dữ liệu
        labels: (n,) — nhãn cluster
        
    Returns:
        Dunn Index (float), càng lớn càng tốt
    """
    unique_labels = np.unique(labels)
    c = len(unique_labels)
 
    if c < 2:
        return 0.0
    # Tách data theo cluster
    clusters = [X[labels == label] for label in unique_labels]
    # Max intra-cluster diameter (khoảng cách xa nhất trong mỗi cluster)
    max_intra = 0.0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        # pdist trả về vector condensed → chỉ cần max
        intra_dists = pdist(cluster, metric='euclidean')
        d = np.max(intra_dists)
        if d > max_intra:
            max_intra = d
    if max_intra == 0.0:
        return 0.0
 
    # Min inter-cluster distance (khoảng cách gần nhất giữa 2 cluster)
    min_inter = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            # cdist: (n_i, n_j) — không tạo tensor 3D
            inter_dists = cdist(clusters[i], clusters[j], metric='euclidean')
            d = np.min(inter_dists)
            if d < min_inter:
                min_inter = d
 
    return min_inter / max_intra


# DB -
def davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(data, labels)


# PCI fuzzy +
def partition_coefficient(membership: np.ndarray) -> float:
    N, C = membership.shape
    _pc = np.sum(np.square(membership)) / N  # PC fuzzy
    _1dc = 1/C
    return (_pc - _1dc) / (1 - _1dc)


# CE fuzzy -
def classification_entropy(membership: np.ndarray, a: float = np.e) -> float:
    """
    CE đo lường mức độ không chắc chắn trong việc gán điểm vào các cụm, giá trị càng thấp, độ hợp lệ
    của phân cụm càng tốt. CE thường kết hợp với PC, một phân cụm tốt thường có PC cao và CE thấp,
    0 ≤ 1 − P C ≤ CE
    """
    N = membership.shape[0]

    # Tránh log(0) bằng cách thêm một epsilon nhỏ cho tất cả các phần tử
    epsilon = np.finfo(float).eps
    membership = np.clip(membership, epsilon, 1)

    # Tính tỉ lệ phần trăm điểm dữ liệu thuộc về mỗi cụm
    log_u = np.log(membership) / np.log(a)  # Chuyển đổi cơ số logarit
    return -np.sum(membership * log_u) / N


# PE fuzzy -
def partition_entropy(membership: np.ndarray) -> float:
    """
    Tính chỉ số Partition Entropy index

    Parameters
    ----------
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Partition Entropy index
    """
    return classification_entropy(membership=membership, a=np.e)

# S fuzzy +
def separation(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    _N, C = membership.shape
    _ut = membership.T
    numerator = 0
    for i in range(C):
        diff = data - centroids[i]
        squared_diff = np.sum(diff**2, axis=1)
        numerator += np.sum((_ut[i] ** m) * squared_diff)
    center_dists = np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2)
    np.fill_diagonal(center_dists, np.inf)
    min_center_dist = np.min(center_dists)
    return numerator / division_by_zero(min_center_dist)

# SI +
def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    # N = len(data)
    # silhouette_vals = np.zeros(N)
    # for i in range(N):
    #     a_i = 0
    #     b_i = np.inf
    #     for j in range(N):
    #         if i != j:
    #             distance = np.sqrt(np.sum((data[i] - data[j])**2))
    #             if labels[i] == labels[j]:
    #                 a_i += distance
    #             else:
    #                 b_i = min(b_i, distance)

    #     if np.sum(labels == labels[i]) > 1:
    #         a_i /= (np.sum(labels == labels[i]) - 1)
    #     else:
    #         a_i = 0
    #     silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    # return np.mean(silhouette_vals)
    from sklearn.metrics import silhouette_score
    return silhouette_score(data, labels)


# AC +
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    # total_samples = len(y_true)
    # return correct_predictions / total_samples
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


# F1 +
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:  # binary|weighted
    # tp, fp, fn = tp_fp_fn(y_true, y_pred)
    # # Tính precision và recall
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # total = precision + recall
    # return 2 * (precision * recall) / total if total > 0 else 0
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)


# FHV fuzzy +
def hypervolume(membership: np.ndarray, m: float = 2) -> float:
    C = membership.shape[1]
    result = 0
    for i in range(C):
        cluster_u = membership[:, i]
        n_i = np.sum(cluster_u > 0)
        if n_i > 0:
            result += np.sum(cluster_u ** m) / n_i
    return result

# XB fuzzy -
def Xie_Benie(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray) -> float:
    """
    Tính chỉ số Xie-Benie index

    Parameters
    ----------
    data: dữ liệu chưa được phân cụm.
    centroids: Ma trận tâm cụm 
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Xie-Benie index
    """
    _, C = membership.shape
    labels = extract_labels(membership)
    clusters = [data[labels == i] for i in range(C)]

    from sklearn.metrics import pairwise_distances
    S_iq = np.asanyarray([np.mean([np.linalg.norm(point - centroids[i]) for point in cluster]) for i, cluster in enumerate(clusters)])
    tu = np.sum(np.square(membership) * np.square(S_iq))
    distance = pairwise_distances(centroids)
    distance[distance == 0] = math.inf
    mau = len(data) * np.min(np.square(distance))
    return tu / mau

def Q(membership: np.ndarray, m: float) -> float: 
    S = 0.0
    diff = membership[:, :, np.newaxis] - membership[:, np.newaxis,:] 
    S = np.sum(np.abs(diff) ** m)
    # Tính C
    U_bar = 1.0 - membership
    C = np.sum(U_bar ** m)
    if C == 0:
        return np.inf
    return S / C

def calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(data, labels)

def cs(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    N, C = membership.shape
    numerator = 0
    for i in range(C):
        numerator += np.sum((membership[:, i]**m)[:, np.newaxis] *
                            np.sum((data - centroids[i])**2, axis=1)[:, np.newaxis])
    min_center_dist = np.min([np.sum((centroids[i] - centroids[j])**2)
                              for i in range(C)
                              for j in range(i+1, C)])
    return numerator / (N * min_center_dist)

# Dice +
def dice_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Tính chỉ số Dice (Sørensen–Dice coefficient) cho bài toán phân cụm.
    
    Dice = 2*|A ∩ B| / (|A| + |B|) = 2*TP / (2*TP + FP + FN)
    
    Parameters
    ----------
    y_true: nhãn thực tế
    y_pred: nhãn dự đoán từ thuật toán phân cụm (sẽ được align tự động)
    average: 'macro' | 'weighted' | 'binary'
        - 'macro': trung bình cộng Dice của các lớp (không trọng số)
        - 'weighted': trung bình có trọng số theo số lượng mẫu mỗi lớp
        - 'binary': chỉ tính cho lớp dương (lớp 1)
    
    Return
    ------
    Giá trị Dice trong [0, 1], càng cao càng tốt
    """
    y_true = np.asarray(y_true).astype(np.int64).flatten()
    y_pred = align_labels(y_true, np.asarray(y_pred))
    
    classes = np.unique(y_true)
    dice_per_class = []
    weights = []
    
    for c in classes:
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        
        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)
        
        denom = 2 * tp + fp + fn
        dice_c = (2 * tp) / denom if denom > 0 else 0.0
        
        dice_per_class.append(dice_c)
        weights.append(np.sum(true_c))
    
    dice_per_class = np.array(dice_per_class)
    weights = np.array(weights, dtype=float)
    
    if average == 'binary':
        # Giả định lớp dương là lớp có giá trị lớn nhất (thường là 1)
        return dice_per_class[-1]
    elif average == 'weighted':
        return np.sum(dice_per_class * weights) / np.sum(weights)
    else:  # macro
        return np.mean(dice_per_class)
 
 
# Jaccard +
def jaccard_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Tính chỉ số Jaccard (Intersection over Union - IoU) cho bài toán phân cụm.
    
    Jaccard = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)
    
    Quan hệ với Dice: Jaccard = Dice / (2 - Dice)
    
    Parameters
    ----------
    y_true: nhãn thực tế
    y_pred: nhãn dự đoán từ thuật toán phân cụm (sẽ được align tự động)
    average: 'macro' | 'weighted' | 'binary'
    
    Return
    ------
    Giá trị Jaccard trong [0, 1], càng cao càng tốt
    """
    y_true = np.asarray(y_true).astype(np.int64).flatten()
    y_pred = align_labels(y_true, np.asarray(y_pred))
    
    classes = np.unique(y_true)
    jac_per_class = []
    weights = []
    
    for c in classes:
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        
        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)
        
        denom = tp + fp + fn
        jac_c = tp / denom if denom > 0 else 0.0
        
        jac_per_class.append(jac_c)
        weights.append(np.sum(true_c))
    
    jac_per_class = np.array(jac_per_class)
    weights = np.array(weights, dtype=float)
    
    if average == 'binary':
        return jac_per_class[-1]
    elif average == 'weighted':
        return np.sum(jac_per_class * weights) / np.sum(weights)
    else:  # macro
        return np.mean(jac_per_class)
 
 
# Kappa +
def kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính chỉ số Cohen's Kappa cho bài toán phân cụm.
    
    Kappa = (p_o - p_e) / (1 - p_e)
        - p_o: độ chính xác quan sát được (observed agreement)
        - p_e: độ chính xác kỳ vọng ngẫu nhiên (expected agreement)
    
    Ý nghĩa giá trị:
        < 0    : tệ hơn ngẫu nhiên
        0-0.2  : rất yếu
        0.2-0.4: yếu
        0.4-0.6: trung bình
        0.6-0.8: tốt
        0.8-1.0: rất tốt
    
    Parameters
    ----------
    y_true: nhãn thực tế
    y_pred: nhãn dự đoán từ thuật toán phân cụm (sẽ được align tự động)
    
    Return
    ------
    Giá trị Kappa trong [-1, 1], càng cao càng tốt
    """
    from sklearn.metrics import cohen_kappa_score
    
    y_true = np.asarray(y_true).astype(np.int64).flatten()
    y_pred = align_labels(y_true, np.asarray(y_pred))
    
    return cohen_kappa_score(y_true, y_pred)



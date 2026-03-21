from fcm import *
import numpy as np
from newdataloader import *
import pandas as pd

from utility import *


class OFCM(FCM):
    def __init__(self, n_clusters=3, m=2, max_iter=100, error=0.00001, random_state=42):
        super().__init__(n_clusters, m, max_iter, error, random_state)

    def update_member(self, X: np.ndarray) -> np.ndarray:

        dist_matrix = distance_cdist(X, self.V, metric='cityblock')  # N x C
        power = 1.0 / (self.m - 1)

        dist_matrix = division_by_zero(dist_matrix ** power)

        reverse_dist_matrix = 1.0 / (dist_matrix)

        U_new = (dist_matrix) * \
            np.sum(reverse_dist_matrix, axis=1, keepdims=True)
        U_new = 1.0 / U_new
        # Xử lý các trường hợp khoảng cách bằng 0
        zero_indices = np.argwhere(dist_matrix == 0)
        row = zero_indices[:,0]
        U_new[row,:] = 0
        U_new[zero_indices] = 1.0


        return U_new

    def calculate_centroids(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape

        # 1. Sort từng cột của X
        sorted_indices = np.argsort(X, axis=0)  # (N, D)

        # 2. Sort dữ liệu theo từng cột
        data_sorted = np.take_along_axis(X, sorted_indices, axis=0)  # (N, D)

        # 3. Trọng số (N, C)
        weights_matrix = self.U ** self.m  # (N, C)
        C = weights_matrix.shape[1]

        # 4. Sort trọng số theo từng cột D
        weights_sorted = weights_matrix[sorted_indices.T, :]  # (D, N, C)

        # 5. Tính tích lũy trọng số theo N
        cumsum = np.cumsum(weights_sorted, axis=1)  # (D, N, C)

        # 6. Cutoff = 1/2 tổng trọng số của mỗi (D, C)
        cutoff = cumsum[:, -1, :] * 0.5  # (D, C)

        # 7. Tìm median index cho từng (d, c)
        median_idx = np.zeros((D, C), dtype=int)

        for c in range(C):
            for d in range(D):
                median_idx[d, c] = np.searchsorted(
                    cumsum[d, :, c], cutoff[d, c], side='left'
                )

        # 8. Lấy giá trị median tại mỗi (d, c)
        centroids = np.zeros((C, D))

        for c in range(C):
            for d in range(D):
                centroids[c, d] = data_sorted[median_idx[d, c], d]

        return centroids

    def fit(self, X: np.ndarray):
        np.random.seed(42)
        X = np.array(X)
        n, d = X.shape
        self.U = self.init_membership_matrix(n)
        random_indices = np.random.choice(n, self.c, replace=False)
        self.V = X[random_indices].copy()
        for i in range(0, self.max_iter):
            U_prev = self.U.copy()
            V_prev = self.V.copy()
            self.U = self.update_member(X)
            self.V = self.calculate_centroids(X)
            diff = np.linalg.norm(self.U-U_prev)
            if (diff < self.error):
                break

        return self.U, self.V, i


if __name__ == "__main__":
    data, class_data = load_data_with_outliers('data_iris.csv','class')

    from utility import round_float, extract_labels
    # from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie, classification_entropy, silhouette, f1_score, hypervolume, accuracy_score

    # Chuyển đổi nhãn từ chuỗi sang số
    unique_labels = np.unique(class_data)
    label_to_index = {label: index for index,
                      label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[label] for label in class_data])

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(Xie_Benie(X, V, U)),  # XB
            wdvl(classification_entropy(U)),  # CE
            wdvl(silhouette(X, labels)),  # SI
            wdvl(hypervolume(U)),  # FHV
            # F1 - sử dụng numeric_labels thay vì class_data
            wdvl(f1_score(numeric_labels, labels)),
            # AC - sử dụng numeric_labels thay vì class_data
            wdvl(accuracy_score(numeric_labels, labels))

        ]
        return SPLIT.join(kqdg)

    ofcm = OFCM(n_clusters=3)  # c : số cụm
    U, V, i = ofcm.fit(data)
    # print('Tâm cụm:\n', centroids)

    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+',
              'PE-', 'XB-', 'CE-', 'SI+', 'FHV+', 'F1+', 'AC+']
    print(SPLIT.join(titles))

    print(write_report_fcm(alg='OFCM', index=0,
          process_time=ofcm.process_time, step=i, X=data, V=V, U=U))

import time
import numpy as np
from scipy.spatial.distance import cdist
from .MYFCM import FCM, division_by_zero
from .SSFCM import SSFCM

class SSCFCM:
    def __init__(self, n_clusters: int, m: float = 2.0, epsilon: float = 1e-5, max_iter: int = 1000):
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.data_sites = []  # Danh sách các đối tượng SSFCM
        self.j_ii = []
        self.u_bar_list = []
        self.len_datasite = 0

    # CT 18: Cập nhật tâm cụm kết hợp Semi-supervised và Collaborative
    def update_centroids(self, data: np.ndarray, membership: np.ndarray, u_bar: np.ndarray, centroids_fall: np.ndarray, beta):
        # (U - U_bar)^m
        u_diff = (membership - u_bar) ** self.m
        
        # Phần 1: dot product với dữ liệu cục bộ
        part1 = u_diff.T @ data  # (C, N) x (N, D) -> (C, D)
        
        # Phần 2: Ảnh hưởng từ các site khác (Collaborative)
        # C x D * (Tổng trọng số dòng của u_diff)
        part2 = centroids_fall * np.sum(u_diff, axis=0)[:, np.newaxis]
        
        numerator = part1 + beta * part2
        denominator = (1 + beta) * np.sum(u_diff, axis=0)
        
        return numerator / division_by_zero(denominator[:, np.newaxis])

    # CT 17: Cập nhật ma trận thành viên
    def update_membership(self, data: np.ndarray, centroids: np.ndarray, u_bar: np.ndarray, centroids_fall: np.ndarray, beta) -> np.ndarray:
        # Khoảng cách tới tâm cụm hiện tại
        d_ik = cdist(data, centroids) 
        # Khoảng cách tới "tâm cụm rơi" (ảnh hưởng từ site khác)
        v_diff = np.linalg.norm(centroids - centroids_fall, axis=1) # C
        
        # Tính toán mẫu số của công thức FCM cải tiến
        # numerator ở đây là 1 / (d^2 + beta * v_diff^2)
        dist_term = d_ik**2 + beta * (v_diff**2)
        
        # Tránh chia cho 0
        dist_term = division_by_zero(dist_term)
        
        inv_dist = 1.0 / (dist_term ** (1 / (self.m - 1)))
        denumerator = np.sum(inv_dist, axis=1, keepdims=True)
        
        # Kết hợp với phần semi-supervised (u_bar)
        fcm_term = inv_dist / division_by_zero(denumerator)
        return u_bar + (1 - np.sum(u_bar, axis=1, keepdims=True)) * fcm_term

    def calculate_centroids_fall(self):
        # Tính trung bình tâm cụm của tất cả các site khác cho mỗi site i
        c = self.n_clusters
        d = self.data_sites[0].V.shape[1]
        v_fall_all = np.zeros((self.len_datasite, c, d))

        for i in range(self.len_datasite):
            other_centers = [self.data_sites[k].V for k in range(self.len_datasite) if k != i]
            v_fall_all[i] = np.mean(other_centers, axis=0)
        return v_fall_all

    def calculate_j_fall(self, data: np.ndarray, membership: np.ndarray, centers: np.ndarray):
        distances = cdist(data, centers)
        return np.sum((membership ** self.m) * (distances ** 2))

    def calculate_beta_matrix(self):
        beta_matrix = np.zeros((self.len_datasite, self.len_datasite))
        for i in range(self.len_datasite):
            # Lấy data từ biến local_data mình đã gán thủ công
            data_i = self.data_sites[i].local_data 
            j_ii = self.j_ii[i]
            for j in range(self.len_datasite):
                if i == j: continue
                centers_j = self.data_sites[j].V
                # Tính J_ij
                distances = cdist(data_i, centers_j)
                j_ij = np.sum((self.data_sites[i].U ** self.m) * (distances ** 2))
                beta_matrix[i][j] = min(1, j_ii / j_ij) if j_ij > 0 else 1
        return beta_matrix

    def phase1(self, sub_datasets, sub_labels):
          # Chạy SSFCM độc lập cho từng site
          self.len_datasite = len(sub_datasets)
          for i in range(self.len_datasite):
              ssfcm = SSFCM(c=self.n_clusters, m=self.m, max_iter=self.max_iter, eps=self.epsilon)
              ssfcm.local_data = sub_datasets[i] 
              
              ssfcm.fit(sub_datasets[i], sub_labels[i])
              self.data_sites.append(ssfcm)
              
              # Tính J_ii dựa trên local_data vừa gán
              dist = cdist(ssfcm.local_data, ssfcm.V)
              j_val = np.sum((ssfcm.U ** self.m) * (dist ** 2))
              self.j_ii.append(j_val)

    def phase2(self):
        for iteration in range(self.max_iter):
            converged_count = 0
            beta_matrix = self.calculate_beta_matrix()
            v_fall_all = self.calculate_centroids_fall()

            for i in range(self.len_datasite):
                site = self.data_sites[i]
                old_U = site.U.copy()
                
                mask = np.arange(self.len_datasite) != i
                beta_avg = np.mean(beta_matrix[i, mask])

                site.U = self.update_membership(
                    site.local_data, site.V, site.u_bar, v_fall_all[i], beta_avg
                )
                site.V = self.update_centroids(
                    site.local_data, site.U, site.u_bar, v_fall_all[i], beta_avg
                )

                if np.linalg.norm(site.U - old_U) < self.epsilon:
                    converged_count += 1
            if converged_count == self.len_datasite:
                return iteration + 1   # trả về số step thực tế
        return self.max_iter   # nếu không hội tụ

    def fit(self, sub_datasets, sub_labels):
        start_time = time.time()
        
        self.phase1(sub_datasets, sub_labels)
        
        phase2_steps = self.phase2()
        
        total_time = time.time() - start_time
        
        return  phase2_steps

# --- Demo sử dụng ---
if __name__ == "__main__":
    from algo.SSFCM import init_semi_data
    import pandas as pd
    
    # Giả sử bạn có data_iris.csv
    try:
        df = pd.read_csv('data_iris.csv')
        X = df.iloc[:, :-1].values
        y = df['class'].values
        
        # Chia dữ liệu làm 2 site (ví dụ)
        mid = len(X) // 2
        sub_datasets = [X[:mid], X[mid:]]
        sub_labels_raw = [y[:mid], y[mid:]]
        
        # Tạo nhãn bán giám sát cho từng site
        sub_labels = [init_semi_data(labels, 0.3) for labels in sub_labels_raw]
        
        model = SSCFCM(n_clusters=3)
        total_time = model.fit(sub_datasets, sub_labels)
        
        print(f"Tổng thời gian thực hiện: {total_time:.4f}s")
        for i, site in enumerate(model.data_sites):
            print(f"Site {i+1} centers:\n{site.V}")
            
    except FileNotFoundError:
        print("Vui lòng đảm bảo có file data_iris.csv hoặc thay thế bằng dữ liệu của bạn.")
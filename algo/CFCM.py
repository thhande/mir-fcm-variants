import numpy as np
import time
from .MYFCM import FCM, division_by_zero  # Import class FCM mới
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class Dcfcm:
    def __init__(self, n_clusters: int, m: float = 2.0, beta: float = 0.1, epsilon: float = 1e-5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.beta = beta  # Hệ số cộng tác
        self.data_sites = [] 
        self.steps = []
        self.num_sites = 0
    
    def align_centroids(self, standard_centroid=None):
        """
        Đồng bộ thứ tự các tâm cụm của các site.
        Nếu có standard_centroid, sẽ gióng tất cả site theo tâm chuẩn này.
        Nếu không, sẽ gióng theo Site 0.
        """
        if self.num_sites <= 0:
            return

        # Nếu có standard_centroid (Lựa chọn 2 - Ép theo nhãn thật)
        if standard_centroid is not None:
            ref_V = standard_centroid
            start_idx = 0  # Phải gióng TẤT CẢ các site, bao gồm cả site 0
        else:
            # Lựa chọn 1 - Lấy Site 0 làm chuẩn (Cách chuẩn Unsupervised)
            if self.num_sites <= 1:
                return
            ref_V = self.data_sites[0].V  
            start_idx = 1  # Bỏ qua site 0 vì lấy nó làm chuẩn

        for i in range(start_idx, self.num_sites):
            site = self.data_sites[i]
            
            # Tính ma trận khoảng cách giữa tâm chuẩn (ref_V) và Site i
            dist_matrix = cdist(ref_V, site.V, metric='euclidean')
            
            # Tìm cặp match tối ưu nhất bằng thuật toán Hungarian
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # Tạo bản sao để sắp xếp lại theo thứ tự của tâm chuẩn
            new_V = np.zeros_like(site.V)
            new_U = np.zeros_like(site.U)
            
            for ref_idx, site_idx in zip(row_ind, col_ind):
                new_V[ref_idx] = site.V[site_idx]
                new_U[:, ref_idx] = site.U[:, site_idx]
                
            # Cập nhật lại dữ liệu cho Site i
            site.V = new_V
            site.U = new_U
        
    def phase1(self, list_data: list):
        """
        Giai đoạn 1: Chạy FCM độc lập trên từng data site.
        """
        self.num_sites = len(list_data)
        self.data_sites = []
        
        for i, data in enumerate(list_data):

            fcm_instance = FCM(c=self.n_clusters, m=self.m, 
                               max_iter=self.max_iter, eps=self.epsilon)
             

            V, U, labels, steps = fcm_instance.fit(data)
            
            # Lưu dữ liệu vào object để dùng cho phase 2
            fcm_instance.local_data = np.array(data) 
            self.data_sites.append(fcm_instance)

    # Gọi trực tiếp bằng class không cần khởi tạo object mới:
    def calculate_induced_matrix(self, fcm_instance, data_i, centroids_j):
        # Lưu tạm V hiện tại
        temp_V = fcm_instance.V 
        # Gán V của site j vào
        fcm_instance.V = centroids_j
        # Tính U cảm ứng
        u_induced = fcm_instance.update_membership_matrix(data_i)
        # Trả lại V cũ
        fcm_instance.V = temp_V 
        return u_induced

    def update_collaborative_U(self, fcm_i, u_fall_sum):
        """
        Cập nhật ma trận U theo công thức cộng tác (A.11).
        """
        # Tính U chuẩn (standard FCM update) dựa trên logic TRUEFCM
        u_standard = fcm_i.update_membership_matrix(fcm_i.local_data)

        P = self.num_sites
        denominator = 1 + self.beta * (P - 1)
        
        # Term liên quan đến Beta
        interaction_term = (self.beta * u_fall_sum) / denominator
        
        # Hệ số điều chỉnh (1 - sum(interaction))
        adjustment_factor = 1 - np.sum(interaction_term, axis=1, keepdims=True)
        
        # Công thức A.11:
        u_new = (u_standard * adjustment_factor) + interaction_term
        return u_new

    def update_collaborative_V(self, fcm_i, u_new, u_fall_sum_sq_diff):
        """
        Cập nhật Centroids theo công thức (A.14).
        """
        X = fcm_i.local_data
        U_sq = u_new ** self.m # Sử dụng m từ class
        
        # Tử số: Sum(U^m * X) + Beta * Sum((U - U~)^m * X)
        num_part1 = U_sq.T @ X 
        num_part2 = self.beta * (u_fall_sum_sq_diff.T @ X)
        numerator = num_part1 + num_part2
        
        # Mẫu số: Sum(U^m) + Beta * Sum((U - U~)^m)
        den_part1 = np.sum(U_sq, axis=0, keepdims=True).T
        den_part2 = self.beta * np.sum(u_fall_sum_sq_diff, axis=0, keepdims=True).T
        denominator = division_by_zero(den_part1 + den_part2)
        
        return numerator / denominator

    def phase2(self):
        """
        Giai đoạn 2: Vòng lặp cộng tác (Collaboration Loop)
        """
        self.steps = [0] * self.num_sites
        
        for iteration in range(self.max_iter):
            max_change = 0.0
            all_u_induced = {} 
            
            # 1. Communication Phase
            for i in range(self.num_sites):
                all_u_induced[i] = {}
                for j in range(self.num_sites):
                    if i == j: continue
                    u_tilde = self.calculate_induced_matrix(
                        self.data_sites[i],
                        self.data_sites[i].local_data, 
                        self.data_sites[j].V
                    )
                    all_u_induced[i][j] = u_tilde

            # 2. Optimization Phase
            convergence_flags = [False] * self.num_sites
            for i in range(self.num_sites):
                fcm_i = self.data_sites[i]
                
                u_fall_sum = np.zeros_like(fcm_i.U)
                for j in range(self.num_sites):
                    if i == j: continue
                    u_fall_sum += all_u_induced[i][j]
                
                # Cập nhật U mới
                U_new = self.update_collaborative_U(fcm_i, u_fall_sum)
                
                # Cập nhật lại sai số bình phương với U mới cho V
                u_fall_sum_sq_diff_new = np.zeros_like(U_new)
                for j in range(self.num_sites):
                    if i == j: continue
                    u_fall_sum_sq_diff_new += (U_new - all_u_induced[i][j]) ** self.m

                V_new = self.update_collaborative_V(fcm_i, U_new, u_fall_sum_sq_diff_new)
                
                # Kiểm tra hội tụ (sử dụng max abs diff cho ổn định)
                diff_u = np.max(np.abs(U_new - fcm_i.U))
                
                if diff_u < self.epsilon:
                    convergence_flags[i] = True
                else:
                    self.steps[i] += 1
                
                fcm_i.U = U_new
                fcm_i.V = V_new
                if diff_u > max_change: max_change = diff_u

            if all(convergence_flags):
                break

    def fit(self, list_data, standard_centroid=None):
        """
        Hàm fit đã được cập nhật để nhận thêm standard_centroid
        """
        self.phase1(list_data)
        self.align_centroids(standard_centroid)
        self.phase2()
        return self.data_sites
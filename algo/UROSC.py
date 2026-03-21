import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

class UROSC:
    """
    Implementierung des UROSC-Algorithmus (Uncorrelated Ridge Regression model 
    with Optimal Scaling for Semi-supervised Clustering) basierend auf dem 
    Paper NEUCOM-D-25-12653.
    Tham số:
    ----------
    n_clusters : int
        Số lượng cụm (giá trị 'c').      
    gamma : float
        Tham số điều chuẩn (giá trị 'γ').
        
    max_iter : int, optional (default=100)
        Số vòng lặp tối đa cho thuật toán tối ưu (Algorithm 2).
        
    tol : float, optional (default=1e-5)
        Ngưỡng hội tụ. Dừng lại nếu sự thay đổi trong ma trận F 
        nhỏ hơn giá trị này.
    """

    def __init__(self, n_clusters, gamma, max_iter=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        
        # Các biến kết quả sẽ được lưu trữ sau khi fit
        self.W_ = None
        self.b_ = None
        self.Q_ = None
        self.labels_ = None

    def _matrix_sqrt_inv(self, M):
        """
        Tính nghịch đảo căn bậc hai của một ma trận đối xứng M.
        Sử dụng phân rã trị riêng: M = V * D * V^T
        M^{-1/2} = V * D^{-1/2} * V^T
        """
        # Đảm bảo M là đối xứng
        M = (M + M.T) / 2
        try:
            # Phân rã trị riêng
            eigvals, eigvecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            print("Lỗi: Phân rã trị riêng thất bại. Thêm nhiễu nhỏ vào đường chéo.")
            M_reg = M + np.eye(M.shape[0]) * 1e-10
            eigvals, eigvecs = np.linalg.eigh(M_reg)

        # Tính D^{-1/2}
        # Xử lý các trị riêng rất nhỏ/âm để tránh lỗi chia cho 0
        eigvals_inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
        
        # Dựng lại ma trận
        D_inv_sqrt = np.diag(eigvals_inv_sqrt)
        M_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T
        return M_inv_sqrt

    def _update_F(self, Z, F_current, labeled_idx, unlabeled_idx, c):
        """
        Cập nhật ma trận chỉ thị F sử dụng Coordinate Descent cải tiến.
        Đây là việc triển khai Algorithm 1 trong bài báo. """
        n, _ = Z.shape
        F_new = F_current.copy()
        # 1. Tiền tính toán (Algorithm 1, Step 3)
        # a_k = Z^T * f_k. 'a' là ma trận (c x c) nơi cột k là a_k.
        a = Z.T @ F_new  # (c x n) @ (n x c) = (c x c)
        # b_k = f_k^T * f_k (số lượng điểm trong cụm k)
        b = np.sum(F_new, axis=0)  # shape (c,)
     
        # term_k = f_k^T * Z * Z^T * f_k = a_k^T * a_k
        term = np.einsum('ij,ij->j', a, a) # Tương đương np.diag(a.T @ a)
        # z_i_sq = z_i * (z_i)^T (norm L2 bình phương của mỗi hàng z_i)
        z_sq_norms = np.einsum('ij,ij->i', Z, Z) # shape (n,)   
        # 2. Lặp qua các điểm chưa dán nhãn (Algorithm 1, Step 5)
        for i in unlabeled_idx:
            z_i = Z[i, :]  # (1 x c)
            z_i_sq = z_sq_norms[i]        
            # Cụm hiện tại của điểm i
            p = np.argmax(F_new[i, :])        
           # Tiền tính z_i * a_k cho tất cả k
            z_i_a = z_i @ a  # (1 x c) @ (c x c) = (1 x c)
            
            # Tính các tử số cho Eq. (23)
            term_add_num = term + 2 * z_i_a + z_i_sq
            term_sub_num = term - 2 * z_i_a + z_i_sq
            term_orig_num = term
            
            # Tính các mẫu số cho Eq. (23)
            b_add = b + 1
            b_sub = b - 1
            b_orig = b
            
            # Tính phi(k) cho tất cả k (Eq. 23)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Trường hợp k != p
                phi = term_add_num / b_add - term_orig_num / b_orig
                
                # Trường hợp k == p
                # Xử lý trường hợp b_sub[p] == 0 (dòng 257)
                if b_sub[p] <= 0:
                    phi_p = -np.inf
                else:
                    phi_p = term_orig_num[p] / b_orig[p] - term_sub_num[p] / b_sub[p]
                
                phi[p] = phi_p
                phi[np.isnan(phi)] = -np.inf # Xử lý các trường hợp 0/0
            
            # 3. Tìm cụm mới (Algorithm 1, Step 6)
            q = np.argmax(phi)
            
            # 4. Cập nhật nếu cụm thay đổi (Algorithm 1, Step 8-10)
            if p != q:
                # Cập nhật F
                F_new[i, p] = 0.0
                F_new[i, q] = 1.0
                
                # Chuyển z_i (1, c) thành cột (c, 1)
                z_i_col = Z[i, :].T 
                
                # Cập nhật a (Eq. 25)
                a[:, p] = a[:, p] - z_i_col
                a[:, q] = a[:, q] + z_i_col
                
                # Cập nhật b (Eq. 25)
                b[p] -= 1
                b[q] += 1
                
                # Cập nhật term (Eq. 26 - sử dụng mẹo ở dòng 281)
                term[p] = term_sub_num[p] # Đã được tính toán trước
                term[q] = term_add_num[q] # Đã được tính toán trước
                
        return F_new

    def fit(self, X_data, y_data):
        """
        Huấn luyện mô hình UROSC trên dữ liệu.
        (ĐÃ TỐI ƯU HÓA BỘ NHỚ ĐỂ TRÁNH TẠO MA TRẬN (n, n))
        """
        
        # 1. Xử lý dữ liệu đầu vào
        if isinstance(X_data, pd.DataFrame):
            X = X_data.values
        else:
            X = np.asarray(X_data)
            
        if isinstance(y_data, pd.Series):
            y = y_data.values
        else:
            y = np.asarray(y_data)

        y = np.nan_to_num(y, nan=-1)

        n, d = X.shape  # n = n_samples, d = n_features
        c = self.n_clusters
        gamma = self.gamma
        
        # Chuyển X sang ký hiệu của bài báo: X_p (d x n)
        X_p = X.T

        # 2. Xác định các chỉ số (index)
        labeled_idx = np.where(y >= 0)[0]
        unlabeled_idx = np.where(y < 0)[0]
        n_l = len(labeled_idx)
        n_u = len(unlabeled_idx)
        
        if n_l == 0:
            print("Cảnh báo: Không có dữ liệu được dán nhãn. Chạy ở chế độ không giám sát.")
        
        # 3. Khởi tạo (Algorithm 2, Step 2)
        Q = np.eye(c)
        F = np.zeros((n, c), dtype=float)
        
        if n_l > 0:
            F[labeled_idx, y[labeled_idx].astype(int)] = 1.0
            
        rand_labels = np.random.randint(0, c, size=n_u)
        F[unlabeled_idx, rand_labels] = 1.0
        
        # 4. Tiền tính toán các ma trận không đổi
        I_d = np.eye(d)
        ones_n = np.ones((n, 1))
        
        # --- TÍNH S_t (Ma trận tán xạ) HIỆU QUẢ ---
        # S_t = X_p * H * X_p^T = X_p * (I - 1/n * 1_n * 1_n^T) * X_p^T
        # S_t = (X_p * X_p^T) - (1/n) * (X_p * 1_n) * (1_n^T * X_p^T)
        
        # Xp_XpT là (d, d) - nhỏ
        Xp_XpT = X_p @ X_p.T  
        
        # Xp_ones là (d, 1) - nhỏ
        Xp_ones = X_p @ ones_n
        
        # S_t là (d, d) - nhỏ. Phép toán này tránh được ma trận (n, n)
        S_t = Xp_XpT - (1/n) * (Xp_ones @ Xp_ones.T)

        try:
            # (S_t + γI)^(-1/2) - Kích thước (d, d) - nhỏ
            S_t_reg_inv_sqrt = self._matrix_sqrt_inv(S_t + gamma * I_d)
        except np.linalg.LinAlgError:
            print("Lỗi: Không thể tính S_t_reg_inv_sqrt. Ma trận có thể suy biến.")
            return self

        # --- Tính X_p * H (ma trận đặc trưng đã tâm hóa) HIỆU QUẢ ---
        # X_p_H = X_p * (I - 1/n * 1_n * 1_n^T)
        # X_p_H = X_p - (1/n) * (X_p * 1_n) * 1_n^T
        # X_p_H = X_p - mean_vec_p * 1_n^T
        
        # mean_vec_p là (d, 1) - nhỏ
        mean_vec_p = (1/n) * Xp_ones
        
        # X_p_H là (d, n) - lớn nhưng có thể quản lý được (30, 284807)
        # Phép trừ này (broadcasting) tương đương với (mean_vec_p @ ones_n.T)
        X_p_H = X_p - mean_vec_p 

        # 5. Vòng lặp tối ưu (Algorithm 2, Step 3)
        for i in range(self.max_iter):
            F_old = F.copy()
            
            # --- Cập nhật W và b ---
            # (Giữ F và Q cố định)
            
            # B = (S_t + γI)^(-1/2) * (X_p * H) * F * Q
            # Kích thước: (d,d) @ (d,n) @ (n,c) @ (c,c) -> (d, c)
            B = S_t_reg_inv_sqrt @ X_p_H @ F @ Q
            
            # SVD trên B (kích thước d, c - nhỏ)
            U, s, V_t = np.linalg.svd(B, full_matrices=False)
            
            # Cập nhật W (kích thước d, c - nhỏ)
            W = S_t_reg_inv_sqrt @ U @ V_t
            
            # Cập nhật b (kích thước c, 1 - nhỏ)
            b = (1/n) * (Q.T @ F.T - W.T @ X_p) @ ones_n
            
            # --- Cập nhật F (Algorithm 2, Step 8) ---
            # (Giữ W, b, Q cố định)
            
            # Z = X * W + 1_n * b^T (kích thước n, c)
            Z = X @ W + ones_n @ b.T
            
            # Gọi Algorithm 1 (cập nhật F từng hàng)
            F = self._update_F(Z, F, labeled_idx, unlabeled_idx, c)

            # --- Cập nhật Q (Algorithm 2, Step 9) ---
            # (Giữ W, b, F cố định)
            
            F_T_F = F.T @ F # (c, c)
            F_T_Z = F.T @ Z # (c, c)
            reg = 1e-6 * np.eye(c)
            
            try:
                Q = np.linalg.solve(F_T_F + reg, F_T_Z) # (c, c)
            except np.linalg.LinAlgError:
                # print(f"Vòng lặp {i}: Ma trận F^T*F suy biến. Sử dụng Q cũ.")
                Q = self.Q_ if self.Q_ is not None else np.eye(c)

            # --- Kiểm tra hội tụ (Algorithm 2, Step 10) ---
            diff = np.sum(np.abs(F - F_old))
            if diff < self.tol:
                print(f"Hội tụ sau {i+1} vòng lặp.")
                break
        
        if i == self.max_iter - 1:
            print(f"Đã đạt số vòng lặp tối đa ({self.max_iter}) mà chưa hội tụ.")

        # 6. Lưu kết quả
        self.W_ = W
        self.b_ = b
        self.Q_ = Q
        self.labels_ = np.argmax(F, axis=1)
        
        return self

    def predict(self, X_data=None):
        """
        Trả về các nhãn cụm đã được gán sau khi fit.
        Lưu ý: Thuật toán này gán nhãn cho dữ liệu huấn luyện (transductive).
        """
        if self.labels_ is None:
            raise ValueError("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước.")
        return self.labels_
    


#    def write_report_urosc(alg, process_time, step, X, labels):
#         kqdg = [
#             alg,
#             wdvl(process_time, n=2),
#             str(step),
#             wdvl(dunn(X, labels)),
#             wdvl(davies_bouldin(X, labels)),
#             'NA',  # PC
#             'NA',  # PE
#             'NA',  # XB
#             'NA',  # CE
#             wdvl(silhouette(X, labels)),
#             'NA',  # FHV
#             wdvl(f1_score(numeric_labels, labels)),
#             wdvl(accuracy_score(numeric_labels, labels)),
#         ]
#         return SPLIT.join(kqdg)

# from dataloader import *

# FILE_DU_LIEU = "semion/semeion.data"   
# TEN_COT_NHAN = None            
# PERCENTAGE_LABELED = 0.3       

# X_data, y_semi, y_true, SO_CUM = load_dataset(
#     filepath=FILE_DU_LIEU,
#     percentage_labeled=PERCENTAGE_LABELED,
#     label_column=TEN_COT_NHAN
# )
# try:
    
#     print("Bắt đầu huấn luyện mô hình UROSC...") 
#     # Cấu hình tham số Gamma
#     THAM_SO_GAMMA = 1.0  # Bạn có thể cần thử nghiệm giá trị này  
#     # Khởi tạo mô hình
#     urosc_model = UROSC(n_clusters=SO_CUM, 
#                         gamma=THAM_SO_GAMMA, 
#                         max_iter=3)
    
#     # Huấn luyện mô hình trên X và y_semi
#     urosc_model.fit(X_data, y_semi)
#     # Lấy kết quả dự đoán
#     predicted_labels = urosc_model.predict()
#     print("Huấn luyện hoàn tất.")
#     print("-" * 30)
#     # --- 4. XEM KẾT QUẢ VÀ ĐÁNH GIÁ ---
#     # So sánh kết quả dự đoán với nhãn thật (y_true)
#     # Lấy chỉ số của các điểm *không* được dán nhãn
#     unlabeled_indices_mask = (y_semi == -1)
#     # Tính độ chính xác trên các điểm *chưa được dán nhãn*
#     acc_unlabeled = accuracy_score(
#         y_true[unlabeled_indices_mask], 
#         predicted_labels[unlabeled_indices_mask]
#     )
#     # Tính độ chính xác tổng thể
#     acc_total = accuracy_score(y_true, predicted_labels)
    
#     print("--- ĐÁNH GIÁ HIỆU SUẤT ---")
#     print(f"Độ chính xác tổng thể (trên 100% dữ liệu): {acc_total:.4f}")
#     print(f"Độ chính xác trên các điểm chưa dán nhãn (90% dữ liệu): {acc_unlabeled:.4f}")
    
#     print("\nSo sánh (20 mẫu đầu):")
#     print(pd.DataFrame({
#         'NhanThat (y_true)': y_true[:20],
#         'NhanBanGiamSat (y_semi)': y_semi[:20],
#         'DuDoan (predicted)': predicted_labels[:20]
#     }))


# except FileNotFoundError:
#     print(f"LỖI: Không tìm thấy file '{FILE_DU_LIEU}'.")
#     print("Hãy đảm bảo bạn đã lưu file dữ liệu và đặt tên đúng.")
# except KeyError:
#     print(f"LỖI: Không tìm thấy cột '{TEN_COT_NHAN}' trong file.")
#     print(f"Các cột tìm thấy là: {list(X_data.columns)}")
# except Exception as e:
#     print(f"Đã xảy ra lỗi không xác định: {e}")

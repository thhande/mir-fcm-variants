import numpy as np
from scipy.spatial.distance import cdist
class FCM:
    
    # hàm khởi tạo
    def __init__(self, c: int, m: int = 2, max_iter: int = 1000, eps: float = 1e-5):
        self.c=c
        self.m=m
        self.max_iter=max_iter
        self.eps=eps
        self.U=None
        self.V=None
        self.process_time=0
        
    # khởi tạo ma trận thành viên ngẫu nhiên
    def initialize_U(self, data : np.ndarray)-> np.ndarray:
        n=data.shape[0] #lấy số lượng điểm dữ liệu
        np.random.seed(seed=42)
        U = np.random.rand(n, self.c)
        U = U/np.sum(U, axis=1, keepdims=True)
        return U
    
    #tính tâm cụm
    def calculate_V(self, data: np.ndarray) -> np.ndarray:
        um = self.U ** self.m
        return (um.T @ data) / division_by_zero(np.sum(um.T, axis=1, keepdims=True))
    
    #Cập nhật ma trận thành viên
    def update_membership_matrix(self, data: np.ndarray) -> np.ndarray: 
       # Tính khoảng cách gốc
        distance = cdist(data, self.V, metric='euclidean')

        # Chặn dưới bằng epsilon để né số 0, sau đó mới mũ
        distance = np.fmax(distance, np.finfo(float).eps) ** (2 / (self.m - 1))
        D = [distance[:, j] for j in range(self.c)]
        numerator = 1 / np.array(D)
        denominator = np.sum(numerator, axis=0)
        U = numerator / division_by_zero(denominator)
        return np.squeeze(U).T

    def fit(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        import time
        self.U = self.initialize_U(data)
        start_time = time.time()
        
        for i in range(self.max_iter):
            U_old = self.U.copy()
            self.V = self.calculate_V(data)
            self.U = self.update_membership_matrix(data)
            if np.linalg.norm(self.U - U_old) < self.eps:
                break
        end_time = time.time()
        self.process_time = end_time - start_time
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, i + 1  # Trả về số vòng lặp thực tế
    
    def get_labels(self):
        """get the label."""
        return np.argmax(self.U, axis=1)
    

def division_by_zero(data: np.ndarray | float) -> np.ndarray | float:
        if isinstance(data, np.ndarray):
            data[data == 0] = np.finfo(float).eps
            return data
        return np.finfo(float).eps if data == 0 else data
    
        

# if __name__ == "__main__":
#     data, class_data = docDuLieu()
#     print('Dữ liệu:\n', data)
#     # print('Nhãn dữ liệu:\n', data_class)
    
    
#     from utility import round_float, extract_labels
#     # from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
#     from validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie,classification_entropy,silhouette,f1_score,hypervolume,accuracy_score

#      # Chuyển đổi nhãn từ chuỗi sang số
#     unique_labels = np.unique(class_data)
#     label_to_index = {label: index for index, label in enumerate(unique_labels)}
#     numeric_labels = np.array([label_to_index[label] for label in class_data])
    
#     ROUND_FLOAT = 3
#     EPSILON = 1e-5
#     MAX_ITER = 1000
#     M = 2
#     SEED = 42
#     SPLIT = '\t'
#     # =======================================

#     def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
#         return str(round_float(val, n=n))

#     def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
#         labels = extract_labels(U)  # Giai mo
#         kqdg = [
#             alg,
#             wdvl(process_time, n=2),
#             str(step),
#             wdvl(dunn(X, labels)),  # DI
#             wdvl(davies_bouldin(X, labels)),  # DB
#             wdvl(partition_coefficient(U)),  # PC
#             wdvl(partition_entropy(U)),  # PE
#             wdvl(Xie_Benie(X, V, U)),  # XB
#             wdvl(classification_entropy(U)),  # CE
#             wdvl(silhouette(X, labels)),  # SI
#             wdvl(hypervolume(U)),  # FHV
#             wdvl(f1_score(numeric_labels, labels)),  # F1 - sử dụng numeric_labels thay vì class_data
#             wdvl(accuracy_score(numeric_labels, labels))  # AC - sử dụng numeric_labels thay vì class_data
            
#         ]
#         return SPLIT.join(kqdg)
    
#     fcm=FCM(c=3) # c : số cụm
#     centroids,U,labels,i=fcm.fit(data)
#     print('Tâm cụm:\n', centroids)
    
    
#     titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-', 'CE-', 'SI+', 'FHV+', 'F1+', 'AC+']
#     print(SPLIT.join(titles))
    
#     print(write_report_fcm(alg='FCM', index=0, process_time=fcm.process_time, step=i, X=data, V=centroids, U=U))
  
    
    


#      #Ảnh viễn thám
#     # Đường dẫn đến các file ảnh
#     # images=['100/landsat8_b2_100.tif' ,'100/landsat8_b3_100.tif','100/landsat8_b4_100.tif', '100/landsat8_b5_100.tif']
#     # #images=['100/z20_B2_dn.tif' ,'100/z20_B3_dn.tif','100/z20_B4_dn.tif', '100/z20_B5_dn.tif']
#     # bands = read_img(images)   # trả về (W, H, nBands)
#     # data=docDuLieu()
#     # data, W, H = calculate_data(bands)
#     # print("data: ")
#     # print(data)
#     # #phân cụm FCM
   
#     # show_img([bands[:,:,i] for i in range(bands.shape[2])])  # hiển thị từng band
#     # print("số kênh ảnh là : ")
#     # print(bands)
    
#     # fcm=FCM(c=6)
#     # centroids,U,labels,i=fcm.fit(data)
#     # labels = labels.reshape(-1, 1)
#     # print("Kích thước labels:", labels.shape)
#     # print("W, H:", W, H)
    
    
    
#     # #bộ dữ liệu ảnh viễn thám
   
#     # #in kết quả
#     # # lables=fcm.calculate_labels(U)
#     # print("tâm cụm : ")
#     # print(centroids)
#     # print("Ma trận thành viên : ")
#     # print(U)
#     # print("Cụm dự đoán : ")
#     # print(labels)
#     # print("số bước : ")
#     # print(i)
    
#     # from utility import round_float, extract_labels
#     # # from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
#     # from validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie,classification_entropy,silhouette,f1_score,hypervolume,accuracy_score
#     # # Chuyển đổi nhãn từ chuỗi sang số
#     # # unique_labels = np.unique(class_data)
#     # # label_to_index = {label: index for index, label in enumerate(unique_labels)}
#     # # numeric_labels = np.array([label_to_index[label] for label in class_data])
    
#     # ROUND_FLOAT = 3
#     # EPSILON = 1e-5
#     # MAX_ITER = 1000
#     # M = 2
#     # SEED = 42
#     # SPLIT = '\t'
#     # # =======================================

#     # def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
#     #     return str(round_float(val, n=n))

#     # def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
#     #     labels = extract_labels(U)  # Giai mo
#     #     kqdg = [
#     #         alg,
#     #         wdvl(process_time, n=2),
#     #         str(step),
#     #         # wdvl(dunn(X, labels)),  # DI
#     #         wdvl(davies_bouldin(X, labels)),  # DB
#     #         wdvl(partition_coefficient(U)),  # PC
#     #         wdvl(partition_entropy(U)),  # PE
#     #         wdvl(Xie_Benie(X, V, U)),  # XB
#     #         wdvl(classification_entropy(U)),  # CE
#     #         wdvl(silhouette(X, labels)),  # SI
#     #         wdvl(hypervolume(U)),  # FHV
#     #         # wdvl(f1_score(numeric_labels, labels)),  # F1 - sử dụng numeric_labels thay vì class_data
#     #         # wdvl(accuracy_score(numeric_labels, labels))  # AC - sử dụng numeric_labels thay vì class_data
                
            
#     #     ]
#     #     return SPLIT.join(kqdg)
#     # # In tiêu đề
#     # titles = ['Alg', 'Time', 'Step', 'DB-', 'PC+', 'PE-', 'XB-', 'CE-', 'FHV+', 'SI+']
#     # print(SPLIT.join(titles))
#     # # In kết quả
#     # print(write_report_fcm(alg='FCM ', index=0, process_time=fcm.process_time, step=i, X=data, V=centroids, U=U))
#     # # Visualize kết quả
#     # visualize_img(labels,W,H)
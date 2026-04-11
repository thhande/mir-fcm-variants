from algo.MYFCM import *           
from utility import *
from validity import *                  
import numpy as np  
from algo.newdataloader import *
from algo.CFCM import Dcfcm
from algo.SSFCM import *
from algo.SSCFCM import *
import time
from sklearn.preprocessing import StandardScaler



if  __name__ == "__main__":
    data,class_data = load_data_with_labels('dataset/Iris/data.csv','class') 
    C = 3
    SEMI_DATA_RATIO = 0.2
    unique_labels = np.unique(class_data)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
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
   
    print(
    f"{'Alg':<10}"
    f"{'Time':>10}"
    f"{'Step':>10}"
    f"{'DI+':>10}"
    f"{'DB-':>10}"
    f"{'PC+':>10}"
    f"{'PE-':>12}"
    f"{'XB-':>10}"
    f"{'CE-':>10}"
    f"{'SI+':>10}"
    f'{'FHV+':>10}'
    f'{'F1+':>10}'
    f'{'AC+':>10}'

)
    x = data
    np.random.seed(SEED)
    #run fcm
    fcm = FCM(c = C)
    fv,fu,fl,fs = fcm.fit(data=data)
    
    
    #run ssfcm
    ssfcm = SSFCM(c = C)
    sv,su,sl,ss = ssfcm.fit(data=data,labels=init_semi_data(labels=class_data,ratio=SEMI_DATA_RATIO))
    
    #run cfcm
    unique_labels = np.unique(class_data)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # Trộn index
    
    split_indices = np.array_split(indices, 3) # Chia làm 3 phần
    
    list_datas = [data[idx] for idx in split_indices]
    list_labels = [numeric_labels[idx] for idx in split_indices] # Biến này thay cho sub_datasets_labels

    standard_centroid = np.array([
        data[numeric_labels == c].mean(axis=0) for c in range(C)
    ])

    # 2. Khởi tạo thuật toán
    start_time = time.time()
    dcfcm = Dcfcm(n_clusters=C, m=2, beta=0.5, max_iter=MAX_ITER) 

    # 3. Truyền tâm chuẩn vào hàm fit
    results = dcfcm.fit(list_datas, standard_centroid=standard_centroid) 
    phase2_time = time.time() - start_time


    #run sscfcm
    n_start_time = time.time() 
    sscfcm = SSCFCM(n_clusters=C,max_iter=MAX_ITER)
    sub_labels =[init_semi_data(labels, SEMI_DATA_RATIO) for labels in list_labels]
    sscfcm_steps = sscfcm.fit(sub_datasets=list_datas,sub_labels=sub_labels)
    ss_phase2_time = time.time()-n_start_time 

    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
        labels = np.argmax(U, axis=1)
        aligned_labels = align_labels(true_labels, labels)
        
        return (
            f"{alg_name:<10}"
            f"{process_time:>10.3f}"
            f"{steps:>10}"
            f"{wdvl(dunn(X, labels)):>10}"
            f"{wdvl(davies_bouldin(X, labels)):>10}"
            f"{wdvl(partition_coefficient(U)):>10}"
            f"{wdvl(partition_entropy(U)):>10}"
            f"{wdvl(Xie_Benie(X, V, U)):>10}"
            f"{wdvl(classification_entropy(U)):>10}"
            f"{wdvl(silhouette(X, labels)):>10}"
            f"{wdvl(hypervolume(U)):>10}"
            f"{wdvl(f1_score(true_labels, aligned_labels,'weighted')):>10}"
            f"{wdvl(accuracy_score(true_labels, aligned_labels)):>10}"
        )
        
    for i in range(len(list_datas)):
        X_site = list_datas[i]
        y_site = list_labels[i]
        # ===== FCM =====
        # lấy U toàn cục nhưng chỉ tính metric trên site
        U_fcm_site = fu[split_indices[i]]
        V_fcm = fv
        print(build_row("FCM", X_site, U_fcm_site, V_fcm, y_site, fcm.process_time,steps = fs))
        # ===== SSFCM =====
        U_ss_site = su[split_indices[i]]
        V_ss = sv
        print(build_row("SSFCM", X_site, U_ss_site, V_ss, y_site, ssfcm.process_time,steps = ss))
        # ===== CFCM =====
        site_cf = dcfcm.data_sites[i]
        print(build_row("CFCM",
                        site_cf.local_data,
                        site_cf.U,
                        site_cf.V,
                        y_site,
                       phase2_time,
                       steps = sum(dcfcm.steps)))
        # ===== SSCFCM =====
        site_sscf = sscfcm.data_sites[i]
        print(build_row("SSCFCM",
                        site_sscf.local_data,
                        site_sscf.U,
                        site_sscf.V,
                        y_site,
                        ss_phase2_time,
                        steps = sscfcm_steps))
        print('________________________________________________________________________________________________________________________')
    
    

   



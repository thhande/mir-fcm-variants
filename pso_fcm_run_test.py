from algo.PSO_FCM import *     
from utility import *
from validity import *                  
import numpy as np  
from algo.newdataloader import *
from algo.MYFCM import *
import time

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

    pso_fcm  = PSO_V_FCM(c=C,swarm_size=30,max_iter = 1000)
    pv,pu,pl,ps = pso_fcm.fit(data)
    
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
    
    print(build_row(alg_name='FCM',X = data,U = fu,V = fv,true_labels=numeric_labels,process_time=fcm.process_time,steps = fs)) 
    print(build_row(alg_name='PSO_V_FCM',X = data,U = pu,V = pv,true_labels=numeric_labels,process_time=pso_fcm.process_time,steps = ps))
    
  
        
    
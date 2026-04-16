from algo.PSO_FCM import *     
from utility import *
from validity import *                  
import numpy as np  
from algo.newdataloader import *
from algo.MYFCM import *
import time
from sklearn.preprocessing import StandardScaler
from algo.SSPSO import *
from algo.SSFCM import *

if  __name__ == "__main__":
    data,class_data = load_data_with_labels('dataset/Drybean/data.csv','Class')
    C = 7
    SEMI_DATA_RATIO = 0.3
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
    for _ in range(3):
        np.random.seed(SEED)
        #run fcm
        fcm = FCM(c = C)
        fv,fu,fl,fs = fcm.fit(data=data)

        pso_fcm  = PSO_V_FCM(c=C,swarm_size=30,patience= 30)
        pv,pu,pl,ps = pso_fcm.fit(data=data)

        sspso = SSPSO_dlsBBPSO(
            c=C,
            m=M,
            max_iter=20,          # paper khuyến nghị max_iter=20
            swarm_size=20,        # paper khuyến nghị swarm_size=20
            semi_mode='ssFCM',    # hoặc 'IS' nếu muốn IS-dlsBBPSO
            seed=SEED
        )
        semi_labels = init_semi_data(labels=numeric_labels, ratio=SEMI_DATA_RATIO)
        psv, psu, psl, pss = sspso.fit(
            data=data,
            labels=semi_labels
        )

        ssfcm = SSFCM(c = C)
        sv,su,sl,ss = ssfcm.fit(data=data,labels=init_semi_data(labels=class_data,ratio=SEMI_DATA_RATIO))

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
        print(build_row(alg_name='SSPSO',X = data,U = psu,V = psv,true_labels=numeric_labels,process_time=sspso.process_time,steps = pss))
        print(build_row(alg_name='SSFCM',X = data,U = su,V = sv,true_labels=numeric_labels,process_time=ssfcm.process_time,steps = ss))
        print('__________________________________________________________________________________________________________________________')
  
        
    
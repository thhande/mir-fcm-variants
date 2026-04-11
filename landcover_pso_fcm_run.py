import os
import numpy as np
import matplotlib.pyplot as plt
from xu_ly_du_lieu import read_img, calculate_data, init_semi_data_optimized, align_labels
from doc_du_lieu import prepare_mask_labels

#import the 4 algorithms
from algo.SSFCM import SSFCM
from algo.MYFCM import *
from algo.CFCM import *
from algo.SSCFCM import *
from algo.PSO_FCM import *
from algo.SSPSO import *

#lib to print validity
from utility import *
from validity import *
import random

import time
from sklearn.metrics import confusion_matrix

MAX_ITER = 1000  # số vòng lặp tối đa
SEMI_DATA_RATIO = 0.3 # chỉ số bán giám sát
C = 5 #số cụm
ROUND_FLOAT = 3 # số làm tròn

if __name__ == "__main__":
    # 1. Cấu hình đường dẫn dữ liệu LandCover.ai
    os.environ["KAGGLEHUB_CACHE"] = "D:/Kaggle_Data"    
    import kagglehub
    
    # print("Đang tải dữ liệu LandCover.ai...")
    path = kagglehub.dataset_download("adrianboguszewski/landcoverai")
    
    img_dir = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")
    
    # Lấy file ảnh đầu tiên để xử lý
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith('.tif')]
    random_index = random.randint(0,len(all_images)-1)#get a random image for testing
    target_name = all_images[15] 
    
    image_path = [os.path.join(img_dir, target_name)]
    mask_path = os.path.join(mask_dir, target_name)

    print(f"Ảnh đang xử lý: {target_name}")

    # 2. Đọc và tiền xử lý ảnh 1000x1000 (3 bands RGB)
    bands = read_img(image_path) 
    mask_true = prepare_mask_labels(mask_path) 
    
        
    data, W, H = calculate_data(bands) #
    data = data.astype(float) / 255.0

    # 3. Tạo nhãn bán giám sát (Giữ lại 30%)
    semi_labels, numeric_true = init_semi_data_optimized(mask_true, ratio=SEMI_DATA_RATIO)

    # 4. Chạy 4 thuật toán (4 lớp đối tượng)

    fcm = FCM(c=C, m=2)
    fv, fu, fl, fs = fcm.fit(data=data) 
    pso_fcm =SSPSO_dlsBBPSO(
        c=C,
        m=2,
        max_iter=20,          # paper khuyến nghị max_iter=20
        swarm_size=20,        # paper khuyến nghị swarm_size=20
        semi_mode='ssFCM',    # hoặc 'IS' nếu muốn IS-dlsBBPSO
        seed=42
    )
    pv,pu,pl,ps = pso_fcm.fit(data=data,labels=semi_labels)
  
    # ssfcm = SSFCM(c = C)
    # ss_v,ss_u,ss_l,ss_s = ssfcm.fit(data=data,labels=semi_labels) 

    # Chuẩn bị dữ liệu cho mô hình phân cụm cộng tác (CFCM)

    # 1. Chuẩn hóa nhãn gốc thành các số nguyên tuần tự (0, 1, 2,...) để dễ quản lý
    unique_labels = np.unique(mask_true)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[label] for label in mask_true])

    # 2. Khởi tạo mảng chỉ số (index) và trộn ngẫu nhiên để đảm bảo phân bố dữ liệu đồng đều
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # Trộn index ngẫu nhiên

    # 5. Báo cáo chỉ số đánh giá đầy đủ
    ROUND_FLOAT = 3
    SPLIT = '\t'
    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    # Căn chỉnh nhãn dự đoán để khớp với nhãn thực tế
    




    def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
        labels = np.argmax(U, axis=1)
        aligned_labels = align_labels(true_labels, labels)        
        # Map nhãn phân cụm ngẫu nhiên với nhãn Ground Truth (Thuật toán Hungarian)  
        return (
            f"{alg_name:<10}"
            f"{process_time:>10.3f}"
            f"{int(steps):>10}"
            f"{davies_bouldin(X, labels):>10.3f}"
            f"{partition_coefficient(U):>10.3f}"
            f"{partition_entropy(U):>10.3f}"
            f"{Xie_Benie(X, V, U):>10.3f}"
            f"{classification_entropy(U):>10.3f}"
            f"{hypervolume(U):>10.3f}"
            f"{f1_score(true_labels, aligned_labels, average='weighted'):>10.3f}"
            f"{accuracy_score(true_labels, aligned_labels):>10.3f}"
        )
    print(
    f"{'Alg':<10}"
    f"{'Time':>10}"
    f"{'Step':>10}"
    f"{'DB-':>10}"
    f"{'PC+':>10}"
    f"{'PE-':>12}"
    f"{'XB-':>10}"
    f"{'CE-':>10}"
    f'{'FHV+':>10}'
    f'{'F1+':>10}'
    f'{'AC+':>10}'
)
    print(build_row(alg_name='FCM',X= data,V = fv,U = fu,process_time=fcm.process_time,true_labels=numeric_labels,steps= fs))
    print(build_row(alg_name='PSO_V_FCM',X=data,V = pv,U =pu,true_labels=numeric_labels,process_time=pso_fcm.process_time,steps=ps))

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import numpy as np

    def visualize_clustering_results(img_rgb, ground_truth, results, H, W, class_names, color_map_rgb):
        """
        Hàm hiển thị ảnh gốc, nhãn gốc và kết quả của nhiều thuật toán.
        
        results: List các tuple [(Tên_Alg, U_matrix), ...]
        """
        # 1. Chuẩn bị Colormap
        color_map_normalized = np.array(color_map_rgb) / 255.0
        custom_cmap = ListedColormap(color_map_normalized)
        vmin, vmax = 0, len(class_names) - 1

        # 2. Xử lý ảnh RGB (đảm bảo dải 0-1)
        img_display = img_rgb.copy()
        if img_display.max() > 1.0:
            img_display = img_display / 255.0

        # 3. Tính toán số lượng subplot (Ảnh RGB + GT + các thuật toán + Legend)
        num_algs = len(results)
        total_plots = 2 + num_algs + 1 # RGB + GT + Algs + Legend
        cols = 4 # Số cột cố định
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten()

        # --- Plot 1: Ảnh RGB Gốc ---
        axes[0].imshow(img_display)
        axes[0].set_title("Ảnh RGB Gốc", fontsize=12, fontweight='bold')

        # --- Plot 2: Ground Truth ---
        axes[1].imshow(ground_truth.reshape(H, W), cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title("Nhãn Gốc (Ground Truth)", fontsize=12, color='darkgreen', fontweight='bold')

        # --- Plot các thuật toán ---
        for i, (alg_name, U) in enumerate(results):
            # Lấy nhãn cứng (hard labels) từ ma trận membership U
            labels = np.argmax(U, axis=1)
            # Căn chỉnh nhãn khớp với Ground Truth (sử dụng hàm của bạn)
            aligned_labels = align_labels(ground_truth, labels).reshape(H, W)
            
            ax_idx = i + 2
            axes[ax_idx].imshow(aligned_labels, cmap=custom_cmap, vmin=vmin, vmax=vmax)
            axes[ax_idx].set_title(f"Kết quả {alg_name}", fontsize=12)

        # --- Tạo Legend ở ô cuối cùng ---
        axes[-1].axis('off')
        legend_patches = [mpatches.Patch(color=color_map_normalized[i], label=f"{i}: {class_names[i]}") 
                        for i in range(len(class_names))]
        axes[-1].legend(handles=legend_patches, loc='center', fontsize=12, 
                        title="Lớp đối tượng", frameon=True, shadow=True)

        # Tắt trục cho tất cả các ô
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            if not ax.get_images() and ax != axes[-1]: # Ẩn các ô thừa
                ax.axis('off')

        plt.tight_layout()
        print("Đang hiển thị kết quả trực quan...")
        plt.show()
    # =========================================================================
    # BƯỚC RECONSTRUCT VÀ VẼ ẢNH HIỂN THỊ TRỰC QUAN
    # =========================================================================
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    print("Đang tiến hành ghép dữ liệu và tô màu theo Colormap chuyên dụng...")

    # 1. Sử dụng luôn H và W từ calculate_data() (Sửa lỗi shape câu hỏi trước)
    h_img, w_img = H, W 

    # 3. Lấy nhãn của FCM và SSFCM
    fcm_full_labels = np.argmax(fu, axis=1)
    # ssfcm_full_labels = np.argmax(ss_u, axis=1)

    # 4. Căn chỉnh nhãn (Align Labels) khớp với Ground Truth và reshape về dạng 2D
    # (Đảm bảo lớp Building của FCM cũng màu Nâu đỏ giống GT)
    fcm_aligned = align_labels(numeric_labels, fcm_full_labels).reshape((h_img, w_img))
        # ssfcm_aligned = align_labels(numeric_labels, ssfcm_full_labels).reshape((h_img, w_img))
        
    # =========================================================================
    # 5. ĐỊNH NGHĨA VÀ CHUYỂN ĐỔI CUSTOM COLORMAP (Theo yêu cầu)
    # =========================================================================
    # Danh sách tên các lớp (theo đúng thứ tự chỉ số 0-4)
    # ... (giữ nguyên phần chạy thuật toán bên trên)

    # Gom kết quả vào một danh sách để truyền vào hàm
    clustering_results = [
            ("FCM", fu),
            ("SSPSO_dlsBBPSO", pu)
            # Bạn có thể thêm các thuật toán khác vào đây: ("SSFCM", ss_u), v.v.
        ]

    # Cấu hình màu sắc (giữ nguyên của bạn)
    class_names = ['Background', 'Buildings', 'Woodlands', 'Water', 'Roads']
    color_map_rgb = [
            [0, 0, 0],         # 0: Background
            [165, 42, 42],    # 1: Buildings
            [34, 139, 34],    # 2: Woodlands
            [0, 105, 148],    # 3: Water
            [128, 128, 128]   # 4: Roads
        ]

    # Gọi hàm hiển thị
    visualize_clustering_results(
            img_rgb=bands, 
            ground_truth=numeric_labels, 
            results=clustering_results, 
            H=H, W=W, 
            class_names=class_names, 
            color_map_rgb=color_map_rgb
        )
        
   
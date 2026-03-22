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
    target_name = all_images[10] 
    
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

    ssfcm = SSFCM(c = C)
    ss_v,ss_u,ss_l,ss_s = ssfcm.fit(data=data,labels=semi_labels) 

    # Chuẩn bị dữ liệu cho mô hình phân cụm cộng tác (CFCM)

    # 1. Chuẩn hóa nhãn gốc thành các số nguyên tuần tự (0, 1, 2,...) để dễ quản lý
    unique_labels = np.unique(mask_true)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[label] for label in mask_true])

    # 2. Khởi tạo mảng chỉ số (index) và trộn ngẫu nhiên để đảm bảo phân bố dữ liệu đồng đều
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # Trộn index ngẫu nhiên

    # 3. Chia đều tập dữ liệu và tập nhãn thành 3 phần (đại diện cho 3 nút/tác tử trong CFCM)
    split_indices = np.array_split(indices, 3) # Chia làm 3 mảng index con
    list_datas = [data[idx] for idx in split_indices] # Tạo danh sách chứa 3 tập dữ liệu con
    list_labels = [numeric_labels[idx] for idx in split_indices] # Tạo danh sách chứa 3 tập nhãn con tương ứng


    
    start_time = time.time() # Bắt đầu tính giờ
    dcfcm = Dcfcm(n_clusters=C, m=2, beta=0.11, max_iter=MAX_ITER)
    results = dcfcm.fit(list_datas)
    phase2_time = time.time()-start_time 

    n_start_time = time.time() 
    sscfcm = SSCFCM(n_clusters=C,max_iter=MAX_ITER)
    sub_labels = [init_semi_data_optimized(labels, SEMI_DATA_RATIO)[0] for labels in list_labels]
    sscfcm_steps = sscfcm.fit(sub_datasets=list_datas,sub_labels=sub_labels)
    ss_phase2_time = time.time()-n_start_time 




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

    for i in range(len(list_datas)):
        X_site = list_datas[i]
        y_site = list_labels[i]
        # ===== FCM =====
        # lấy U toàn cục nhưng chỉ tính metric trên site
        U_fcm_site = fu[split_indices[i]]
        V_fcm = fv
        print(build_row("FCM", X_site, U_fcm_site, V_fcm, y_site, fcm.process_time,steps = fs))
        # ===== SSFCM =====
        U_ss_site = ss_u[split_indices[i]]
        V_ss = ss_v
        print(build_row("SSFCM", X_site, U_ss_site, V_ss, y_site, ssfcm.process_time,steps = ss_s))
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
    
# =========================================================================
    # BƯỚC RECONSTRUCT VÀ VẼ ẢNH HIỂN THỊ TRỰC QUAN
    # =========================================================================
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    print("Đang tiến hành ghép dữ liệu và tô màu theo Colormap chuyên dụng...")

    # 1. Sử dụng luôn H và W từ calculate_data() (Sửa lỗi shape câu hỏi trước)
    h_img, w_img = H, W 
    
    # 2. Khôi phục nhãn toàn cục cho CFCM và SSCFCM
    cfcm_full_labels = np.zeros(n_samples, dtype=int)
    sscfcm_full_labels = np.zeros(n_samples, dtype=int)

    for i in range(3):
        cfcm_site_labels = np.argmax(dcfcm.data_sites[i].U, axis=1)
        cfcm_full_labels[split_indices[i]] = cfcm_site_labels
        
        sscfcm_site_labels = np.argmax(sscfcm.data_sites[i].U, axis=1)
        sscfcm_full_labels[split_indices[i]] = sscfcm_site_labels

    # 3. Lấy nhãn của FCM và SSFCM
    fcm_full_labels = np.argmax(fu, axis=1)
    ssfcm_full_labels = np.argmax(ss_u, axis=1)

    # 4. Căn chỉnh nhãn (Align Labels) khớp với Ground Truth và reshape về dạng 2D
    # (Đảm bảo lớp Building của FCM cũng màu Nâu đỏ giống GT)
    fcm_aligned = align_labels(numeric_labels, fcm_full_labels).reshape((h_img, w_img))
    ssfcm_aligned = align_labels(numeric_labels, ssfcm_full_labels).reshape((h_img, w_img))
    cfcm_aligned = align_labels(numeric_labels, cfcm_full_labels).reshape((h_img, w_img))
    sscfcm_aligned = align_labels(numeric_labels, sscfcm_full_labels).reshape((h_img, w_img))
    
    # =========================================================================
    # 5. ĐỊNH NGHĨA VÀ CHUYỂN ĐỔI CUSTOM COLORMAP (Theo yêu cầu)
    # =========================================================================
    # Danh sách tên các lớp (theo đúng thứ tự chỉ số 0-4)
    class_names = ['Background', 'Buildings', 'Woodlands', 'Water', 'Roads']
    
    # Mảng RGB bạn cung cấp
    color_map_rgb = np.array([
        [0, 0, 0],         # 0: Background (Đen)
        [165, 42, 42],    # 1: Buildings (Nâu đỏ)
        [34, 139, 34],    # 2: Woodlands (Xanh rừng)
        [0, 105, 148],    # 3: Water (Xanh biển)
        [128, 128, 128]   # 4: Roads (Xám bê tông)
    ])

    # Chuyển đổi màu từ thang 0-255 sang 0-1 (Matplotlib yêu cầu)
    color_map_normalized = color_map_rgb / 255.0
    
    # Tạo ListedColormap chuyên dụng
    custom_cmap = ListedColormap(color_map_normalized)
    
    # Xác định giới hạn phân vùng màu (0 đến C-1)
    vmin = 0
    vmax = C - 1 

    # =========================================================================
    # 6. VẼ ẢNH VỚI CUSTOM COLORMAP
    # =========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))

    # Xử lý dải màu ảnh gốc (đảm bảo hiển thị đúng 0.0-1.0)
    if bands.max() <= 1.0:
        img_rgb = bands
    else:
        img_rgb = bands / 255.0

    # --- HÀNG TRÊN: Kênh màu đơn, Ảnh gốc, Nhãn gốc (Tô màu chuyên dụng) ---
    axes[0, 0].imshow(img_rgb[:, :, 0], cmap='Reds')
    axes[0, 0].set_title("Kênh Đỏ (Red)", fontsize=14)
    
    axes[0, 1].imshow(img_rgb[:, :, 1], cmap='Greens')
    axes[0, 1].set_title("Kênh Xanh lá (Green)", fontsize=14)
    
    axes[0, 2].imshow(img_rgb[:, :, 2], cmap='Blues')
    axes[0, 2].set_title("Kênh Xanh lam (Blue)", fontsize=14)
    
    axes[0, 3].imshow(img_rgb)
    axes[0, 3].set_title("Ảnh RGB Gốc", fontsize=14)
    
    # Ảnh Nhãn Ground Truth -> Tô màu custom, xác định rõ vmin, vmax
    axes[0, 4].imshow(numeric_labels.reshape(h_img, w_img), cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[0, 4].set_title("Nhãn Gốc (Ground Truth)\n[Tô theo Colormap Custom]", fontsize=14, color='darkgreen', fontweight='bold')

    # --- HÀNG DƯỚI: 4 thuật toán phân cụm (Đều tô theo Colormap Custom) ---
    axes[1, 0].imshow(fcm_aligned, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Kết quả FCM", fontsize=14)
    
    axes[1, 1].imshow(ssfcm_aligned, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Kết quả SSFCM", fontsize=14)
    
    axes[1, 2].imshow(cfcm_aligned, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("Kết quả CFCM", fontsize=14)
    
    axes[1, 3].imshow(sscfcm_aligned, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[1, 3].set_title("Kết quả SSCFCM", fontsize=14)

    # --- BỔ SUNG CHÚ THÍCH MÀU (LEGEND) CHUYÊN NGHIỆP ---
    # Tắt trục tọa độ và vẽ legend ở vị trí subplot bị thừa
    axes[1, 4].axis('off') 
    
    # Tạo các "miếng vá" màu (patches) để làm legend
    legend_patches = []
    for i in range(len(class_names)):
        patch = mpatches.Patch(color=color_map_normalized[i], label=f"{i}: {class_names[i]}")
        legend_patches.append(patch)
        
    # Vẽ legend
    axes[1, 4].legend(handles=legend_patches, loc='center', fontsize=14, title="Bảng màu lớp đối tượng", title_fontsize=15, frameon=True, shadow=True, borderpad=1)

    # Tắt trục tọa độ cho toàn bộ các ảnh để nhìn gọn gàng hơn
    for ax in axes.flatten():
        # Kiểm tra nếu trục chưa bị tắt (trường hợp plot legend)
        if ax.axison:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Điều chỉnh khoảng cách
    
    # In ra một dòng thông báo
    print("Xong! Đang hiển thị ảnh...")
    plt.show()
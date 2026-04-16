







import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from algo.MYFCM import *#my own fcm module
from algo.SSFCM import *
from algo.CFCM import *
from algo.SSCFCM import *
from sklearn.metrics import confusion_matrix
from validity import *
from utility import *
from algo.my_util import *
from algo.PSO_FCM import *
from algo.SSPSO import *

# ==============================
# THAM SỐ
# ==============================

ROUND_FLOAT = 4
SPLIT = '\t'
SEMI_DATA_RATIO = 0.3
file_path = 'mri_data/2387.mat'
cluster_num = 6
MAX_ITER =1000

def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
    return str(round_float(val, n=n))

from utility import round_float
from validity import (
    davies_bouldin,
    partition_coefficient,
    partition_entropy,
    Xie_Benie,
    classification_entropy,
    hypervolume,
    extract_labels
)

from sklearn.metrics import accuracy_score, f1_score

from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_closing,
    binary_fill_holes
)



def init_semi_data_fixed(labels, ratio):
    """
    Tạo dữ liệu bán giám sát:
    - Lấy ratio% dữ liệu của TOÀN ẢNH
    - Đảm bảo mỗi nhãn có ít nhất 1 pixel labeled
    """

    labels = labels.flatten()

    # map label về dạng 0..k
    unique_labels = np.unique(labels)
    label_to_index = {l: i for i, l in enumerate(unique_labels)}
    mapped_labels = np.array([label_to_index[l] for l in labels])

    total_pixels = len(mapped_labels)
    n_labeled = int(total_pixels * ratio)

    # khởi tạo tất cả là unlabeled
    semi_labels = np.full(total_pixels, -1)

    np.random.seed(42)

    labeled_indices = []

    # đảm bảo mỗi nhãn có ít nhất 1 điểm
    for label in np.unique(mapped_labels):
        pos = np.where(mapped_labels == label)[0]
        chosen = np.random.choice(pos, 1)
        labeled_indices.extend(chosen)

    # số điểm còn cần lấy thêm
    remaining = n_labeled - len(labeled_indices)

    if remaining > 0:
        all_indices = np.arange(total_pixels)

        # bỏ các điểm đã chọn
        remaining_pool = np.setdiff1d(all_indices, labeled_indices)

        extra = np.random.choice(remaining_pool, remaining, replace=False)

        labeled_indices.extend(extra)

    labeled_indices = np.array(labeled_indices)

    # gán nhãn
    semi_labels[labeled_indices] = mapped_labels[labeled_indices]

    return semi_labels

# =======================================
# 1. ĐỌC DỮ LIỆU
# =======================================

print("-" * 100)
print("Đang đọc dữ liệu ...")

with h5py.File(file_path, 'r') as f:
    cjdata = f['cjdata']
    img_raw = np.array(cjdata['image']).astype(np.float32)
    true_mask = np.array(cjdata['tumorMask']).flatten().astype(int) # 2 cum
    # print(np.unique(true_mask))
    # exit()

H, W = img_raw.shape


unique_labels = np.unique(true_mask)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
# Giữ nguyên biến này cho toàn bộ ảnh (nếu bạn có dùng ở các bước tái tạo ảnh)
numeric_labels = np.array([label_to_index[label] for label in true_mask]) 


# =======================================
# 2. TIỀN XỬ LÝ
# =======================================

# ----- Normalize -----
data_min = img_raw.min()
data_max = img_raw.max()

if data_max - data_min > 0:
    img_norm = (img_raw - data_min) / (data_max - data_min)
else:
    img_norm = img_raw.copy()

# ----- Smooth -----
img_smooth = gaussian_filter(img_norm, sigma=1)


# =======================================
# 3. OTSU + LOẠI SỌ
# =======================================

def otsu_threshold(image, bins=256):
    hist, bin_edges = np.histogram(image.ravel(), bins=bins)
    hist = hist.astype(float)

    total = image.size
    sum_total = np.dot(hist, bin_edges[:-1])

    sum_bg = 0
    weight_bg = 0
    max_var = 0
    threshold = 0

    for i in range(bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_edges[i]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = bin_edges[i]

    return threshold


t = otsu_threshold(img_smooth)

upper_thresh = np.percentile(img_smooth, 97)
binary = (img_smooth > t) & (img_smooth < upper_thresh)


# =======================================
# 4. LẤY NÃO (CONNECTED COMPONENT)
# =======================================

labeled_array, num_features = label(binary)

component_sizes = np.bincount(labeled_array.ravel())
component_sizes[0] = 0

largest_component = np.argmax(component_sizes)
brain_mask = labeled_array == largest_component

brain_mask = binary_closing(brain_mask, structure=np.ones((5, 5)))
brain_mask = binary_fill_holes(brain_mask)


# =======================================
# 5. CHUẨN BỊ DỮ LIỆU FCM
# =======================================

flat_img = img_smooth.flatten()
valid_mask = brain_mask.flatten()

cluster_data = flat_img[valid_mask].reshape(-1, 1)
cluster_true_labels = true_mask[valid_mask]

print(f"Kích thước ảnh: {img_raw.shape}")
print(f"Số pixel đưa vào FCM: {cluster_data.shape[0]}")


# =======================================
# 6. CHẠY FCM
# =======================================
semi_labels = init_semi_data_fixed(cluster_true_labels,ratio=SEMI_DATA_RATIO)

#run fcm
fcm_model = FCM(c  = cluster_num, m = 2, eps = 1e-5,max_iter=1000)#khởi tạo thuật toán
fv, fu, fl, fs = fcm_model.fit(cluster_data) # lưu kết quả chạy
fcm_process_time = fcm_model.process_time#thời gian chạy của thuật toán

pso_fcm = PSO_V_FCM(c = cluster_num)
pv,pu,pl,ps = pso_fcm.fit(cluster_data)

sspso =  pso_fcm =SSPSO_dlsBBPSO(
        c=cluster_num,
        m=2,
        max_iter=20,          # paper khuyến nghị max_iter=20
        swarm_size=20,        # paper khuyến nghị swarm_size=20
        semi_mode='ssFCM',    # hoặc 'IS' nếu muốn IS-dlsBBPSO
        seed=42
    )
psv,psu,psl,pss = sspso.fit(data=cluster_data,labels=semi_labels)

ssfcm = SSFCM(c = cluster_num)
sv,su,sl,ss = ssfcm.fit(cluster_data,labels = semi_labels)







pred_subset = np.argmax(fu, axis=1)#this part is for coloring the image, i skip it by now
# =======================================
# 7. CHUẨN HÓA THỨ TỰ CỤM (GIẢI QUYẾT HOÁN VỊ)
# =======================================

sorted_indices = np.argsort(fv.flatten())
new_labels = np.zeros_like(pred_subset)

for new_idx, old_idx in enumerate(sorted_indices):
    new_labels[pred_subset == old_idx] = new_idx + 1

pred_subset = new_labels


# =======================================
# 8. TÁI TẠO ẢNH LABEL
# =======================================

full_labels = np.zeros(flat_img.shape, dtype=int)# khởi tạo khung hình trống
full_labels[valid_mask] = pred_subset#đưa về đúng vị trí


label_image_2d = full_labels.reshape(H, W)

# ---- smoothing to remove salt-pepper noise ----
from scipy.ndimage import median_filter, binary_closing
label_image_2d = median_filter(label_image_2d, size=5)


# =======================================
# 9. CHỌN CỤM TUMOR + CỐ ĐỊNH MÀU
# =======================================

best_f1 = 0
best_label = 0

for i in range(cluster_num):
    temp_pred = (label_image_2d.flatten() == (i + 1)).astype(int)
    temp_f1 = f1_score(true_mask, temp_pred)

    if temp_f1 > best_f1:
        best_f1 = temp_f1
        best_label = i + 1

# ---- ÉP tumor thành cụm cao nhất ----
if best_label != cluster_num:
    temp = label_image_2d.copy()

    label_image_2d[label_image_2d == best_label] = cluster_num
    label_image_2d[temp == cluster_num] = best_label

target_label = cluster_num
y_pred_binary = (label_image_2d == target_label).astype(int)


# =======================================
# 10. TÔ MÀU CỐ ĐỊNH
# =======================================

color_map = {
        1: [40, 80, 180],     # Deep blue (CSF)
        2: [180, 180, 180],   # Light gray (GM)
        3: [220, 220, 220],   # Near white (WM)
        4: [150, 200, 150],   # Soft green (tissue)
        5: [200, 200, 120],   # Pale yellow
        6: [220, 30, 30],     # Strong red (Tumor)
    }

seg_image = np.zeros((H, W, 3), dtype=np.uint8)

for lbl, color in color_map.items():
    seg_image[label_image_2d == lbl] = color#gán màu cho mỗi pixel

#=============================
#Print validities
#================================
        

def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
        # Lấy nhãn phân cụm thô (từ 0 đến 5)
        labels = extract_labels(U)
        
        # ==========================================
        # 1. GOM 6 CỤM THÀNH 2 CỤM ĐỂ TÍNH F1 VÀ ACC
        # ==========================================
        best_f1 = 0
        tumor_cluster_idx = 0
        # Quét qua 6 cụm để tìm cụm đại diện cho Khối u (Tumor) tốt nhất
        for c in range(U.shape[1]):
            # Tạo mask nhị phân thử nghiệm cho cụm c
            temp_pred = (labels == c).astype(int)
            # Tính F1 (nhị phân) để xem cụm này khớp với khối u thực tế đến đâu
            # Giả định nhãn 1 trong true_labels là khối u
            temp_f1 = f1_score(true_labels, temp_pred, zero_division=0)
            
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                tumor_cluster_idx = c
                
        # Gom cụm: Cụm được chọn làm Khối u -> 1, tất cả các cụm não/nền còn lại -> 0
        binary_pred_labels = (labels == tumor_cluster_idx).astype(int)

        # ==========================================
        # 2. XUẤT CHỈ SỐ
        # ==========================================
        # Lưu ý: Các chỉ số DB, PC, PE, XB, CE, SI, FHV vẫn truyền 'labels', 'U', 'V' gốc (6 cụm)
        # Các chỉ số F1, AC truyền 'binary_pred_labels' (2 cụm)
        
        return (
            f"{alg_name:<10}"
            f"{process_time:>10.3f}"
            f"{steps:>10}"
            f"{0:>10}" # dunn(X, labels):>10.3f
            f"{davies_bouldin(X, labels):>10.3f}"
            f"{partition_coefficient(U):>10.3f}"
            f"{partition_entropy(U):>10.3f}"
            f"{Xie_Benie(X, V, U):>10.3f}"
            f"{classification_entropy(U):>10.3f}"
            f"{silhouette(X,labels):>10.3f}"
            f"{hypervolume(U):>10.3f}"
            f"{f1_score(true_labels, binary_pred_labels, average='weighted'):>10.3f}"
            f"{accuracy_score(true_labels, binary_pred_labels):>10.3f}"
            f"{dice_score(true_labels, binary_pred_labels):>10.3f}"
            f"{jaccard_score(true_labels, binary_pred_labels):>10.3f}"
        )

#print titles
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
    f'{'DIC':>10}'
    f'{'JAC':>10}'
)

print(build_row("FCM",
                        cluster_data,
                        fu,
                        fv,
                        numeric_labels[valid_mask],
                        fcm_model.process_time,
                        steps = fs))
print(build_row("SSFCM",
                        cluster_data,
                        su,
                        sv,
                        numeric_labels[valid_mask],
                        ssfcm.process_time,
                        steps = ss))
print(build_row("PSO_FCM",
                        cluster_data,
                        pu,
                        pv,
                        numeric_labels[valid_mask],
                        pso_fcm.process_time,
                        steps = ps))
print(build_row("SSPSO",
                        cluster_data,
                        psu,
                        psv,
                        numeric_labels[valid_mask],
                        sspso.process_time,
                        steps = pss))
print('_______________________________________________________')


# =======================================
# 11. HÀM HỖ TRỢ HIỂN THỊ
# =======================================

def get_colored_image(U, V, valid_mask, H, W, true_mask_flat, cluster_num, color_map):
    """
    Xử lý hậu kỳ: Sắp xếp cụm, lọc nhiễu, xác định Tumor và gán màu.
    """
    # 1. Lấy nhãn thô
    pred_subset = np.argmax(U, axis=1)
    
    # 2. Sắp xếp thứ tự cụm theo cường độ tâm cụm (giảm hoán vị)
    sorted_indices = np.argsort(V.flatten())
    new_labels = np.zeros_like(pred_subset)
    for new_idx, old_idx in enumerate(sorted_indices):
        new_labels[pred_subset == old_idx] = new_idx + 1
    
    # 3. Tái tạo ảnh 2D
    full_labels = np.zeros(H * W, dtype=int)
    full_labels[valid_mask] = new_labels
    label_image_2d = full_labels.reshape(H, W)
    
    # 4. Lọc nhiễu (Median filter)
    label_image_2d = median_filter(label_image_2d, size=5)
    
    # 5. Xác định cụm Tumor (F1-score cao nhất) và ép về nhãn cluster_num (Màu đỏ)
    best_f1 = 0
    best_label = 0
    for i in range(1, cluster_num + 1):
        temp_pred = (label_image_2d.flatten() == i).astype(int)
        # true_mask ở đây là mask gốc của toàn bộ ảnh
        temp_f1 = f1_score(true_mask, temp_pred, zero_division=0)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_label = i

    if best_label != cluster_num and best_label != 0:
        temp = label_image_2d.copy()
        label_image_2d[temp == best_label] = cluster_num
        label_image_2d[temp == cluster_num] = best_label

    # 6. Tô màu
    colored_img = np.zeros((H, W, 3), dtype=np.uint8)
    for lbl, color in color_map.items():
        colored_img[label_image_2d == lbl] = color
        
    return colored_img

# =======================================
# 12. HIỂN THỊ KẾT QUẢ TRÊN HÀNG NGANG
# =======================================

# Danh sách kết quả các thuật toán
results = [
    {"name": "FCM", "U": fu, "V": fv},
    {"name": "SSFCM", "U": su, "V": sv},
    {"name": "PSO-FCM", "U": pu, "V": pv},
    {"name": "SSPSO", "U": psu, "V": psv}
]

plt.figure(figsize=(20, 5))

# In thêm ảnh gốc để đối chiếu (tùy chọn)
plt.subplot(1, len(results) + 1, 1)
plt.imshow(img_raw, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

# Vòng lặp in kết quả segmentation
for i, res in enumerate(results):
    img_colored = get_colored_image(
        res["U"], res["V"], valid_mask, H, W, 
        true_mask, cluster_num, color_map
    )
    
    plt.subplot(1, len(results) + 1, i + 2)
    plt.imshow(img_colored)
    plt.title(res["name"])
    plt.axis('off')

plt.tight_layout()
plt.show()
        

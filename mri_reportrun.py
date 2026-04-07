







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

# ==============================
# THAM SỐ
# ==============================

ROUND_FLOAT = 4
SPLIT = '\t'
SEMI_DATA_RATIO = 0.3
file_path = 'mri_data/1159.mat'
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

#run fcm
fcm_model = FCM(c  = cluster_num, m = 2, eps = 1e-5,max_iter=1000)#khởi tạo thuật toán
fv, fu, fl, fs = fcm_model.fit(cluster_data) # lưu kết quả chạy
fcm_process_time = fcm_model.process_time#thời gian chạy của thuật toán

#run ssfcm
ssfcm_model = SSFCM(c = cluster_num, max_iter = 1000)
ss_v,ss_u,ss_l,ss_i = ssfcm_model.fit(data = cluster_data,labels= init_semi_data_fixed(cluster_true_labels,SEMI_DATA_RATIO))#V,U,label, iterations
ssfcm_precess_time = ssfcm_model.process_time


#prepare the data for CFCM and SSCFCM
# =======================================
# CHUẨN BỊ DỮ LIỆU CFCM VÀ SSCFCM
# =======================================

unique_labels = np.unique(true_mask)
label_to_index = {label: index for index, label in enumerate(unique_labels)}

# Giữ nguyên biến này cho toàn bộ ảnh (nếu bạn có dùng ở các bước tái tạo ảnh)
numeric_labels = np.array([label_to_index[label] for label in true_mask]) 

# Tạo biến mới: MAPPING ĐÚNG với kích thước của cluster_data (160,249)
numeric_labels_masked = np.array([label_to_index[label] for label in cluster_true_labels])

n_samples = cluster_data.shape[0]
indices = np.arange(n_samples)
np.random.shuffle(indices) # Trộn index
    
split_indices = np.array_split(indices, 3) # Chia làm 3 phần
    
list_datas = [cluster_data[idx] for idx in split_indices]

# SỬA LỖI 1: Sử dụng biến masked để chia nhãn chuẩn xác cho từng phần dữ liệu
list_labels = [numeric_labels_masked[idx] for idx in split_indices] 


# SỬA LỖI 2: Tính standard_centroid an toàn (Chống lỗi NaN)
standard_centroid = np.zeros((cluster_num, cluster_data.shape[1]))

for c in range(cluster_num):
    mask_c = (numeric_labels_masked == c)
    if np.any(mask_c): 
        # Nếu nhãn c tồn tại trong ground truth -> tính trung bình bình thường
        standard_centroid[c] = cluster_data[mask_c].mean(axis=0)
    else:
        # Nếu nhãn c KHÔNG tồn tại (do mask chỉ có 2 nhãn mà cluster_num = 6)
        # -> Mượn tạm tâm của cụm tương ứng từ thuật toán FCM đã chạy thành công ở trên (biến fv)
        standard_centroid[c] = fv[c]

#run cfcm

start_time = time.time()
dcfcm = Dcfcm(n_clusters=cluster_num, m=2, beta=0.5, max_iter=MAX_ITER) 

# 3. Truyền tâm chuẩn vào hàm fit
results = dcfcm.fit(list_datas, standard_centroid=standard_centroid) 
phase2_time = time.time() - start_time

#run sscfcm
n_start_time = time.time() 
sscfcm = SSCFCM(n_clusters=cluster_num,max_iter=1000)
sub_labels =[init_semi_data_fixed(labels, SEMI_DATA_RATIO) for labels in list_labels]
sscfcm_steps = sscfcm.fit(sub_datasets=list_datas,sub_labels=sub_labels)
ss_phase2_time = time.time()-n_start_time 


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
    0: [180, 180, 180],       # đen → nền
    1: [220, 30, 30],     # đỏ → tumor
}

seg_image = np.zeros((H, W, 3), dtype=np.uint8)

for lbl, color in color_map.items():
    seg_image[label_image_2d == lbl] = color#gán màu cho mỗi pixel



#print report
def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
        labels = extract_labels(U)
        aligned_labels = align_labels(true_labels,labels)
        # Map nhãn phân cụm ngẫu nhiên với nhãn Ground Truth (Thuật toán Hungarian)
        
        return (
            f"{alg_name:<10}"
            f"{process_time:>10.3f}"
            f"{steps:>10}"
            f"{0:>10}"#dunn(X, labels):>10.3f
            f"{davies_bouldin(X, labels):>10.3f}"
            f"{partition_coefficient(U):>10.3f}"
            f"{partition_entropy(U):>10.3f}"
            f"{Xie_Benie(X, V, U):>10.3f}"
            f"{classification_entropy(U):>10.3f}"
            f"{silhouette(X,labels):>10.3f}"
            f"{hypervolume(U):>10.3f}"
            f"{f1_score(true_labels, aligned_labels, average='weighted'):>10.3f}"
            f"{accuracy_score(true_labels, aligned_labels):>10.3f}"
        )

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
            f"{dunn(X, labels):>10.3f}" # dunn(X, labels):>10.3f
            f"{davies_bouldin(X, labels):>10.3f}"
            f"{partition_coefficient(U):>10.3f}"
            f"{partition_entropy(U):>10.3f}"
            f"{Xie_Benie(X, V, U):>10.3f}"
            f"{classification_entropy(U):>10.3f}"
            f"{silhouette(X,labels):>10.3f}"
            f"{hypervolume(U):>10.3f}"
            f"{f1_score(true_labels, binary_pred_labels, average='weighted'):>10.3f}"
            f"{accuracy_score(true_labels, binary_pred_labels):>10.3f}"
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

)

        
for i in range(len(list_datas)):
        X_site = list_datas[i]
        y_site = list_labels[i]
        # ===== FCM =====
        # lấy U toàn cục nhưng chỉ tính metric trên site
        U_fcm_site = fu[split_indices[i]]
        V_fcm = fv
        print(build_row("FCM", X_site, U_fcm_site, V_fcm, y_site, fcm_model.process_time,steps = fs))
        # ===== SSFCM =====
        U_ss_site = ss_u[split_indices[i]]
        V_ss = ss_v
        print(build_row("SSFCM", X_site, U_ss_site, V_ss, y_site, ssfcm_model.process_time,steps = ss_i))
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

# =======================================
# 11. ĐỒNG BỘ MÀU VÀ IN 4 THUẬT TOÁN 1 HÀNG NGANG
# =======================================

def unshuffle_U(model_sites, n_samples, cluster_num, split_indices):
    """Gom U từ các site về đúng vị trí ban đầu"""
    U_global = np.zeros((n_samples, cluster_num))
    for i, site in enumerate(model_sites):
        U_global[split_indices[i]] = site.U
    return U_global

def compute_global_V(U_global, X_data):
    """Tính lại tâm cụm (V) dựa trên U_global để xếp hạng màu chuẩn xác"""
    # Cộng thêm 1e-8 để tránh lỗi chia cho 0
    return np.dot(U_global.T, X_data) / (U_global.sum(axis=0)[:, None] + 1e-8)

def get_synced_rgb_image(U_matrix, V_matrix, valid_mask, flat_shape, H, W, true_mask, cluster_num):
    """Đồng bộ màu và trả về ảnh RGB hoàn chỉnh"""
    # 1. Lấy nhãn thô
    pred_subset = np.argmax(U_matrix, axis=1)
    
    # 2. Xếp hạng nhãn theo độ sáng của tâm cụm (từ tối đến sáng)
    sorted_indices = np.argsort(V_matrix.flatten())
    aligned_labels = np.zeros_like(pred_subset)
    for new_idx, old_idx in enumerate(sorted_indices):
        aligned_labels[pred_subset == old_idx] = new_idx + 1 # Nhãn từ 1 đến C
        
    # 3. Lắp lại vào ảnh 2D
    full_labels = np.zeros(flat_shape, dtype=int)
    full_labels[valid_mask] = aligned_labels
    
    label_image_2d = full_labels.reshape(H, W)

# ---- smoothing to remove salt-pepper noise ----
    from scipy.ndimage import median_filter, binary_closing
    label_image_2d = median_filter(label_image_2d, size=5)


    # from scipy.ndimage import median_filter
    # from scipy.ndimage import median_filter
    # label_image_2d = median_filter(label_image_2d, size=5) # size=5 hoặc 3 tùy độ mịn bạn muốn
    
    # 4. Tìm cụm Khối u (F1 cao nhất) và ép nó thành số 6 (Để luôn tô màu đỏ)
    best_f1 = 0
    best_label = 0
    for i in range(1, cluster_num + 1):
        temp_pred = (label_image_2d == i).astype(int)
        temp_f1 = f1_score(true_mask, temp_pred.flatten())
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_label = i
            
    # Swap nhãn khối u với nhãn số 6 (cluster_num)
    if best_label != cluster_num:
        temp = label_image_2d.copy()
        label_image_2d[label_image_2d == best_label] = cluster_num
        label_image_2d[temp == cluster_num] = best_label
        
    # 5. Đổ màu chuẩn theo thư viện từ điển (Color Map)

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
        seg_image[label_image_2d == lbl] = color
        
    return seg_image




# --- GOM DỮ LIỆU U VÀ TÍNH LẠI TÂM V CHO TỪNG THUẬT TOÁN ---
U_FCM = fu 
V_FCM = fv

U_SSFCM = ss_u 
V_SSFCM = ss_v

U_CFCM = unshuffle_U(dcfcm.data_sites, n_samples, cluster_num, split_indices)
V_CFCM = compute_global_V(U_CFCM, cluster_data)

U_SSCFCM = unshuffle_U(sscfcm.data_sites, n_samples, cluster_num, split_indices)
V_SSCFCM = compute_global_V(U_SSCFCM, cluster_data)

algorithms = {
    "FCM": (U_FCM, V_FCM),
    "SSFCM": (U_SSFCM, V_SSFCM),
    "CFCM": (U_CFCM, V_CFCM),
    "SSCFCM": (U_SSCFCM, V_SSCFCM)
}

# --- VẼ LÊN 1 HÀNG NGANG DUY NHẤT ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # 1 hàng, 4 cột

for ax, (alg_name, (U_mat, V_mat)) in zip(axes, algorithms.items()):
    # Trích xuất ảnh RGB đã được đồng bộ màu
    rgb_img = get_synced_rgb_image(
        U_mat, V_mat, valid_mask, flat_img.shape, H, W, true_mask, cluster_num
    )
    
    ax.imshow(rgb_img)
    ax.set_title(alg_name, fontsize=16, fontweight='bold') # Chỉ in tên thuật toán, bỏ F1
    ax.axis('off')

plt.tight_layout()
plt.show()


# =======================================
# 12. VISUALIZE VÀ KIỂM TRA GROUND TRUTH & SEMI-SUPERVISED LABLES
# =======================================

def visualize_ground_truth_and_semi(true_mask, H, W, ratio):
    print("-" * 100)
    print("Đang tạo ảnh visualize kiểm tra nhãn và màu...")
    
    # 1. Đưa true_mask về lại dạng 2D
    true_mask_2d = true_mask.reshape(H, W)
    
    # 2. Sinh nhãn bán giám sát trên toàn bộ ảnh (để test)
    # init_semi_data_fixed sẽ trả về mảng 1D có kích thước bằng true_mask
    semi_labels_full = init_semi_data_fixed(true_mask, ratio)
    print("Unique:", np.unique(semi_labels_full))
    print("Count:", np.unique(semi_labels_full, return_counts=True))
    print("Total pixels:", len(semi_labels_full))
    print("Labeled pixels:", np.sum(semi_labels_full != -1))

    semi_labels_2d = semi_labels_full.reshape(H, W)
    
    # 3. Chuẩn bị bảng màu kiểm tra
    # Lấy đúng mã màu Đỏ của Tumor (nhãn số 6) từ color_map của bạn
    tumor_color = [220, 30, 30] 
    
    # Khởi tạo ảnh RGB nền đen
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    semi_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Xác định giá trị thực của Tumor trong file .mat (thường là 1)
    tumor_index = 1 if len(np.unique(true_mask)) > 1 else 0

    # --- TÔ MÀU GROUND TRUTH ---
    # Chỉ tô màu khối u thành màu đỏ, còn lại để đen
    gt_rgb[true_mask_2d == tumor_index] = tumor_color

    # --- TÔ MÀU SEMI-SUPERVISED ---
    # -1: Unlabeled -> Tô màu xám đậm [50, 50, 50] để thấy rõ phần chưa gán nhãn
    # 0: Background đã gán nhãn -> Đen [0, 0, 0]
    # tumor_index: Tumor đã gán nhãn -> Đỏ giống Ground Truth
    # semi_rgb[semi_labels_2d == -1] = [50, 50, 50]
    # semi_rgb[semi_labels_2d == 1] = [0, 0, 0]
    # semi_rgb[semi_labels_2d == tumor_index] = tumor_color

    # unlabeled
    semi_rgb[semi_labels_2d == -1] = [50, 50, 50]

    # background
    semi_rgb[semi_labels_2d == 0] = [0, 0, 0]

    # tumor
    semi_rgb[semi_labels_2d == tumor_index] = tumor_color
    
    # 4. Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(gt_rgb)
    axes[0].set_title('Full Label (Ground Truth)\nTumor: Red [220, 30, 30]', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(semi_rgb)
    axes[1].set_title(f'Semi Labels (Ratio={ratio})\nRed=Tumor, Gray=Unlabeled', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Gọi hàm kiểm tra
visualize_ground_truth_and_semi(true_mask, H, W, SEMI_DATA_RATIO)
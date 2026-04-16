import os
import h5py
import csv
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

# ============================================================
# CẤU HÌNH DIỆN TÍCH
# ============================================================
# 0=Background (ngoài não), 1-6 là các cụm mô theo color_map
AREA_CLASS_NAMES = ['Background', 'Brain', 'Tumor']
AREA_ALG_NAMES   = ['FCM', 'SSFCM', 'PSO_FCM', 'SSPSO']
N_AREA_CLASSES   = len(AREA_CLASS_NAMES)  # 3: Background / Brain(1-5) / Tumor(6)
CSV_PATH         = 'mri_area_results.csv'

# ============================================================
# HÀM TÍNH DIỆN TÍCH
# ============================================================

def get_label_image_for_area(U, V, valid_mask_flat, H, W, true_mask_flat, cluster_num):
    """Trả về label_image_2d (H,W) với 0=background, 1..cluster_num=cụm (Tumor=cluster_num)."""
    from scipy.ndimage import median_filter as _mf
    pred   = np.argmax(U, axis=1)
    si     = np.argsort(V.flatten())
    nl     = np.zeros_like(pred)
    for ni, oi in enumerate(si):
        nl[pred == oi] = ni + 1
    full              = np.zeros(H * W, dtype=int)
    full[valid_mask_flat] = nl
    lbl2d             = full.reshape(H, W)
    lbl2d             = _mf(lbl2d, size=5)
    # Ép Tumor về nhãn cluster_num
    best_f1, best_lbl = 0, 0
    from sklearn.metrics import f1_score as _f1
    for i in range(1, cluster_num + 1):
        tp = (lbl2d.flatten() == i).astype(int)
        tf = _f1(true_mask_flat, tp, zero_division=0)
        if tf > best_f1:
            best_f1, best_lbl = tf, i
    if best_lbl not in (0, cluster_num):
        tmp = lbl2d.copy()
        lbl2d[tmp == best_lbl]    = cluster_num
        lbl2d[tmp == cluster_num] = best_lbl
    return lbl2d

def compute_area_pixels_mri(lbl2d):
    """Gộp cụm 1-5 thành Brain, cụm 6 là Tumor."""
    counts = np.zeros(N_AREA_CLASSES, dtype=np.int64)
    counts[0] = np.sum(lbl2d == 0)                          # Background
    counts[1] = np.sum((lbl2d >= 1) & (lbl2d <= 5))        # Brain (gộp 5 cụm mô)
    counts[2] = np.sum(lbl2d == 6)                          # Tumor
    return counts

def compute_gt_area_mri(true_mask_flat, valid_mask_flat, H, W):
    """GT gộp thành 3 lớp: Background / Brain / Tumor."""
    counts    = np.zeros(N_AREA_CLASSES, dtype=np.int64)
    counts[0] = np.sum(~valid_mask_flat)                              # Background (ngoài não)
    counts[1] = np.sum(valid_mask_flat & (true_mask_flat == 0))      # Brain bình thường
    counts[2] = np.sum(valid_mask_flat & (true_mask_flat == 1))      # Tumor
    return counts

# ============================================================
# HÀM IN / CSV / BIỂU ĐỒ DIỆN TÍCH
# ============================================================

def print_area_table(area_per_alg, gt_counts, img_name=""):
    """In bảng pixel + % cho 1 ảnh."""
    CW  = 13
    SEP = "-" * (10 + 8 + CW * N_AREA_CLASSES * 2 + CW)
    print(f"\n  {'─'*80}")
    print(f"  DIỆN TÍCH TỪNG LỚP: {img_name}")
    print(f"  {'─'*80}")
    hdr1 = f"  {'Alg':<10}{'Metric':<8}"
    hdr2 = f"  {'':10}{'':8}"
    for cls in AREA_CLASS_NAMES:
        hdr1 += f"{cls:^{CW*2}}"
        hdr2 += f"{'(px)':^{CW}}{'(%)':^{CW}}"
    hdr1 += f"{'Tổng px':>{CW}}"
    hdr2 += f"{'':>{CW}}"
    print(hdr1); print(hdr2); print(f"  {SEP}")
    for alg_name, counts in list(area_per_alg.items()) + [('GT', gt_counts)]:
        total_px = counts.sum()
        pcts     = counts / total_px * 100 if total_px > 0 else np.zeros(N_AREA_CLASSES)
        row = f"  {alg_name:<10}{'':8}"
        for px, pct in zip(counts, pcts):
            row += f"{px:>{CW},}{pct:>{CW}.1f}"
        row += f"{total_px:>{CW},}"
        print(f"\033[92m{row}\033[0m" if alg_name == 'GT' else row)
    print(f"  {SEP}")

def init_area_csv(csv_path):
    cols   = []
    for c in AREA_CLASS_NAMES:
        cols += [f"{c}_px", f"{c}_pct"]
    header = ["img_name", "algorithm"] + cols + ["total_px"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
    print(f"  [CSV] Khởi tạo: {csv_path}")

def append_area_csv(csv_path, img_name, area_per_alg, gt_counts):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for alg_name, counts in list(area_per_alg.items()) + [("GT", gt_counts)]:
            total_px = counts.sum()
            pcts     = counts / total_px * 100 if total_px > 0 else np.zeros(N_AREA_CLASSES)
            row = [img_name, alg_name]
            for px, pct in zip(counts, pcts):
                row += [int(px), round(float(pct), 2)]
            row.append(int(total_px))
            writer.writerow(row)

def plot_area_bar(area_per_alg, gt_counts, img_name=""):
    """Biểu đồ bar chart diện tích cho 1 ảnh."""
    bar_colors = [
        [0.10, 0.10, 0.10], [0.16, 0.31, 0.71], [0.71, 0.71, 0.71],
        [0.86, 0.86, 0.86], [0.59, 0.78, 0.59], [0.78, 0.78, 0.47],
        [0.86, 0.12, 0.12],
    ]
    all_entries = list(area_per_alg.items()) + [('GT', gt_counts)]
    n = len(all_entries)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Diện tích từng lớp — {img_name}", fontsize=12, fontweight='bold')
    for ax, (alg_name, counts) in zip(axes, all_entries):
        total_px = counts.sum()
        pcts     = counts / total_px * 100 if total_px > 0 else np.zeros(N_AREA_CLASSES)
        bars = ax.bar(AREA_CLASS_NAMES, pcts, color=bar_colors,
                      edgecolor='black', linewidth=0.7)
        ax.set_title(alg_name, fontsize=11, fontweight='bold',
                     color='darkgreen' if alg_name == 'GT' else 'black')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=35)
        ax.set_ylabel("%" if alg_name == all_entries[0][0] else "")
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{pct:.1f}%", ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    safe_name = img_name.replace('.', '_').replace('/', '_')
    out_path  = f"mri_area_{safe_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  [Chart] Đã lưu: {out_path}")
    plt.show()






def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
    labels = extract_labels(U)
    best_f1 = 0
    tumor_cluster_idx = 0
    for c in range(U.shape[1]):
        temp_pred = (labels == c).astype(int)
        temp_f1   = f1_score(true_labels, temp_pred, zero_division=0)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            tumor_cluster_idx = c
    binary_pred_labels = (labels == tumor_cluster_idx).astype(int)
    return (
        f"{alg_name:<10}"
        f"{process_time:>10.3f}"
        f"{steps:>10}"
        f"{0:>10}"
        f"{davies_bouldin(X, labels):>10.3f}"
        f"{partition_coefficient(U):>10.3f}"
        f"{partition_entropy(U):>10.3f}"
        f"{Xie_Benie(X, V, U):>10.3f}"
        f"{classification_entropy(U):>10.3f}"
        f"{silhouette(X, labels):>10.3f}"
        f"{hypervolume(U):>10.3f}"
        f"{f1_score(true_labels, binary_pred_labels, average='weighted'):>10.3f}"
        f"{accuracy_score(true_labels, binary_pred_labels):>10.3f}"
        f"{dice_score(true_labels, binary_pred_labels):>10.3f}"
        f"{jaccard_score(true_labels, binary_pred_labels):>10.3f}"
    )

def otsu_threshold(image, bins=256):
    hist, bin_edges = np.histogram(image.ravel(), bins=bins)
    hist = hist.astype(float)
    total = image.size
    sum_total = np.dot(hist, bin_edges[:-1])
    sum_bg = weight_bg = max_var = threshold = 0
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
            max_var   = var_between
            threshold = bin_edges[i]
    return threshold


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

# ============================================================
# VÒNG LẶP TẤT CẢ FILE .mat TRONG THƯ MỤC mri_data
# ============================================================

all_files = sorted([f for f in os.listdir('mri_data') if f.endswith('.mat')])
total     = len(all_files)
print(f"Tổng số file MRI: {total}")

# Bộ tích luỹ diện tích qua tất cả ảnh
area_accumulator = {'SSFCM': [], 'GT': []}

init_area_csv(CSV_PATH)

for file_idx, fname in enumerate(all_files):
    file_path = os.path.join('mri_data', fname)
    print("-" * 100)
    print(f"[{file_idx+1}/{total}] Đang xử lý: {fname}")
    print("-" * 100)

    # ── Đọc dữ liệu ──────────────────────────────────────────
    try:
        with h5py.File(file_path, 'r') as f:
            cjdata    = f['cjdata']
            img_raw   = np.array(cjdata['image']).astype(np.float32)
            true_mask = np.array(cjdata['tumorMask']).flatten().astype(int)
    except Exception as e:
        print(f"  [SKIP] Lỗi đọc file: {e}")
        continue

    H, W = img_raw.shape

    unique_labels  = np.unique(true_mask)
    label_to_index = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[l] for l in true_mask])

    # ── Tiền xử lý ───────────────────────────────────────────
    try:
        data_min  = img_raw.min()
        data_max  = img_raw.max()
        img_norm  = (img_raw - data_min) / (data_max - data_min) if data_max > data_min else img_raw.copy()
        img_smooth = gaussian_filter(img_norm, sigma=1)

        t            = otsu_threshold(img_smooth)
        upper_thresh = np.percentile(img_smooth, 97)
        binary       = (img_smooth > t) & (img_smooth < upper_thresh)

        labeled_array, _  = label(binary)
        component_sizes   = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        brain_mask        = labeled_array == np.argmax(component_sizes)
        brain_mask        = binary_closing(brain_mask, structure=np.ones((5, 5)))
        brain_mask        = binary_fill_holes(brain_mask)
    except Exception as e:
        print(f"  [SKIP] Lỗi tiền xử lý: {e}")
        continue

    flat_img            = img_smooth.flatten()
    valid_mask          = brain_mask.flatten()
    cluster_data        = flat_img[valid_mask].reshape(-1, 1)
    cluster_true_labels = true_mask[valid_mask]
    semi_labels         = init_semi_data_fixed(cluster_true_labels, ratio=SEMI_DATA_RATIO)

    print(f"  Kích thước ảnh: {img_raw.shape} | Pixel FCM: {cluster_data.shape[0]}")

    # ── Chạy 4 thuật toán ────────────────────────────────────
    try:
        ssfcm = SSFCM(c=cluster_num)
        sv, su, sl, ss = ssfcm.fit(cluster_data, labels=semi_labels)
    except Exception as e:
        print(f"  [SKIP] Lỗi chạy thuật toán: {e}")
        continue

    # ── In metric ────────────────────────────────────────────
    # print(
    #     f"{'Alg':<10}{'Time':>10}{'Step':>10}{'DI+':>10}"
    #     f"{'DB-':>10}{'PC+':>10}{'PE-':>12}{'XB-':>10}"
    #     f"{'CE-':>10}{'SI+':>10}{'FHV+':>10}{'F1+':>10}"
    #     f"{'AC+':>10}{'DIC':>10}{'JAC':>10}"
    # )
    # print(build_row("SSFCM", cluster_data, su, sv, numeric_labels[valid_mask], ssfcm.process_time, ss))
    print('_' * 100)

    # ── Tính diện tích SSFCM ─────────────────────────────────
    area_per_alg = {}
    for _alg_name, (_U, _V) in [('SSFCM', (su, sv))]:
        _lbl2d = get_label_image_for_area(_U, _V, valid_mask, H, W, true_mask, cluster_num)
        area_per_alg[_alg_name] = compute_area_pixels_mri(_lbl2d)
        area_accumulator[_alg_name].append(area_per_alg[_alg_name])

    gt_area = compute_gt_area_mri(true_mask, valid_mask, H, W)
    area_accumulator['GT'].append(gt_area)

    print_area_table(area_per_alg, gt_area, img_name=fname)
    append_area_csv(CSV_PATH, fname, area_per_alg, gt_area)
    print(f"  [CSV] Đã ghi [{file_idx+1}/{total}]: {fname}")

# ============================================================
# TỔNG KẾT SAU KHI CHẠY HẾT
# ============================================================

n_processed = len(area_accumulator['GT'])
print(f"\nĐã xử lý thành công: {n_processed}/{total} file")

for key in area_accumulator:
    area_accumulator[key] = np.array(area_accumulator[key])  # (n, N_CLASSES)

# ── Ghi summary vào CSV ──────────────────────────────────────
def save_summary_to_csv(csv_path, area_accumulator):
    cols = []
    for c in AREA_CLASS_NAMES:
        cols += [f"{c}_px", f"{c}_pct"]
    header = ["img_name", "algorithm"] + cols + ["total_px"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow(header)
        for alg_name in ['SSFCM', 'GT']:
            arr = area_accumulator[alg_name]
            if arr.ndim < 2 or arr.shape[0] == 0:
                continue
            mean_px  = arr.mean(axis=0)
            std_px   = arr.std(axis=0)
            total    = mean_px.sum()
            mean_pct = mean_px / total * 100 if total > 0 else np.zeros(N_AREA_CLASSES)
            row_mean = ["MEAN", alg_name]
            for px, pct in zip(mean_px.tolist(), mean_pct.tolist()):
                row_mean += [round(float(px), 1), round(float(pct), 2)]
            row_mean.append(round(float(total), 1))
            writer.writerow(row_mean)
            row_std = ["STD", alg_name]
            for std in std_px.tolist():
                row_std += [round(float(std), 1), ""]
            row_std.append("")
            writer.writerow(row_std)

save_summary_to_csv(CSV_PATH, area_accumulator)
print(f"[CSV] Đã lưu tổng kết: {CSV_PATH}")

# ── In bảng tổng kết ─────────────────────────────────────────
print("\n" + "=" * 90)
print(f"  THỐNG KÊ DIỆN TÍCH TRUNG BÌNH TRÊN {n_processed} ẢNH MRI")
print("=" * 90)
CW = 13
hdr = f"{'Alg':<10}{'Metric':<12}" + "".join(f"{c:>{CW}}" for c in AREA_CLASS_NAMES) + f"{'Tổng':>{CW}}"
print(hdr)
print("-" * len(hdr))
for alg_name in ['SSFCM', 'GT']:
    arr      = area_accumulator[alg_name]
    mean_px  = arr.mean(axis=0)
    std_px   = arr.std(axis=0)
    mean_pct = mean_px / mean_px.sum() * 100
    print(f"{alg_name:<10}{'Mean(px)':<12}" + "".join(f"{v:>{CW},.0f}" for v in mean_px) + f"{mean_px.sum():>{CW},.0f}")
    print(f"{'':10}{'Std(px)':<12}"          + "".join(f"{v:>{CW},.0f}" for v in std_px))
    print(f"{'':10}{'Mean(%)':<12}"           + "".join(f"{v:>{CW}.2f}" for v in mean_pct) + f"{'100.00':>{CW}}")
    print("-" * len(hdr))
print("=" * 90)

# ── Biểu đồ tổng kết ─────────────────────────────────────────
bar_colors = [
    [0.10, 0.10, 0.10], [0.16, 0.31, 0.71], [0.71, 0.71, 0.71],
    [0.86, 0.86, 0.86], [0.59, 0.78, 0.59], [0.78, 0.78, 0.47],
    [0.86, 0.12, 0.12],
]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Diện tích trung bình trên {n_processed} ảnh MRI (%)", fontsize=13, fontweight='bold')
for ax, alg_name in zip(axes, ['SSFCM', 'GT']):
    arr      = area_accumulator[alg_name]
    mean_pct = arr.mean(axis=0) / arr.mean(axis=0).sum() * 100
    std_pct  = arr.std(axis=0)  / arr.mean(axis=0).sum() * 100
    bars = ax.bar(AREA_CLASS_NAMES, mean_pct, color=bar_colors,
                  edgecolor='black', linewidth=0.7,
                  yerr=std_pct, capsize=4,
                  error_kw={'elinewidth': 1.2, 'ecolor': 'gray'})
    ax.set_title(alg_name, fontsize=12, fontweight='bold',
                 color='darkgreen' if alg_name == 'GT' else 'black')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylabel("%")
    for bar, pct in zip(bars, mean_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("mri_area_summary.png", dpi=150, bbox_inches='tight')
print("[Chart] Đã lưu: mri_area_summary.png")
plt.show()
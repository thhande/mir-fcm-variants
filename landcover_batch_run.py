import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random
import time
import warnings
warnings.filterwarnings('ignore')

from xu_ly_du_lieu import read_img, calculate_data, init_semi_data_optimized, align_labels
from doc_du_lieu import prepare_mask_labels

from algo.SSFCM import SSFCM
from algo.MYFCM import *
from algo.CFCM import *
from algo.SSCFCM import *
from algo.PSO_FCM import *
from algo.SSPSO import *

from utility import *
from validity import *
from sklearn.metrics import confusion_matrix

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
MAX_ITER    = 1000
SEMI_RATIO  = 0.3
C           = 5
ROUND_FLOAT = 3

CLASS_NAMES   = ['Background', 'Buildings', 'Woodlands', 'Water', 'Roads']
COLOR_MAP_RGB = [
    [0,   0,   0  ],   # 0: Background
    [165, 42,  42 ],   # 1: Buildings
    [34,  139, 34 ],   # 2: Woodlands
    [0,   105, 148],   # 3: Water
    [128, 128, 128],   # 4: Roads
]

ALG_NAMES = ['SSFCM']

CSV_PATH = "landcover_area_results.csv"

# ============================================================
# HÀM CSV
# ============================================================

def init_csv(csv_path, class_names):
    """Tạo file CSV mới với header."""
    cls_cols = []
    for cls in class_names:
        cls_cols += [f"{cls}_px", f"{cls}_pct"]
    header = ["img_name", "algorithm"] + cls_cols + ["total_px"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"  [CSV] Khởi tạo file: {csv_path}")


def append_area_to_csv(csv_path, img_name, area_per_alg, gt_counts):
    """Ghi diện tích của 1 ảnh (tất cả thuật toán + GT) vào CSV."""
    all_entries = list(area_per_alg.items()) + [("GT", gt_counts)]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for alg_name, counts in all_entries:
            total_px = counts.sum()
            pcts = counts / total_px * 100 if total_px > 0 else np.zeros(len(counts))
            row = [img_name, alg_name]
            for px, pct in zip(counts, pcts):
                row += [int(px), round(float(pct), 2)]
            row.append(int(total_px))
            writer.writerow(row)


def save_summary_to_csv(csv_path, area_accumulator, class_names):
    """Ghi bảng tổng kết (mean ± std) vào cuối CSV."""
    all_keys = ALG_NAMES + ["GT"]
    cls_cols = []
    for cls in class_names:
        cls_cols += [f"{cls}_px", f"{cls}_pct"]
    header = ["img_name", "algorithm"] + cls_cols + ["total_px"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow(header)
        for alg_name in all_keys:
            arr      = area_accumulator[alg_name]
            mean_px  = arr.mean(axis=0)
            std_px   = arr.std(axis=0)
            total    = mean_px.sum()
            mean_pct = mean_px / total * 100 if total > 0 else np.zeros(len(mean_px))

            row_mean = ["MEAN", alg_name]
            for px, pct in zip(mean_px, mean_pct):
                row_mean += [round(float(px), 1), round(float(pct), 2)]
            row_mean.append(round(float(total), 1))
            writer.writerow(row_mean)

            row_std = ["STD", alg_name]
            for std in std_px:
                row_std += [round(float(std), 1), ""]
            row_std.append("")
            writer.writerow(row_std)


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def normalize_labels(mask_true):
    """Chuyển nhãn gốc thành chỉ số nguyên 0..C-1."""
    unique_labels = np.unique(mask_true)
    label_to_index = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return np.array([label_to_index[lbl] for lbl in mask_true])


def compute_area_pixels(aligned_labels, n_classes=C):
    """
    Đếm số pixel của từng lớp trong kết quả phân cụm đã align.
    Trả về mảng shape (n_classes,) — đơn vị: pixel.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for cls in range(n_classes):
        counts[cls] = np.sum(aligned_labels == cls)
    return counts


def compute_area_percent(pixel_counts):
    """Chuyển số pixel thành % so với tổng ảnh."""
    total = pixel_counts.sum()
    if total == 0:
        return np.zeros_like(pixel_counts, dtype=float)
    return pixel_counts / total * 100.0


def build_row(alg_name, X, U, V, true_labels, process_time, steps=0):
    labels = np.argmax(U, axis=1)
    aligned_labels = align_labels(true_labels, labels)
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
        f"{kappa_score(true_labels, aligned_labels):>10.3f}"
    )


def print_header():
    print(
        f"{'Alg':<10}"
        f"{'Time':>10}"
        f"{'Step':>10}"
        f"{'DB-':>10}"
        f"{'PC+':>10}"
        f"{'PE-':>12}"
        f"{'XB-':>10}"
        f"{'CE-':>10}"
        f"{'FHV+':>10}"
        f"{'F1+':>10}"
        f"{'AC+':>10}"
        f"{'KA':>10}"
    )


def visualize_clustering_results(img_rgb, ground_truth, results,
                                  H, W, class_names, color_map_rgb,
                                  img_name=""):
    """2 hàng x 5 cột: Row1=RGB+GT+R+G+B, Row2=4 alg+Legend."""
    color_map_normalized = np.array(color_map_rgb) / 255.0
    custom_cmap = ListedColormap(color_map_normalized)
    vmin, vmax = 0, len(class_names) - 1

    img_display = img_rgb.copy().astype(float)
    if img_display.max() > 1.0:
        img_display /= 255.0

    R_ch = img_display[:, :, 0]
    G_ch = img_display[:, :, 1]
    B_ch = img_display[:, :, 2]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Ảnh: {img_name}", fontsize=13, fontweight='bold')

    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Ảnh RGB Gốc", fontsize=12, fontweight='bold')

    axes[0, 1].imshow(ground_truth.reshape(H, W),
                      cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Nhãn Gốc (GT)", fontsize=12,
                          color='darkgreen', fontweight='bold')

    axes[0, 2].imshow(R_ch, cmap='Reds')
    axes[0, 2].set_title("Kênh Đỏ (R)", fontsize=11)
    axes[0, 3].imshow(G_ch, cmap='Greens')
    axes[0, 3].set_title("Kênh Lục (G)", fontsize=11)
    axes[0, 4].imshow(B_ch, cmap='Blues')
    axes[0, 4].set_title("Kênh Lam (B)", fontsize=11)

    for i in range(4):
        if i < len(results):
            alg_name, U = results[i]
            labels = np.argmax(U, axis=1)
            aligned = align_labels(ground_truth, labels).reshape(H, W)
            axes[1, i].imshow(aligned, cmap=custom_cmap, vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Kết quả {alg_name}", fontsize=12,
                                  fontweight='bold')
        else:
            axes[1, i].axis('off')

    axes[1, 4].axis('off')
    patches = [mpatches.Patch(color=color_map_normalized[i],
                               label=f"{i}: {class_names[i]}")
               for i in range(len(class_names))]
    axes[1, 4].legend(handles=patches, loc='center', fontsize=11,
                      title="Lớp đối tượng", frameon=True, shadow=True)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# ============================================================
# HÀM IN DIỆN TÍCH TỪNG ẢNH
# ============================================================

def print_area_per_image(img_name, img_idx, total_imgs,
                          area_per_alg, gt_counts, class_names):
    """
    In bảng diện tích (pixel + %) của từng thuật toán và GT cho 1 ảnh.

    area_per_alg : dict { alg_name: np.array(C,) pixel counts }
    gt_counts    : np.array(C,) pixel counts của Ground Truth
    """
    CW   = 13   # column width
    SEP  = "-" * (10 + 8 + CW * len(class_names) * 2 + CW)

    print(f"\n  {'─'*70}")
    print(f"  ĐỐI TƯỢNG | ẢNH [{img_idx}/{total_imgs}]: {img_name}")
    print(f"  {'─'*70}")

    # Header hai dòng: tên lớp (mỗi lớp chiếm 2 cột px + %)
    hdr1 = f"  {'Alg':<10}{'Metric':<8}"
    hdr2 = f"  {'':10}{'':8}"
    for cls in class_names:
        hdr1 += f"{cls:^{CW*2}}"
        hdr2 += f"{'(px)':^{CW}}{'(%)':^{CW}}"
    hdr1 += f"{'Tổng px':>{CW}}"
    hdr2 += f"{'':>{CW}}"
    print(hdr1)
    print(hdr2)
    print(f"  {SEP}")

    all_entries = list(area_per_alg.items()) + [('GT', gt_counts)]
    for alg_name, counts in all_entries:
        total_px = counts.sum()
        pcts     = counts / total_px * 100 if total_px > 0 else np.zeros(len(counts))

        row = f"  {alg_name:<10}{'':8}"
        for px, pct in zip(counts, pcts):
            row += f"{px:>{CW},}{pct:>{CW}.1f}"
        row += f"{total_px:>{CW},}"
        # Đánh dấu GT bằng màu (dùng ANSI nếu terminal hỗ trợ)
        if alg_name == 'GT':
            print(f"\033[92m{row}\033[0m")   # xanh lá
        else:
            print(row)

    print(f"  {SEP}")


# ============================================================
# HÀM IN BẢNG DIỆN TÍCH TRUNG BÌNH
# ============================================================

def print_area_summary(area_accumulator, n_images, class_names):
    """
    area_accumulator: dict { alg_name: np.array shape (n_images, C) }
    In bảng trung bình pixel + % diện tích cho từng lớp.
    """
    print("\n" + "=" * 80)
    print(f"  THỐNG KÊ DIỆN TÍCH TRUNG BÌNH TRÊN {n_images} ẢNH")
    print("=" * 80)

    # Header
    col_w = 14
    header = f"{'Alg':<10}" + f"{'Lớp':<16}"
    for cls in class_names:
        header += f"{cls:>{col_w}}"
    header += f"{'Tổng (px)':>{col_w}}"
    print(header)
    print("-" * len(header))

    for alg_name in ALG_NAMES:
        arr = area_accumulator[alg_name]           # (n_images, C)
        mean_px  = arr.mean(axis=0)                # (C,)
        std_px   = arr.std(axis=0)                 # (C,)
        mean_pct = mean_px / mean_px.sum() * 100   # (C,)

        # Dòng trung bình pixel
        row_px = f"{alg_name:<10}{'Mean(px)':<16}"
        for v in mean_px:
            row_px += f"{v:>{col_w},.0f}"
        row_px += f"{mean_px.sum():>{col_w},.0f}"
        print(row_px)

        # Dòng std pixel
        row_std = f"{'':10}{'Std(px)':<16}"
        for v in std_px:
            row_std += f"{v:>{col_w},.0f}"
        row_std += f"{'':>{col_w}}"
        print(row_std)

        # Dòng %
        row_pct = f"{'':10}{'Mean(%)' :<16}"
        for v in mean_pct:
            row_pct += f"{v:>{col_w}.2f}"
        row_pct += f"{'100.00':>{col_w}}"
        print(row_pct)

        print("-" * len(header))

    # Ground Truth riêng
    gt_arr = area_accumulator['GT']
    gt_mean_px  = gt_arr.mean(axis=0)
    gt_std_px   = gt_arr.std(axis=0)
    gt_mean_pct = gt_mean_px / gt_mean_px.sum() * 100

    row_px = f"{'GT':<10}{'Mean(px)':<16}"
    for v in gt_mean_px:
        row_px += f"{v:>{col_w},.0f}"
    row_px += f"{gt_mean_px.sum():>{col_w},.0f}"
    print(row_px)

    row_std = f"{'':10}{'Std(px)':<16}"
    for v in gt_std_px:
        row_std += f"{v:>{col_w},.0f}"
    print(row_std)

    row_pct = f"{'':10}{'Mean(%)':<16}"
    for v in gt_mean_pct:
        row_pct += f"{v:>{col_w}.2f}"
    row_pct += f"{'100.00':>{col_w}}"
    print(row_pct)
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ------ 1. Đọc dataset ------
    os.environ["KAGGLEHUB_CACHE"] = "D:/Kaggle_Data"
    import kagglehub

    path     = kagglehub.dataset_download("adrianboguszewski/landcoverai")
    img_dir  = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")

    all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.tif')])
    total = len(all_images)
    print(f"Tổng số ảnh tìm thấy: {total}")

    # Khởi tạo CSV
    init_csv(CSV_PATH, CLASS_NAMES)

    # ------ 2. Khởi tạo bộ tích luỹ diện tích ------
    # area_accumulator[alg_name] = list of np.array(C,) cho mỗi ảnh
    area_accumulator = {name: [] for name in ALG_NAMES + ['GT']}

    # Bảng tổng hợp metric qua các ảnh (tính trung bình cuối)
    # metric_accumulator[alg_name] = list of dict
    metric_accumulator = {name: [] for name in ALG_NAMES}

    print_header()

    # ------ 3. Vòng lặp qua 42 ảnh ------
    for img_idx, img_name in enumerate(all_images):
        print(f"\n{'='*80}")
        print(f"[{img_idx+1}/{total}] Đang xử lý: {img_name}")
        print(f"{'='*80}")

        image_path = [os.path.join(img_dir, img_name)]
        mask_path  = os.path.join(mask_dir, img_name)

        # --- Đọc & tiền xử lý ---
        try:
            bands      = read_img(image_path)
            mask_true  = prepare_mask_labels(mask_path)
            data, W, H = calculate_data(bands)
            data       = data.astype(float) / 255.0
        except Exception as e:
            print(f"  [SKIP] Lỗi đọc ảnh: {e}")
            continue

        # Chuẩn hóa nhãn
        numeric_labels = normalize_labels(mask_true)

        # Nhãn bán giám sát
        semi_labels, _ = init_semi_data_optimized(mask_true, ratio=SEMI_RATIO)

        # --- Chạy SSFCM ---
        try:
            ssfcm = SSFCM(c=C)
            ss_v, ss_u, ss_l, ss_s = ssfcm.fit(data=data, labels=semi_labels)

        except Exception as e:
            print(f"  [SKIP] Lỗi chạy thuật toán: {e}")
            continue

        # --- In metric ảnh hiện tại ---
        print(build_row('SSFCM', data, ss_u, ss_v, numeric_labels, ssfcm.process_time, ss_s))

        # --- Tính diện tích từng lớp (pixel count) ---
        results_map = {
            'SSFCM': ss_u,
        }

        for alg_name, U in results_map.items():
            labels        = np.argmax(U, axis=1)
            aligned_lbl   = align_labels(numeric_labels, labels)
            pixel_counts  = compute_area_pixels(aligned_lbl, n_classes=C)
            area_accumulator[alg_name].append(pixel_counts)

        # Ground Truth
        gt_counts = compute_area_pixels(numeric_labels, n_classes=C)
        area_accumulator['GT'].append(gt_counts)

        # In diện tích từng lớp cho ảnh hiện tại
        area_this_img = {'SSFCM': area_accumulator['SSFCM'][-1]}
        print_area_per_image(
            img_name     = img_name,
            img_idx      = img_idx + 1,
            total_imgs   = total,
            area_per_alg = area_this_img,
            gt_counts    = area_accumulator['GT'][-1],
            class_names  = CLASS_NAMES,
        )
        append_area_to_csv(CSV_PATH, img_name, area_this_img, area_accumulator['GT'][-1])
        print(f"  [CSV] Đã ghi ảnh {img_idx+1}/{total}: {img_name}")

        # # --- Trực quan hoá ảnh hiện tại ---
        # clustering_results = [
        #     ("SSFCM", ss_u),
        # ]
        # visualize_clustering_results(
        #     img_rgb=bands,
        #     ground_truth=numeric_labels,
        #     results=clustering_results,
        #     H=H, W=W,
        #     class_names=CLASS_NAMES,
        #     color_map_rgb=COLOR_MAP_RGB,
        #     img_name=img_name
        # )

    # ------ 4. Tổng kết diện tích trung bình ------
    n_processed = len(area_accumulator['GT'])
    print(f"\nĐã xử lý thành công: {n_processed}/{total} ảnh")

    # Chuyển list -> np.array để tính thống kê
    for key in area_accumulator:
        area_accumulator[key] = np.array(area_accumulator[key])  # (n_images, C)

    save_summary_to_csv(CSV_PATH, area_accumulator, CLASS_NAMES)
    print(f"[CSV] Đã lưu tổng kết vào: {CSV_PATH}")

    print_area_summary(area_accumulator, n_processed, CLASS_NAMES)

    # ------ 5. Vẽ biểu đồ tổng kết diện tích trung bình ------
    color_norm = np.array(COLOR_MAP_RGB) / 255.0

    fig, axes = plt.subplots(1, len(ALG_NAMES) + 1, figsize=(6 * (len(ALG_NAMES) + 1), 6))
    fig.suptitle(f"Diện tích trung bình từng lớp trên {n_processed} ảnh (%)",
                 fontsize=14, fontweight='bold')

    all_keys = ALG_NAMES + ['GT']
    for ax_idx, alg_name in enumerate(all_keys):
        arr      = area_accumulator[alg_name]      # (n, C)
        mean_pct = arr.mean(axis=0) / arr.mean(axis=0).sum() * 100
        std_pct  = arr.std(axis=0)  / arr.mean(axis=0).sum() * 100

        bars = axes[ax_idx].bar(
            CLASS_NAMES, mean_pct,
            color=color_norm,
            edgecolor='black',
            linewidth=0.8,
            yerr=std_pct,
            capsize=4,
            error_kw={'elinewidth': 1.2, 'ecolor': 'gray'}
        )
        axes[ax_idx].set_title(
            alg_name, fontsize=12,
            fontweight='bold',
            color='darkgreen' if alg_name == 'GT' else 'black'
        )
        axes[ax_idx].set_ylabel("Diện tích (%)" if ax_idx == 0 else "")
        axes[ax_idx].set_ylim(0, 100)
        axes[ax_idx].tick_params(axis='x', rotation=30)

        # Ghi số % lên đỉnh mỗi cột
        for bar, pct in zip(bars, mean_pct):
            axes[ax_idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha='center', va='bottom', fontsize=8
            )

    plt.tight_layout()
    plt.savefig("area_summary.png", dpi=150, bbox_inches='tight')
    print("\nĐã lưu biểu đồ tổng kết: area_summary.png")
    plt.show()

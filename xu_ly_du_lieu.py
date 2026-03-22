import numpy as np
import matplotlib.pyplot as plt
import rasterio

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def init_semi_data_optimized(labels, ratio):
    """
    Tạo dữ liệu bán giám sát, tối ưu bộ nhớ (hàng triệu pixel).
        """
    # Lọc ra các nhãn duy nhất để map về int (nếu nhãn gốc chưa từ 0)
    convert_label = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(convert_label)}
    # Dùng list comprehension có thể chậm với dữ liệu lớn, chuyển sang vectorize
    labels_mapped = np.vectorize(label_to_index.get)(labels)
    # Tính số lượng điểm bị ẩn nhãn
    n_not_semi = int(len(labels_mapped) * (1 - ratio))
    np.random.seed(42)
    # TỐI ƯU: Truyền trực tiếp len(labels_mapped) thay vì tạo list comprehension
    unlabel_indices = np.random.choice(len(labels_mapped), size=n_not_semi, replace=False)
    # Tạo bản sao để tránh thay đổi nhãn gốc và gán -1 cho các điểm bị ẩn
    semi_labels = labels_mapped.copy()
    semi_labels[unlabel_indices] = -1
    return semi_labels, labels_mapped # Trả về cả semi_labels và nhãn thực tế (để đánh giá)

def align_labels(y_true, y_pred):
    """
    Map nhãn dự đoán từ thuật toán phân cụm về đúng nhãn thực tế
    dựa trên độ phủ lớn nhất (Hungarian algorithm).
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64).flatten()
    
    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    # Tìm cách ghép cặp (row, col) sao cho tổng các phần tử trên đường chéo là lớn nhất
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Tạo mảng nhãn mới đã được map
    aligned_labels = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[y_pred == j] = i
        
    return aligned_labels

# def read_img(img):
#     for path in img:
#         print("Đang đọc:", path)
#         with rasterio.open(path) as src:
#             band = src.read()  # Đọc toàn bộ phổ, trả về (Bands, W, H)
#             print("Kích thước:", band.shape)
#     data = np.transpose(band, (1, 2, 0))  # (Bands, W, H) -> (W, H, Bands)
#     return data

from rasterio.enums import Resampling

def read_img(img_paths):
    path = img_paths[0]
    # print(f"Đang thực hiện Resampling ảnh từ kích thước gốc về 1000x1000...")
    
    with rasterio.open(path) as src:
        # Đọc và thay đổi kích thước toàn bộ các band
        data = src.read(
            out_shape=(src.count, 1000, 1000),
            resampling=Resampling.bilinear # Giữ chất lượng mượt mà cho ảnh phổ
        )
        
        # print(f"Kích thước mới: {data.shape}")
        # Chuyển vị về (W, H, Bands)
        data = np.transpose(data, (1, 2, 0))
        
    return data
def show_img(bands):
    # Tạo lưới 2x2 để hiển thị 4 ảnh
    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
# Hiển thị từng ảnh lên subplot tương ứng
    for i, ax in enumerate(axs.flatten()):
        # ax.imshow(bands[i].squeeze(), cmap='grey')  # Hiển thị ảnh xám
        ax.set_title(f'Band {i+2} landsat8')
        ax.axis('off')  # Tắt các trục 
# Hiển thị lưới các ảnh
    plt.tight_layout()
    plt.show()
    


def calculate_data(bands):
    # bands hiện tại có dạng (W, H, nBands)
    W, H, Bands = bands.shape
    # print("Kích thước ảnh ban đầu:", W, H, Bands)
    data = bands.reshape(W * H, Bands)
    # print("Kích thước dữ liệu cần phân cụm là:", data.shape)
    return data, W, H
    

def visualize_img(labels, W, H):
    # Tính tổng số pixel của ảnh gốc
    num_pixels = W * H
    
    # --- PHẦN SỬA LỖI RESHAPE ---
    # Kiểm tra nếu số lượng nhãn khớp với số pixel (Trường hợp ảnh 1 band hoặc thuật toán n chiều)
    if labels.size == num_pixels:
        labels_reshaped = labels.reshape(W, H)
        
    # Kiểm tra nếu số lượng nhãn gấp N lần số pixel (Trường hợp BFCM chạy trên ảnh đa kênh)
    elif labels.size > num_pixels and labels.size % num_pixels == 0:
        n_bands = labels.size // num_pixels
        print(f"⚠️ Cảnh báo: Kích thước nhãn ({labels.size}) gấp {n_bands} lần số pixel ({num_pixels}).")
        print("-> Đang xử lý: Lấy nhãn tương ứng của kênh (band) đầu tiên để hiển thị.")
        
        # Dữ liệu gốc trước khi vào BFCM có dạng (W*H, Bands)
        # Nên labels cũng sẽ có thứ tự tương ứng.
        # Ta reshape về (W*H, Bands)
        labels_temp = labels.reshape(num_pixels, n_bands)
        # Cách đơn giản nhất: Lấy kết quả phân cụm của kênh đầu tiên
        # (Hoặc bạn có thể code thêm logic lấy mode - nhãn xuất hiện nhiều nhất trong các kênh)
        labels_reshaped = labels_temp[:, 0]
        # Cuối cùng reshape về dạng ảnh 2D
        labels_reshaped = labels_reshaped.reshape(W, H)
    else:
        # Trường hợp lỗi không xác định
        raise ValueError(f"Lỗi kích thước: Labels có {labels.size} phần tử, nhưng ảnh {W}x{H} chỉ có {num_pixels} pixels.")
    # ----------------------------

    colored_segmented_image = np.zeros((W, H, 3), dtype=np.uint8)
    color_map = np.array([
        [255, 0, 0],      # Buildings (Tòa nhà) - Màu Đỏ rực
        [0, 255, 0],      # Woodlands (Cây cối/Rừng) - Màu Xanh lá rực
        [0, 0, 255],      # Water (Nước) - Màu Xanh dương
        [255, 255, 0],    # Roads (Đường sá) - Màu Vàng chói (tương phản mạnh với cụm kia)
    ], dtype=np.uint8)

    # Đảm bảo labels_reshaped nằm trong phạm vi của color_map
    # (Nếu thuật toán ra nhãn > 5 thì sẽ bị lỗi index, nên cần cẩn thận số cụm c=6)
    for i in range(len(color_map)):
        colored_segmented_image[labels_reshaped == i] = color_map[i]

    plt.figure(figsize=(10, 10))
    plt.imshow(colored_segmented_image)
    plt.title("Hình ảnh phân cụm")
    plt.axis('off')
    plt.show()
    
    
    
    


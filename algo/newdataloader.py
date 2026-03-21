import pandas as pd
import numpy as np



def divide_data_for_collaborative(data, labels, n_data_site):
    #Xáo trộn bọ dữ liệu rồi đem chia đều cho n_data_site
    np.random.seed(42)

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    X_shuffled = data[indices]
    y_shuffled = labels[indices]

    sub_data = np.array_split(X_shuffled, n_data_site, axis=0)
    sub_data_labels = np.array_split(y_shuffled, n_data_site, axis=0)

    return sub_data, sub_data_labels

def load_data(path, dropna=True):
    """
    Load dữ liệu từ file CSV/Excel và trả về numpy array để dùng cho FCM.

    Parameters:
    - path (str): đường dẫn đến file .csv hoặc .xlsx
    - dropna (bool): nếu True thì tự động loại bỏ các dòng có giá trị NaN

    Returns:
    - X (ndarray): dữ liệu dạng numpy array, chỉ gồm các cột numeric
    """


    # đọc file tùy theo phần mở rộng
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("File phải là .csv hoặc .xlsx, đừng chơi mấy định dạng dở hơi.")

    # loại bỏ dòng NaN nếu muốn
    if dropna:
        df = df.dropna()

    # chỉ lấy cột numeric để tránh lỗi FCM
    df_numeric = df.select_dtypes(include=[float, int])

    if df_numeric.shape[1] == 0:
        raise ValueError("Không tìm thấy cột numeric nào để phân cụm.")

    return df_numeric.to_numpy()


def load_data_with_labels(path, label_col="label", dropna=True):
    """
    Load dữ liệu từ file CSV/Excel và tách ra:
    - X: các cột numeric (trừ cột label)
    - y: cột nhãn (label)

    Parameters:
    - path (str): đường dẫn file .csv hoặc .xlsx
    - label_col (str): tên cột nhãn trong file
    - dropna (bool): nếu True thì loại bỏ dòng NaN

    Returns:
    - X (ndarray)
    - y (ndarray)
    """
    import pandas as pd
    import numpy as np

    # đọc file
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("File phải là .csv hoặc .xlsx.")

    # kiểm tra label column
    if label_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn '{label_col}' trong dữ liệu.")

    # xử lý NaN
    if dropna:
        df = df.dropna()

    # tách nhãn
    y = df[label_col].to_numpy()
    df = df.drop(columns=[label_col])

    # lấy cột numeric cho X
    df_numeric = df.select_dtypes(include=[float, int])
    if df_numeric.shape[1] == 0:
        raise ValueError("Không có cột numeric nào để làm feature.")

    X = df_numeric.to_numpy()
    return X, y

def load_data_with_outliers(path,label = "label", p=0.05, F=10, random_state=42):
    X,y = load_data_with_labels(path,label_col=label)
    X = inject_outliers(X,p=p, F=F, random_state=random_state)
    return X,y

def inject_outliers(data, p=0.5, F=10, random_state=42):
    """
    Thêm ngoại lai vào tập dữ liệu dựa trên quy trình mô tả trong hình ảnh.
    
    Tham số:
    ----------
    data : pandas.DataFrame hoặc numpy.ndarray
        Tập dữ liệu đầu vào (N mẫu, d đặc trưng).
    p : float, mặc định = 0.05 (tương đương 5%)
        Tỷ lệ phần trăm các giá trị trong ma trận sẽ bị thay thế bởi ngoại lai.
        (Lưu ý: nhập 0.05 cho 5%, 0.10 cho 10%).
    F : float, mặc định = 10
        Hệ số tỉ lệ (scaling factor) để kiểm soát độ lớn của ngoại lai.
        Theo bài báo, F = 10.
    random_state : int, mặc định = 42
        Hạt giống ngẫu nhiên (seed) để đảm bảo tính tái lập (reproducibility).
        
    Trả về:
    -------
    data_outlier : cùng kiểu với data đầu vào
        Tập dữ liệu đã được thêm ngoại lai.
    """
    
    # Chuyển đổi sang numpy array nếu đầu vào là DataFrame để dễ tính toán
    is_dataframe = False
    if isinstance(data, pd.DataFrame):
        is_dataframe = True
        columns = data.columns
        index = data.index
        X = data.values.copy()
    else:
        X = data.copy()
        
    # Thiết lập hạt giống ngẫu nhiên (Fixed random seed)
    np.random.seed(random_state)
    N, d = X.shape # Số lượng mẫu (samples) và đặc trưng (features)
    
    # Tính số lượng giá trị cần thay thế (p * N * d)
    # Hàm ceil hoặc round để lấy số nguyên
    n_outliers = int(np.ceil(p * N * d))
    
    # Tính Mean (mu) và Standard Deviation (sigma) cho từng đặc trưng (cột)
    mus = np.mean(X, axis=0)
    sigmas = np.std(X, axis=0)
    
    # Chọn ngẫu nhiên các vị trí (indices) trong ma trận để thay thế
    # Chúng ta chọn từ danh sách phẳng (flattened) từ 0 đến N*d - 1
    total_elements = N * d
    flat_indices = np.random.choice(total_elements, n_outliers, replace=False)
    # exit()
    # Chuyển đổi index phẳng sang toạ độ (hàng, cột)
    rows, cols = np.unravel_index(flat_indices, (N, d))
    
    # Áp dụng công thức: outlier_value = mu_j + F * sigma_j
    # Thay thế giá trị gốc bằng giá trị ngoại lai
    # Sử dụng fancy indexing của numpy để thay thế hàng loạt
    X[rows, cols] = mus[cols] + F * sigmas[cols]
    
    # Trả về định dạng đúng như ban đầu
    if is_dataframe:
        return pd.DataFrame(X, columns=columns, index=index)
    
    return X
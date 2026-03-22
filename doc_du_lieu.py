import pandas as pd
import numpy as np
import rasterio
import os

import rasterio

import rasterio
import numpy as np

def prepare_mask_labels(mask_path):
    """
    Đọc file mask .tif và chuyển đổi thành mảng 1D.
    
    Args:
        mask_path (str): Đường dẫn tới file mask .tif
        
    Returns:
        y (numpy.ndarray): Mảng 1D chứa nhãn (labels) của từng pixel.
    """
    with rasterio.open(mask_path) as src:
        # File mask thường chỉ có 1 band, ta đọc band 1
        mask_data = src.read(1)
        
        # Trải phẳng thành mảng 1D
        # Hàm flatten() sẽ làm phẳng mảng 2D (rows, cols) thành 1D (rows * cols)
        y = mask_data.flatten()
        
        return y

def docDuLieu():
#    Đường dẫn đến folder chứa ảnh
    folder_path = "archive/images"
    images = []

    # Lặp qua tất cả các file trong folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.tifF'):  # Chỉ đọc các file .tif
            file_path = os.path.join(folder_path, filename)
            with rasterio.open(file_path) as src:
                band = src.read()  # Đọc kênh đầu tiên
                images.append(band)  # Thêm vào danh sách
    return images  

from rasterio.enums import Resampling

def prepare_mask_labels(mask_path):
    with rasterio.open(mask_path) as src:
        # Thay đổi kích thước mask về 1000x1000
        mask_data = src.read(
            1,
            out_shape=(1000, 1000),
            resampling=Resampling.nearest # Quan trọng: giữ nguyên giá trị nhãn gốc
        )
        return mask_data.flatten()
 

   
   
   


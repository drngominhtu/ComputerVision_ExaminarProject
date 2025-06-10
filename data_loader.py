import numpy as np

def load_data(csv_path):
    """Tải dữ liệu từ file CSV"""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    y = data[:, 0].astype(int)  # Labels (0-25)
    X = data[:, 1:].reshape(-1, 28, 28)  # Ảnh 28x28 pixel
    return X, y

def preprocess_data(X):
    """Chuẩn hóa dữ liệu pixel về [0, 1]"""
    return X / 255.0
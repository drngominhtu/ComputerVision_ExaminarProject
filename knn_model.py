import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        """Huấn luyện mô hình KNN"""
        self.X_train = X_train.reshape(len(X_train), -1)
        self.y_train = y_train
        print(f"Đã train xong mô hình KNN với {len(self.X_train)} mẫu")
    
    def predict(self, X_test):
        """Dự đoán nhãn cho dữ liệu test"""
        print(f"Bắt đầu dự đoán {len(X_test)} mẫu")
        X_test = X_test.reshape(len(X_test), -1)
        predictions = []
        for i, x in enumerate(X_test):
            if i % 10 == 0: 
                print(f"Đang dự đoán mẫu {i}/{len(X_test)}")
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            pred = np.bincount(k_labels).argmax()
            predictions.append(pred)
        return np.array(predictions)
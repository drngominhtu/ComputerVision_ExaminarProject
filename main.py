import numpy as np
from data_loader import load_data, preprocess_data
from knn_model import KNN
from visualization import save_prediction_images
from evaluation import test_with_fixed_dataset, test_with_different_train_sizes, draw_bar_chart_for_accuracy

def main():
    # Tải dữ liệu
    print("Loading data...")
    X_train, y_train = load_data("data/sign_mnist_train.csv")
    X_test, y_test = load_data("data/sign_mnist_test.csv")

    # Chuẩn hóa dữ liệu
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Sử dụng mẫu nhỏ để test ban đầu
    sample_size = 10  # Fix cứng 10 mẫu
    print(f"Dự đoán trên {sample_size} mẫu đầu tiên")

    # Huấn luyện mô hình
    model = KNN(k=5)
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test[:sample_size])

    # Đánh giá
    accuracy = np.mean(y_pred == y_test[:sample_size])
    print(f"Accuracy trên {sample_size} mẫu test: {accuracy*100:.2f}%")

    # Lưu ảnh kết quả
    save_prediction_images(X_test, y_test, y_pred, sample_size)

    # Test với các kích thước mẫu train khác nhau
    test_with_different_train_sizes(X_train, y_train, X_test, y_test)
    
    # Vẽ biểu đồ cột
    draw_bar_chart_for_accuracy(X_train, y_train, X_test, y_test)
    
    # Test với dataset cố định
    test_with_fixed_dataset(model, X_test, y_test)

if __name__ == "__main__":
    main()
    print("Done")
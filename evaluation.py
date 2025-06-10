import numpy as np
from knn_model import KNN
from visualization import plot_line_chart, plot_bar_chart, print_accuracy_table

def test_with_fixed_dataset(model, X_test, y_test):
    """Test với dataset cố định 10 mẫu"""
    sample_size = 10  # Fix cứng 10 mẫu
    print(f"Dự đoán trên {sample_size} mẫu đầu tiên")
    X_test_sample = X_test[:sample_size]
    y_test_sample = y_test[:sample_size]
    y_pred_sample = model.predict(X_test_sample)
    accuracy_sample = np.mean(y_pred_sample == y_test_sample)
    print(f"Accuracy trên {sample_size} mẫu: {accuracy_sample*100:.2f}%")

def test_with_different_train_sizes(X_train, y_train, X_test, y_test):
    """Test với các kích thước mẫu train khác nhau"""
    sample_sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    model = KNN(k=5)
    print("Bắt đầu test với các kích thước mẫu train khác nhau")
    accuracies = []
    
    for size in sample_sizes:
        X_train_sample = X_train[:size]
        y_train_sample = y_train[:size]
        
        model.fit(X_train_sample, y_train_sample)
        y_pred_sample = model.predict(X_test[:10])
        accuracy_sample = np.mean(y_pred_sample == y_test[:10])
        accuracies.append(accuracy_sample * 100)
        
        print(f"Accuracy với {size} mẫu train: {accuracy_sample * 100:.2f}%")
    
    # Hiển thị bảng tổng kết và biểu đồ
    print_accuracy_table(sample_sizes, accuracies)
    plot_line_chart(sample_sizes, accuracies)
    
    return sample_sizes, accuracies

def draw_bar_chart_for_accuracy(X_train, y_train, X_test, y_test):
    """Vẽ biểu đồ cột cho accuracy"""
    sample_sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    model = KNN(k=5)
    print("Bắt đầu vẽ biểu đồ cột accuracy")
    
    accuracies = []
    
    for size in sample_sizes:
        X_train_sample = X_train[:size]
        y_train_sample = y_train[:size]
        
        model.fit(X_train_sample, y_train_sample)
        y_pred_sample = model.predict(X_test[:10])
        accuracy_sample = np.mean(y_pred_sample == y_test[:10])
        accuracies.append(accuracy_sample * 100)
        
        print(f"Accuracy với {size} mẫu train: {accuracy_sample * 100:.2f}%")
    
    # Hiển thị bảng tổng kết và biểu đồ
    print_accuracy_table(sample_sizes, accuracies, "(BAR CHART) ")
    plot_bar_chart(sample_sizes, accuracies)
    
    return sample_sizes, accuracies
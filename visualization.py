import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_prediction_images(X_test, y_test, y_pred, sample_size=10):
    """Lưu ảnh kết quả dự đoán"""
    print("Hiển thị kết quả dự đoán trên 10 ảnh")
    for i in range(min(10, sample_size)):
        img = (X_test[i] * 255).astype(np.uint8)
        true_label = chr(y_test[i] + 65)
        pred_label = chr(y_pred[i] + 65)
        
        # Phóng to ảnh
        img_large = cv2.resize(img, (280, 280))
        cv2.putText(img_large, f"True: {true_label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        cv2.putText(img_large, f"Pred: {pred_label}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        
        # Lưu file
        filename = f"result_img_{i}.png"
        cv2.imwrite(filename, img_large)
        print(f"Đã lưu kết quả vào file {filename}")

def plot_line_chart(sample_sizes, accuracies):
    """Vẽ biểu đồ đường accuracy"""
    plt.figure(figsize=(12, 6))
    plt.plot(sample_sizes, accuracies, marker='o')
    plt.xlabel('Kích thước mẫu train')
    plt.ylabel('Độ chính xác (%)')
    plt.title('Độ chính xác của mô hình KNN theo kích thước mẫu train')
    plt.xticks(sample_sizes, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Thêm text hiển thị accuracy trên biểu đồ
    for i, (size, acc) in enumerate(zip(sample_sizes, accuracies)):
        if i % 3 == 0:  # Chỉ hiển thị một số điểm để tránh quá tải
            plt.annotate(f'{acc:.1f}%', (size, acc), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_bar_chart(sample_sizes, accuracies):
    """Vẽ biểu đồ cột accuracy"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sample_sizes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Kích thước mẫu train')
    plt.ylabel('Độ chính xác (%)')
    plt.title('Độ chính xác của mô hình KNN theo kích thước mẫu train (Bar Chart)')
    plt.xticks(sample_sizes, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Thêm text hiển thị accuracy trên các cột
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if i % 2 == 0:  # Chỉ hiển thị một số cột để tránh quá tải
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def print_accuracy_table(sample_sizes, accuracies, chart_type=""):
    """In bảng tổng kết accuracy"""
    print(f"\n--- BẢNG TỔNG KẾT ACCURACY {chart_type}---")
    print("Kích thước mẫu train | Accuracy (%)")
    print("-" * 35)
    for size, acc in zip(sample_sizes, accuracies):
        print(f"{size:>18} | {acc:>9.2f}")
    
    print(f"\nAccuracy cao nhất: {max(accuracies):.2f}% với {sample_sizes[accuracies.index(max(accuracies))]} mẫu train")
    print(f"Accuracy thấp nhất: {min(accuracies):.2f}% với {sample_sizes[accuracies.index(min(accuracies))]} mẫu train")
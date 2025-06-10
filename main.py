import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
import time
from data_loader import load_data, preprocess_data
from knn_model import KNN
from visualization import save_prediction_images
from evaluation import test_with_fixed_dataset, test_with_different_train_sizes
from detailed_evaluation import plot_confusion_matrix, detailed_classification_report
from real_time_prediction import real_time_hand_sign_detection

def show_confusion_matrix(y_test, y_pred, sample_size):
    """Display Confusion Matrix in separate window"""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test[:sample_size], y_pred)
        class_names = [chr(i + 65) for i in range(max(max(y_test[:sample_size]), max(y_pred)) + 1)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()  # Blocking show
        print("Confusion Matrix displayed successfully")
    except Exception as e:
        print(f"Error displaying Confusion Matrix: {e}")

def show_accuracy_charts(X_train, y_train, X_test, y_test):
    """Display accuracy charts in separate window"""
    try:
        sample_sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        accuracies = []
        
        print("Computing accuracy for different training sizes...")
        for i, size in enumerate(sample_sizes):
            temp_model = KNN(k=5)
            temp_model.fit(X_train[:size], y_train[:size])
            temp_pred = temp_model.predict(X_test[:50])  # Reduce sample for speed
            temp_acc = np.mean(temp_pred == y_test[:50])
            accuracies.append(temp_acc * 100)
            print(f"  {i+1}/{len(sample_sizes)}: Train size {size} -> Accuracy {temp_acc*100:.2f}%")
        
        # Create new figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Line Chart
        ax1.plot(sample_sizes, accuracies, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Sample Size')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy vs Train Size (Line Chart)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Annotation for highest point
        max_idx = np.argmax(accuracies)
        ax1.annotate(f'{accuracies[max_idx]:.1f}%', 
                    (sample_sizes[max_idx], accuracies[max_idx]),
                    textcoords="offset points", xytext=(0,10), ha='center')
        
        # Bar Chart
        bars = ax2.bar(sample_sizes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Training Sample Size')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs Train Size (Bar Chart)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Text on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if i % 2 == 0:  # Show only some bars
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()  # Blocking show
        print("Accuracy Charts displayed successfully")
        return accuracies, sample_sizes
        
    except Exception as e:
        print(f"Error displaying Accuracy Charts: {e}")
        return [], []

def show_class_accuracy(y_test, y_pred, sample_size):
    """Display accuracy by class in separate window"""
    try:
        unique_labels = np.unique(y_test[:sample_size])
        class_accuracies = []
        class_letters = []
        
        for label in unique_labels:
            mask = y_test[:sample_size] == label
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred[mask] == y_test[:sample_size][mask])
                class_accuracies.append(class_accuracy * 100)
                class_letters.append(chr(label + 65))
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(class_letters, class_accuracies, color='lightgreen', edgecolor='darkgreen')
        ax.set_xlabel('Class (Letter)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Class', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Text on bars
        for bar, acc in zip(bars, class_accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()  # Blocking show
        print("Class Accuracy Chart displayed successfully")
        return class_accuracies, class_letters
        
    except Exception as e:
        print(f"Error displaying Class Accuracy Chart: {e}")
        return [], []

def show_statistics_summary(X_train_full, sample_size, accuracy, accuracies, sample_sizes, class_accuracies, class_letters, unique_labels):
    """Display statistics summary in separate window"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(accuracies) > 0 and len(class_accuracies) > 0:
            max_idx = np.argmax(accuracies)
            stats_text = f"""
COMPREHENSIVE STATISTICS - KNN ASL RECOGNITION

================================================================

DATASET INFORMATION:
• Training samples: {len(X_train_full):,}
• Test samples: {sample_size:,}
• Image size: 28 × 28 pixels
• Number of classes: {len(unique_labels)}
• Classes: {', '.join([chr(i + 65) for i in unique_labels])}

MODEL PERFORMANCE:
• Algorithm: K-Nearest Neighbors (KNN)
• K-value: 5
• Overall Accuracy: {accuracy*100:.2f}%
• Best Training Size: {sample_sizes[max_idx]:,} samples
• Best Accuracy: {max(accuracies):.2f}%

CLASS PERFORMANCE:
• Best performing class: {class_letters[np.argmax(class_accuracies)]} ({max(class_accuracies):.1f}%)
• Worst performing class: {class_letters[np.argmin(class_accuracies)]} ({min(class_accuracies):.1f}%)
• Average class accuracy: {np.mean(class_accuracies):.1f}%
• Standard deviation: {np.std(class_accuracies):.1f}%

PERFORMANCE METRICS:
• Training speed: Fast (KNN lazy learning)
• Prediction speed: Real-time capable
• Memory usage: Moderate
• Scalability: Good for small-medium datasets
• Full dataset used: YES ({len(X_train_full):,} samples)

================================================================
            """
        else:
            stats_text = f"""
COMPREHENSIVE STATISTICS - KNN ASL RECOGNITION

Dataset Information:
• Training samples: {len(X_train_full):,}
• Test samples: {sample_size:,}
• Image size: 28 × 28 pixels
• Number of classes: {len(unique_labels)}

Model Performance:
• Overall Accuracy: {accuracy*100:.2f}%
• Algorithm: KNN (k=5)
• Full dataset used: YES
            """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.axis('off')
        ax.set_title('STATISTICS SUMMARY', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()  # Blocking show
        print("Statistics Summary displayed successfully")
        
    except Exception as e:
        print(f"Error displaying Statistics Summary: {e}")

def show_sample_predictions(X_test, y_test, y_pred, sample_size):
    """Display sample predictions in separate window"""
    try:
        n_samples = 12
        sample_indices = np.random.choice(sample_size, n_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            axes[i].imshow(X_test[idx], cmap='gray')
            true_label = chr(y_test[idx] + 65)
            pred_label = chr(y_pred[idx] + 65)
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label} | Pred: {pred_label}', 
                             color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
            
        fig.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()  # Blocking show
        print("Sample Predictions displayed successfully")
        
    except Exception as e:
        print(f"Error displaying Sample Predictions: {e}")

def safe_visualization_display():
    """Display visualization safely"""
    try:
        # Try different backends
        backends = ['TkAgg', 'Qt5Agg', 'Agg']
        
        for backend in backends:
            try:
                matplotlib.use(backend)
                plt.figure()
                plt.close()
                print(f"Using matplotlib backend: {backend}")
                return True
            except:
                continue
        
        print("Warning: Could not find suitable backend")
        return False
    except Exception as e:
        print(f"Cannot initialize matplotlib: {e}")
        return False

def main():
    # Initialize matplotlib
    if not safe_visualization_display():
        print("Warning: May encounter issues with matplotlib display")
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_data("data/sign_mnist_train.csv")
    X_test, y_test = load_data("data/sign_mnist_test.csv")

    # Preprocess data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    print("Dataset Info:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Use small sample for initial testing
    sample_size = 100
    print(f"Predicting on first {sample_size} samples")

    # Use full training set
    print(f"Training KNN model with full {len(X_train):,} training samples...")
    print("Note: Training with full dataset may take some time...")
    
    model = KNN(k=5)
    model.fit(X_train, y_train)

    # Prediction
    print("Performing predictions...")
    y_pred = model.predict(X_test[:sample_size])

    # Basic evaluation
    accuracy = np.mean(y_pred == y_test[:sample_size])
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Lưu ảnh kết quả
    print("Saving result images...")
    save_prediction_images(X_test, y_test, y_pred, min(10, sample_size))

    print("\n" + "="*60)
    print("DISPLAY VISUALIZATIONS (separate windows)")
    print("="*60)

    # Hỏi người dùng muốn xem cái nào
    print("\nChọn visualization muốn xem (có thể chọn nhiều):")
    print("1. Confusion Matrix")
    print("2. Accuracy Charts (Line + Bar)")
    print("3. Class Accuracy Chart") 
    print("4. Statistics Summary")
    print("5. Sample Predictions")
    print("6. All (từng cái một)")
    print("0. Bỏ qua visualizations")
    print("\n💡 Hướng dẫn:")
    print("  - Chọn một số: ví dụ '1' hoặc '2'")
    print("  - Chọn nhiều số: ví dụ '1 2 3' hoặc '1,2,3'")
    print("  - Chọn tất cả: '6' hoặc 'all'")
    print("  - Bỏ qua: '0' hoặc 'skip'")
    
    choices_input = input("\nNhập lựa chọn: ").strip().lower()
    
    # Xử lý input
    choices = []
    if choices_input in ['0', 'skip']:
        choices = ['0']
    elif choices_input in ['6', 'all']:
        choices = ['6']
    else:
        # Tách các lựa chọn bằng dấu cách hoặc dấu phẩy
        choices_raw = choices_input.replace(',', ' ').split()
        choices = [choice for choice in choices_raw if choice in ['1', '2', '3', '4', '5']]
        
        if not choices:
            print("No valid choices! Displaying all...")
            choices = ['1', '2', '3', '4', '5']
    
    print(f"🎯 Các visualization được chọn: {', '.join(choices)}")
    
    unique_labels = np.unique(y_test[:sample_size])
    
    # Khởi tạo biến để tránh lỗi
    accuracies = []
    sample_sizes = []
    class_accuracies = []
    class_letters = []
    
    # Thực hiện các visualization theo lựa chọn
    if '0' in choices:
        print("Skipping all visualizations")
    elif '6' in choices:
        print("\nDisplaying all visualizations (one by one)...")
        
        print("Displaying Confusion Matrix...")
        show_confusion_matrix(y_test, y_pred, sample_size)
        
        print("Displaying Accuracy Charts...")
        accuracies, sample_sizes = show_accuracy_charts(X_train, y_train, X_test, y_test)
        
        print("Displaying Class Accuracy...")
        class_accuracies, class_letters = show_class_accuracy(y_test, y_pred, sample_size)
        
        print("Displaying Statistics Summary...")
        show_statistics_summary(X_train, sample_size, accuracy, accuracies, sample_sizes, class_accuracies, class_letters, unique_labels)
        
        print("Displaying Sample Predictions...")
        show_sample_predictions(X_test, y_test, y_pred, sample_size)
        
    else:
        # Hiển thị theo từng lựa chọn cụ thể
        print(f"\nDisplaying {len(choices)} visualization(s)...")
        
        for i, choice in enumerate(choices):
            print(f"\n[{i+1}/{len(choices)}] Displaying visualization {choice}...")
            
            if choice == '1':
                print("Displaying Confusion Matrix...")
                show_confusion_matrix(y_test, y_pred, sample_size)
                
            elif choice == '2':
                print("Displaying Accuracy Charts...")
                accuracies, sample_sizes = show_accuracy_charts(X_train, y_train, X_test, y_test)
                
            elif choice == '3':
                print("Displaying Class Accuracy Chart...")
                class_accuracies, class_letters = show_class_accuracy(y_test, y_pred, sample_size)
                
            elif choice == '4':
                print("Displaying Statistics Summary...")
                # Cần tính toán trước nếu chưa có
                if not accuracies:
                    print("  Computing accuracy charts first...")
                    accuracies, sample_sizes = show_accuracy_charts(X_train, y_train, X_test, y_test)
                if not class_accuracies:
                    print("  Computing class accuracy first...")
                    class_accuracies, class_letters = show_class_accuracy(y_test, y_pred, sample_size)
                show_statistics_summary(X_train, sample_size, accuracy, accuracies, sample_sizes, class_accuracies, class_letters, unique_labels)
                
            elif choice == '5':
                print("Displaying Sample Predictions...")
                show_sample_predictions(X_test, y_test, y_pred, sample_size)

    # In báo cáo chi tiết ra console
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    try:
        detailed_classification_report(y_test[:sample_size], y_pred)
    except Exception as e:
        print(f"Error creating classification report: {e}")

    # Thống kê chi tiết theo class
    print("\n" + "="*60)
    print("DETAILED STATISTICS BY CLASS")
    print("="*60)
    print(f"Number of classes in test set: {len(unique_labels)}")
    print(f"Classes: {[chr(i + 65) for i in unique_labels]}")
    
    print("\nAccuracy by class:")
    for i, label in enumerate(unique_labels):
        mask = y_test[:sample_size] == label
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred[mask] == y_test[:sample_size][mask])
            class_letter = chr(label + 65)
            print(f"  Class {class_letter}: {class_accuracy*100:6.2f}% ({np.sum(mask):3d} samples)")

    # Real-time prediction
    print("\n" + "="*60)
    print("REAL-TIME PREDICTION")
    print("="*60)
    try:
        user_choice = input("Do you want to try real-time recognition? (y/n): ").lower()
        if user_choice == 'y' or user_choice == 'yes':
            print("Starting real-time recognition...")
            print("Instructions:")
            print("  - Place hand in green rectangle")
            print("  - Press 'q' to quit")
            print("  - Ensure sufficient lighting and dark background")
            real_time_hand_sign_detection(model)
        else:
            print("Skipping real-time prediction feature")
    except Exception as e:
        print(f"Error running real-time prediction: {e}")
        print("This may be due to missing webcam or incorrect OpenCV installation")

def menu_driven_main():
    """Run program with menu selection"""
    print("ASL CHARACTER RECOGNITION PROGRAM")
    print("="*50)
    print("1. Run all features")
    print("2. Training and basic testing only")
    print("3. Real-time prediction only")
    print("4. Charts and confusion matrix only")
    
    choice = input("Choose function (1-4): ")
    
    # Tải dữ liệu chung
    print("Loading data...")
    X_train, y_train = load_data("data/sign_mnist_train.csv")
    X_test, y_test = load_data("data/sign_mnist_test.csv")
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # Sử dụng toàn bộ tập train
    print(f"Training with full {len(X_train):,} samples...")
    model = KNN(k=5)
    model.fit(X_train, y_train)
    
    if choice == "1":
        main()
    elif choice == "2":
        sample_size = 100
        y_pred = model.predict(X_test[:sample_size])
        accuracy = np.mean(y_pred == y_test[:sample_size])
        print(f"Accuracy: {accuracy*100:.2f}%")
        save_prediction_images(X_test, y_test, y_pred, min(10, sample_size))
    elif choice == "3":
        print("Starting real-time prediction...")
        real_time_hand_sign_detection(model)
    elif choice == "4":
        sample_size = 100
        y_pred = model.predict(X_test[:sample_size])
        show_confusion_matrix(y_test, y_pred, sample_size)
        show_accuracy_charts(X_train, y_train, X_test, y_test)
        show_class_accuracy(y_test, y_pred, sample_size)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    # Choose to run with menu or full mode
    run_mode = input("Run with menu (m) or full mode (f)? [m/f]: ").lower()
    
    if run_mode == 'm':
        menu_driven_main()
    else:
        main()
    
    print("\nCOMPLETED!")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    if class_names is None:
        class_names = [chr(i + 65) for i in range(26)]  # A-Z
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def detailed_classification_report(y_true, y_pred):
    """Chi tiết classification report"""
    class_names = [chr(i + 65) for i in range(26)]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
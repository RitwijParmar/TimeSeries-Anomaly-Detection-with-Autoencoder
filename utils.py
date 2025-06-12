import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def find_threshold(model, dataloader, device):
    """
    Finds the anomaly threshold from the training data's reconstruction error.
    """
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for seq, _ in dataloader:
            seq = seq.to(device)
            reconstructed = model(seq)
            errors = torch.mean((reconstructed - seq) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
    
    # Set threshold to be a value higher than most training errors
    threshold = np.percentile(reconstruction_errors, 95)
    print(f"Anomaly threshold set to: {threshold:.4f}")
    return threshold

def plot_reconstruction_errors(errors, threshold):
    """Plots the distribution of reconstruction errors."""
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Anomaly Threshold')
    plt.title('Distribution of Reconstruction Errors on Test Set')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
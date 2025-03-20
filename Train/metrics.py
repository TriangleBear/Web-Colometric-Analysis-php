import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

def compute_per_class_metrics(targets, preds, num_classes=12):
    """Compute precision, recall and F1 score for each class"""
    # Ensure we're working with numpy arrays
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    
    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np, preds_np, labels=range(num_classes), 
        zero_division=0, average=None
    )
    
    # Create a DataFrame for easier reading
    metrics = {}
    for i in range(num_classes):
        metrics[i] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    return metrics

def plot_class_metrics(metrics, class_names=None, save_path=None):
    """Plot metrics for each class as bar charts"""
    class_ids = list(metrics.keys())
    precision = [metrics[i]['precision'] for i in class_ids]
    recall = [metrics[i]['recall'] for i in class_ids]
    f1 = [metrics[i]['f1'] for i in class_ids]
    
    # Use class names if provided, otherwise use class IDs
    x_labels = [class_names.get(i, f"Class {i}") if class_names else f"Class {i}" for i in class_ids]
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision plot
    ax[0].bar(x_labels, precision)
    ax[0].set_title('Precision by Class')
    ax[0].set_ylim(0, 1)
    ax[0].set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Recall plot
    ax[1].bar(x_labels, recall)
    ax[1].set_title('Recall by Class')
    ax[1].set_ylim(0, 1)
    ax[1].set_xticklabels(x_labels, rotation=45, ha='right')
    
    # F1 plot
    ax[2].bar(x_labels, f1)
    ax[2].set_title('F1 Score by Class')
    ax[2].set_ylim(0, 1)
    ax[2].set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    else:
        plt.show(block=False)
        return fig

def evaluate_class_learning(model, dataloader, device, class_names=None, save_path=None):
    """Evaluate how well each class is being learned"""
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[class_names.get(i, f"Class {i}") for i in range(len(class_names))]
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        plt.show(block=False)
    
    # Compute and return per-class metrics
    metrics = compute_per_class_metrics(all_targets, all_preds, len(class_names))
    return metrics, cm

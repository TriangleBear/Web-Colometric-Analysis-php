import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel  # Import the required class
from datasets import UrineStripDataset
from models import UNetYOLO
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, NUM_CLASSES, IMAGE_SIZE  # Import more constants
from train_unet_yolo import train_model  # Import train_model function
from train_svm import train_svm_classifier  # Import train_svm_classifier function
from tqdm import tqdm  # Import tqdm for progress bars

# Add DetectionModel to safe globals for model loading
torch.serialization.add_safe_globals([DetectionModel])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")

def load_data(batch_size=16):
    dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Evaluate the model with additional diagnostics and progress bar
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nDiagnostic information for model evaluation:")
    print(f"Using device: {device}")
    
    with torch.no_grad():
        # Add progress bar for evaluation
        eval_progress = tqdm(enumerate(dataloader), desc="Evaluating model", total=len(dataloader))
        
        for i, (images, targets, _) in eval_progress:
            images, targets = images.to(device), targets.to(device)  # Move to GPU
            outputs = model(images)
            
            # Get the predicted class for each image 
            preds = torch.argmax(outputs, dim=1)  # Shape: (batch_size, height, width)
            
            # Get the most common prediction for each image
            batch_preds = []
            for j in range(preds.shape[0]):
                # Get most common class in the prediction (mode)
                img_pred = preds[j].flatten()
                values, counts = torch.unique(img_pred, return_counts=True)
                mode_idx = torch.argmax(counts)
                most_common_class = values[mode_idx].item()
                batch_preds.append(most_common_class)
                
                # Print diagnostic info for first few batches only
                if i < 2 and j < 3:  # First 2 batches, first 3 images in each
                    print(f"Image {i*len(images)+j}:")
                    target_cls = targets[j].item()
                    print(f"  Target class: {target_cls}")
                    print(f"  Predicted class: {most_common_class}")
                    print(f"  Unique predicted pixel classes: {values.cpu().numpy()}")
            
            all_preds.extend(batch_preds)
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar with current batch info
            eval_progress.set_postfix({"Batch": f"{i+1}/{len(dataloader)}"})
    
    # Convert to numpy arrays and ensure they have the same shape
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    
    # Make sure we have enough data points
    if len(targets_np) < 2 or len(preds_np) < 2:
        print("WARNING: Not enough data points for meaningful evaluation")
        return targets_np, preds_np
    
    print(f"\nDataset evaluation:")
    print(f"Number of predictions: {len(preds_np)}")
    print(f"Number of targets: {len(targets_np)}")
    print(f"Unique target classes: {np.unique(targets_np)}")
    print(f"Unique predicted classes: {np.unique(preds_np)}")
    
    return targets_np, preds_np

# Calculate metrics with better handling of edge cases
def calculate_metrics(targets, preds):
    # Check if we have enough classes for meaningful metrics
    unique_targets = np.unique(targets)
    unique_preds = np.unique(preds)
    
    print(f"Calculating metrics for {len(targets)} samples")
    print(f"Target classes: {unique_targets}")
    print(f"Predicted classes: {unique_preds}")
    
    # If we have a single class or no variation in predictions, 
    # standard metrics may not be meaningful
    if len(unique_targets) <= 1 or len(unique_preds) <= 1:
        print("WARNING: Not enough class variation for standard metrics")
        print(f"Using accuracy only")
        
        # Calculate simple accuracy instead
        accuracy = np.mean(targets == preds)
        return accuracy, 0.0, 0.0  # Return only accuracy with zeros for other metrics
    
    try:
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        recall = recall_score(targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        return precision, recall, f1
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print("Falling back to accuracy")
        accuracy = np.mean(targets == preds)
        return accuracy, 0.0, 0.0

def check_if_model_has_weights(model):
    """Check if model has meaningful weights loaded"""
    # Check if at least some parameters are non-zero
    for name, param in model.named_parameters():
        if param.requires_grad and param.sum().abs().item() > 0:
            return True  # Found at least one non-zero parameter
    return False  # All parameters are zeros or close to zeros

# Main function
if __name__ == "__main__":

    # Load model
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES).to(device)  # Move model to GPU
    
    # Check if model has meaningful weights
    has_weights = check_if_model_has_weights(model)
    
    if not has_weights:
        print("\nWARNING: Model appears to be untrained (no weights loaded).")
        print("The initial evaluation will serve as a baseline with random predictions.")
    
    # Load data
    dataloader = load_data()

    # Check model
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # Evaluate model
    print("\nEvaluating model...")
    targets, preds = evaluate_model(model, dataloader)

    # Calculate metrics
    print("\nCalculating metrics...")
    precision, recall, f1 = calculate_metrics(targets, preds)
    
    # Also calculate accuracy directly
    accuracy = np.mean(targets == preds) if len(targets) > 0 else 0
    
    # Print metrics
    print(f'\nModel Evaluation Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    if not has_weights:
        print("These metrics are from an untrained model - training will now begin.")
    
    # Generate and save confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Explicitly specify all possible class labels (0 to NUM_CLASSES-1)
    all_labels = list(range(NUM_CLASSES))
    
    cm = confusion_matrix(targets, preds, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Evaluation complete! Confusion matrix saved to confusion_matrix.png")
    
    print("\nStarting model training...")
    # Train UNet-YOLO model
    train_model(num_epochs=10, batch_size=16, learning_rate=0.001)
    
    print("\nTraining complete! Now evaluating trained model...")
    # Evaluate trained model
    targets, preds = evaluate_model(model, dataloader)
    precision, recall, f1 = calculate_metrics(targets, preds)
    print(f'Trained Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # Generate and save confusion matrix for trained model
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(targets, preds, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Trained Model Confusion Matrix')
    plt.savefig('trained_confusion_matrix.png')
    plt.close()
    
    # Train SVM classifier
    unet_model_path = r'D:\Programming\urine_interpret\models\weights.pt'
    train_svm_classifier(unet_model_path)

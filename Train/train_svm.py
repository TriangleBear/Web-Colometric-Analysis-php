import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from utils import extract_features_and_labels, save_svm_model
from datasets import UrineStripDataset
from config import (
    TRAIN_IMAGE_FOLDER,
    TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER,
    VALID_MASK_FOLDER,
    TEST_IMAGE_FOLDER,
    TEST_MASK_FOLDER,
    device,
    NUM_CLASSES,
    get_model_folder
)
from models import UNetYOLO
from ultralytics.nn.tasks import DetectionModel
from icecream import ic
import os
import cv2
from tqdm import tqdm  # Import tqdm for progress bars
from torch.utils.data import DataLoader  # Add this import

# Define class names
CLASS_NAMES = {
    0: 'Bilirubin',
    1: 'Blood',
    2: 'Glucose',
    3: 'Ketone',
    4: 'Leukocytes',
    5: 'Nitrite',
    6: 'Protein',
    7: 'SpGravity',
    8: 'Urobilinogen',
    9: 'background',
    10: 'pH',
    11: 'strip'  # Add background class
}

def extract_polygon_features(image, mask, class_id):
    """Extract features for polygon-shaped bounding boxes."""
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        region = image[y:y+h, x:x+w]
        features.append(region.mean(axis=(0, 1)))  # Mean color in the region
    if features:
        return np.mean(features, axis=0)  # Average features across all polygons
    else:
        return np.zeros(3)  # Return zeros if no polygons are found

def extract_features_from_segmentation(image, segmentation_map):
    """Extract features from segmentation map for SVM classification."""
    features = []
    
    # Get unique classes in the segmentation map
    unique_classes = np.unique(segmentation_map)
    
    # For each class, extract region properties
    for class_id in range(NUM_CLASSES):
        # Create binary mask for this class
        class_mask = (segmentation_map == class_id)
        
        # Skip if class not present
        if not class_mask.any():
            # Add zeros for absent classes
            features.extend([0, 0, 0, 0, 0, 0])
            continue
            
        # Calculate region properties
        # 1. Area ratio (area of class / total area)
        area_ratio = np.sum(class_mask) / segmentation_map.size
        features.append(area_ratio)
        
        # 2. Mean color within the class region
        for channel in range(image.shape[2]):
            mean_color = np.mean(image[:, :, channel][class_mask])
            features.append(mean_color)
            
        # 3. Mean position (center of mass)
        y_indices, x_indices = np.where(class_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            mean_y = np.mean(y_indices) / segmentation_map.shape[0]
            mean_x = np.mean(x_indices) / segmentation_map.shape[1]
            features.extend([mean_y, mean_x])
        else:
            features.extend([0, 0])
    
    return np.array(features, dtype=np.float32)

def extract_features_and_labels_with_progress(dataset, model):
    """
    Extract features and labels from the dataset using the given model.
    Fixes class preservation issues by directly using the original class labels.
    """
    features = []
    labels = []
    
    # Debug information for diagnosis
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    model.eval()
    with torch.no_grad():
        # Process dataset in batches for efficiency, but track items individually
        batch_size = 8
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Don't shuffle to maintain order
            num_workers=4,
            pin_memory=True
        )
        
        progress_bar = tqdm(dataloader, desc="Extracting features", total=len(dataloader))
        
        for batch_idx, batch in enumerate(progress_bar):
            images, targets, _ = batch
            
            # Use the original labels from the dataset instead of deriving from outputs
            batch_labels = targets.cpu().numpy()
            
            # Count classes for debugging
            for label in batch_labels:
                class_counts[label] += 1
            
            # Ensure images have the right shape for the model [B, C, H, W]
            if images.dim() == 3:  # Single image: [C, H, W]
                images = images.unsqueeze(0)
                
            # Move to device
            images = images.to(device)
            
            # IMPROVED: Extract better features from multiple layers of the model
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                outputs = model(images)
                
                # Extract bottleneck features from encoder (richer representation)
                x1 = model.unet.inc(images)
                x2 = model.unet.down1(x1)
                x3 = model.unet.down2(x2)
                x4 = model.unet.down3(x3)
                x5 = model.unet.down4(x4)
                
                # Combine features from multiple layers
                bottleneck = model.unet.dropout(x5)
                
                # Global average pooling to get fixed-size feature vectors
                pooled_x5 = torch.nn.functional.adaptive_avg_pool2d(bottleneck, (1, 1))
                pooled_x4 = torch.nn.functional.adaptive_avg_pool2d(x4, (1, 1))
                
                # Concatenate features from different layers
                multi_scale_features = torch.cat([
                    pooled_x5.view(pooled_x5.size(0), -1),
                    pooled_x4.view(pooled_x4.size(0), -1)
                ], dim=1)
                
                # Convert to numpy
                batch_features = multi_scale_features.cpu().numpy()
            
            # Extend our lists with batch data
            features.extend(batch_features)
            labels.extend(batch_labels)
            
            # Update progress bar with class stats every 10 batches
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                # Get the number of unique classes found so far
                unique_classes = len([c for c, count in class_counts.items() if count > 0])
                progress_bar.set_postfix({
                    "Features": len(features),
                    "Classes": unique_classes
                })
    
    # Final diagnostic info
    print("\nClass distribution in extracted features:")
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        class_name = CLASS_NAMES.get(label, f"Unknown class {label}")
        print(f"  Class {label} ({class_name}): {count} samples")
    
    if len(label_counts) <= 1:
        print("\nWARNING: Only one class detected after feature extraction!")
        print("This will cause SVM training to fail.")
    
    return np.array(features), np.array(labels)

def train_svm_with_real_data():
    """Train an SVM classifier using features from the weights.pt model on real datasets"""
    # Load the UNetYOLO model from weights.pt
    model_path = r'D:\Programming\urine_interpret\models\weights.pt'
    model = load_model(model_path)
    
    if model is None:
        print("Failed to load the model. Cannot train SVM.")
        return None
    
    # Load actual datasets
    print("Loading datasets...")
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    # Directly examine the raw labels in the dataset
    print("Examining raw labels in dataset to verify diversity...")
    raw_labels = []
    for i in tqdm(range(min(100, len(train_dataset))), desc="Checking labels"):
        try:
            _, label, _ = train_dataset[i]
            raw_labels.append(label)
        except Exception as e:
            print(f"Error accessing dataset item {i}: {e}")
    
    raw_label_counts = {}
    for label in raw_labels:
        raw_label_counts[label] = raw_label_counts.get(label, 0) + 1


def train_svm_classifier(unet_model_path, model_path=None, binary_mode=True):
    ic("Loading training, validation, and test datasets...")
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, debug_level=1)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)
    
    # Print raw annotation distribution to debug
    ic("Raw annotation class distribution from dataset:")
    if hasattr(train_dataset, 'raw_annotations_count'):
        for cls_id, count in sorted(train_dataset.raw_annotations_count.items()):
            if count > 0:
                ic(f"Class {cls_id} ({CLASS_NAMES.get(cls_id, 'Unknown')}): {count} raw annotations")
    
    ic("Loading trained UNet model...")
    unet_model = UNetYOLO(3, NUM_CLASSES).to(device)
    torch.serialization.add_safe_globals([DetectionModel])
    unet_model.load_state_dict(torch.load(unet_model_path, map_location=device), strict=False)
    unet_model.eval()
    
    # Print the number of classes present in the dataset
    ic("Verifying class distribution in datasets...")
    train_class_dist = train_dataset.class_distribution
    ic(f"Classes in training set: {sorted(train_class_dist.keys())}")
    for cls_id, count in sorted(train_class_dist.items()):
        ic(f"Class {cls_id} ({CLASS_NAMES.get(cls_id, 'Unknown')}): {count} samples")
    
    # Add synthetic samples for missing classes if needed
    missing_classes = [i for i in range(NUM_CLASSES) if i not in train_class_dist]
    if missing_classes:
        ic(f"WARNING: Missing classes: {missing_classes}")
        if hasattr(train_dataset, 'generate_synthetic_samples'):
            train_dataset.generate_synthetic_samples(missing_classes, num_samples=20)
            ic("Added synthetic samples for missing classes")
    
    ic("Extracting features and labels...")
    train_features, train_labels = extract_features_and_labels_with_progress(train_dataset, unet_model)
    valid_features, valid_labels = extract_features_and_labels_with_progress(valid_dataset, unet_model)
    test_features, test_labels = extract_features_and_labels_with_progress(test_dataset, unet_model)
    
    ic(f"Training set: {len(train_features)} samples")
    ic(f"Validation set: {len(valid_features)} samples")
    ic(f"Test set: {len(test_features)} samples")
    
    # Count unique classes
    unique_train_classes = np.unique(train_labels)
    ic(f"Unique classes in training set: {unique_train_classes}")
    
    # Check if we have enough unique classes for multi-class classification
    if len(unique_train_classes) == 0:
        ic("ERROR: No classes detected in the training set.")
        return None
    
    if len(unique_train_classes) <= 1:
        if binary_mode:
            ic("Only one class detected. Switching to binary classification mode.")
            ic("Will train classifier to distinguish this class from artificially generated 'other' class")
            
            # Create synthetic data for the 'other' class
            existing_class = unique_train_classes[0]
            other_class = 1 if existing_class == 0 else 0  # Choose a different class ID
            
            # Generate synthetic features
            num_synthetic = len(train_features) // 3  # Generate 1/3 as many samples
            
            # Create synthetic features by perturbing existing features
            synthetic_features = []
            for i in range(num_synthetic):
                # Take a random sample and perturb it
                idx = np.random.randint(0, len(train_features))
                feature = train_features[idx].copy()
                
                # Add random noise
                feature += np.random.normal(0, 0.5, size=feature.shape)
                synthetic_features.append(feature)
            
            synthetic_labels = np.full(num_synthetic, other_class)
            
            # Combine with original data
            train_features = np.vstack([train_features, synthetic_features])
            train_labels = np.concatenate([train_labels, synthetic_labels])
            
            ic(f"Added {num_synthetic} synthetic samples for class {other_class}")
            ic(f"New training set: {len(train_features)} samples")
            ic(f"Unique classes after augmentation: {np.unique(train_labels)}")
        else:
            ic("WARNING: Only one class detected in the training set.")
            ic("Multi-class SVM requires at least two different classes.")
            ic("Either add more data with different classes or enable binary_mode=True")
            return None
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    valid_features_scaled = scaler.transform(valid_features)
    test_features_scaled = scaler.transform(test_features)
    
    # IMPROVED: Train SVM with better parameters for improved accuracy
    ic("Training SVM classifier with optimized parameters...")
    with tqdm(total=100, desc="SVM Training") as progress:
        svm_model = SVC(
            kernel='rbf',
            C=10.0,  # Increased C for less regularization
            gamma='auto',  # Let sklearn determine gamma automatically
            probability=True,
            class_weight='balanced',  # Use balanced class weights
            verbose=False
        )
        
        # Fit the model
        svm_model.fit(train_features_scaled, train_labels)
        progress.update(100)  # Update to 100% when done
    
    # Evaluate on validation set
    valid_pred = svm_model.predict(valid_features_scaled)
    valid_accuracy = accuracy_score(valid_labels, valid_pred) * 100
    ic(f"Validation Accuracy: {valid_accuracy:.2f}%")
    
    # Print detailed classification report for validation set
    ic("Classification Report on Validation Set:")
    valid_report = classification_report(
        valid_labels, valid_pred,
        target_names=[CLASS_NAMES[i] for i in np.unique(np.concatenate([train_labels, valid_labels, valid_pred]))],
        zero_division=0
    )
    ic(valid_report)
    
    # Evaluate on test set
    test_pred = svm_model.predict(test_features_scaled)
    test_accuracy = accuracy_score(test_labels, test_pred) * 100
    ic(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Print detailed classification report for test set
    ic("Classification Report on Test Set:")
    test_report = classification_report(
        test_labels, test_pred,
        target_names=[CLASS_NAMES[i] for i in np.unique(np.concatenate([train_labels, test_labels, test_pred]))],
        zero_division=0
    )
    ic(test_report)
    
    # Save the model and preprocessing objects
    if model_path is None:
        model_folder = get_model_folder()
        model_path = os.path.join(model_folder, "svm_rbf_model.pkl")
    
    model_data = {
        'model': svm_model,
        'scaler': scaler,
        'class_names': CLASS_NAMES,
        'binary_mode': len(unique_train_classes) <= 1,
        'classes': list(np.unique(train_labels))
    }
    save_svm_model(model_data, model_path)
    ic(f"Model saved to {model_path}")
    
    return model_data

if __name__ == "__main__":
    unet_model_path = r'D:\Programming\urine_interpret\models\weights.pt'
    train_svm_classifier(unet_model_path)

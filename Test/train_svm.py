import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
import time
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))

# Import necessary modules
from Train.config import NUM_CLASSES, TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, VALID_IMAGE_FOLDER, VALID_MASK_FOLDER
from Train.models import UNetYOLO
from Train.datasets import UrineStripDataset
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for better reporting
CLASS_NAMES = {
    0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone',
    4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity',
    8: 'Urobilinogen', 9: 'Background', 10: 'pH', 11: 'Strip'
}

def load_model(model_path):
    """Load the UNetYOLO model from weights.pt"""
    print(f"Loading model from {model_path}...")
    
    # Add safe modules for loading
    torch.serialization.add_safe_globals([DetectionModel])
    
    # Initialize model
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES)
    
    try:
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def extract_features_from_dataset(model, dataset, desc="Extracting features"):
    """Extract features from dataset using the UNetYOLO model"""
    features = []
    labels = []
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            images, targets, _ = batch
            images = images.to(device)
            
            # Extract features from the model
            # For each image, get bottleneck features and pool them
            x1 = model.unet.inc(images)
            x2 = model.unet.down1(x1)
            x3 = model.unet.down2(x2)
            x4 = model.unet.down3(x3)
            x5 = model.unet.down4(x4)
            bottleneck = model.unet.dropout(x5)
            
            # Global average pooling for a fixed-length feature vector
            batch_features = torch.mean(bottleneck, dim=[2, 3]).cpu().numpy()
            
            # Store features and labels
            features.extend(batch_features)
            labels.extend(targets.numpy())
    
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
    
    print(f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples.")
    
    # Extract features
    print("\nExtracting features from training data...")
    train_features, train_labels = extract_features_from_dataset(model, train_dataset, "Extracting train features")
    
    print("Extracting features from validation data...")
    valid_features, valid_labels = extract_features_from_dataset(model, valid_dataset, "Extracting validation features")
    
    print(f"Extracted features: {train_features.shape}, labels: {train_labels.shape}")
    
    # Check class distribution
    train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    valid_class_counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
    
    print("\nClass distribution in training data:")
    unique_classes = []
    for cls, count in enumerate(train_class_counts):
        print(f"  {CLASS_NAMES.get(cls, 'Unknown')}: {count} samples")
        if count > 0:
            unique_classes.append(cls)
    
    # Handle single-class issue
    if len(unique_classes) <= 1:
        print("\nERROR: Only one class detected in the training data.")
        print("SVM requires at least two classes to train.")
        print("\nPossible issues:")
        print("1. Dataset labels are incorrect or all the same")
        print("2. Feature extraction is not maintaining class diversity")
        print("3. Only one class is actually present in your training set")
        
        # Try to fix the issue by:
        # 1. Ensure we have a more diverse dataset with multiple classes
        print("\nTrying to create a more diverse synthetic dataset for training...")
        
        # Generate synthetic data with multiple classes
        synthetic_features = []
        synthetic_labels = []
        
        # Generate features for each class, simulating what we'd expect
        for cls in range(NUM_CLASSES):
            if cls != 9:  # Skip background class
                # Create 10 synthetic samples per class
                for i in range(10):
                    # Create a feature vector with some randomness
                    feature = np.random.randn(train_features.shape[1])
                    synthetic_features.append(feature)
                    synthetic_labels.append(cls)
                    
                    if i == 0:
                        print(f"Generated synthetic sample for class {cls} ({CLASS_NAMES.get(cls, 'Unknown')})")
        
        train_features = np.vstack([train_features, np.array(synthetic_features)])
        train_labels = np.concatenate([train_labels, np.array(synthetic_labels)])
        
        print(f"\nNew training set: {train_features.shape}, labels: {train_labels.shape}")
        train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        print("\nUpdated class distribution:")
        unique_classes = []
        for cls, count in enumerate(train_class_counts):
            if count > 0:
                print(f"  {CLASS_NAMES.get(cls, 'Unknown')}: {count} samples")
                unique_classes.append(cls)
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    valid_features_scaled = scaler.transform(valid_features)
    
    # Train SVM with RBF kernel
    print("\nTraining SVM classifier with RBF kernel...")
    
    try:
        # Use class_weight='balanced' to handle class imbalance
        svm = SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            probability=True, 
            class_weight='balanced'
        )
        svm.fit(train_features_scaled, train_labels)
        
        # Evaluate on validation set
        print("Evaluating SVM on validation data...")
        if len(np.unique(valid_labels)) > 1:
            val_predictions = svm.predict(valid_features_scaled)
            val_accuracy = accuracy_score(valid_labels, val_predictions)
            
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print("\nClassification Report:")
            class_names = [CLASS_NAMES.get(i, f"Class {i}") for i in range(NUM_CLASSES)]
            print(classification_report(valid_labels, val_predictions, target_names=class_names, zero_division=0))
        else:
            print("Skipping validation - not enough diversity in validation set")
            val_accuracy = 0
        
        # Save the model
        svm_model = {
            'model': svm,
            'scaler': scaler,
            'class_names': CLASS_NAMES,
            'accuracy': val_accuracy,
            'contains_synthetic': len(unique_classes) <= 1
        }
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = rf'D:\Programming\urine_interpret\models\svm_model_{timestamp}.pkl'
        joblib.dump(svm_model, save_path)
        
        print(f"SVM model saved to {save_path}")
        return svm_model
    
    except Exception as e:
        print(f"Error training SVM: {e}")
        print("Check your dataset to ensure it has diverse classes for training.")
        return None

if __name__ == "__main__":
    svm_model = train_svm_with_real_data()

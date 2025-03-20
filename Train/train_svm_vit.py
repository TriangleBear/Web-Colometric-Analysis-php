from transformers import ViTModel, ViTImageProcessor, ViTForImageClassification
import numpy as np
import os
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils import save_svm_model
from datasets import UrineStripDataset
from torch.utils.data import DataLoader
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, VALID_IMAGE_FOLDER, VALID_MASK_FOLDER, TEST_IMAGE_FOLDER, TEST_MASK_FOLDER, device, get_model_folder

# Add ViTForImageClassification and custom modules to safe globals for model loading
torch.serialization.add_safe_globals([ViTForImageClassification])
torch.serialization.add_safe_globals({'src.classification.vit_model.ViTForImageClassification': ViTForImageClassification})

# Load pre-trained ViT model and image processor
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
vit_image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def extract_vit_features(dataset, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    features = []
    labels = []

    vit_model.eval()
    with torch.no_grad():
        for images, targets, _ in dataloader:
            # Preprocess images - add do_rescale=False since images are already scaled to [0,1]
            inputs = vit_image_processor(images, return_tensors="pt", do_rescale=False).to(device)
            outputs = vit_model(**inputs)
            # Extract the [CLS] token representation
            cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(cls_features)
            labels.extend(targets.cpu().numpy())

    return np.array(features), np.array(labels)

def load_pretrained_model(model_path):
    """Load a pre-trained model from a .pt file"""
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    # Use weights_only=True to avoid needing custom modules
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    # Load datasets
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)

    # Load pre-trained model if available
    pretrained_model_path = r'D:\Programming\urine_interpret\models\vitmodel.pt'
    if os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from {pretrained_model_path}...")
        try:
            # Try loading with weights_only=True
            print("Attempting to load weights only...")
            vit_model = load_pretrained_model(pretrained_model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Using default ViT model instead.")
    else:
        print("No pre-trained model found. Using default ViT model.")

    # Extract features
    print("Extracting features from training data...")
    train_features, train_labels = extract_vit_features(train_dataset)
    print("Extracting features from validation data...")
    valid_features, valid_labels = extract_vit_features(valid_dataset)
    print("Extracting features from test data...")
    test_features, test_labels = extract_vit_features(test_dataset)

    # Save features and labels
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('valid_features.npy', valid_features)
    np.save('valid_labels.npy', valid_labels)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)

    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    valid_features_scaled = scaler.transform(valid_features)
    test_features_scaled = scaler.transform(test_features)

    # Train SVM with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_model.fit(train_features_scaled, train_labels)

    # Evaluate on validation set
    valid_pred = svm_model.predict(valid_features_scaled)
    valid_accuracy = accuracy_score(valid_labels, valid_pred) * 100
    print(f"Validation Accuracy: {valid_accuracy:.2f}%")
    print("Classification Report on Validation Set:")
    print(classification_report(valid_labels, valid_pred))

    # Evaluate on test set
    test_pred = svm_model.predict(test_features_scaled)
    test_accuracy = accuracy_score(test_labels, test_pred) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Classification Report on Test Set:")
    print(classification_report(test_labels, test_pred))

    # Save the model and preprocessing objects
    model_folder = get_model_folder()
    model_path = os.path.join(model_folder, "svm_rbf_vit_model.pkl")
    model_data = {
        'model': svm_model,
        'scaler': scaler
    }
    save_svm_model(model_data, model_path)
    print(f"Model saved to {model_path}")

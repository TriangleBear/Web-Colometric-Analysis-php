import torch
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from torch.utils.data import DataLoader
from datasets import UrineStripDataset
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, VALID_IMAGE_FOLDER, VALID_MASK_FOLDER, TEST_IMAGE_FOLDER, TEST_MASK_FOLDER, device

# Load pre-trained ViT model and feature extractor
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def extract_vit_features(dataset, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    features = []
    labels = []

    vit_model.eval()
    with torch.no_grad():
        for images, targets, _ in dataloader:
            # Preprocess images - add do_rescale=False since images are already scaled to [0,1]
            inputs = vit_feature_extractor(images, return_tensors="pt", do_rescale=False).to(device)
            outputs = vit_model(**inputs)
            # Extract the [CLS] token representation
            cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(cls_features)
            labels.extend(targets.cpu().numpy())

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Load datasets
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)

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

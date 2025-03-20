import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models import UNet
from datasets import UrineStripDataset
from config import device, IMAGE_FOLDER, MASK_FOLDER, NUM_CLASSES

# Define a color map for class IDs
class_colors = {
    0: (0, 0, 0),       # Background - Black
    1: (255, 0, 0),     # Class 1 - Red
    2: (0, 255, 0),     # Class 2 - Green
    3: (0, 0, 255),     # Class 3 - Blue
    4: (255, 255, 0),   # Class 4 - Yellow
    5: (255, 0, 255),   # Class 5 - Magenta
    6: (0, 255, 255),   # Class 6 - Cyan
    7: (128, 0, 0),     # Class 7 - Maroon
    8: (0, 128, 0),     # Class 8 - Dark Green
    9: (0, 0, 128),     # Class 9 - Navy
    10: (128, 128, 0)   # Class 10 - Olive
}

def load_model(model_path):
    model = UNet(in_channels=3, out_channels=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def predict(model, image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = F.softmax(prediction, dim=1)
        predicted_class = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return predicted_class

def visualize_prediction(image, mask, predicted_mask):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()

def evaluate_model(model, dataset):
    y_true = []
    y_pred = []
    for i in range(len(dataset)):
        image, mask = dataset[i]
        image_pil = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))
        predicted_mask = predict(model, image_pil)
        y_true.extend(mask.flatten().numpy())
        y_pred.extend(predicted_mask.flatten())

        # Visualize the prediction for the first few samples
        if i < 5:
            visualize_prediction(image_pil, mask.numpy(), predicted_mask)

        # Debugging: Check unique values in the predicted mask
        unique_values = np.unique(predicted_mask)
        print(f"Sample {i}: Unique values in predicted mask: {unique_values}")

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(NUM_CLASSES)])

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    model_path = r'D:\Programming\urine_interpret\models\unet_model_20250220-213018.pth_epoch_62.pth'
    model = load_model(model_path)

    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    evaluate_model(model, dataset)

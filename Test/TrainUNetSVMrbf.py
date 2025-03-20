import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from skimage import color
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.model_selection import train_test_split
import random
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Additional imports
import torch.utils.checkpoint as checkpoint
import torch._dynamo
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False  # Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True  # Enable cuDNN for performance

timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"
model_filename_svm = f"svm_model_{timestamp}.pkl"


def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    return torch.from_numpy(mask)

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
    targets = targets.squeeze(1)  # Shape becomes [B, H, W]
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (outputs * targets_one_hot).sum(dim=(2,3))
    union = outputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()

def focal_loss(outputs, targets, alpha=0.25, gamma=2):
    """ Focal Loss for multi-class segmentation. """
    targets = targets.squeeze(1)  # Remove channel dim (B, H, W)
    targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
    pt = torch.exp(-bce_loss)
    focal = (alpha * (1 - pt) ** gamma * bce_loss).mean()
    return focal

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=11, dropout_prob=0.2):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        
        up3 = self.upconv3(bottleneck)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        output = self.output(up1)
        return output


class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.horizontal and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if self.vertical and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return {'image': image, 'mask': mask}

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return {'image': image, 'mask': mask}

class RandomAffine:
    def __init__(self, translate=(0.1, 0.1)):
        self.translate = translate
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        params = transforms.RandomAffine.get_params(
            degrees=[-10, 10],
            translate=self.translate,
            scale_ranges=None,
            shears=None,
            img_size=image.size
        )
        image = transforms.functional.affine(image, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        mask = transforms.functional.affine(mask, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        return {'image': image, 'mask': mask}

class RandomTrainTransformations:
    def __init__(self, mean, std):
        self.joint_transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=10),
            RandomAffine(translate=(0.1, 0.1))
        ])
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            mask_to_tensor
        ])
    def __call__(self, sample):
        sample = self.joint_transform(sample)
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}

class SimpleValTransformations:
    def __init__(self, mean, std):
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            mask_to_tensor
        ])
    def __call__(self, sample):
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}

class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))
        self.transform = transform
        if len(self.image_files) != len(self.txt_files):
            raise ValueError("Mismatch between number of images and masks")

        # Define default transformation if none is provided
        self.default_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        txt_file = self.txt_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        txt_path = os.path.join(self.mask_folder, txt_file)

        # Open the image and convert to RGB
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        mask = self.create_mask_from_yolo(txt_path)
        mask = Image.fromarray(mask).resize((256, 256), Image.NEAREST)

        # Convert image and mask to tensors
        image = self.default_transform(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask



    def create_mask_from_yolo(self, txt_path, image_size=(256, 256)):  
        mask = np.zeros(image_size, dtype=np.uint8)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts)
                center_x *= image_size[1]
                center_y *= image_size[0]
                bbox_width *= image_size[1]
                bbox_height *= image_size[0]
                xmin = max(0, int(center_x - bbox_width / 2))
                ymin = max(0, int(center_y - bbox_height / 2))
                xmax = min(image_size[1], int(center_x + bbox_width / 2))
                ymax = min(image_size[0], int(center_y + bbox_height / 2))
                mask[ymin:ymax, xmin:xmax] = int(class_id)
            elif len(parts) > 5:
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                if len(coords) % 2 != 0:
                    continue  # Skip if not even number of coordinates
                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= image_size[1]
                pts[:, 1] *= image_size[0]
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], int(class_id))
        return mask

# New helper functions for SVM feature extraction
def extract_bounding_boxes(mask_np):
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes

def extract_contours(mask_np):
    """
    Extract contours from the mask.
    """
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compute_polygon_average_lab(image_pil, contour):
    """
    Compute the average LAB color within the polygon defined by the contour.
    """
    image_np = np.array(image_pil)
    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    lab = color.rgb2lab(image_np)
    lab_pixels = lab[mask == 255]
    if lab_pixels.size == 0:
        return np.array([0, 0, 0])
    avg_lab = lab_pixels.mean(axis=0)
    return avg_lab

def simulated_label_from_filename(filename):
    # Placeholder: return 0 or derive label from filename if available
    return 0

def extract_features_and_labels(dataset, unet_model):
    """
    Run the trained UNet on the dataset to extract segmentation masks,
    then extract LAB color features from each polygon (contour),
    and collect corresponding labels determined by the mode of the predicted mask
    within the bounding rectangle.
    """
    features = []
    labels = []
    for i in range(len(dataset)):
        image_file = dataset.image_files[i]
        image_path = os.path.join(dataset.image_folder, image_file)
        # Open and resize the image.
        image_pil = Image.open(image_path).convert("RGB").resize((256, 256))
        
        # Get the ground-truth mask directly from the annotation file.
        txt_file = dataset.txt_files[i]
        txt_path = os.path.join(dataset.mask_folder, txt_file)
        gt_mask = dataset.create_mask_from_yolo(txt_path, image_size=(256, 256))
        
        # Debug: print unique classes in the ground truth mask.
        print(f"Processing {image_file} - Unique classes in GT mask: {np.unique(gt_mask)}")
        
        # Extract contours from the ground-truth mask.
        contours = extract_contours(gt_mask)
        if len(contours) == 0:
            print("No contours found in image (GT):", image_file)
        for contour in contours:
            feat = compute_polygon_average_lab(image_pil, contour)
            # Use the bounding rectangle of the contour and get the predominant label from the GT mask.
            box = cv2.boundingRect(contour)
            lbl = get_box_label(gt_mask, box)
            features.append(feat)
            labels.append(lbl)
    return np.array(features), np.array(labels)

# Define functions for UNet segmentation and SVM feature extraction
def segment_test_strip(unet_model, image):
    """Segment the full urine test strip (class ID 10)"""
    unet_model.eval()
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cuda().numpy()
    return pred_mask

def crop_test_strip(image, mask):
    """Crop the urine test strip from the image using the mask"""
    contours, _ = cv2.findContours((mask == 10).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image.crop((x, y, x + w, y + h)), (x, y, w, h)
    return image, (0, 0, image.width, image.height)

def segment_reagent_pads(unet_model, cropped_image):
    """Segment the reagent pads (class IDs 0-9) from the cropped test strip."""
    input_tensor = transforms.ToTensor()(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cuda().numpy()
    return pred_mask

def extract_features(image, mask, offset):
    """Extract LAB color features from segmented reagent pads."""
    x_offset, y_offset, _, _ = offset
    features, labels = [], []
    for class_id in range(10):
        contours, _ = cv2.findContours((mask == class_id).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = image.crop((x + x_offset, y + y_offset, x + x_offset + w, y + y_offset + h))
            roi_lab = color.rgb2lab(np.array(roi))
            features.append(roi_lab.mean(axis=(0, 1)))
            labels.append(class_id)
    return np.array(features), np.array(labels)

def get_box_label(mask, box):
    """
    Given a mask and a bounding box (x, y, w, h),
    compute the mode (most frequent value) of the mask within that box.
    """
    x, y, w, h = box
    region = mask[y:y+h, x:x+w]
    values, counts = np.unique(region, return_counts=True)
    return values[np.argmax(counts)]

def train_svm_classifier_with_early_stopping(features, labels, class_names, patience=5):
    if features.size == 0:
        raise ValueError("No features extracted. Please check your segmentation and feature extraction pipeline.")
    
    # Map integer labels to class names.
    mapped_labels = [class_names[int(lbl)] for lbl in labels]
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        features, mapped_labels, test_size=0.2, random_state=42
    )
    
    import itertools
    best_score = -np.inf
    best_params = None
    best_model = None
    no_improve_count = 0
    
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in param_combinations:
        svm = SVC(kernel='rbf', **params)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)
        print(f"Params: {params}, Val Accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
            best_model = svm
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered in SVM grid search.")
                break
    
    print("Best SVM parameters:", best_params, "with validation accuracy:", best_score)
    return best_model

def compute_mean_std(image_folder, mask_folder):
    transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL images to tensors
    dataset = UrineStripDataset(image_folder, mask_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (last batch can be smaller)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten spatial dimensions
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Main Training Process
def main():
    image_folder = r"Datasets/Test test/images"
    mask_folder = r"Datasets/Test test/labels"

    # Compute mean and std for normalization
    mean, std = compute_mean_std(image_folder, mask_folder)

    # Create two datasets with different transforms:
    full_dataset = UrineStripDataset(image_folder, mask_folder, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = list(range(train_size)), list(range(train_size, len(full_dataset)))
    
    train_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=RandomTrainTransformations(mean, std)),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=SimpleValTransformations(mean, std)),
        val_indices
    )   

    # DataLoaders with persistent workers:
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=False)

    
    torch.cuda.empty_cache()
    
    # Initialize Model and Optimizer
    num_classes = 11
    unet_model = UNet(in_channels=3, out_channels=num_classes)
    unet_model.to(device)
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 100
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        unet_model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc='Training Epoch', leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = unet_model(images)
                loss_focal = focal_loss(outputs, masks)
                loss_dice = dice_loss(outputs, masks)
                loss = 0.3 * focal_loss(outputs, masks) + 0.7 * dice_loss(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=0.5)
            for name, param in unet_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 100:
                        print(f"🔍 High Grad Norm [{name}]: {grad_norm:.2f}")
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        unet_model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_pixels = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = unet_model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == masks).sum().item()
                total_pixels += masks.shape[0] * masks.shape[1] * masks.shape[2]
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct_predictions / total_pixels
        val_accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(unet_model.state_dict(), model_filename)
            print("✅ Model improved and saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement in validation loss for {early_stop_counter}/{patience} epochs")
        if early_stop_counter >= patience:
            print("⛔ Early stopping triggered! Training stopped.")
            break

    torch.save(unet_model.state_dict(), model_filename)

    # Plot training results
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.show()

    # ----- SVM Training Section -----
    names = ['Bilirubin', 'Blood', 'Glucose', 'Ketone', 'Leukocytes', 'Nitrite', 'Protein', 'SpGravity', 'Urobilinogen', 'pH', 'strip']
    print("Extracting features for SVM training...")
    svm_features, svm_labels = extract_features_and_labels(full_dataset, unet_model)
    print("Training SVM classifier on extracted features with early stopping...")
    svm_model = train_svm_classifier_with_early_stopping(svm_features, svm_labels, names, patience=5)
    joblib.dump(svm_model, "svm_model.pkl")
    print("SVM classifier saved!")

if __name__ == '__main__':
    print(f"Using device: {device}")
    main()
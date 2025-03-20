import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch._dynamo
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt  # For plotting

# Additional imports for SVM and image processing
import cv2
from skimage import color
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True  # Enable cuDNN for performance

# Define the model filename with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    return torch.from_numpy(mask)

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = F.softmax(outputs, dim=1)  # Convert logits to probabilities

    # Ensure targets are squeezed to remove extra channel dimension
    targets = targets.squeeze(1)  # Shape becomes [B, H, W]

    # Ensure targets are integer type before one-hot encoding
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (outputs * targets_one_hot).sum(dim=(2,3))
    union = outputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))

    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()

def focal_loss(outputs, targets, alpha=0.25, gamma=2):
    """ Focal Loss for multi-class segmentation. """
    targets = targets.squeeze(1)  # Remove channel dim (B, H, W)
    targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1])  # Convert to one-hot
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Reshape to (B, C, H, W)

    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
    pt = torch.exp(-bce_loss)
    focal = (alpha * (1 - pt) ** gamma * bce_loss).mean()
    return focal

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10, dropout_prob=0.5):
        super(UNet, self).__init__()

        # Encoder path with BatchNorm and LeakyReLU
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),  # 🔥 Added BatchNorm
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 🔥 Changed to LeakyReLU
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # Decoder path
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

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

        # 🚀 Apply Xavier Initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Apply Kaiming Initialization for Conv2d and ConvTranspose2d layers (better for LeakyReLU)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use kaiming_uniform_ with 'leaky_relu' nonlinearity
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


    def forward(self, x):
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        #print(f"🔍 Activation after enc1: mean={enc1.abs().mean().item()}, max={enc1.abs().max().item()}")

        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        #print(f"🔍 Activation after enc2: mean={enc2.abs().mean().item()}, max={enc2.abs().max().item()}")

        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        #print(f"🔍 Activation after enc3: mean={enc3.abs().mean().item()}, max={enc3.abs().max().item()}")

        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)
        #print(f"🔍 Activation after enc4: mean={enc4.abs().mean().item()}, max={enc4.abs().max().item()}")

        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)
        #print(f"🔍 Activation after bottleneck: mean={bottleneck.abs().mean().item()}, max={bottleneck.abs().max().item()}")

        up3 = checkpoint.checkpoint(self.upconv3, bottleneck, use_reentrant=False)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)
        #print(f"🔍 Activation after up3: mean={up3.abs().mean().item()}, max={up3.abs().max().item()}")

        up2 = checkpoint.checkpoint(self.upconv2, up3, use_reentrant=False)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)
        #print(f"🔍 Activation after up2: mean={up2.abs().mean().item()}, max={up2.abs().max().item()}")

        up1 = checkpoint.checkpoint(self.upconv1, up2, use_reentrant=False)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)
        #print(f"🔍 Activation after up1: mean={up1.abs().mean().item()}, max={up1.abs().max().item()}")

        output = self.output(up1)
        #print(f"🔍 Activation after output: mean={output.abs().mean().item()}, max={output.abs().max().item()}")

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
    def __init__(self):
        self.joint_transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=10),
            RandomAffine(translate=(0.1, 0.1))
        ])
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Works with PIL
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),  # Converts PIL to Tensor
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Works with Tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert masks to tensor
            mask_to_tensor
        ])

    def __call__(self, sample):
        sample = self.joint_transform(sample)  # Apply augmentations that work with PIL
        image = self.image_transform(sample['image'])  # Convert to Tensor after PIL augmentations
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}



class SimpleValTransformations:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        txt_file = self.txt_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        txt_path = os.path.join(self.mask_folder, txt_file)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))
        mask = self.create_mask_from_yolo(txt_path)
        mask = Image.fromarray(mask).resize((128, 128), Image.NEAREST)
        if self.transform:
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def create_mask_from_yolo(self, txt_path, image_size=(128, 128)):  
        mask = np.zeros(image_size, dtype=np.uint8)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts[:5])

            # Convert YOLO normalized coordinates to absolute pixels
            center_x *= image_size[1]
            center_y *= image_size[0]
            bbox_width *= image_size[1]
            bbox_height *= image_size[0]

            xmin = max(0, int(center_x - bbox_width / 2))
            ymin = max(0, int(center_y - bbox_height / 2))
            xmax = min(image_size[1], int(center_x + bbox_width / 2))
            ymax = min(image_size[0], int(center_y + bbox_height / 2))

            # Set mask values to class ID
            mask[ymin:ymax, xmin:xmax] = int(class_id)

        return mask


def extract_bounding_boxes(mask_np):
    """
    Given a numpy mask, extract bounding boxes using OpenCV.
    Returns a list of boxes [(x, y, w, h), ...]
    """
    # Ensure mask is in uint8 format for OpenCV
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes

def compute_average_lab(image_pil, box):
    """
    Given a PIL image and a bounding box (x, y, w, h),
    crop the region, convert to Lab, and compute the mean Lab color.
    """
    x, y, w, h = box
    cropped = image_pil.crop((x, y, x + w, y + h))
    cropped_np = np.array(cropped)
    # Convert from RGB to Lab using skimage.color
    lab = color.rgb2lab(cropped_np)
    mean_lab = lab.mean(axis=(0, 1))
    return mean_lab

def simulated_label_from_filename(filename):
    """
    Placeholder: extract or simulate a label for the ROI.
    In practice, use a mapping based on ground truth.
    """
    # For demonstration, simply return 0 (or parse the filename)
    return 0

def extract_features_and_labels(dataset, unet_model):
    """
    Run the trained U-Net on the dataset to extract segmentation masks,
    then extract features (e.g., average Lab color) from each bounding box,
    and collect corresponding labels.
    """
    unet_model.eval()
    features = []
    labels = []
    # Use the original PIL images from the dataset (without transforms)
    for i in range(len(dataset)):
        # Load original image (without transform) for feature extraction
        image_file = dataset.image_files[i]
        image_path = os.path.join(dataset.image_folder, image_file)
        image_pil = Image.open(image_path).convert("RGB").resize((128, 128))
        # Get the corresponding mask using the dataset's method
        txt_file = dataset.txt_files[i]
        txt_path = os.path.join(dataset.mask_folder, txt_file)
        mask_pil = Image.fromarray(dataset.create_mask_from_yolo(txt_path)).resize((128, 128), Image.NEAREST)
        # Convert image to tensor and add batch dimension for U-Net inference
        input_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = unet_model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        # Extract bounding boxes from the predicted mask
        boxes = extract_bounding_boxes(pred_mask)
        for box in boxes:
            feat = compute_average_lab(image_pil, box)
            features.append(feat)
            # Here, we simulate label extraction (in a real scenario, use annotated data)
            labels.append(simulated_label_from_filename(image_file))
    return np.array(features), np.array(labels)

def train_svm_classifier(features, labels):
    """
    Train an SVM classifier with an RBF kernel using GridSearchCV.
    """
    param_grid = {'C': [0.1, 1, 10],
                   'gamma': [0.001, 0.01, 0.1]}
    svm = SVC(kernel='rbf')
    grid = GridSearchCV(svm, param_grid, cv=5)
    grid.fit(features, labels)
    print("Best SVM parameters:", grid.best_params_)
    return grid.best_estimator_

def main():
    image_folder = r"almost 1k dataset/train/images"
    mask_folder = r"almost 1k dataset/train/labels"

    # Create two datasets with different transforms:
    full_dataset = UrineStripDataset(image_folder, mask_folder, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = list(range(train_size)), list(range(train_size, len(full_dataset)))
    
    train_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=RandomTrainTransformations()),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=SimpleValTransformations()),
        val_indices
    )   

    # DataLoaders with persistent workers:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)


    # Initialize Model and Optimizer
    num_classes = 10
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

    best_val_loss = float('inf')  # Set to infinity so the first validation loss will always be lower.
    early_stop_counter = 0
    patience = 10  # Or whichever number of epochs you want to wait without improvement.

    for epoch in range(num_epochs):
        unet_model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc='Training Epoch', leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):  # Mixed precision training
                outputs = unet_model(images)
                loss_focal = focal_loss(outputs, masks)  # Remove channel dimension
                loss_dice = dice_loss(outputs, masks)
                loss = 0.3 * focal_loss(outputs, masks) + 0.7 * dice_loss(outputs, masks)



            scaler.scale(loss).backward()

            # 🚀 Apply Gradient Clipping Before Updating
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=0.5)  # 🔥 Reduced from 0.1 to 0.05
            
            # 🚀 Monitor Gradient Norms
            for name, param in unet_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 100:  # 🚀 Only print if grad norm is large
                        print(f"🔍 High Grad Norm [{name}]: {grad_norm:.2f}")

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # 🚀 Validation Phase
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

                # 🚀 Fixed total_pixels calculation
                total_pixels += masks.shape[0] * masks.shape[1] * masks.shape[2]  # Only H×W, ignoring batch & channels

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

if __name__ == '__main__':
    print(f"Using device: {device}")
    main()
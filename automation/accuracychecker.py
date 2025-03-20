import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Additional Conv2d layers after concatenation to reduce channels
        self.conv_up4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_up3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck layer
        bottleneck = self.bottleneck(enc4)

        # Decoder path
        up4 = self.upconv4(bottleneck)
        up4 = F.interpolate(up4, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up4 = torch.cat([up4, enc4], dim=1)  # Concatenate encoder feature maps with upsampled bottleneck
        up4 = self.conv_up4(up4)  # Apply Conv2d to match the expected number of channels

        up3 = self.upconv3(up4)
        up3 = F.interpolate(up3, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc3], dim=1)  # Concatenate encoder feature maps with upsampled feature maps
        up3 = self.conv_up3(up3)  # Apply Conv2d to match the expected number of channels

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc2], dim=1)  # Concatenate encoder feature maps with upsampled feature maps
        up2 = self.conv_up2(up2)  # Apply Conv2d to match the expected number of channels

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc1], dim=1)  # Concatenate encoder feature maps with upsampled feature maps
        up1 = self.conv_up1(up1)  # Apply Conv2d to match the expected number of channels

        # Final output layer
        output = self.output(up1)

        return output

class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        # List image files
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.txt')]  # Change to .txt files

        # Ensure both lists have the same length
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) does not match number of masks ({len(self.mask_files)})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image and mask
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]

        image_path = os.path.join(self.image_folder, image_file)
        mask_path = os.path.join(self.mask_folder, mask_file)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded")

        # Create mask from YOLO annotation
        mask = create_mask_from_yolo(mask_path)

        # Preprocess image (resize, normalize, etc.)
        image = cv2.resize(image, (256, 256))  # Resize to 256x256
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format

        # Convert the NumPy array to a PIL Image with proper dtype
        image = Image.fromarray((image * 255).astype(np.uint8).transpose(1, 2, 0))  # Convert back to uint8

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)  # Transform will handle ToTensor properly

        # Ensure the mask is a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

# Load the trained model
unet_model = UNet(in_channels=3, out_channels=1)
unet_model.load_state_dict(torch.load('best_unet_model.pth'))  # Path to your model

# Set the model to evaluation mode
unet_model.eval()

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model.to(device)

# Define your dataset again (no transformation, only original)
dataset = UrineStripDataset(image_folder, mask_folder, transform=None)

# Create DataLoader for validation
val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Function to calculate IoU and Dice score
def calculate_metrics(preds, targets):
    # Flatten the predictions and targets for evaluation
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    # Calculate IoU
    iou = jaccard_score(targets, preds, average='binary')

    # Calculate Dice score
    intersection = np.sum(targets * preds)
    dice = 2 * intersection / (np.sum(targets) + np.sum(preds))

    return iou, dice

# Store metrics for each batch
ious = []
dices = []

# Evaluate the model
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Evaluating", ncols=100, leave=True):
        images, masks = images.to(device), masks.to(device)

        # Get the model's predictions
        outputs = unet_model(images)
        preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities (since it's a binary mask)
        preds = (preds > 0.5).float()  # Binarize predictions

        # Calculate IoU and Dice score
        iou, dice = calculate_metrics(preds, masks)

        ious.append(iou)
        dices.append(dice)

# Calculate the average metrics
avg_iou = np.mean(ious)
avg_dice = np.mean(dices)

print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice score: {avg_dice:.4f}")

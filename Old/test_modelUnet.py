import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from icecream import ic
from torch.utils.checkpoint import checkpoint
from ultralytics.nn.tasks import DetectionModel  # Import the required module

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

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=11, dropout_prob=0.3):
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

model = UNet(in_channels=3, out_channels=11).to(device)
torch.serialization.add_safe_globals([DetectionModel])  # Add DetectionModel to safe globals
model.load_state_dict(torch.load(r'D:\Programming\urine_interpret\models\unet_model_20250221-100214.pt_epoch_78.pt', map_location=device, weights_only=False), strict=False)  # Set weights_only to False
model.eval()
ic("Model loaded.")

# ==== Load and Transform Image ====
image_path = r'D:\Programming\urine_interpret\Datasets\Test test\images\IMG_2978_png.rf.847fdb3c89e2b7d091f00881edb10506.jpg'
image = Image.open(image_path).convert("RGB")

# Normalize using mean and std deviation for better brightness preservation
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0).to(device)
print("Input image tensor min/max:", torch.min(image_tensor), torch.max(image_tensor))

# Visualize the input image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")
plt.show()

# ==== Run Model ====
with torch.no_grad():
    model.eval()
    prediction = model(image_tensor)

    # Apply Softmax for multi-class segmentation
    prediction = torch.softmax(prediction, dim=1)

    # Get the class with the highest probability
    predicted_class = torch.argmax(prediction, dim=1)

    print("Unique predicted classes:", torch.unique(predicted_class))  # Debugging output
    print("Prediction shape:", prediction.shape)  # Debugging output
    print("Prediction min/max:", torch.min(prediction), torch.max(prediction))  # Debugging output

# Visualize the predicted mask
predicted_mask = predicted_class.squeeze(0).cpu().numpy()
plt.figure(figsize=(6, 6))
plt.imshow(predicted_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")
plt.show()

# ==== Process Predicted Mask ====
mask = prediction.squeeze(0).cpu().numpy()
mask = np.argmax(mask, axis=0).astype(np.uint8)

print("Unique values in mask:", np.unique(mask))  # Debugging output

# Visualize the processed mask
plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap='gray')
plt.title("Processed Mask")
plt.axis("off")
plt.show()

# Create a color mask
color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for class_id, color in class_colors.items():
    color_mask[mask == class_id] = color

# Visualize the color mask
plt.figure(figsize=(6, 6))
plt.imshow(color_mask)
plt.title("Color Mask")
plt.axis("off")
plt.show()

# ==== Debugging Output ====
ic(np.unique(mask))  # Should show different values if segmentation works

# ==== Detect Full Urine Test Strip ====
full_strip_mask = (mask > 0).astype(np.uint8) * 255
cv2.imwrite("full_strip_mask.jpg", full_strip_mask)  # Save for debugging

contours, _ = cv2.findContours(full_strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_np = np.array(image)

if len(contours) == 0:
    ic("No full strip detected!")

# Scale factor to align bounding boxes with the original image
scale_x = image_np.shape[1] / mask.shape[1]
scale_y = image_np.shape[0] / mask.shape[0]

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)
    cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for full strip

# ==== Detect Reagent Pads with Bounding Boxes ====
for class_id in range(1, 10):  # Ignore background
    pad_mask = (mask == class_id).astype(np.uint8) * 255
    cv2.imwrite(f"pad_mask_{class_id}.jpg", pad_mask)  # Save for debugging

    contours, _ = cv2.findContours(pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        ic(f"No reagent pads detected for class {class_id}")

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        cv2.rectangle(image_np, (x, y), (x + w, y + h), class_colors[class_id], 2)  # Use class color for bounding box

print("Raw logits:", torch.min(prediction), torch.max(prediction))

# ==== Display Result ====
plt.figure(figsize=(6, 6))
plt.imshow(image_np)
plt.title("Urine Test Strip & Reagent Pads Detection with Bounding Boxes")
plt.axis("off")
plt.show()

# Convert multi-class prediction to single-channel mask
binary_mask = torch.argmax(prediction, dim=1).squeeze(0)  # Shape: (512, 512)

plt.imshow(color_mask)
plt.title("Predicted Mask with Colors")
plt.axis("off")
plt.show()

print(f"binary_mask shape: {binary_mask.shape}")  # Should be (512, 512)


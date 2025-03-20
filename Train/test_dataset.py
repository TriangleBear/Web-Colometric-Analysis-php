import os
from datasets import UrineStripDataset

# Define paths for images and masks
image_dir = '/content/urine_interpret/Datasets/Split_70_20_10/train/images'  # Update with your image directory
mask_dir = '/content/urine_interpret/Datasets/Split_70_20_10/train/labels'    # Update with your mask directory

# Create an instance of the dataset
dataset = UrineStripDataset(image_dir=image_dir, mask_dir=mask_dir)

# Iterate through a few samples to check class distribution
for i in range(5):  # Change the range as needed
    image_tensor, label, class_distribution = dataset[i]
    print(f"Sample {i}: Label = {label}, Class Distribution = {class_distribution}")

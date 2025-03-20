import os
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))

from Train.config import (
    TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, 
    VALID_IMAGE_FOLDER, VALID_MASK_FOLDER
)
from Train.datasets import UrineStripDataset

# Class names
CLASS_NAMES = {
    0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone',
    4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity',
    8: 'Urobilinogen', 9: 'Background', 10: 'pH', 11: 'Strip'
}

def analyze_dataset(name, dataset):
    """Analyze the class distribution in a dataset"""
    print(f"\nAnalyzing {name} dataset...")
    class_counts = {}
    
    for i, (_, label, _) in enumerate(tqdm(dataset, desc=f"Scanning {name}")):
        label_value = label.item() if hasattr(label, 'item') else label
        
        if label_value not in class_counts:
            class_counts[label_value] = 0
        class_counts[label_value] += 1
    
    print(f"\nClass distribution in {name} dataset:")
    for cls in sorted(class_counts.keys()):
        cls_name = CLASS_NAMES.get(cls, f"Class {cls}")
        count = class_counts[cls]
        percentage = 100 * count / len(dataset)
        print(f"  {cls_name}: {count} samples ({percentage:.1f}%)")
    
    return class_counts

def check_annotation_files(mask_folder):
    """Analyze the actual annotation files directly"""
    print(f"\nChecking annotation files in {mask_folder}...")
    
    label_files = [f for f in os.listdir(mask_folder) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files")
    
    class_counts = {}
    samples_with_class = {}
    
    for file in tqdm(label_files, desc="Reading labels"):
        file_path = os.path.join(mask_folder, file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        
                        if class_id not in class_counts:
                            class_counts[class_id] = 0
                            samples_with_class[class_id] = 0
                        
                        class_counts[class_id] += 1
                        samples_with_class[class_id] = samples_with_class.get(class_id, 0) + 1
                    except ValueError:
                        continue
    
    print("\nClass distribution in annotation files:")
    for cls in sorted(class_counts.keys()):
        cls_name = CLASS_NAMES.get(cls, f"Class {cls}")
        count = class_counts[cls]
        sample_count = samples_with_class.get(cls, 0)
        print(f"  {cls_name}: {count} annotations in {sample_count} files")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(
        [CLASS_NAMES.get(cls, f"Class {cls}") for cls in sorted(class_counts.keys())],
        [samples_with_class.get(cls, 0) for cls in sorted(class_counts.keys())]
    )
    plt.title('Class Distribution in Annotation Files')
    plt.xlabel('Classes')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print(f"Class distribution plot saved to 'class_distribution.png'")
    
    return class_counts, samples_with_class

if __name__ == "__main__":
    # Load datasets
    print("Loading datasets...")
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(valid_dataset)} samples")
    
    # Analyze datasets
    train_counts = analyze_dataset("Train", train_dataset)
    valid_counts = analyze_dataset("Validation", valid_dataset)
    
    # Check annotation files directly
    train_annot_counts, train_sample_counts = check_annotation_files(TRAIN_MASK_FOLDER)
    valid_annot_counts, valid_sample_counts = check_annotation_files(VALID_MASK_FOLDER)
    
    # Show files for a specific class if present
    missing_classes = []
    for cls in range(12):
        if cls not in train_counts and cls not in valid_counts:
            missing_classes.append(cls)
    
    if missing_classes:
        print("\nMISSING CLASSES:")
        for cls in missing_classes:
            print(f"  {CLASS_NAMES.get(cls, f'Class {cls}')} is missing from both datasets")
    
    print("\nAnalysis Complete")

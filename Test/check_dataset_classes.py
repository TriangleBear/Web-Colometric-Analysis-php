import os
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))

# Import required modules
from Train.config import (
    TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER, VALID_MASK_FOLDER,
    TEST_IMAGE_FOLDER, TEST_MASK_FOLDER
)
from Train.datasets import UrineStripDataset, CLASS_NAMES

def analyze_dataset_classes(name, dataset_path, label_path):
    """Analyze the class distribution in a dataset folder"""
    print(f"\n{'='*30} {name.upper()} DATASET ANALYSIS {'='*30}")
    
    # Examine raw label files directly
    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files in {label_path}")
    
    # Initialize counters
    all_file_classes = []
    class_counts = Counter()
    file_class_counts = Counter()
    files_with_multiple_classes = 0
    
    # Examine each label file
    for label_file in tqdm(label_files, desc=f"Analyzing {name} labels"):
        try:
            file_path = os.path.join(label_path, label_file)
            file_classes = set()
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:  # Valid YOLO format needs at least 5 parts
                        class_id = int(parts[0])
                        file_classes.add(class_id)
                        class_counts[class_id] += 1
            
            # Track file level statistics
            if file_classes:
                all_file_classes.append(list(file_classes))
                for cls in file_classes:
                    file_class_counts[cls] += 1
                    
                if len(file_classes) > 1:
                    files_with_multiple_classes += 1
                    
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    # Print raw label file analysis
    print("\nRAW LABEL FILE ANALYSIS:")
    print(f"Total files with classes: {sum(file_class_counts.values())}")
    print(f"Files with multiple classes: {files_with_multiple_classes} ({files_with_multiple_classes/len(label_files)*100:.1f}% of files)")
    print(f"Number of unique classes found: {len(class_counts)}")
    
    print("\nClass distribution in raw files:")
    for class_id, count in sorted(file_class_counts.items()):
        class_name = CLASS_NAMES.get(class_id, f"Unknown class {class_id}")
        percentage = count / len(label_files) * 100
        print(f"  Class {class_id} ({class_name}): {count} files ({percentage:.1f}%)")
    
    # Now load the dataset and analyze how it's reading the classes
    print("\nLoading dataset with UrineStripDataset class...")
    try:
        dataset = UrineStripDataset(dataset_path, label_path)
        print(f"Dataset loaded: {len(dataset)} samples")
        
        dataset_classes = []
        dataset_class_counts = Counter()
        
        # Look at a larger sample to be sure
        sample_size = min(len(dataset), 500)  # Check first 500 samples
        
        for i in tqdm(range(sample_size), desc="Sampling dataset"):
            try:
                _, label, _ = dataset[i]
                dataset_classes.append(label)
                dataset_class_counts[label] += 1
                
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
        
        print("\nClass distribution in dataset samples:")
        for class_id, count in sorted(dataset_class_counts.items()):
            class_name = CLASS_NAMES.get(class_id, f"Unknown class {class_id}")
            percentage = count / sample_size * 100
            print(f"  Class {class_id} ({class_name}): {count} samples ({percentage:.1f}%)")
            
        # Create histograms
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Raw Label Files - {name}")
        plt.bar([CLASS_NAMES.get(cls, f"Class {cls}") for cls in sorted(file_class_counts.keys())], 
                [file_class_counts[cls] for cls in sorted(file_class_counts.keys())])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Number of files")
        
        plt.subplot(1, 2, 2)
        plt.title(f"Dataset Samples - {name}")
        plt.bar([CLASS_NAMES.get(cls, f"Class {cls}") for cls in sorted(dataset_class_counts.keys())], 
                [dataset_class_counts[cls] for cls in sorted(dataset_class_counts.keys())])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Number of samples")
        
        plt.tight_layout()
        plt.savefig(f"{name}_class_distribution.png")
        print(f"Saved class distribution plot to {name}_class_distribution.png")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
    
    return file_class_counts, dataset_class_counts if 'dataset_class_counts' in locals() else None

def inspect_annotation_files(label_path, num_samples=5):
    """Inspect the content of a few random annotation files"""
    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    
    if not label_files:
        print(f"No label files found in {label_path}")
        return
    
    print(f"\nInspecting {num_samples} random annotation files from {label_path}:")
    
    # Randomly select some files
    import random
    random.shuffle(label_files)
    samples = label_files[:num_samples]
    
    for idx, file_name in enumerate(samples):
        file_path = os.path.join(label_path, file_name)
        print(f"\nSample {idx+1}: {file_name}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content:
                    print("File content:")
                    for line_idx, line in enumerate(content.split('\n')):
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_name = CLASS_NAMES.get(class_id, f"Unknown class {class_id}")
                            print(f"  Line {line_idx+1}: Class {class_id} ({class_name}) - {line}")
                else:
                    print("File is empty")
        except Exception as e:
            print(f"  Error reading file: {e}")

def main():
    """Main function to check dataset classes across all splits"""
    # Create plots directory
    os.makedirs("class_analysis_plots", exist_ok=True)
    
    # First, let's inspect some actual annotation files
    print("\n== SAMPLE ANNOTATION FILE INSPECTION ==")
    inspect_annotation_files(TRAIN_MASK_FOLDER, num_samples=10)
    
    # Analyze each dataset split
    train_file_counts, train_dataset_counts = analyze_dataset_classes("Training", TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_file_counts, valid_dataset_counts = analyze_dataset_classes("Validation", VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_file_counts, test_dataset_counts = analyze_dataset_classes("Test", TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)
    
    # Summarize findings and check dataset balance
    print("\n" + "="*80)
    print("DATASET SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    # Check how many unique classes we have in the entire dataset
    all_classes = set(list(train_file_counts.keys()) + list(valid_file_counts.keys()) + list(test_file_counts.keys()))
    
    print(f"Total unique classes found across all splits: {len(all_classes)}")
    
    if len(all_classes) <= 1:
        print("\nCRITICAL ISSUE: Your dataset contains only one class!")
        print("For multi-class classification, you need examples of multiple classes.")
        print("\nPossible solutions:")
        print("1. Check your data annotation process - make sure different objects are assigned different class IDs.")
        print("2. If your dataset genuinely only has one class, consider using binary classification instead.")
        print("   - Modify your SVM training to use binary classification (class vs. background)")
        print("3. Add more labeled data with different classes.")
        
    else:
        # Check for class imbalance
        imbalanced_splits = []
        for split_name, counts in [("Training", train_dataset_counts), 
                                ("Validation", valid_dataset_counts), 
                                ("Test", test_dataset_counts)]:
            if counts and len(counts) <= 1:
                imbalanced_splits.append(split_name)
                
        if imbalanced_splits:
            print(f"\nWARNING: The following splits contain only one class: {', '.join(imbalanced_splits)}")
            print("SVM training requires multiple classes in the training data.")
            print("\nPossible solutions:")
            print("1. Redistribute your data to ensure multiple classes in each split.")
            print("2. Use data augmentation to balance classes.")
            print("3. Ensure your dataset's __getitem__ method is correctly extracting class information.")
        else:
            print("\nNo critical class balance issues detected.")

if __name__ == "__main__":
    main()

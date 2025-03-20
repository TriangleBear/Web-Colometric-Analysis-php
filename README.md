# Urine Interpretation Project

This project is part of the thesis titled "Colorimetric Analysis of Urine Test Strips Using U-Net and Support Vector Machine" by Buenaventura, Ubando, and Zacarias. The goal of this project is to develop a deep learning model using a UNet architecture for segmenting urine test strips and reagent pads, and to classify the segmented regions based on LAB color features using an SVM classifier.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TriangleBear/urine_interpret.git
    cd urine_interpret
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Compute Dataset Statistics**:
    ```python
    from config import IMAGE_FOLDER, MASK_FOLDER
    from datasets import UrineStripDataset
    from utils import compute_mean_std

    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    mean, std = compute_mean_std(dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    ```

2. **Train the UNet Model**:
    ```python
    from train_unet import train_unet

    unet_model, train_losses, val_losses, val_accuracies = train_unet()
    ```

3. **Extract Features and Train SVM**:
    ```python
    from utils import extract_features_and_labels, train_svm_classifier, save_svm_model

    features, labels = extract_features_and_labels(dataset, unet_model)
    svm_model = train_svm_classifier(features, labels)
    save_svm_model(svm_model, get_svm_filename())
    ```

4. **Evaluate the SVM Model**:
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    svm_accuracy = svm_model.score(X_test, y_test) * 100
    print(f"SVM RBF Accuracy: {svm_accuracy:.2f}%")
    ```

## Project Structure

```
urine_interpret/
├── automation/          # Automation scripts
├── datasets/            # Dataset-related files
├── models/              # Model definitions and saved models
├── php/                 # PHP files
├── tests/               # Unit and integration tests
├── train/               # Training scripts and configurations
├── utils/               # Utility functions
├── assets/              # Static assets like images or XML files
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

## Disclaimer

This project requires the stable version of CUDA. Ensure that you have the correct version of CUDA installed and configured on your system. You can check the compatibility of your CUDA version with PyTorch [here](https://pytorch.org/get-started/previous-versions/) and check the Release compatibility Matric [here](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).

## Thesis Reference

For more detailed information about the project, please refer to the thesis document:
**"Colorimetric Analysis of Urine Test Strips Using U-Net and Support Vector Machine"** by Buenaventura, Ubando, and Zacarias.

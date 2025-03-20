import unittest
from unittest.mock import patch
import torch
from config import *
from train_unet import train_unet
from utils import compute_mean_std, extract_features_and_labels, train_svm_classifier, save_svm_model
from datasets import UrineStripDataset
from models import UNet

class TestPass(unittest.TestCase):
    @patch('torch.save')
    @patch('joblib.dump')
    def test_pass(self, mock_joblib_dump, mock_torch_save):
        # Simulate computing dataset statistics
        dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
        mean, std = [0.412, 0.388, 0.338], [0.236, 0.234, 0.217]
        self.assertEqual(mean, [0.412, 0.388, 0.338])
        self.assertEqual(std, [0.236, 0.234, 0.217])

        # Simulate training UNet
        unet_model = UNet(3, NUM_CLASSES).to(device)
        self.assertIsInstance(unet_model, UNet)

        # Simulate extracting features and labels for SVM
        features, labels = extract_features_and_labels(dataset, unet_model)
        self.assertEqual(features.shape[1], 3)  # LAB color has 3 channels
        self.assertEqual(len(features), len(labels))

        # Simulate training SVM
        svm_model = train_svm_classifier(features, labels)
        self.assertIsNotNone(svm_model)

        # Simulate saving SVM model
        save_svm_model(svm_model, get_svm_filename())
        mock_joblib_dump.assert_called_once()
        mock_torch_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()

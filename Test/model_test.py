import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import pickle
from inference_sdk import InferenceHTTPClient
from torchvision import transforms
from Train.utils import compute_mean_std  # Import the compute_mean_std function

# Define color scheme for visualization
CLASS_COLORS = [
    (255, 0, 0),    # 0: Bilirubin (Red)
    (0, 0, 255),    # 1: Blood (Blue)
    (0, 255, 0),    # 2: Glucose (Green)
    (255, 255, 0),  # 3: Ketone (Yellow)
    (255, 0, 255),  # 4: Leukocytes (Magenta)
    (0, 255, 255),  # 5: Nitrite (Cyan)
    (128, 0, 0),    # 6: Protein (Maroon)
    (0, 128, 0),    # 7: SpGravity (Dark Green)
    (0, 0, 128),    # 8: Urobilinogen (Navy)
    (128, 128, 128),# 9: Background (Gray)
    (255, 165, 0),  # 10: pH (Orange)
    (75, 0, 130)    # 11: Strip (Indigo)
]

# Class names for reporting
CLASS_NAMES = {
    0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone',
    4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity',
    8: 'Urobilinogen', 9: 'background', 10: 'pH', 11: 'strip'
}

# Try importing local models, but continue if they're not available
try:
    # Add parent directory to path for local imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Import directly from train directory
    from Train.models import UNetYOLO
    from Train.config import NUM_CLASSES, IMAGE_SIZE
    LOCAL_MODELS_AVAILABLE = True
    print("Local models successfully imported")
except ImportError as e:
    print(f"Warning: Could not import local models: {e}")
    LOCAL_MODELS_AVAILABLE = False
    NUM_CLASSES = 12  # Default if can't import from config
    
class RoboflowUrineAnalyzer:
    def __init__(self, api_key, workspace_name, workflow_id):
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        
    def analyze_image(self, image_path, confidence_threshold=0.5):
        """Process an image using Roboflow API"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None
            
        print(f"Analyzing image: {image_path}")
        
        # Run inference through Roboflow
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                use_cache=True
            )
            
            # Debug: Print the raw result structure
            print("\nRoboflow API Response Structure:")
            print(f"Result type: {type(result)}")
            # Rest of the debug output
            
            # Load original image for visualization
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Process and visualize results
            visualization = self.visualize_results(image_np, result, confidence_threshold)
            
            return {
                'original_image': image_np,
                'result_image': visualization,
                'predictions': result,
            }
        except Exception as e:
            print(f"Error calling Roboflow API: {e}")
            # Return a minimal result with just the original image
            image = Image.open(image_path)
            image_np = np.array(image)
            return {
                'original_image': image_np,
                'result_image': image_np.copy(),  # No modifications
                'predictions': None,
                'error': str(e)
            }
    
    def visualize_results(self, image_np, result, confidence_threshold=0.5):
        """Draw bounding boxes and labels on the image"""
        result_image = image_np.copy()
        
        # Check if we received a modified image output directly
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'output' in result[0]:
                # For base64 encoded images, we already handled this in analyze_image
                # and the processed image is already in result_image
                print("Using pre-processed image output from API response")
                return result_image
        
        # Check for empty result
        if result is None:
            print("Result is None")
            return result_image
            
        # Try different formats that Roboflow API might return
        try:
            # Format 1: List of tasks with individual tasks results
            if isinstance(result, list):
                for task_result in result:
                    if isinstance(task_result, dict):
                        # Debug print for each task
                        print(f"Task keys: {list(task_result.keys())}")
                        
                        # Try to find predictions in any of these common structures
                        if 'predictions' in task_result:
                            self._draw_predictions(result_image, task_result['predictions'], confidence_threshold)
                        elif 'Bounding Box' in task_result and 'predictions' in task_result['Bounding Box']:
                            self._draw_predictions(result_image, task_result['Bounding Box']['predictions'], confidence_threshold)
                        elif 'tasks' in task_result:
                            for subtask in task_result['tasks']:
                                if 'predictions' in subtask:
                                    self._draw_predictions(result_image, subtask['predictions'], confidence_threshold)
                                    
            # Format 2: Dictionary with 'tasks' or direct 'predictions'
            elif isinstance(result, dict):
                # Debug print for dictionary result
                print(f"Result dict keys: {list(result.keys())}")
                
                if 'predictions' in result:
                    self._draw_predictions(result_image, result['predictions'], confidence_threshold)
                elif 'tasks' in result:
                    for task in result['tasks']:
                        if 'predictions' in task:
                            self._draw_predictions(result_image, task['predictions'], confidence_threshold)
                
            else:
                print(f"Unexpected result type: {type(result)}")
        
        except Exception as e:
            print(f"Error during visualization: {e}")
            
        return result_image
        
    def _draw_predictions(self, image, predictions, confidence_threshold):
        """Helper method to draw predictions on image"""
        if not isinstance(predictions, list):
            print(f"Predictions is not a list: {type(predictions)}")
            return
            
        print(f"Processing {len(predictions)} predictions")
        
        for pred in predictions:
            if not isinstance(pred, dict):
                print(f"Prediction is not a dictionary: {type(pred)}")
                continue
                
            # Debug print first prediction to see its structure
            if predictions.index(pred) == 0:
                print(f"Example prediction: {pred}")
                
            # Check confidence threshold
            confidence = pred.get('confidence', 0)
            if confidence < confidence_threshold:
                continue
            
            # Extract coordinates - handle different possible formats
            if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                # YOLO-style centerpoint format
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                
                # Calculate absolute coordinates (top-left and bottom-right)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
            elif 'x' in pred and 'y' in pred and 'w' in pred and 'h' in pred:
                # Alternative naming
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('w', 0)
                h = pred.get('h', 0)
                
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
            elif 'bbox' in pred:
                # Bbox array format [x1, y1, width, height] or [x1, y1, x2, y2]
                bbox = pred['bbox']
                if len(bbox) == 4:
                    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # Likely [x1,y1,x2,y2] format
                        x1, y1, x2, y2 = bbox
                    else:  # Likely [x1,y1,w,h] format
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h
            elif 'box' in pred:
                # Another common format
                box = pred['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            else:
                # Skip if no valid coordinates
                print(f"Skipping prediction - no valid coordinates: {pred}")
                continue
                
            # Get class information
            class_name = pred.get('class', pred.get('class_name', 'Unknown'))
            class_id = pred.get('class_id', 9)  # Default to background if not found
            
            # Get color based on class_id
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            
            # Make sure coordinates are valid - FIX: Strict boundary checking
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1]-1, int(x2))  # Subtract 1 to stay within bounds
            y2 = min(image.shape[0]-1, int(y2))  # Subtract 1 to stay within bounds
            
            # Skip if invalid after clamping
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                print(f"Invalid coordinates after clamping: [{x1}, {y1}, {x2}, {y2}] for image shape {image.shape}")
                continue
            
            # Draw bounding box and label
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{class_name}: {confidence:.2f}", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Error drawing on image: {e}")
    
    def save_results(self, results, save_path):
        """Save results to a file"""
        if results and 'result_image' in results:
            cv2.imwrite(save_path, cv2.cvtColor(results['result_image'], cv2.COLOR_RGB2BGR))
            print(f"Results saved to {save_path}")
    
    def extract_strip(self, image_np, predictions, confidence_threshold=0.5):
        """Extract the strip region if detected"""
        strip_image = None
        strip_bbox = None
        
        print("Attempting to extract strip from image...")
        print(f"Image shape: {image_np.shape}")
        
        # Debug the predictions structure
        if isinstance(predictions, list):
            print(f"Predictions is a list with {len(predictions)} items")
            for i, pred_item in enumerate(predictions):
                if isinstance(pred_item, dict):
                    print(f"Item {i} keys: {list(pred_item.keys())}")
                    
                    # Look for classification results
                    if 'classification_label_visualization' in pred_item:
                        vis_data = pred_item['classification_label_visualization']
                        if isinstance(vis_data, dict) and 'strip' in vis_data:
                            print(f"Found 'strip' in classification with confidence: {vis_data['strip']}")
                    
                    # Look for predictions in various formats
                    for key in ['predictions', 'Bounding Box']:
                        if key in pred_item and isinstance(pred_item[key], dict) and 'predictions' in pred_item[key]:
                            print(f"Checking predictions in {key}")
                            for bbox_pred in pred_item[key]['predictions']:
                                if isinstance(bbox_pred, dict) and 'class' in bbox_pred:
                                    class_name = bbox_pred.get('class', '').lower()
                                    if 'strip' in class_name:
                                        print(f"Found strip bounding box: {bbox_pred}")
        
        # Try different ways to find the strip
        try:
            # Method 1: Try to find strip in Roboflow predictions - more thorough search
            if isinstance(predictions, list):
                # Iterate through all prediction items
                for pred_item in predictions:
                    # Check direct predictions list
                    if isinstance(pred_item, dict):
                        # Check standard predictions format
                        if 'predictions' in pred_item:
                            for pred in pred_item['predictions']:
                                class_name = str(pred.get('class', '')).lower()
                                if 'strip' in class_name and pred.get('confidence', 0) >= confidence_threshold:
                                    print(f"Found strip in standard predictions: {pred}")
                                    strip_image, strip_bbox = self._extract_region_from_prediction(image_np, pred)
                                    if strip_image is not None:
                                        return strip_image, strip_bbox
                        
                        # Check nested format (Bounding Box)
                        if 'Bounding Box' in pred_item and 'predictions' in pred_item['Bounding Box']:
                            for pred in pred_item['Bounding Box']['predictions']:
                                class_name = str(pred.get('class', '')).lower()
                                if 'strip' in class_name and pred.get('confidence', 0) >= confidence_threshold:
                                    print(f"Found strip in Bounding Box predictions: {pred}")
                                    strip_image, strip_bbox = self._extract_region_from_prediction(image_np, pred)
                                    if strip_image is not None:
                                        return strip_image, strip_bbox
                        
                        # Check tasks format
                        if 'tasks' in pred_item:
                            for task in pred_item['tasks']:
                                if isinstance(task, dict) and 'predictions' in task:
                                    for pred in task['predictions']:
                                        class_name = str(pred.get('class', '')).lower()
                                        if 'strip' in class_name and pred.get('confidence', 0) >= confidence_threshold:
                                            print(f"Found strip in tasks predictions: {pred}")
                                            strip_image, strip_bbox = self._extract_region_from_prediction(image_np, pred)
                                            if strip_image is not None:
                                                return strip_image, strip_bbox
            
            # Method 2: Try to use aspect ratio-based detection (most urine strips are long and narrow)
            print("No strip found in predictions, trying aspect ratio detection...")
            strip_image, strip_bbox = self._detect_strip_by_aspect_ratio(image_np)
            if strip_image is not None:
                print(f"Found strip using aspect ratio detection: {strip_bbox}")
                return strip_image, strip_bbox
                
            # Method 3: Fallback to basic thresholding-based detection
            print("Trying threshold-based detection...")
            strip_image, strip_bbox = self._detect_strip_by_thresholding(image_np)
            if strip_image is not None:
                print(f"Found strip using threshold detection: {strip_bbox}")
                return strip_image, strip_bbox
                
        except Exception as e:
            print(f"Error during strip extraction: {e}")
        
        print("Failed to detect strip in image")
        return None, None
    
    def _extract_region_from_prediction(self, image_np, pred):
        """Extract region from prediction dictionary in various formats"""
        try:
            # YOLO-style centerpoint format
            if 'x' in pred and 'y' in pred and ('width' in pred or 'w' in pred) and ('height' in pred or 'h' in pred):
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', pred.get('w', 0))
                h = pred.get('height', pred.get('h', 0))
                
                # Calculate absolute coordinates - FIX: Ensure strict bounds checking
                x1 = max(0, int(x - w/2))
                y1 = max(0, int(y - h/2))
                x2 = min(image_np.shape[1]-1, int(x + w/2))  # Subtract 1 to stay strictly within bounds
                y2 = min(image_np.shape[0]-1, int(y + h/2))  # Subtract 1 to stay strictly within bounds
                
                # Extract the region - ensure dimensions are valid
                if x1 < x2 and y1 < y2:
                    return image_np[y1:y2+1, x1:x2+1].copy(), (x1, y1, x2, y2)  # +1 to include last pixel
            
            # Format with bbox array
            elif 'bbox' in pred:
                bbox = pred['bbox']
                if len(bbox) == 4:
                    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # Likely [x1,y1,x2,y2] format
                        x1, y1, x2, y2 = bbox
                    else:  # Likely [x1,y1,w,h] format
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h
                        
                    # Ensure coordinates are valid - FIX: Strict boundary checking
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(image_np.shape[1]-1, int(x2))
                    y2 = min(image_np.shape[0]-1, int(y2))
                    
                    if x1 < x2 and y1 < y2:
                        return image_np[y1:y2+1, x1:x2+1].copy(), (x1, y1, x2, y2)  # +1 to include last pixel
            
            # Format with box dictionary
            elif 'box' in pred:
                box = pred['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                
                # Ensure coordinates are valid - FIX: Strict boundary checking
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image_np.shape[1]-1, int(x2))
                y2 = min(image_np.shape[0]-1, int(y2))
                
                # Extract the region if valid
                if x1 < x2 and y1 < y2:
                    return image_np[y1:y2+1, x1:x2+1].copy(), (x1, y1, x2, y2)  # +1 to include last pixel
                
        except Exception as e:
            print(f"Error extracting region: {e}")
            
        return None, None
    
    def _detect_strip_by_aspect_ratio(self, image_np, min_aspect_ratio=3.0, max_area_ratio=0.8):
        """Detect strip based on aspect ratio - strips are usually long and narrow"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding to handle varying lighting conditions
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Perform morphological operations to remove noise and connect fragments
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
            
            # Filter contours by aspect ratio and size
            img_area = image_np.shape[0] * image_np.shape[1]
            strip_candidates = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / float(min(w, h))
                area_ratio = (w * h) / float(img_area)
                
                # Skip small contours or those that are too large relative to the image
                if area_ratio < 0.05 or area_ratio > max_area_ratio:
                    continue
                
                # Strips have high aspect ratio
                if aspect_ratio > min_aspect_ratio:
                    strip_candidates.append((x, y, w, h, aspect_ratio, area_ratio))
            
            # Select the best strip candidate (highest aspect ratio with reasonable size)
            if strip_candidates:
                # Sort by aspect ratio (higher is better for strips)
                strip_candidates.sort(key=lambda c: c[4], reverse=True)
                x, y, w, h, _, _ = strip_candidates[0]
                
                # Add some padding - FIX: Apply strict bounds checking
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image_np.shape[1]-1, x + w + pad)  # Subtract 1 to stay in bounds
                y2 = min(image_np.shape[0]-1, y + h + pad)  # Subtract 1 to stay in bounds
                
                # Extract the region - ensure dimensions are valid
                if x1 < x2 and y1 < y2:
                    strip_image = image_np[y1:y2+1, x1:x2+1].copy()  # +1 to include last pixel
                    strip_bbox = (x1, y1, x2, y2)
                    return strip_image, strip_bbox
        
        except Exception as e:
            print(f"Error in aspect ratio detection: {e}")
            
        return None, None
    
    def _detect_strip_by_thresholding(self, image_np):
        """Detect strip using basic thresholding method"""
        try:
            # Convert to HSV for better color-based segmentation
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            
            # Extract value channel
            h, s, v = cv2.split(hsv)
            
            # Use Otsu's binarization for automatic thresholding
            _, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Remove small noise
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assumed to be the strip)
            if contours:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Get bounding rectangle of largest contour
                x, y, w, h = cv2.boundingRect(contours[0])
                
                # Add padding - FIX: Apply strict bounds checking
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image_np.shape[1]-1, x + w + pad)  # Subtract 1 to stay in bounds
                y2 = min(image_np.shape[0]-1, y + h + pad)  # Subtract 1 to stay in bounds
                
                # Extract the region if it's a reasonable size
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    strip_image = image_np[y1:y2+1, x1:x2+1].copy()  # +1 to include last pixel
                    strip_bbox = (x1, y1, x2, y2)
                    return strip_image, strip_bbox
        
        except Exception as e:
            print(f"Error in threshold detection: {e}")
            
        return None, None

# Dataset for normalization calculation
class SimpleImageDataset:
    def __init__(self, image_folder):
        """Create a simple dataset from image folder for mean/std computation"""
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image)
        return tensor, 0  # Return dummy label

# Compute normalization statistics from actual dataset
def compute_normalization_stats():
    cache_path = os.path.join(os.path.dirname(__file__), 'norm_cache.npz')
    
    # Try to load cached statistics
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return data['mean'], data['std']
        except Exception as e:
            print(f"Error loading cached stats: {e}")
    
    # Look for dataset directories
    dataset_dirs = [
        r'D:\Programming\urine_interpret\Datasets\original_images',
        r'D:\Programming\urine_interpret\Datasets\Split_70_20_10\train\images',
        # Add more potential directories here
    ]
    
    # Find first valid directory
    image_dir = None
    for dir_path in dataset_dirs:
        if os.path.exists(dir_path) and os.listdir(dir_path):
            image_dir = dir_path
            break
    
    if not image_dir:
        print("No valid dataset directory found. Using ImageNet statistics.")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Create dataset and compute statistics
    print(f"Computing mean and std from images in {image_dir}")
    dataset = SimpleImageDataset(image_dir)
    
    if len(dataset) == 0:
        print("No images found. Using ImageNet statistics.")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    print(f"Computing statistics from {len(dataset)} images...")
    mean, std = compute_mean_std(dataset)
    
    # Cache results
    try:
        np.savez(cache_path, mean=mean, std=std)
        print(f"Saved normalization statistics to {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {e}")
    
    return mean, std

# Compute mean and std (will use cache if available)
MEAN, STD = compute_normalization_stats()
print(f"Using normalization - Mean: {MEAN}, Std: {STD}")

class LocalModelAnalyzer:
    """Class to handle inference with locally trained models"""
    def __init__(self):
        self.unet_model = None
        self.svm_model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use computed normalization values
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load UNet and SVM models from disk"""
        try:
            # Load UNet model only if imports were successful
            if LOCAL_MODELS_AVAILABLE:
                unet_path = r'D:\Programming\urine_interpret\models\weights.pt'
                if os.path.exists(unet_path):
                    self.unet_model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES).to(self.device)
                    self.unet_model.load_state_dict(torch.load(unet_path, map_location=self.device))
                    self.unet_model.eval()
                    print("UNet model loaded successfully")
                else:
                    print(f"UNet model not found at {unet_path}")
                
                # Load SVM model
                svm_path = r'D:\Programming\urine_interpret\models\svm_rbf_model.pkl'
                if os.path.exists(svm_path):
                    with open(svm_path, 'rb') as f:
                        model_data = pickle.load(f)
                        self.svm_model = model_data['model']
                        self.scaler = model_data['scaler']
                    print("SVM model loaded successfully")
                else:
                    print(f"SVM model not found at {svm_path}")
                    
                return self.unet_model is not None and self.svm_model is not None
            else:
                print("Local models not available due to import errors")
                return False
        except Exception as e:
            print(f"Error loading local models: {e}")
            return False
    
    def are_models_loaded(self):
        """Check if both models are loaded and ready"""
        return self.unet_model is not None and self.svm_model is not None
    
    def analyze_image(self, image_path, confidence_threshold=0.5):
        """Process an image using local models"""
        # Implementation here if models loaded successfully
        print("Local model analysis not implemented")
        return None

def update_image_on_canvas(image_np):
    """Update the image shown on the canvas"""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.itemconfig(image_on_canvas, image=image_tk)
    canvas.image = image_tk

def display_strip_in_plot(strip_image):
    """Display the cropped strip in a new matplotlib window"""
    if strip_image is None:
        print("No strip image to display")
        return
    
    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Create subplot layout for the strip image
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(strip_image)
    ax1.set_title("Cropped Strip")
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Remove the numbers/ticks from the axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add color intensity profile
    if strip_image.shape[0] > 20:  # Only if the strip has reasonable height
        # Calculate average color intensity across the strip height
        intensity = np.mean(strip_image, axis=0)  # Average across height
        
        # Create bottom subplot for the color profile
        ax2 = plt.subplot(2, 1, 2)
        
        # Plot RGB channels
        x = np.arange(intensity.shape[0])
        ax2.plot(x, intensity[:, 0], 'r-', alpha=0.7, label='Red')
        ax2.plot(x, intensity[:, 1], 'g-', alpha=0.7, label='Green')
        ax2.plot(x, intensity[:, 2], 'b-', alpha=0.7, label='Blue')
        
        ax2.set_title("Color Intensity Profile")
        ax2.set_xlabel("Position along strip (pixels)")
        ax2.set_ylabel("Intensity")
        ax2.legend(loc='upper right')
        ax2.grid(True)
    
    # Adjust layout and show the figure
    plt.tight_layout(pad=3.0)
    plt.show(block=False)  # non-blocking so the main application continues running

def crop_strip_button_click():
    """Handle the crop strip button click event"""
    global last_results
    
    if last_results is None or 'original_image' not in last_results:
        print("No results available to crop strip from")
        return
    
    original_image = last_results['original_image']
    predictions = last_results.get('predictions', [])
    
    # Try to extract the strip from the original image
    strip_image, strip_bbox = analyzer.extract_strip(original_image, predictions)
    
    if strip_image is not None:
        print(f"Strip extracted: {strip_bbox}")
        display_strip_in_plot(strip_image)
    else:
        print("Failed to extract strip from image")

def process_and_display(image_path, confidence_threshold, save_results=False, save_path=None):
    """Process an image first with local models, then fall back to Roboflow if needed"""
    global last_results
    
    # Try local models first if they are available
    if LOCAL_MODELS_AVAILABLE and local_analyzer.are_models_loaded():
        print("Attempting analysis with local models...")
        local_results = local_analyzer.analyze_image(image_path, confidence_threshold)
        
        if local_results:
            status_var.set("Analyzed with local models")
            last_results = local_results
            
            # Update the display
            update_image_on_canvas(local_results['result_image'])
            
            # Update results text
            result_text = "Local Model Analysis Results:\n\n"
            
            # Show classification results
            if 'class_prediction' in local_results:
                class_id = local_results['class_prediction']
                class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
                result_text += f"Predicted Class: {class_name}\n\n"
            
            # Show all classes with probabilities above threshold
            if 'class_probs' in local_results and 'predictions' in local_results:
                result_text += "Class Probabilities:\n"
                for pred in local_results['predictions']:
                    result_text += f"{pred['class']}: {pred['confidence']:.2f}\n"
            
            results_display.config(state=tk.NORMAL)
            results_display.delete(1.0, tk.END)
            results_display.insert(tk.END, result_text)
            results_display.config(state=tk.DISABLED)
            
            # Save results if requested
            if save_results and save_path:
                cv2.imwrite(save_path, cv2.cvtColor(local_results['result_image'], cv2.COLOR_RGB2BGR))
                status_var.set(f"Results saved to {save_path} (local analysis)")
            
            return local_results
    
    # If local analysis failed or is not available, fall back to Roboflow
    print("Using Roboflow API for analysis...")
    
    # Process the image with Roboflow
    results = analyzer.analyze_image(image_path, confidence_threshold)
    last_results = results  # Store results for later use
    
    if not results:
        status_var.set("Error processing image")
        return None
    
    # Update the display
    update_image_on_canvas(results['result_image'])
    
    # Update results text
    result_text = "Urine Analysis Results:\n"
    
    # Display raw JSON for debugging if no structured info was found
    if result_text == "Analysis Results:\n" and 'predictions' in results:
        result_text += "\nRaw Results (Debug):\n"
        result_text += str(results['predictions'])[:500] + '...'  # Truncate if too long
    
    results_display.config(state=tk.NORMAL)
    results_display.delete(1.0, tk.END)
    results_display.insert(tk.END, result_text)
    results_display.config(state=tk.DISABLED)
    
    # Save results if requested
    if save_results and save_path:
        analyzer.save_results(results, save_path)
        status_var.set(f"Results saved to {save_path} (analysis)")
    else:
        status_var.set("Ready")
    
    return results

def select_image():
    """Open a file dialog to select an image to process"""
    global current_image_path
    file_types = [
        ('Image files', '*.jpg;*.jpeg;*.png'),
        ('JPEG files', '*.jpg;*.jpeg'),
        ('PNG files', '*.png'),
        ('All files', '*.*')
    ]
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=file_types)
    
    if image_path:
        current_image_path = image_path
        status_var.set(f"Processing {os.path.basename(image_path)}...")
        root.update()  # Update UI to show status message
        
        # Get confidence threshold from slider
        confidence_threshold = float(scale.get()) if 'scale' in globals() else 0.5
        
        # Get save settings
        should_save = save_results_var.get() if 'save_results_var' in globals() else False
        save_path = save_path_var.get() if 'save_path_var' in globals() and should_save else None
        
        # Process the selected image
        process_and_display(image_path, confidence_threshold, should_save, save_path)
    else:
        status_var.set("No image selected")

# Add a function to update the confidence threshold if it's missing
def update_confidence_threshold(val):
    """Update the confidence threshold and re-process the current image"""
    global current_image_path
    if current_image_path:
        confidence_threshold = float(val)
        process_and_display(current_image_path, confidence_threshold, 
                          save_results_var.get() if 'save_results_var' in globals() else False, 
                          save_path_var.get() if 'save_path_var' in globals() else None)

# Main GUI setup
if __name__ == "__main__":
    # Initialize the local models analyzer
    local_analyzer = LocalModelAnalyzer()
    
    # Initialize Roboflow client as fallback
    API_KEY = "rmrJF4g00wIf5b5xyYht"  # Your Roboflow API key
    WORKSPACE_NAME = "urine-test-strips-qq9jx"  # Your Roboflow workspace name
    WORKFLOW_ID = "detect-and-classify"  # Your Roboflow workflow ID
    
    analyzer = RoboflowUrineAnalyzer(API_KEY, WORKSPACE_NAME, WORKFLOW_ID)
    
    # Initialize global variables
    current_image_path = None
    last_results = None
    
    # Create the main window
    root = tk.Tk()
    root.title("Urine Strip Analyzer")
    root.geometry("1200x600")
    
    # Create frames for layout
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X)
    
    # Add analysis method indicator
    analysis_method_var = tk.StringVar()
    if LOCAL_MODELS_AVAILABLE and local_analyzer.are_models_loaded():
        analysis_method_var.set("Using: Local Models + Fallback")
    else:
        analysis_method_var.set("Using: API Only")
    analysis_method_label = tk.Label(top_frame, textvariable=analysis_method_var, 
                                    fg="blue", font=("Arial", 10, "bold"))
    analysis_method_label.pack(side=tk.TOP, padx=10, pady=5)
    
    # Add a button to select an image
    select_button = tk.Button(top_frame, text="Select Image", command=select_image)
    select_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add confidence threshold slider if it's not already there
    threshold_label = tk.Label(top_frame, text="Confidence Threshold:")
    threshold_label.pack(side=tk.LEFT, padx=10, pady=10)
    scale = tk.Scale(top_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                   resolution=0.01, command=update_confidence_threshold, length=200)
    scale.set(0.5)  # Default value
    scale.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add checkbox to save results
    save_results_var = tk.BooleanVar()
    save_results_checkbox = tk.Checkbutton(top_frame, text="Save Results", variable=save_results_var)
    save_results_checkbox.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add entry for save path
    save_path_var = tk.StringVar()
    save_path_entry = tk.Entry(top_frame, textvariable=save_path_var, width=30)
    save_path_entry.pack(side=tk.LEFT, padx=10, pady=10)
    save_path_var.set(r'D:\Programming\urine_interpret\results.jpg')  # Default save path
    
    # Add crop strip button
    crop_button = tk.Button(top_frame, text="Crop Strip & Plot", command=crop_strip_button_click)
    crop_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Create a canvas to display the image
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(canvas_frame, width=256, height=256, bd=2, relief=tk.SUNKEN)
    canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Initialize the image on the canvas
    image_np = np.zeros((256, 256, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_np)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    
    # Text area for displaying results
    results_display = tk.Text(canvas_frame, height=10, width=30, state=tk.DISABLED)
    results_display.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Status bar with analysis method indicator
    status_frame = tk.Frame(root)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = tk.Label(status_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Run the application
    root.mainloop()

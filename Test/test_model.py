import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch.serialization
from PIL import Image, ImageTk
from ultralytics.nn.tasks import DetectionModel
import tkinter as tk
from tkinter import ttk, filedialog
import joblib
from skimage import color
import sys
import os
from transformers import ViTModel, ViTFeatureExtractor

# Add both the parent directory and Train directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))

# Import config first to ensure it's in the module cache
from Train.config import NUM_CLASSES
# Then import the model
from Train.models import UNetYOLO

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names and configuration
CLASS_NAMES = {
    0: 'Bilirubin',
    1: 'Blood',
    2: 'Glucose',
    3: 'Ketone',
    4: 'Leukocytes',
    5: 'Nitrite',
    6: 'Protein',
    7: 'SpGravity',
    8: 'Urobilinogen',
    9: 'Background',
    10: 'pH',
    11: 'Strip'
}

# Lists of class IDs for multi-stage approach
STRIP_CLASS = 11  # Strip class ID
PAD_CLASSES = list(range(9)) + [10]  # Reagent pad classes (0-8, 10)

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

# Fixed normalization: using fixed mean/std (e.g., ImageNet values)
def fixed_normalization(image):
    image = image.resize((512, 512))  # Change size to 512x512
    tensor_image = T.ToTensor()(image)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    return normalize(tensor_image)

def load_model(model_path):
    """Load the UNetYOLO model from weights.pt"""
    print(f"Loading model from {model_path}...")
    
    # Add safe modules for loading
    torch.serialization.add_safe_globals([DetectionModel])
    
    # Initialize model
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES)
    
    # Load weights
    try:
        # First try: Standard loading
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Second try: Try loading with DetectionModel in safe globals
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with alternate method")
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            print("Using uninitialized model (results will be poor)")
    
    model.to(device)
    model.eval()
    return model

def load_svm_model(model_path):
    """Load the trained SVM model from disk"""
    try:
        print(f"Loading SVM model from {model_path}...")
        svm_data = joblib.load(model_path)
        print("SVM model loaded successfully")
        return svm_data
    except Exception as e:
        print(f"Error loading SVM model: {e}")
        print("Creating a default SVM model instead")
        return create_default_svm_model()

def create_default_svm_model():
    """Create a simple SVM classifier as a fallback"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # Create SVM classifier with RBF kernel
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True
    )
    
    # Return with an empty scaler
    return {
        'model': svm,
        'scaler': StandardScaler()
    }

def create_svm_classifier():
    """Create an SVM classifier with RBF kernel on the fly"""
    from sklearn.svm import SVC
    
    # Create SVM classifier with RBF kernel
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        verbose=False
    )
    
    return svm

def train_svm_on_features(features, labels):
    """Train SVM classifier on extracted features"""
    from sklearn.preprocessing import StandardScaler
    
    # Create and fit the classifier
    svm = create_svm_classifier()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Train SVM
    svm.fit(scaled_features, labels)
    
    return {
        'model': svm,
        'scaler': scaler
    }

def predict_masks(model, image_tensor):
    """Run the model to predict segmentation masks"""
    with torch.no_grad():
        # Get model prediction
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        
        # Get segmentation mask (pixel-wise prediction)
        mask = torch.argmax(probs, dim=1).cpu().numpy()[0]  # Shape: (H, W)
        
        # Get confidence map (max probability per pixel)
        confidence_map = torch.max(probs, dim=1)[0].cpu().numpy()[0]  # Shape: (H, W)
        
    return mask, confidence_map, probs.cpu()

def extract_strip_and_pads(mask):
    """
    Two-stage extraction:
    1. First extract the strip (class 11)
    2. Then extract the reagent pads (classes 0-8, 10) within the strip area
    """
    # Create strip mask (class 11)
    strip_mask = (mask == STRIP_CLASS).astype(np.uint8)
    
    # Find contours of the strip
    strip_contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no strip found, return empty results
    if not strip_contours:
        print("No strip detected!")
        return None, []
    
    # Take the largest contour as the strip
    strip_contour = max(strip_contours, key=cv2.contourArea)
    strip_bbox = cv2.boundingRect(strip_contour)
    
    # Create a mask for reagent pads (classes 0-8, 10)
    pad_masks = []
    for pad_class in PAD_CLASSES:
        pad_mask = (mask == pad_class).astype(np.uint8)
        
        # Find contours of this pad class
        pad_contours, _ = cv2.findContours(pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour of this pad class
        for contour in pad_contours:
            # Filter only pads that have significant overlap with the strip
            pad_bbox = cv2.boundingRect(contour)
            pad_masks.append({
                'class_id': pad_class,
                'contour': contour,
                'bbox': pad_bbox
            })
    
    return strip_bbox, pad_masks

def draw_results(image_np, strip_bbox, pad_masks, confidence_map, confidence_threshold=0.5):
    """Draw the strip and pads bounding boxes with labels"""
    # Create a copy of the image for drawing
    result_image = image_np.copy()
    
    # Draw strip bbox
    if strip_bbox:
        x, y, w, h = strip_bbox
        cv2.rectangle(result_image, (x, y), (x+w, y+h), CLASS_COLORS[STRIP_CLASS], 2)
        cv2.putText(result_image, f"Strip", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[STRIP_CLASS], 2)
    
    # Draw each pad with sufficient confidence
    for pad in pad_masks:
        x, y, w, h = pad['bbox']
        class_id = pad['class_id']
        
        # Calculate average confidence in this region
        region_confidence = confidence_map[y:y+h, x:x+w]
        avg_confidence = np.mean(region_confidence)
        
        if avg_confidence >= confidence_threshold:
            color = CLASS_COLORS[class_id]
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, f"{CLASS_NAMES[class_id]}", 
                      (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

def extract_vit_features(image):
    """Extract features from an image using ViT"""
    # Add do_rescale=False since image is already scaled to [0,1] by fixed_normalization
    inputs = vit_feature_extractor(image, return_tensors="pt", do_rescale=False).to(device)
    outputs = vit_model(**inputs)
    cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_features

def classify_pads(image_np, mask, svm_data):
    """Classify reagent pads using the SVM model"""
    pad_results = {}
    
    # Extract SVM model and scaler from data
    if isinstance(svm_data, dict) and 'model' in svm_data and 'scaler' in svm_data:
        svm_model = svm_data['model']
        scaler = svm_data['scaler']
    else:
        print("Warning: SVM data format not recognized, using raw model")
        svm_model = svm_data
        scaler = None
    
    # Extract features for each pad class
    for pad_class in PAD_CLASSES:
        # Create binary mask for this class
        pad_mask = (mask == pad_class)
        if np.any(pad_mask):  # If this class exists in the mask
            # Extract features from the pad region
            lab_image = color.rgb2lab(image_np)
            pad_region = lab_image[pad_mask]
            
            if len(pad_region) > 0:
                # Get only the color features (L, a, b)
                color_features = pad_region.mean(axis=0)  # Mean LAB color
                
                # Create feature vector with only the color features
                # SVM model expects 3 features (L, a, b)
                feature_vector = np.array(color_features).reshape(1, -1)
                
                # Check if feature dimensions match what model expects
                expected_features = getattr(svm_model, 'n_features_in_', 3)
                if feature_vector.shape[1] != expected_features:
                    print(f"Warning: Feature mismatch. Model expects {expected_features} features, got {feature_vector.shape[1]}")
                    # Adjust feature vector to match expected size
                    if feature_vector.shape[1] > expected_features:
                        feature_vector = feature_vector[:, :expected_features]
                    else:
                        # Pad with zeros
                        padding = np.zeros((1, expected_features - feature_vector.shape[1]))
                        feature_vector = np.hstack((feature_vector, padding))
                
                # Extract features using ViT
                vit_features = extract_vit_features(image_np)
                
                # Apply scaler if available
                if scaler is not None:
                    try:
                        vit_features = scaler.transform(vit_features)
                    except Exception as e:
                        print(f"Error scaling features: {e}")
                
                try:
                    # Run SVM prediction
                    prediction = svm_model.predict(vit_features)
                    confidence = 1.0  # Default confidence
                    
                    if hasattr(svm_model, 'predict_proba'):
                        probas = svm_model.predict_proba(vit_features)
                        confidence = np.max(probas)
                    
                    # Store results
                    pad_results[pad_class] = {
                        'class_name': CLASS_NAMES[pad_class],
                        'predicted_class': prediction[0],
                        'predicted_name': CLASS_NAMES.get(prediction[0], 'Unknown'),
                        'confidence': confidence,
                    }
                    
                except Exception as e:
                    print(f"Error during prediction for class {pad_class}: {e}")
                    # Fallback to default prediction on error
                    pad_results[pad_class] = {
                        'class_name': CLASS_NAMES[pad_class],
                        'predicted_class': pad_class,
                        'predicted_name': CLASS_NAMES[pad_class],
                        'confidence': 0.0,
                        'error': str(e)
                    }
    
    return pad_results

def save_strip_region(image_np, strip_bbox, save_path):
    """Save the bounding box region of the strip"""
    if strip_bbox:
        x, y, w, h = strip_bbox
        strip_region = image_np[y:y+h, x:x+w]
        cv2.imwrite(save_path, strip_region)
        print(f"Strip region saved to {save_path}")

def predict_image(model, svm_model, image_path, confidence_threshold=0.5, save_strip=False, strip_save_path=None):
    """Main function to predict and visualize results on a single image"""
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image.resize((512, 512)))  # Change size to 512x512
    image_tensor = fixed_normalization(image).unsqueeze(0).to(device)
    
    # Get segmentation prediction
    mask, confidence_map, probs = predict_masks(model, image_tensor)
    
    # Extract strip and pads using two-stage approach
    strip_bbox, pad_masks = extract_strip_and_pads(mask)
    
    # Save the strip region if requested
    if save_strip and strip_bbox:
        save_strip_region(image_np, strip_bbox, strip_save_path)
    
    # Draw results on the image
    result_image = draw_results(image_np, strip_bbox, pad_masks, 
                               confidence_map, confidence_threshold)
    
    # Extract features using ViT
    vit_features = extract_vit_features(image)
    
    # Classify pads using SVM
    pad_results = classify_pads(image_np, mask, svm_model)
    
    # Print classification results
    print("\nClassification Results:")
    for pad_class, result in pad_results.items():
        print(f"{result['class_name']}: Predicted as {CLASS_NAMES.get(result['predicted_class'], 'Unknown')} with confidence {result['confidence']:.2f}")
    
    return {
        'original_image': image_np,
        'segmentation_mask': mask,
        'confidence_map': confidence_map,
        'result_image': result_image,
        'pad_results': pad_results,
        'strip_bbox': strip_bbox,
        'pad_masks': pad_masks,
        'probs': probs.numpy(),
        'vit_features': vit_features
    }

def update_image_on_canvas(image_np):
    """Update the image shown on the canvas"""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.itemconfig(image_on_canvas, image=image_tk)
    canvas.image = image_tk

def process_and_display(image_path, confidence_threshold, save_strip=False, strip_save_path=None):
    """Process an image and display the results"""
    # Process the image
    results = predict_image(model, svm_model, image_path, confidence_threshold, save_strip, strip_save_path)
    
    # Update the display
    update_image_on_canvas(results['result_image'])
    
    # Update results text
    result_text = "Classification Results:\n"
    for pad_class, result in results['pad_results'].items():
        result_text += f"{result['class_name']}: Predicted as {CLASS_NAMES.get(result['predicted_class'], 'Unknown')}"
        result_text += f" with confidence {result['confidence']:.2f}\n"
    
    results_display.config(state=tk.NORMAL)
    results_display.delete(1.0, tk.END)
    results_display.insert(tk.END, result_text)
    results_display.config(state=tk.DISABLED)
    
    return results

def update_confidence_threshold(val):
    """Update the confidence threshold and re-process the current image"""
    global current_image_path
    confidence_threshold = float(val)
    if current_image_path:
        process_and_display(current_image_path, confidence_threshold, save_strip_var.get(), strip_save_path_var.get())

def select_image():
    """Open a file dialog to select an image"""
    global current_image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        current_image_path = image_path
        confidence_threshold = float(scale.get())
        process_and_display(image_path, confidence_threshold, save_strip_var.get(), strip_save_path_var.get())

# Main GUI setup
if __name__ == "__main__":
    # Only use weights.pt for the neural network model
    model_path = r'D:\Programming\urine_interpret\models\weights.pt'
    svm_path = r'D:\Programming\urine_interpret\models\vitmodel.pt'
    
    # Load the neural network model from weights.pt
    print("Loading UNetYOLO model from weights.pt...")
    model = load_model(model_path)
    
    # Load pre-trained ViT model and feature extractor
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Load the SVM RBF model
    print("Loading SVM RBF classifier...")
    svm_model = load_svm_model(svm_path)
    
    # Initialize global variables
    current_image_path = None
    
    # Create the main window
    root = tk.Tk()
    root.title("Urine Strip Analyzer")
    root.geometry("1200x600")
    
    # Create frames for layout
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X)
    
    # Add a button to select an image
    select_button = tk.Button(top_frame, text="Select Image", command=select_image)
    select_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add confidence threshold slider
    threshold_label = tk.Label(top_frame, text="Confidence Threshold:")
    threshold_label.pack(side=tk.LEFT, padx=10, pady=10)
    scale = tk.Scale(top_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                    resolution=0.01, command=update_confidence_threshold, length=200)
    scale.set(0.5)  # Default value
    scale.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add checkbox to save strip region
    save_strip_var = tk.BooleanVar()
    save_strip_checkbox = tk.Checkbutton(top_frame, text="Save Strip Region", variable=save_strip_var)
    save_strip_checkbox.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add entry for strip save path
    strip_save_path_var = tk.StringVar()
    strip_save_path_entry = tk.Entry(top_frame, textvariable=strip_save_path_var, width=30)
    strip_save_path_entry.pack(side=tk.LEFT, padx=10, pady=10)
    strip_save_path_var.set(r'D:\Programming\urine_interpret\strip_region.jpg')  # Default save path
    
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
    
    # Status bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Run the application
    root.mainloop()

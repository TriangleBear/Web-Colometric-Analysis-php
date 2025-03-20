import os
import numpy as np
import cv2
import colorsys
from PIL import Image
from sklearn.cluster import KMeans

class StripDetector:
    """Class for detecting urine strips and their reagent pads using image processing"""
    
    def __init__(self):
        """Initialize the strip detector"""
        pass
        
    def detect_strip(self, image):
        """
        Detect the strip in an image using image processing techniques
        
        Args:
            image: RGB numpy array or path to image file
            
        Returns:
            Tuple of (x1, y1, x2, y2) bounding box and confidence score,
            or (None, 0.0) if no strip is detected
        """
        # Load image if path was provided
        if isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Make a copy to not modify the original
        img = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find strip-like shapes (elongated rectangles)
        strip_candidates = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Calculate rect area
            rect_area = w * h
            
            # Skip small contours
            if area < 1000:
                continue
                
            # Calculate fill ratio (how much of the bounding rect is filled)
            fill_ratio = float(area) / rect_area if rect_area > 0 else 0
            
            # Strip should be elongated and have high fill ratio
            if aspect_ratio > 3.0 and fill_ratio > 0.5:
                score = aspect_ratio * fill_ratio * (area / 10000)  # Score based on aspect ratio, fill ratio and size
                strip_candidates.append((x, y, x+w, y+h, score))
                
        # If we found candidates, return the one with the highest score
        if strip_candidates:
            strip_candidates.sort(key=lambda x: x[4], reverse=True)
            x1, y1, x2, y2, score = strip_candidates[0]
            confidence = min(score / 10.0, 1.0)  # Normalize score to [0,1]
            return (x1, y1, x2, y2), confidence
            
        # If no good contour found, try a different approach with color
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define range for whitish/yellowish color (common for test strips)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 5000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if aspect_ratio > 2.0:  # Strip should be elongated
                    confidence = min((area / 10000) * aspect_ratio / 10, 1.0)
                    return (x, y, x+w, y+h), confidence
        
        # If all else fails, return None
        return None, 0.0
    
    def detect_pads(self, strip_image, min_pads=10, max_pads=12):
        """
        Detect reagent pads on a strip using image processing
        
        Args:
            strip_image: Cropped strip image (RGB numpy array)
            min_pads: Minimum expected number of pads
            max_pads: Maximum expected number of pads
            
        Returns:
            List of dictionaries containing pad information
        """
        # Make a copy to not modify the original
        img = strip_image.copy()
        
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Create masks for different color ranges
        masks = []
        
        # Define color ranges for common pad colors
        color_ranges = [
            # Yellow/Orange
            (np.array([15, 100, 100]), np.array([35, 255, 255])),
            # Green
            (np.array([35, 50, 50]), np.array([85, 255, 255])),
            # Blue
            (np.array([85, 50, 50]), np.array([130, 255, 255])),
            # Red (wraps around, so two ranges)
            (np.array([0, 100, 100]), np.array([15, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255])),
            # Magenta/Purple
            (np.array([130, 50, 50]), np.array([160, 255, 255])),
        ]
        
        # Create a combined mask
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Add a mask for any colored area (saturation-based)
        sat_mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        masks.append(sat_mask)
        combined_mask = cv2.bitwise_or(combined_mask, sat_mask)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find pad-like shapes (small rectangles/circles)
        pad_candidates = []
        
        # Calculate expected pad width based on strip width
        expected_pad_width = strip_image.shape[1] / (max_pads * 1.5)  # Estimate
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 50:
                continue
                
            # Skip very large contours (relative to the image)
            if area > (strip_image.shape[0] * strip_image.shape[1] * 0.25):
                continue
                
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Pads should be roughly square (aspect ratio close to 1)
            if 0.5 <= aspect_ratio <= 2.0:
                # Check if width is reasonable for a pad
                if 0.2 * expected_pad_width <= w <= 3 * expected_pad_width:
                    confidence = min((area / expected_pad_width**2) * (1.0 / (abs(aspect_ratio - 1.0) + 0.5)), 1.0)
                    pad_candidates.append({
                        'bbox': (x, y, x+w, y+h),
                        'class': 'pad',
                        'confidence': confidence,
                        'area': area
                    })
        
        # Sort by x-coordinate (left to right)
        pad_candidates.sort(key=lambda p: p['bbox'][0])
        
        # If we have too many candidates, keep the most confident ones
        if len(pad_candidates) > max_pads:
            pad_candidates.sort(key=lambda p: p['confidence'], reverse=True)
            pad_candidates = pad_candidates[:max_pads]
            # Re-sort by x-coordinate
            pad_candidates.sort(key=lambda p: p['bbox'][0])
            
        # Classify pads based on position and color
        pad_classes = ['Leukocytes', 'Nitrite', 'Urobilinogen', 'Protein', 'pH', 
                      'Blood', 'SpGravity', 'Ketone', 'Bilirubin', 'Glucose']
        
        # If we have approximately the expected number of pads, try to assign classes
        if min_pads <= len(pad_candidates) <= max_pads:
            for i, pad in enumerate(pad_candidates):
                if i < len(pad_classes):
                    pad['class'] = pad_classes[i]
        
        return pad_candidates
    
    def extract_pad_colors(self, strip_image, pads):
        """Extract color values from each pad"""
        pad_values = []
        
        for pad in pads:
            x1, y1, x2, y2 = pad['bbox']
            pad_image = strip_image[y1:y2, x1:x2].copy()
            
            # Extract dominant colors using K-means clustering
            # This is more robust than simple averaging
            try:
                # Reshape image to be a list of pixels
                pixels = pad_image.reshape(-1, 3).astype(np.float32)
                
                # Use K-means to find dominant colors (3 clusters)
                kmeans = KMeans(n_clusters=3, random_state=42)
                kmeans.fit(pixels)
                
                # Get the colors
                colors = kmeans.cluster_centers_.astype(np.uint8)
                
                # Calculate cluster sizes
                labels = kmeans.labels_
                counts = np.bincount(labels)
                
                # Sort colors by cluster size (largest first)
                sorted_indices = np.argsort(counts)[::-1]
                dominant_colors = colors[sorted_indices]
                
                # Convert to different color spaces for analysis
                rgb_color = dominant_colors[0]
                hsv_color = colorsys.rgb_to_hsv(rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)
                
                # Convert to Lab color space (commonly used for color difference evaluation)
                rgb_8bit = np.array([[rgb_color]], dtype=np.uint8)
                lab_color = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2Lab)[0][0]
                
                pad_values.append({
                    'class': pad['class'],
                    'rgb': rgb_color.tolist(),
                    'hsv': [hsv_color[0]*360, hsv_color[1]*100, hsv_color[2]*100],  # Scale to common ranges
                    'lab': lab_color.tolist(),
                    'dominant_colors': dominant_colors.tolist()
                })
                
            except Exception as e:
                print(f"Error extracting colors from pad: {e}")
                # Fallback to simple averaging
                mean_color = np.mean(pad_image, axis=(0, 1)).astype(int).tolist()
                pad_values.append({
                    'class': pad['class'],
                    'rgb': mean_color,
                    'hsv': [0, 0, 0],  # Placeholder
                    'lab': [0, 0, 0]    # Placeholder
                })
        
        return pad_values

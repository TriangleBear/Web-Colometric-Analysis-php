import torch.nn.functional as F
import torch
import numpy as np
import cv2
import gc

NUM_CLASSES = 11

def fix_target_indices(targets, num_classes):
    """
    Fix target indices to ensure they are within the valid range.
    
    Args:
        targets: Tensor containing class indices
        num_classes: Number of valid classes
    
    Returns:
        Tensor with indices clamped to valid range
    """
    if targets.max() >= num_classes:
    # print(f"Fixing invalid target indices: max value {targets.max().item()} >= {num_classes}")

        # Option 1: Clamp to the last valid class
        targets = torch.clamp(targets, 0, num_classes-1)
        
        # Option 2: Use a mapping approach for better debugging
        value_counts = torch.bincount(targets.flatten())
    # print(f"Value counts after clamping: {value_counts}")

    
    return targets

def dice_loss(outputs, targets, smooth=1e-6, max_size=512, class_weights=None):

    # First, handle the case where targets have the background/empty label value (NUM_CLASSES)
    # Create a mask to identify which samples should be included in loss computation
    if targets.dim() == 1:  # Class indices [B]
        valid_samples = targets != NUM_CLASSES  # Binary mask of valid samples
        if not valid_samples.any():
            # If all samples are background, return 0 loss
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        # Filter out background samples
        outputs = outputs[valid_samples]
        targets = targets[valid_samples]
        
        # If no samples left, return 0 loss
        if outputs.size(0) == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    # Resize outputs to save memory if necessary
    original_size = outputs.shape[2:]
    if outputs.shape[2] > max_size or outputs.shape[3] > max_size:
        scale_factor = max_size / max(outputs.shape[2], outputs.shape[3])
        new_h, new_w = int(outputs.shape[2] * scale_factor), int(outputs.shape[3] * scale_factor)
        outputs = F.interpolate(outputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # DON'T use softmax here, since we'll manually handle the probability conversion for BCE
    # Instead, we'll use sigmoid on each output channel for binary classification per class
    
    # Handle target dimensions
    if targets.dim() == 5:
        targets = targets.squeeze(1)
    elif targets.dim() == 2:  
        batch_size = outputs.shape[0]
        height, width = outputs.shape[2], outputs.shape[3]
        targets = targets.view(batch_size, height, width)
    elif targets.dim() == 3:
        targets = targets.unsqueeze(1)
    elif targets.dim() == 1:  # Class indices
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()
        # Use binary_cross_entropy_with_logits instead of binary_cross_entropy for autocast safety
        mean_outputs = outputs.mean([2, 3])  # Average over spatial dimensions to get logits
        return F.binary_cross_entropy_with_logits(mean_outputs, targets_one_hot)
    
    # Resize targets to match outputs size
    if targets.shape[2:] != outputs.shape[2:]:
        targets = F.interpolate(targets.float().unsqueeze(1), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
    
    # Fix any invalid target indices
    targets = fix_target_indices(targets, outputs.shape[1])
    
    # Add an optimization for when targets are one-hot
    if isinstance(targets, torch.Tensor) and targets.dim() == 2 and targets.shape[1] > 1:
        # If targets are already one-hot encoded (B, C)
        targets_one_hot = targets  # Use directly
    else:
        # Handle conversion as before
        targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    
    # Use vectorized operations when possible
    with torch.no_grad():
        # If the batch is small enough, process all at once
        if outputs.shape[0] <= 2:  # For very small batches
            # Apply sigmoid to convert logits to probabilities (autocast safe)
            probs = torch.sigmoid(outputs)
            
            intersection = (probs * targets_one_hot).sum(dim=(2,3))
            union = probs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
            dice = 1 - (2 * intersection + smooth) / (union + smooth)
            if class_weights is not None:
                class_weights = class_weights[targets]  # Get weights for the current targets
                dice = dice * class_weights  # Apply class weights to the dice loss

            # Add focal component for rare classes with a higher gamma
            if class_weights is not None:
                # Get class indices
                if targets.dim() == 1:  # [B] class labels
                    target_classes = targets
                else:  # [B, H, W] segmentation masks
                    # Get most common class in each target mask
                    target_classes = []
                    for i in range(targets.shape[0]):
                        values, counts = torch.unique(targets[i], return_counts=True)
                        if len(values) > 0:
                            idx = torch.argmax(counts)
                            target_classes.append(values[idx])
                        else:
                            target_classes.append(0)  # Default to class 0 if empty
                    target_classes = torch.tensor(target_classes, device=targets.device)
                
                # Apply higher weighting if target belongs to an underrepresented class (weight > 3)
                high_weight_mask = class_weights[target_classes] > 3.0
                if high_weight_mask.any():
                    # Further increase loss for rare classes to improve learning
                    dice = dice * torch.where(high_weight_mask.float(), 1.2, 1.0)[:, None]

            return dice.mean()
        else:
            # Otherwise, process batch by batch as before
            batch_size = outputs.shape[0]
            dice_sum = 0
            
            for i in range(batch_size):
                # Process one sample at a time
                output_i = outputs[i:i+1]
                target_i = targets_one_hot[i:i+1]
                
                # Apply sigmoid to convert logits to probabilities (autocast safe)
                probs_i = torch.sigmoid(output_i)
                
                intersection = (probs_i * target_i).sum(dim=(2,3))
                union = probs_i.sum(dim=(2,3)) + target_i.sum(dim=(2,3))
                dice_i = 1 - (2 * intersection + smooth) / (union + smooth)
                dice_sum += dice_i.mean()
                
                # Clean up to save memory
                del output_i, target_i, probs_i, intersection, union, dice_i
            
            # Clean up memory
            del outputs, targets, targets_one_hot
            torch.cuda.empty_cache()
            gc.collect()
            
            return dice_sum / batch_size if batch_size > 0 else torch.tensor(0.0, device='cuda')

# More optimized focal loss with better memory usage
def focal_loss(outputs, targets, alpha=0.25, gamma=2, max_size=512, class_weights=None):
    """
    Compute focal loss for multi-class segmentation with support for class indices.
    Now handles background/empty label (NUM_CLASSES) properly.
    """
    # Handle background class special case
    if targets.dim() == 1:  # [B] - class indices
        valid_samples = targets != NUM_CLASSES  # Binary mask of valid samples
        if not valid_samples.any():
            # If all samples are background, return 0 loss
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        # Filter out background samples
        outputs = outputs[valid_samples]
        targets = targets[valid_samples]
        
        # If no samples left, return 0 loss
        if outputs.size(0) == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    # Resize outputs to save memory if necessary
    original_size = outputs.shape[2:]
    if outputs.shape[2] > max_size or outputs.shape[3] > max_size:
        scale_factor = max_size / max(outputs.shape[2], outputs.shape[3])
        new_h, new_w = int(outputs.shape[2] * scale_factor), int(outputs.shape[3] * scale_factor)
        outputs = F.interpolate(outputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Get dimensions
    batch_size = outputs.shape[0]
    num_classes = outputs.shape[1]
    height, width = outputs.shape[2], outputs.shape[3]
    
    # Handle target dimensions - support both segmentation masks and class indices
    if targets.dim() == 1:  # [B] - class indices
        # Convert class indices to one-hot encoded format
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [B, C]
        
        # Compute binary cross entropy loss
        outputs_pooled = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)  # [B, C]
        bce_loss = F.binary_cross_entropy_with_logits(outputs_pooled, targets_one_hot, reduction='none')
        
        # Replace the main calculation with a more memory-efficient version
        if targets.dim() == 1:  # [B] - class indices (one label per image)
            # Gather only the relevant logits for each target
            # This avoids computing unnecessary probabilities for all classes
            batch_indices = torch.arange(batch_size, device=outputs.device)
            logits_for_targets = outputs_pooled[batch_indices, targets]
            
            # Focal loss for the target class only
            pt = torch.sigmoid(-logits_for_targets)  # Probability of being the target
            focal = alpha * (1 - pt) ** gamma * F.binary_cross_entropy_with_logits(
                logits_for_targets, torch.ones_like(logits_for_targets))
                
            return focal.mean()
        
    else:  # Handle segmentation masks [B, H, W] or [B, 1, H, W]
        # Handle 3D or 4D targets
        if targets.dim() == 4:  # [B, 1, H, W]
            targets = targets.squeeze(1)  # Convert to [B, H, W]
        
        # Resize targets to match outputs size if needed
        if targets.shape[1:] != outputs.shape[2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()
        
        # Fix any invalid target indices
        targets = fix_target_indices(targets, num_classes)
        
        # Convert to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
    # Compute binary cross entropy loss per pixel
    if class_weights is not None:
        class_weights = class_weights[targets]  # Get weights for the current targets

        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
        bce_loss = bce_loss.mean(dim=(2, 3))  # Average over spatial dimensions
    
    # Apply focal loss weighting
    if class_weights is not None:
        focal_loss = focal_loss * class_weights  # Apply class weights to the focal loss

    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss

    # Modify gamma for rare classes to make the model focus more on them
    if class_weights is not None:
        # Identify rare classes (higher weight means rarer)
        rare_classes = []
        common_classes = []
        
        # Check if we have class info from targets
        if targets.dim() == 1:  # Class indices
            for i, target in enumerate(targets):
                if target < len(class_weights) and class_weights[target] > 3.0:
                    rare_classes.append(i)
                else:
                    common_classes.append(i)
        
        # Apply higher gamma (focus parameter) to rare classes
        if rare_classes:
            # Extract batch indices for rare classes
            batch_indices = torch.tensor(rare_classes, device=outputs.device)
            
            # For rare classes, use a higher gamma (e.g., 3 instead of 2)
            if batch_indices.numel() > 0:
                # Increase gamma for rare classes to 3.0
                gamma_rare = 3.0
                pt_rare = torch.exp(-bce_loss[batch_indices])
                focal_loss_rare = alpha * (1 - pt_rare) ** gamma_rare * bce_loss[batch_indices]
                focal_loss[batch_indices] = focal_loss_rare

    if class_weights is not None:
        focal_loss = focal_loss * class_weights  # Apply class weights to the focal loss
    
    # Average over batch and classes
    focal_loss = focal_loss.mean()
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return focal_loss

import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import time
import torchvision.transforms as transforms  # Add this import for data augmentation
import torch.nn.functional as F  # Add this for contrastive_loss function
from torch.utils.data import DataLoader
from datasets import UrineStripDataset, CLASS_NAMES  # Import CLASS_NAMES from datasets
from models import UNetYOLO
from losses import dice_loss, focal_loss
from config import (
    TRAIN_IMAGE_FOLDER, 
    TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER, 
    VALID_MASK_FOLDER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_MIXED_PRECISION,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_WORKERS,
    NUM_CLASSES, 
    NUM_EPOCHS,
    SAVE_INTERVAL,
    DEVICE,
    BATCH_SIZE,
    PATIENCE,
    get_model_folder,
    clean_memory  # Also import the clean_memory function
)
from utils import compute_class_weights, compute_mean_std  # Import the correct function
from torch.amp import GradScaler, autocast  # Import mixed precision training tools
import tracemalloc  # Import for memory profiling
import subprocess  # Add this import for running shell commands
import torch.cuda.amp as amp
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm_dataset = UrineStripDataset(
        TRAIN_IMAGE_FOLDER, 
        TRAIN_MASK_FOLDER,
        transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    )
    
# Compute mean and std using the utility function
mean, std = compute_mean_std(norm_dataset)

# Regularization techniques
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def label_smoothing(targets, num_classes, smoothing=0.1):
    """Apply label smoothing to reduce overconfidence"""
    batch_size = targets.size(0)
    targets_one_hot = torch.zeros(batch_size, num_classes).to(device)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    targets_one_hot = targets_one_hot * (1 - smoothing) + smoothing / num_classes
    return targets_one_hot

def validate_model(model, dataloader, epoch, logger):
    """Run validation on the model and return metrics."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_progress = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}", 
                            position=2, leave=False)
        
        for images, targets, _ in val_progress:
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
            
            # Track loss and accuracy
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy per image (using most common class)
            for i in range(preds.shape[0]):
                img_pred = preds[i].flatten()
                values, counts = torch.unique(img_pred, return_counts=True)
                mode_idx = torch.argmax(counts)
                most_common_class = values[mode_idx].item()
                
                if most_common_class == targets[i].item():
                    correct += 1
                total += 1
                
            val_progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    val_loss = val_loss / len(dataloader)
    val_accuracy = correct / total if total > 0 else 0
    
    return val_loss, val_accuracy

def zip_dataset(model_dir):
    """Zip the dataset directory."""
    dataset_dir = os.path.join(model_dir, f"{model_dir}")
    zip_file = os.path.join(model_dir, f"{model_dir}.zip")
    if os.path.exists(dataset_dir):
        subprocess.run(["zip", "-r", zip_file, dataset_dir])
        print(f"Zipped dataset to {zip_file}")
    else:
        print(f"Dataset directory {dataset_dir} does not exist")

# FIX: Update class_balanced_loss to properly handle segmentation model outputs
def class_balanced_loss(outputs, targets, class_weights=None, gamma=2.0, beta=0.999):
    """
    Class-balanced focal loss that puts more emphasis on rare classes.
    This helps ensure all classes are learned properly.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        targets: Target labels [B] or [B, H, W]
        class_weights: Optional tensor of class weights [C]
        gamma: Focusing parameter for focal loss
        beta: Class balancing parameter
    """
    # Check if we have a classification or segmentation task
    if outputs.dim() == 4:  # [B, C, H, W] - segmentation
        # Spatial dimensions
        H, W = outputs.shape[2], outputs.shape[3]
        
        if targets.dim() == 1:  # [B] - classification labels
            # Global pooling for classification
            outputs_pooled = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)  # [B, C]
            probs = F.softmax(outputs_pooled, dim=1)
            
            # Get one-hot encoding of targets
            batch_size = targets.size(0)
            one_hot_targets = torch.zeros(batch_size, outputs.size(1), device=targets.device)
            one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
            
            # Calculate focal weights
            p_t = (one_hot_targets * probs).sum(1)
            focal_weight = (1 - p_t) ** gamma
            
            if class_weights is not None:
                # Get class weight for each sample based on its target class
                sample_weights = class_weights[targets]
                focal_weight = focal_weight * sample_weights
            
            # Calculate loss
            loss = F.cross_entropy(outputs_pooled, targets, reduction='none')
            balanced_loss = focal_weight * loss
            
            return balanced_loss.mean()
        
        else:  # [B, H, W] - segmentation mask
            # Convert targets to right shape if needed
            if targets.dim() == 3:  # [B, H, W]
                if targets.shape[1:] != (H, W):
                    targets = F.interpolate(targets.float().unsqueeze(1), size=(H, W), 
                                       mode='nearest').squeeze(1).long()
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)  # [B, C, H, W]
            
            # Compute class-wise focal loss
            loss = F.cross_entropy(outputs, targets, reduction='none')  # [B, H, W]
            
            # Calculate pt (probability of true class)
            batch_size = targets.size(0)
            pt = torch.zeros_like(loss)
            
            for b in range(batch_size):
                pt[b] = probs[b, targets[b], torch.arange(H).unsqueeze(1), 
                               torch.arange(W).unsqueeze(0)]
            
            # Focal weight
            focal_weight = (1 - pt) ** gamma
            
            # Apply class weights if provided
            if class_weights is not None:
                # Get weights for each pixel based on its class
                pixel_weights = torch.zeros_like(loss)
                for b in range(batch_size):
                    pixel_weights[b] = class_weights[targets[b]]
                focal_weight = focal_weight * pixel_weights
            
            # Final loss
            balanced_loss = focal_weight * loss
            
            return balanced_loss.mean()
    
    else:  # [B, C] - classification
        # Get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get one-hot encoding of targets
        batch_size = targets.size(0)
        one_hot_targets = torch.zeros(batch_size, outputs.size(1), device=targets.device)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate focal weights
        p_t = (one_hot_targets * probs).sum(1)
        focal_weight = (1 - p_t) ** gamma
        
        if class_weights is not None:
            # Get class weight for each sample based on its target class
            sample_weights = class_weights[targets]
            focal_weight = focal_weight * sample_weights
        
        # Calculate loss
        loss = F.cross_entropy(outputs, targets, reduction='none')
        balanced_loss = focal_weight * loss
        
        return balanced_loss.mean()

def get_rtx4050_optimized_settings():
    """Get optimized training settings for RTX 4050 Mobile GPU (6GB VRAM)."""
    return {
        'batch_size': 2,  # Reduced batch size
        'gradient_accumulation_steps': 8,  # Increased for effective batch size
        'mixed_precision': True,  # Enable mixed precision
        'checkpoint_modules': True,  # Enable gradient checkpointing
        'image_size': (384, 384),  # Reduced image size for efficiency
        'max_gpu_memory_fraction': 0.9,  # Limit memory usage
        'torch_compile': True,  # Use torch.compile for performance
    }

def train_model(num_epochs=None, batch_size=None, learning_rate=None, save_interval=None, 
               weight_decay=None, dropout_prob=0.5, mixup_alpha=0.2, 
               label_smoothing_factor=0.1, grad_clip_value=1.0, use_lite_model=False):
    """
    Train the UNet-YOLO model with memory optimizations for RTX 4050 Mobile
    """
    # Get optimized settings for RTX 4050
    rtx4050_settings = get_rtx4050_optimized_settings()
    
    # Use config values if not specified, with RTX 4050 optimizations
    num_epochs = num_epochs or NUM_EPOCHS
    batch_size = batch_size or rtx4050_settings['batch_size']  # Use smaller batch size
    learning_rate = learning_rate or LEARNING_RATE
    save_interval = save_interval or SAVE_INTERVAL
    weight_decay = weight_decay or WEIGHT_DECAY
    
    # Set CUDA memory management settings
    if torch.cuda.is_available():
        # Reserve specific amount of memory to avoid OOM
        torch.cuda.set_per_process_memory_fraction(rtx4050_settings['max_gpu_memory_fraction'])
        # Enable memory-efficient methods
        torch.backends.cudnn.benchmark = True
        # Use TF32 for faster computation on RTX GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Start memory profiling
    tracemalloc.start()
    
    # Clean memory before starting
    clean_memory()
    
    # Set up logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("training")
    logger.info(f"Training on device: {device}")
    logger.info(f"Using batch size: {batch_size} with {GRADIENT_ACCUMULATION_STEPS} gradient accumulation steps")
    
    # Create output directory for model checkpoints
    model_dir = get_model_folder()
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving models to: {model_dir}")
    
    # Compute dataset normalization statistics
    logger.info("Computing dataset normalization statistics...")
    logger.info(f"Dataset statistics - Mean: {mean}, Std: {std}")
    
    # Create transform with smaller image size for RTX 4050
    rtx_transform = transforms.Compose([
        transforms.Resize(rtx4050_settings['image_size']),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    
    # Load datasets with reduced image size for better memory usage
    logger.info("Loading and preparing datasets with optimized settings for RTX 4050...")
    train_dataset = UrineStripDataset(
        TRAIN_IMAGE_FOLDER, 
        TRAIN_MASK_FOLDER,
        transform=rtx_transform,  # Use RTX optimized transform
        debug_level=1  # Reduce debug level to save memory
    )
    valid_dataset = UrineStripDataset(
        VALID_IMAGE_FOLDER, 
        VALID_MASK_FOLDER, 
        transform=rtx_transform,
        debug_level=0
    )
    
    # Check the labels file in the dataset to ensure all classes have samples
    logger.info("Checking labels file in the dataset...")
    label_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    # NEW: First, add synthetic samples for any missing classes
    missing_raw_classes = [i for i in range(NUM_CLASSES) 
                          if i not in train_dataset.raw_annotations_count or 
                             train_dataset.raw_annotations_count[i] == 0]
    
    if missing_raw_classes:
        logger.warning(f"Missing classes detected in raw annotations: {missing_raw_classes}. Adding synthetic samples.")
        # Generate synthetic samples for missing classes
        if hasattr(train_dataset, 'generate_synthetic_samples'):
            train_dataset.generate_synthetic_samples(missing_raw_classes, num_samples=20)
            logger.info("Added synthetic samples for missing classes. Training will continue.")
        else:
            logger.error("Dataset doesn't support synthetic samples. Training may fail.")
    
    # Get the distribution that includes synthetic samples
    if hasattr(train_dataset, 'get_validated_class_distribution'):
        class_dist = train_dataset.get_validated_class_distribution()
        
        # Use the validated distribution for label counts
        for class_id, count in class_dist.items():
            if 0 <= class_id < NUM_CLASSES:  # Valid class range
                label_counts[class_id] = count
    else:
        # Fall back to regular counting if method not available
        for i in tqdm(range(len(train_dataset)), desc="Counting labels"):
            _, label, _ = train_dataset[i]
            if isinstance(label, list):  # Handle polygon-shaped bounding boxes
                for lbl in label:
                    label_counts[lbl] += 1
            else:
                label_counts[label] += 1
    
    # Print detailed class distribution
    logger.info("Class distribution in training dataset:")
    missing_classes = []
    for class_id in range(NUM_CLASSES):
        count = label_counts.get(class_id, 0)
        if count == 0:
            missing_classes.append(class_id)
            logger.warning(f"⚠️ Class {class_id} has ZERO samples!")
        else:
            logger.info(f"Class {class_id}: {count} samples")
    
    # NEW: Allow training to continue with synthetic data if classes are missing
    if missing_classes:
        logger.warning(f"Missing classes detected: {missing_classes}. Adding synthetic samples.")
        # Generate synthetic samples for missing classes
        if hasattr(train_dataset, 'generate_synthetic_samples'):
            train_dataset.generate_synthetic_samples(missing_classes, num_samples=20)
            logger.info("Added synthetic samples for missing classes. Training will continue.")
        else:
            logger.error("Dataset doesn't support synthetic samples. Training may fail.")
    
    # Now analyze class distribution after the dataset is loaded
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    logger.info("Analyzing full dataset class distribution...")
    for i in tqdm(range(len(train_dataset)), desc="Counting classes"):
        _, label, _ = train_dataset[i]
        if isinstance(label, list):  # Handle polygon-shaped bounding boxes
            for lbl in label:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
        else:
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Print detailed class distribution
    logger.info("Class distribution in training dataset:")
    missing_classes = []
    underrepresented_classes = []
    for class_id in range(NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        if count == 0:
            missing_classes.append(class_id)
            logger.error(f"⚠️ Class {class_id} has ZERO samples!")
        elif count < 10:  # Consider fewer than 10 samples as underrepresented
            underrepresented_classes.append(class_id)
            logger.warning(f"⚠️ Class {class_id} is underrepresented with only {count} samples")
        else:
            logger.info(f"Class {class_id}: {count} samples")
    
    # Check if any class is missing and exit if true
    if missing_classes:
        logger.error("Training aborted due to missing classes. Ensure all classes have samples before training.")
        return None, None
    
    # Handle missing classes through synthetic data generation
    if missing_classes or underrepresented_classes:
        logger.info("Applying class balancing techniques...")
        
        # 1. Oversampling for underrepresented classes
        # Create a sampler that samples underrepresented classes more frequently
        weights = [1.0] * len(train_dataset)
        for i in tqdm(range(len(train_dataset)), desc="Setting sample weights"):
            _, label, _ = train_dataset[i]
            if label in missing_classes or label in underrepresented_classes:
                weights[i] = 10.0  # Sample these 10x more frequently
                
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        # Use the weighted sampler for training
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=sampler,  # Use weighted sampler instead of shuffle=True
            pin_memory=True,
            num_workers=2,
            drop_last=True
        )
        logger.info("Applied weighted sampling for class balance")
    else:
        # Standard DataLoader with shuffling if no class imbalance issues
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
            drop_last=True
        )
    
    # Calculate class weights with higher penalties for rare classes
    class_weights = compute_balanced_class_weights(class_counts, NUM_CLASSES).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Check dataset classes and print distribution
    class_dist = train_dataset.class_distribution if hasattr(train_dataset, 'class_distribution') else {}
    
    # Intelligent handling of class imbalance
    if len(class_dist) > 0:
        max_class_count = max(class_dist.values()) if class_dist else 1
        min_class_count = min(class_dist.values()) if class_dist else 1
        imbalance_ratio = max_class_count / max(min_class_count, 1)
        logger.info(f"⚖️ Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # If severe imbalance, adjust training accordingly
        if imbalance_ratio > 10:
            logger.info("⚠️ Severe class imbalance detected! Applying balancing techniques.")
            # Increase weight decay for regularization with imbalanced data
            weight_decay *= 2
            # Lower learning rate for more stable convergence
            learning_rate *= 0.5
    
    # Configure data loaders with memory optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,  # Changed to False to reduce host memory usage
        num_workers=1,     # Reduced workers
        persistent_workers=False,  # Turn off for memory savings
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,  # Use same batch size for consistency
        shuffle=False,
        pin_memory=False,  # Changed to False
        num_workers=1      # Reduced workers
    )
    
    logger.info(f"Initializing memory-optimized model with dropout={dropout_prob}...")
    
    # Initialize model with optimized architecture for RTX 4050
    if use_lite_model:
        logger.info(f"Using LiteUNet model optimized for RTX 4050...")
        from models import LiteUNet
        model = LiteUNet(in_channels=3, out_channels=NUM_CLASSES, dropout_prob=dropout_prob).to(device)
    else:
        logger.info(f"Using UNetYOLO model optimized for RTX 4050...")
        model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES, dropout_prob=dropout_prob).to(device)
    
    # Enable gradient checkpointing to save memory
    if rtx4050_settings['checkpoint_modules'] and hasattr(model, 'use_checkpointing'):
        model.use_checkpointing = True
        logger.info("Enabled gradient checkpointing for memory efficiency")
    
    # Use model compilation if enabled (PyTorch 2.0+)
    if rtx4050_settings['torch_compile'] and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing with standard model.")
    
    # Use mixed precision training
    scaler = GradScaler() if rtx4050_settings['mixed_precision'] and torch.cuda.is_available() else None
    
    # Compute class weights - FIX: Remove max_weight parameter that's causing the error
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        amsgrad=True  # More stable for convergence
    )
    
    # IMPROVED LEARNING RATE SCHEDULER: Use OneCycle policy which is better for convergence
    # This will rapidly increase LR, then gradually decrease it
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,  # Peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=10,  # Initial lr is max_lr/10
        final_div_factor=100,  # Final lr is max_lr/1000
    )
    
    # ***CRITICAL***: Increase patience for early stopping
    PATIENCE = 30  # Allow more epochs without improvement
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []  # Add this line to track validation accuracies
    lr_history = []
    
    # NEW: Add multi-stage training to focus on different classes
    # Define class groups for targeted training
    reagent_pad_classes = list(range(9)) + [10]  # 0-8, 10
    strip_class = [11]  # Strip class
    background_class = [9]  # Background class
    
    # Track class-specific metrics for better monitoring
    class_accuracies = {i: [] for i in range(NUM_CLASSES)}
    
    # Main training loop with improved monitoring
    logger.info("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        # Adjust focus based on epoch
        # Early epochs: Focus on reagent pads
        # Middle epochs: Equal focus on all classes
        # Later epochs: More focus on underrepresented classes
        epoch_stage = epoch / num_epochs
        
        if epoch_stage < 0.3:  # First 30% of epochs
            logger.info(f"Epoch {epoch+1}: Focus on learning reagent pads")
            focused_classes = reagent_pad_classes
            class_focus_weight = 1.2  # Higher weight for reagent pads
        elif epoch_stage < 0.6:  # Next 30% of epochs
            logger.info(f"Epoch {epoch+1}: Equal focus on all classes")
            focused_classes = list(range(NUM_CLASSES))
            class_focus_weight = 1.0  # Equal weight for all
        else:  # Last 40% of epochs
            logger.info(f"Epoch {epoch+1}: Focus on strip and background classes")
            focused_classes = strip_class + background_class
            class_focus_weight = 1.5  # Higher weight for these classes
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Progress bar for training batches
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Epoch metrics
        batch_losses = []
        
        for batch_idx, (images, targets, _) in enumerate(batch_progress):
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Track unique classes in this batch for diagnostics
            unique_targets = torch.unique(targets).cpu().numpy()
            
            # Apply mixup with probability that increases with epoch
            # This helps stabilize early training while providing regularization later
            mixup_prob = min(0.6, epoch * 0.05)  # Gradually increase up to 60%
            apply_mixup = epoch >= 5 and np.random.random() < mixup_prob
            
            # Use mixed precision
            if scaler:
                with autocast(device_type='cuda'):
                    if apply_mixup:
                        mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
                        outputs = model(mixed_images)
                        loss = lam * dice_loss(outputs, targets_a, class_weights=class_weights) + \
                               (1 - lam) * dice_loss(outputs, targets_b, class_weights=class_weights)
                        loss += lam * focal_loss(outputs, targets_a, class_weights=class_weights) + \
                                (1 - lam) * focal_loss(outputs, targets_b, class_weights=class_weights)
                    else:
                        # Forward pass with the model - handle all return values
                        outputs = model(images)
                        
                        # Handle different model output formats
                        specialized_loss = 0
                        if isinstance(outputs, tuple):
                            if len(outputs) == 2:
                                outputs, aux_output = outputs
                                # Add auxiliary loss if available
                                main_loss = 0.7 * dice_loss(outputs, targets, class_weights=class_weights) + \
                                        0.7 * focal_loss(outputs, targets, class_weights=class_weights)
                                
                                # Convert targets for classification
                                target_classes = targets.view(-1)
                                aux_loss = F.cross_entropy(aux_output, target_classes, weight=class_weights)
                                loss = main_loss + 0.3 * aux_loss
                            
                            elif len(outputs) == 4:  # Handle specialized outputs
                                outputs, aux_output, strip_output, bg_output = outputs
                                
                                # Main loss
                                main_loss = 0.6 * dice_loss(outputs, targets, class_weights=class_weights) + \
                                          0.6 * focal_loss(outputs, targets, class_weights=class_weights)
                                
                                # Auxiliary classification loss
                                target_classes = targets.view(-1)
                                aux_loss = F.cross_entropy(aux_output, target_classes, weight=class_weights)
                                
                                # Specialized losses for strip and background
                                # Create binary masks for these specific classes
                                strip_mask = (targets == 11).float().unsqueeze(1)
                                bg_mask = (targets == 9).float().unsqueeze(1)
                                
                                # BCE loss for specialized heads
                                strip_loss = F.binary_cross_entropy_with_logits(strip_output, strip_mask)
                                bg_loss = F.binary_cross_entropy_with_logits(bg_output, bg_mask)
                                
                                # Add the class-balanced loss
                                cb_loss = class_balanced_loss(outputs, targets, class_weights)
                                
                                # Combine all losses
                                specialized_loss = 0.1 * strip_loss + 0.1 * bg_loss
                                loss = main_loss + 0.2 * aux_loss + 0.1 * cb_loss + specialized_loss
                                
                                # Apply class focus weighting based on epoch stage
                                if any(c in focused_classes for c in targets.cpu().numpy()):
                                    loss = loss * class_focus_weight
                        else:
                            # Regular loss calculation
                            loss = 0.7 * dice_loss(outputs, targets, class_weights=class_weights) + \
                                0.7 * focal_loss(outputs, targets, class_weights=class_weights)
                            
                            # Only add class_balanced_loss if it's a classification task (targets is [B])
                            # or wrap it in a try/except to gracefully handle any dimension issues
                            try:
                                if targets.dim() == 1:  # Classification task
                                    loss += 0.3 * class_balanced_loss(outputs, targets, class_weights)
                                else:
                                    # For segmentation task, use a simpler approach
                                    loss += 0.3 * F.cross_entropy(outputs, targets, 
                                                                 weight=class_weights if class_weights is not None else None)
                            except RuntimeError as e:
                                print(f"Warning: Skipping class_balanced_loss due to error: {e}")
                            
                            # Add a supervised contrastive loss term to better separate classes
                            if epoch > 5:  # Add after a few epochs of basic training
                                loss += 0.2 * contrastive_loss(outputs, targets)
                    
                    # Scale loss by accumulation factor
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Scale loss and compute gradients
                scaler.scale(loss).backward()
                
                # Steps when accumulation is complete
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    # Apply gradient clipping
                    if grad_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    
                    # Optimizer step and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            
            # Track metrics
            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            batch_losses.append(current_loss)
            epoch_loss += current_loss
            
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            batch_progress.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "LR": f"{lr:.6f}",
                "Classes": f"{len(unique_targets)}"  # Show class diversity
            })
            
            # Clear cache periodically
            if (batch_idx + 1) % 5 == 0:
                clean_memory()
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_losses.append(avg_epoch_loss)
        
        # Clear memory before validation
        clean_memory()
        
        # Update learning rate EVERY BATCH with OneCycleLR
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, valid_loader, epoch, logger)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)  # Add this line to append validation accuracy
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "  # Add this line to log validation accuracy
            f"LR: {current_lr:.6f}"
        )
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss
            }, checkpoint_path)
        
        # Save best model (if improved)
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'train_config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                    'dropout_prob': dropout_prob
                }
            }, os.path.join(model_dir, "best_model.pt"))
            
            logger.info(f"✓ Saved best model with val_loss: {val_loss:.4f} (improved by {improvement:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"! No improvement for {epochs_no_improve} epochs (best: {best_val_loss:.4f})")
            
            # Early stopping check with modified strategy
            if epochs_no_improve >= PATIENCE:
                # If we've trained for less than half the epochs, try a learning rate reset
                if epoch < num_epochs / 2 and epochs_no_improve == PATIENCE:
                    logger.info(f"⚠️ No improvement for {epochs_no_improve} epochs - Attempting learning rate reset")
                    
                    # Reset learning rate to initial value * 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * 0.5
                    
                    # Reset early stopping counter to give the model another chance
                    epochs_no_improve = PATIENCE // 2
                else:
                    logger.info(f"⛔ Early stopping triggered after {epoch+1} epochs")
                    early_stop = True
                    break
        
        # Plot and save training progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_training_progress(train_losses, val_losses, val_accuracies, lr_history, 
                                  save_path=os.path.join(model_dir, "training_progress.png"))
        
        # NEW: Track per-class accuracies
        with torch.no_grad():
            class_correct = {i: 0 for i in range(NUM_CLASSES)}
            class_total = {i: 0 for i in range(NUM_CLASSES)}
            
            for images, targets, _ in valid_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take main outputs
                
                _, predicted = torch.max(outputs, 1)
                
                # For each class, count correct predictions
                for i in range(NUM_CLASSES):
                    mask = (targets == i)
                    class_correct[i] += (predicted[mask] == i).sum().item()
                    class_total[i] += mask.sum().item()
            
            # Calculate and log per-class accuracy
            logger.info("Per-class validation accuracy:")
            for i in range(NUM_CLASSES):
                accuracy = 100 * class_correct[i] / max(class_total[i], 1)
                class_accuracies[i].append(accuracy)
                logger.info(f"  Class {i} ({CLASS_NAMES.get(i, 'Unknown')}): {accuracy:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies  # Add this line to save validation accuracies
    }, os.path.join(model_dir, "final_model.pt"))
    
    # Also save a lightweight version with just the weights for easier loading
    torch.save(model.state_dict(), os.path.join(model_dir, "weights.pt"))
    
    # Training summary
    status = "Early stopped" if early_stop else "Completed"
    logger.info(f"Training {status} after {epoch+1} epochs")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
    
    # Stop memory profiling and clean up
    tracemalloc.stop()
    clean_memory()
    
    # Zip the dataset after training completes
    zip_dataset(model_dir)
    
    # Return the model for further use if needed
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,  # Add this line to return validation accuracies
        'best_val_loss': best_val_loss,
        'early_stopped': early_stop,
        'epochs_completed': epoch + 1,
        'model_directory': model_dir
    }

def contrastive_loss(outputs, targets, temperature=0.1):
    """
    Add a contrastive loss component that helps separate class features
    """
    # Get the logits before softmax
    batch_size, num_classes, h, w = outputs.shape
    
    # Global average pooling to get class-level features
    features = F.adaptive_avg_pool2d(outputs, (1, 1))
    features = features.view(batch_size, num_classes)
    
    # Normalize features (important for stable contrastive learning)
    norm_features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(norm_features, norm_features.t()) / temperature
    
    # Create mask for positive pairs (same class)
    labels = targets.view(-1, 1)
    mask = (labels == labels.t()).float()
    
    # Remove self-contrast cases
    mask = mask - torch.eye(batch_size, device=mask.device)
    
    # Compute loss (InfoNCE)
    loss = 0
    for i in range(batch_size):
        if torch.sum(mask[i]) > 0:  # Skip if no positive pairs
            pos_sim = sim_matrix[i][mask[i] > 0]
            neg_sim = sim_matrix[i][mask[i] <= 0]
            
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                loss_i = -torch.log(
                    torch.sum(torch.exp(pos_sim)) / 
                    (torch.sum(torch.exp(pos_sim)) + torch.sum(torch.exp(neg_sim)))
                )
                loss += loss_i
    
    return loss / batch_size if batch_size > 0 else torch.tensor(0.0).to(device)

def compute_balanced_class_weights(class_counts, num_classes, max_weight=150.0):
    """Compute class weights with better handling of rare classes"""
    # Calculate total samples
    total_samples = sum(class_counts.values())
    
    # Compute weights using inverse frequency with enhanced weighting for rare classes
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count == 0:  # Handle missing classes
            weights.append(max_weight)  # Maximum weight for missing classes
        else:
            # The fewer samples, the higher the weight (inverse frequency)
            weight = total_samples / (count * num_classes)
            
            # Apply exponential penalty for very rare classes (fewer than 20 samples)
            if count < 20:
                weight *= 2.0  # Double the weight for very rare classes
                
            weights.append(weight)
    
    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Apply upper bound to avoid exploding gradients
    weights_tensor = torch.clamp(weights_tensor, 0.1, max_weight)
    
    # Normalize weights to have reasonable scale
    if weights_tensor.sum() > 0:
        weights_tensor = weights_tensor * (num_classes / weights_tensor.sum())
        
    print(f"Balanced class weights: {weights_tensor}")
    return weights_tensor

def get_advanced_augmentation(mean=None, std=None):
    """
    Create a more aggressive augmentation pipeline to improve generalization
    """
    # Use provided values or fall back to ImageNet statistics
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet default
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet default
        
    return transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        ], p=0.7),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])

def plot_training_progress(train_losses, val_losses, val_accuracies, lr_history, save_path=None):
    """Plot training progress and save to file"""
    import matplotlib.pyplot as plt
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot training and validation loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate over time
    ax3.plot(epochs, lr_history, 'm-', label='Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_ylabel('Learning Rate')
    ax3.set_xlabel('Epochs')
    ax3.set_yscale('log')  # Log scale for learning rate
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        plt.show()

# Add a function for performing learning rate test
def find_learning_rate(start_lr=1e-7, end_lr=1, batch_size=4, num_iter=100):
    """Find optimal learning rate with learning rate range test."""
    import math
    import matplotlib.pyplot as plt
    
    # Configure dataset and model
    train_dataset = UrineStripDataset(
        TRAIN_IMAGE_FOLDER, 
        TRAIN_MASK_FOLDER,
        transform=get_advanced_augmentation(mean, std),
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )
    
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    
    # Create log scale learning rate range
    gamma = (end_lr / start_lr) ** (1 / num_iter)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
    # Track learning rates and losses
    learning_rates = []
    losses = []
    
    # Run the learning rate finder
    model.train()
    progress = tqdm(range(min(num_iter, len(train_loader))), desc="Finding optimal LR")
    
    for i, (images, targets, _) in enumerate(train_loader):
        if i >= num_iter:
            break
            
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Handle auxiliary output if present
        if isinstance(outputs, tuple) and len(outputs) == 2:
            outputs, _ = outputs
        
        # Calculate loss
        loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
        
        # Record learning rate and loss
        learning_rates.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr_scheduler.step()
        
        progress.update(1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig('learning_rate_finder.png')
    plt.close()
    
    # Find optimal learning rate (point with steepest decrease)
    min_grad_idx = 0
    max_drop = 0
    for i in range(1, len(learning_rates) - 1):
        # Calculate negative gradient (drop)
        drop = (losses[i+1] - losses[i-1]) / (learning_rates[i+1] - learning_rates[i-1])
        if drop < max_drop:
            max_drop = drop
            min_grad_idx = i
    
    optimal_lr = learning_rates[min_grad_idx] / 10  # Divide by 10 as a heuristic
    print(f"Suggested optimal learning rate: {optimal_lr:.8f}")
    
    return optimal_lr

if __name__ == "__main__":
    # Uncomment to run learning rate finder
    # optimal_lr = find_learning_rate()
    # train_model(learning_rate=optimal_lr)
    
    train_model()

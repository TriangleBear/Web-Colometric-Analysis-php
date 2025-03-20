# train_unet.py
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # Correct import for autocast
from tqdm import tqdm
from config import *
from models import UNet
from datasets import UrineStripDataset, RandomTrainTransformations  # Add the import here
from losses import dice_loss, focal_loss
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from utils import compute_mean_std, dynamic_normalization  # Ensure dynamic_normalization is imported

def train_unet(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS, patience=PATIENCE):
    # Compute dataset statistics
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    mean, std = compute_mean_std(dataset)
    
    # Dataset and DataLoader
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER, transform=RandomTrainTransformations(mean, std))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    # Compute class weights
    class_counts = torch.zeros(NUM_CLASSES)
    for _, masks in dataset:
        class_counts += torch.bincount(masks.flatten(), minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights[class_counts == 0] = 0  # Set weights of classes with no samples to 0
    class_weights = class_weights / class_weights.sum()  # Normalize weights to sum to 1
    class_weights = class_weights.to(device)
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Model and Optimizer
    model = UNet(3, NUM_CLASSES, dropout_prob=0.5).to(device)  # Increase dropout rate
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Adjust learning rate and weight decay
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Add class weights
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)  # Use CosineAnnealingWarmRestarts scheduler

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    best_loss = float('inf')
    early_stop_counter = 0
    model_filename = get_model_filename()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            images = dynamic_normalization(images)  # Normalize the input images
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                focal_loss_value = focal_loss(outputs, masks)
                dice_loss_value = dice_loss(outputs, masks)
                loss = 0.3 * focal_loss_value + 0.7 * dice_loss_value  # Combine custom losses
                loss = loss.mean()  # Ensure the loss is a scalar
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            torch.cuda.empty_cache()  # Clear cache after each batch

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                images = dynamic_normalization(images)  # Normalize the input images
                outputs = model(images)
                dice_loss_value = dice_loss(outputs, masks)
                val_loss += dice_loss_value.mean().item()  # Ensure the loss is a scalar
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += masks.numel()
                correct += (predicted == masks).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Adjust learning rate
        scheduler.step(epoch + avg_val_loss)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_filename)
            print("Model improved and saved.")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs.")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"{model_filename}_epoch_{epoch+1}.pth")
        
        # Check early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
    
    return model, train_losses, val_losses, val_accuracies
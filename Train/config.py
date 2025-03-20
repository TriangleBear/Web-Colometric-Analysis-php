import torch
import os
import time
import gc

# Create a flag to track if CUDA info has been printed
_CUDA_INFO_PRINTED = False

def get_device_info():
    """Get device information only once."""
    global _CUDA_INFO_PRINTED
    
    if not _CUDA_INFO_PRINTED:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # Print CUDA details
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            # RTX 4050-optimized CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable TF32 for RTX GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction to avoid OOM errors
            # RTX 4050 mobile typically has 6GB memory
            torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of available memory
            
            # Set environment variables for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
            
        else:
            print("CUDA is not available. Using CPU.")
        
        _CUDA_INFO_PRINTED = True
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get device once during module import
device = get_device_info()

DEVICE = device

BASE_PATH = "D:/Programming/urine_interpret"
DATA_ROOT = os.path.join(BASE_PATH, r"Datasets/Split_70_20_10")
TRAIN_IMAGE_FOLDER = os.path.join(DATA_ROOT, "train/images")
TRAIN_MASK_FOLDER = os.path.join(DATA_ROOT, "train/labels")
VALID_IMAGE_FOLDER = os.path.join(DATA_ROOT, "valid/images")
VALID_MASK_FOLDER = os.path.join(DATA_ROOT, "valid/labels")
TEST_IMAGE_FOLDER = os.path.join(DATA_ROOT, "test/images")
TEST_MASK_FOLDER = os.path.join(DATA_ROOT, "test/labels")

# Create directories if they don't exist
for dir_path in [TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, 
                VALID_IMAGE_FOLDER, VALID_MASK_FOLDER,
                TEST_IMAGE_FOLDER, TEST_MASK_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

# RTX 4050-optimized Training Hyperparameters
BATCH_SIZE = 2  # Reduced for 6GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8  # Increased for effective batch size of 16
NUM_CLASSES = 12
NUM_EPOCHS = 200
PATIENCE = 30
IMAGE_SIZE = (384, 384)  # Reduced for memory efficiency

# Data loading settings
NUM_WORKERS = 1  # Reduced for memory efficiency

# Optimization settings
LEARNING_RATE = 2e-4  # Slightly higher for faster convergence with smaller batches
WEIGHT_DECAY = 1e-4
USE_MIXED_PRECISION = True
USE_GRADIENT_CHECKPOINTING = True

# Learning Rate Scheduler
LR_SCHEDULER_STEP_SIZE = 8
LR_SCHEDULER_GAMMA = 0.2  # Stronger decay

# Memory Management for RTX 4050 Mobile
def clean_memory():
    """Aggressive memory cleanup for RTX 4050 Mobile"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection of CUDA memory
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

# Run memory cleanup on module import
clean_memory()

# RTX 4050-optimized CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'

# Checkpointing strategy - save less frequently to reduce overhead
SAVE_INTERVAL = 5  # Save checkpoints every 5 epochs

# Model Saving
def get_model_folder():
    models_dir = os.path.join(BASE_PATH, "models")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join(models_dir, timestamp)
    os.makedirs(model_folder, exist_ok=True)
    return model_folder

def get_model_filename():
    model_folder = get_model_folder()
    return os.path.join(model_folder, "unet_model.pt")

def get_svm_filename():
    model_folder = get_model_folder()
    return os.path.join(model_folder, "svm_rbf_model.pkl")

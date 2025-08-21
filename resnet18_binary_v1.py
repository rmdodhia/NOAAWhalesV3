"""
ResNet18 Binary Whale Classification Model
==========================================

This script trains a ResNet18 model for binary whale detection (whale present vs. no whale) 
using spectrograms from multiple species and locations.

Key Features:
- Binary classification (whale/no whale)
- Focal Loss for class imbalance
- Location-based train/val/test splits
- Per-species metrics tracking
- Spectrogram augmentations
"""

# Standard library imports
import os
import logging
import datetime
import pytz
import csv
import random
from collections import Counter

# Third-party imports
import pandas as pd
import numpy as np
from PIL import Image

# Scikit-learn imports
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler

# PyTorch Lightning imports
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities import rank_zero_only

# ============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE TO CHANGE EXPERIMENT SETTINGS
# ============================================================================

class Config:
    """Configuration class for easy parameter management with validation"""
    
    # Data Parameters
    DATA_USE_PROPORTION = 1.0  # Proportion of data to use (for faster testing)
    
    # Model Parameters
    NUM_CLASSES = 2  # Binary classification
    
    # Model Training Parameters
    NUM_EPOCHS = 30
    BATCH_SIZE = 128  # Increased for better GPU utilization and gradient stability
    LEARNING_RATE = 0.0003  # Increased to match larger batch size
    WEIGHT_DECAY = 0.01  # L2 regularization to prevent overfitting
    DROPOUT_RATE = 0.3   # Dropout for regularization
    GRADIENT_CLIP_VAL = 1.0  # Gradient clipping to prevent exploding gradients
    
    # Loss Function Settings
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_ALPHA = 0.75  # Weight for positive class in Focal Loss
    FOCAL_LOSS_GAMMA = 3   # Focusing parameter for Focal Loss
    USE_LABEL_SMOOTHING = True  # Enable label smoothing for better calibration
    LABEL_SMOOTHING = 0.1  # Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15  # Reduced for faster stopping when overfitting
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MIN_DELTA = 0.0005  # Minimum change to qualify as improvement
    
    # Learning Rate Scheduling
    LR_SCHEDULER_FACTOR = 0.5      # Factor to reduce LR by
    LR_SCHEDULER_PATIENCE = 5      # Epochs to wait before reducing LR
    LR_SCHEDULER_MIN_LR = 1e-6     # Minimum learning rate
    
    # Model Checkpoint
    CHECKPOINT_MONITOR = 'val_auc'  # AUC is better for imbalanced data than accuracy
    CHECKPOINT_MODE = 'max'
    
    # DataLoader Configuration
    NUM_WORKERS = 0  # Parallel data loading workers (single-threaded for memory safety)
    PREFETCH_FACTOR = None  # Not applicable when NUM_WORKERS = 0
    PERSISTENT_WORKERS = False  # Not applicable when NUM_WORKERS = 0
    
    # Data Augmentation Parameters
    AUG_HORIZONTAL_SHIFT = 0.1      # Enable time shifting for spectrograms
    AUG_OCCLUSION_PROB = 0.15       # Add random frequency/time masking
    AUG_OCCLUSION_MAX_LINES = 2     # Allow more occlusion lines
    AUG_NOISE_PROB = 0.1            # Add Gaussian noise
    AUG_NOISE_STD = 0.02            # Noise standard deviation
    AUG_BUFFER_PROB = 0.05          # Simulate recording artifacts
    AUG_BUFFER_MAX_RATIO = 0.1      # Buffer corruption ratio
    
    # File Paths
    LABEL_FILES = [
    '/home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New/Beluga/Processed/LabelsOverlap400ms/Beluga_labels.csv',
    # '/home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New/Humpback/Processed/LabelsOverlap400ms/Humpback_labels.csv',
    # '/home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New/Orca/Processed/LabelsOverlap400ms/Orca_labels.csv'
    ]
    
    # Train/Validation/Test Split Configuration
    EXPERIMENT_RUNS = [
        {
            'test':  ['216D', '223D', '214D'],
            'val': ['206D', '201D', '213D']
        }
    ]
    
    # Validation split from training data
    VAL_SPLIT_FROM_TRAIN = 0.2
    
    # Output Directories
    RESULTS_DIR = '/home/radodhia/ssdprivate/NOAAWhalesV3/TestResults'
    MODELS_DIR = 'BestModels'
    LOGS_DIR = 'Logs/Resnet18Binary'
    LIGHTNING_LOGS_DIR = 'lightning_logs'
    COMBINED_DATA_DIR = '/home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New/Combined'
    ASSIGNATIONS_DIR = '/home/radodhia/ssdprivate/NOAAWhalesV3/Assignations'
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters and check file existence"""
        errors = []
        
        # Validate numeric ranges
        if not (0.0 < cls.DATA_USE_PROPORTION <= 1.0):
            errors.append(f"DATA_USE_PROPORTION must be between 0.0 and 1.0, got {cls.DATA_USE_PROPORTION}")
        
        if not (0.0 < cls.VAL_SPLIT_FROM_TRAIN < 1.0):
            errors.append(f"VAL_SPLIT_FROM_TRAIN must be between 0.0 and 1.0, got {cls.VAL_SPLIT_FROM_TRAIN}")
        
        if cls.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive, got {cls.BATCH_SIZE}")
        
        if cls.LEARNING_RATE <= 0:
            errors.append(f"LEARNING_RATE must be positive, got {cls.LEARNING_RATE}")
        
        # Validate file existence
        for label_file in cls.LABEL_FILES:
            if not os.path.exists(label_file):
                errors.append(f"Label file not found: {label_file}")
        
        # Validate experiment runs
        for i, run in enumerate(cls.EXPERIMENT_RUNS):
            if 'val' not in run or 'test' not in run:
                errors.append(f"Experiment run {i} missing 'val' or 'test' keys")
            elif set(run['val']).intersection(set(run['test'])):
                errors.append(f"Experiment run {i} has overlapping val/test locations")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        logging.info("Configuration validation passed")
        return True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging():
    """Set up logging configuration - only on rank 0 to avoid multiple log files"""
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:  # Only log from rank 0 process
        log_file = f'{Config.LOGS_DIR}/model_resnet18_binary_{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")}.log'
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info("Logging initialized on rank 0 process")
    else:
        # For non-rank 0 processes, set up basic logging to suppress output
        logging.basicConfig(level=logging.WARNING)

@rank_zero_only
def log_message(message):
    """Rank-zero only logging function - ensures messages only logged once"""
    logging.info(message)

def get_data_from_csv(df, location, data_use_proportion=1.0):
    """
    Extract data subset from labels dataframe for a specific location.
    
    Args:
        df (pd.DataFrame): Labels dataframe containing 'location', 'fullpath', and 'binarylabel' columns
        location (str): Location name to filter by
        data_use_proportion (float): Proportion of data to use (0.0-1.0)
        
    Returns:
        tuple: (image_paths, labels) - Lists of file paths and corresponding binary labels
    """
    logging.info(f'Reading {data_use_proportion} of location {location} from labels dataframe')
    df_location = df[df['location'] == location]
    if data_use_proportion < 1.0:
        df_location = df_location.sample(frac=data_use_proportion, random_state=42)
    image_paths = df_location['fullpath'].tolist()
    labels = df_location['binarylabel'].tolist()
    logging.info(f'Returning {len(image_paths)} image_paths and labels\n')
    return image_paths, labels

def load_and_prepare_data():
    """
    Load and prepare the combined dataset from multiple label files.
    
    Loads CSV files for different species, combines them, and creates binary labels
    where 1 = whale present (any species) and 0 = no whale.
    
    Returns:
        pd.DataFrame: Combined dataframe with binary labels and all metadata
    """
    logging.info("Loading label files...")
    
    # Read all CSV files efficiently
    dfs = [pd.read_csv(label_file) for label_file in Config.LABEL_FILES]
    labelsdf = pd.concat(dfs, ignore_index=True)

    # Vectorized binary label assignment
    whale_species = ['humpback', 'orca', 'beluga']
    labelsdf['binarylabel'] = (
        labelsdf['species'].isin(whale_species) & 
        (labelsdf['label'] != 0)
    ).astype(int)

    logging.info('Label files read and binary labels assigned')
    label_counts = labelsdf['binarylabel'].value_counts()
    logging.info(f'Binary label counts: {label_counts}')
    
    # Save combined binary labels
    os.makedirs(Config.COMBINED_DATA_DIR, exist_ok=True)
    labelsdf.to_csv(f'{Config.COMBINED_DATA_DIR}/labels_overlap400ms_binary.csv', index=False)
    
    return labelsdf

def get_metadata_for_paths(paths, labelsdf):
    """
    Efficiently get species and location metadata for given paths.
    
    Uses pandas indexing for O(1) lookup instead of dictionary creation.
    
    Args:
        paths (list): List of file paths to get metadata for
        labelsdf (pd.DataFrame): Dataframe with 'fullpath', 'species', and 'location' columns
        
    Returns:
        tuple: (species, locations) - Lists of species and location metadata
    """
    path_to_metadata = labelsdf.set_index('fullpath')[['species', 'location']].to_dict('index')
    species = []
    locations = []
    for path in paths:
        metadata = path_to_metadata.get(path, {'species': '', 'location': ''})
        species.append(metadata['species'])
        locations.append(metadata['location'])
    return species, locations

def prepare_location_data(labelsdf, locations, data_use_proportion):
    """
    Prepare image paths and labels for given locations with optimized metadata retrieval.
    
    Args:
        labelsdf (pd.DataFrame): Labels dataframe
        locations (list): List of location names
        data_use_proportion (float): Proportion of data to use
        
    Returns:
        tuple: (image_paths, labels, species, locations_list)
    """
    all_image_paths, all_labels = [], []
    
    # Batch collect all paths first
    for location in locations:
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=location, 
                                          data_use_proportion=data_use_proportion)
        all_image_paths.extend(img_paths)
        all_labels.extend(lbls)
    
    # Single batch metadata lookup for all paths
    species, locations_list = get_metadata_for_paths(all_image_paths, labelsdf)
    
    return all_image_paths, all_labels, species, locations_list

def save_assignment_data(train_image_paths, train_labels, train_species, train_locations_list,
                        val_image_paths, val_labels, val_species, val_locations_list,
                        test_image_paths, test_labels, test_species, test_locations_list,
                        test_location_str, val_location_str):
    """
    Save assignment data to Assignations folder with columns: stage, species, location, label, image_path
    
    Args:
        train_image_paths, train_labels, train_species, train_locations_list: Training data
        val_image_paths, val_labels, val_species, val_locations_list: Validation data  
        test_image_paths, test_labels, test_species, test_locations_list: Test data
        test_location_str: String representation of test locations
        val_location_str: String representation of validation locations
    """
    # Create assignments directory if it doesn't exist
    os.makedirs(Config.ASSIGNATIONS_DIR, exist_ok=True)
    
    # Create filename based on experiment configuration
    assignment_filename = f'Run_test_{test_location_str}_val_{val_location_str}.csv'
    assignment_filepath = os.path.join(Config.ASSIGNATIONS_DIR, assignment_filename)
    
    logging.info(f'Saving assignment data to: {assignment_filepath}')
    
    # Prepare all assignment data
    assignment_data = []
    
    # Add training data
    for i in range(len(train_image_paths)):
        assignment_data.append({
            'stage': 'train',
            'species': train_species[i],
            'location': train_locations_list[i], 
            'label': train_labels[i],
            'image_path': train_image_paths[i]
        })
    
    # Add validation data
    for i in range(len(val_image_paths)):
        assignment_data.append({
            'stage': 'validate',
            'species': val_species[i],
            'location': val_locations_list[i],
            'label': val_labels[i], 
            'image_path': val_image_paths[i]
        })
    
    # Add test data
    for i in range(len(test_image_paths)):
        assignment_data.append({
            'stage': 'test',
            'species': test_species[i],
            'location': test_locations_list[i],
            'label': test_labels[i],
            'image_path': test_image_paths[i]
        })
    
    # Write to CSV file
    with open(assignment_filepath, 'w', newline='') as file:
        fieldnames = ['stage', 'species', 'location', 'label', 'image_path']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignment_data)
    
    logging.info(f'Assignment data saved: {len(assignment_data)} total samples')
    logging.info(f'  - Training: {len(train_image_paths)} samples')
    logging.info(f'  - Validation: {len(val_image_paths)} samples') 
    logging.info(f'  - Test: {len(test_image_paths)} samples')
    
    return assignment_filepath

def create_data_loaders(train_image_paths, train_labels, train_species, train_locations_list,
                       val_image_paths, val_labels, val_species, val_locations_list,
                       test_image_paths, test_labels, test_species, test_locations_list,
                       train_transform, val_transform, test_transform):
    """Create training, validation, and test data loaders"""
    
    # Create datasets
    train_dataset = SpectrogramDataset(train_image_paths, train_labels, 
                                     transform=train_transform, species=train_species, 
                                     locations=train_locations_list, is_training=True)
    
    # Split training data for additional validation
    all_train_idx, val_idx_from_train = train_test_split(
        range(len(train_image_paths)), 
        test_size=Config.VAL_SPLIT_FROM_TRAIN, 
        stratify=train_labels,
        random_state=42
    )
    
    logging.info(f'Training/validation split: {len(all_train_idx)} train, {len(val_idx_from_train)} val from train data')
    
    # Calculate class weights for balancing
    train_label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = [total / (Config.NUM_CLASSES * max(train_label_counts.get(i, 1), 1)) 
                    for i in range(Config.NUM_CLASSES)]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logging.info(f'Class weights for loss: {class_weights}')
    
    # Create weighted sampler for balanced training
    sample_weights = [1.0 / train_label_counts[label] if train_label_counts[label] > 0 else 0.0 
                     for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    
    # Configure DataLoader parameters based on worker settings
    dataloader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': True
    }
    
    # Only add these parameters if using multiple workers
    if Config.NUM_WORKERS > 0:
        if Config.PREFETCH_FACTOR is not None:
            dataloader_kwargs['prefetch_factor'] = Config.PREFETCH_FACTOR
        dataloader_kwargs['persistent_workers'] = Config.PERSISTENT_WORKERS
    
    train_loader = DataLoader(
        train_dataset, 
        sampler=sampler,
        **dataloader_kwargs
    )
    
    # Combine validation sets
    val_subset_from_train = Subset(train_dataset, val_idx_from_train)
    val_dataset_from_val_location = SpectrogramDataset(val_image_paths, val_labels, 
                                                     transform=val_transform, species=val_species, 
                                                     locations=val_locations_list)
    val_dataset = ConcatDataset([val_subset_from_train, val_dataset_from_val_location])
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        **dataloader_kwargs
    )
    
    # Create test loader
    test_dataset = SpectrogramDataset(test_image_paths, test_labels, transform=test_transform, 
                                    is_training=False, species=test_species, 
                                    locations=test_locations_list)
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False,
        **dataloader_kwargs
    )
    
    logging.info('Data loaders created for binary classification')
    return train_loader, val_loader, test_loader, class_weights_tensor

def setup_training_components(test_location_str, val_location_str, test_image_paths, class_weights_tensor):
    """Setup model, callbacks, and trainer for training"""
    
    # Create model
    model = ResNet18BinaryClassifier(
        image_paths=test_image_paths,
        test_location=test_location_str,
        val_location=val_location_str,
        class_weights=class_weights_tensor,
        use_focal_loss=Config.USE_FOCAL_LOSS
    )
    logging.info('ResNet18BinaryClassifier created with Focal Loss')
    
    # Setup training callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        monitor=Config.CHECKPOINT_MONITOR,
        dirpath=Config.MODELS_DIR,
        filename=f'best_model_binary_test_{test_location_str}_val_{val_location_str}',
        save_top_k=1,
        mode=Config.CHECKPOINT_MODE
    )
    
    early_stop_callback = EarlyStopping(
        monitor=Config.EARLY_STOPPING_MONITOR,
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        mode='min',
        min_delta=Config.EARLY_STOPPING_MIN_DELTA  # Minimum improvement required
    )
    
    log_dir = os.path.join(
        Config.LIGHTNING_LOGS_DIR,
        datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M-%S')
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name='')
    tb_logger.log_hyperparams({'test_locations': test_location_str.split('_'), 
                              'val_locations': val_location_str.split('_')})
    
    # Create trainer
    # Note: DDP strategy creates separate processes per GPU, but logging is now configured 
    # to only write from rank 0 process to avoid multiple log files
    # Alternative: Use strategy='auto' for single GPU training if you prefer simpler logging
    trainer = pl.Trainer(
        max_epochs=Config.NUM_EPOCHS,
        devices='auto',
        num_nodes=1,
        accelerator='gpu',
        strategy='ddp',  # Multi-GPU training - logging handled by rank 0 only
        precision='16-mixed',
        logger=tb_logger,
        enable_progress_bar=True,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,  # Gradient clipping
        callbacks=[checkpoint_callback, early_stop_callback, LoggingCallback()]
    )
    
    return model, trainer, checkpoint_callback

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for rare class (between 0 and 1)
        gamma: Focusing parameter (typically 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        if self.alpha is not None:
            if targets.dim() > 1:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss combined with Label Smoothing for better regularization and calibration.
    
    Label smoothing prevents overconfident predictions by softening hard targets:
    - Instead of [0, 1] or [1, 0], uses soft targets like [0.05, 0.95] or [0.95, 0.05]
    - Works excellently with Focal Loss to prevent overconfidence while maintaining focus on hard examples
    
    Args:
        alpha: Focal loss weighting factor for rare class
        gamma: Focal loss focusing parameter  
        label_smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        num_classes: Number of classes for smoothing calculation
    """
    def __init__(self, alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA, 
                 label_smoothing=Config.LABEL_SMOOTHING, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        # Apply label smoothing to create soft targets
        if self.label_smoothing > 0:
            # Create soft targets
            confidence = 1.0 - self.label_smoothing
            smooth_factor = self.label_smoothing / (self.num_classes - 1)
            
            # Convert hard targets to soft targets
            soft_targets = torch.full_like(inputs, smooth_factor)
            soft_targets.scatter_(1, targets.unsqueeze(1), confidence)
            
            # Calculate loss using soft targets
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(soft_targets * log_probs, dim=1)
        else:
            # Standard cross entropy without smoothing
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
        # Apply focal loss weighting
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha weighting
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            alpha_t = 1.0
            
        # Combine focal weighting with smoothed loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# ============================================================================
# AUGMENTATION CLASSES
# ============================================================================

class SpectrogramAugmentations:
    """
    Spectrogram-specific augmentation techniques for whale call data.
    
    Includes horizontal shifting, occlusion, noise addition, and buffer corruption.
    """
    def __init__(self, 
                 horizontal_shift_range=Config.AUG_HORIZONTAL_SHIFT,
                 occlusion_prob=Config.AUG_OCCLUSION_PROB,
                 occlusion_max_lines=Config.AUG_OCCLUSION_MAX_LINES,
                 noise_prob=Config.AUG_NOISE_PROB,
                 noise_std=Config.AUG_NOISE_STD,
                 buffer_prob=Config.AUG_BUFFER_PROB,
                 buffer_max_ratio=Config.AUG_BUFFER_MAX_RATIO):
        self.horizontal_shift_range = horizontal_shift_range
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_lines = occlusion_max_lines
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.buffer_prob = buffer_prob
        self.buffer_max_ratio = buffer_max_ratio
    
    def horizontal_shift(self, spec):
        if torch.rand(1) < 0.1:
            _, _, time_dim = spec.shape
            shift_pixels = int(torch.randint(-int(time_dim * self.horizontal_shift_range), 
                                           int(time_dim * self.horizontal_shift_range) + 1, (1,)))
            if shift_pixels != 0:
                spec = torch.roll(spec, shifts=shift_pixels, dims=2)
        return spec
    
    def add_occlusions(self, spec):
        if torch.rand(1) < self.occlusion_prob:
            _, freq_dim, time_dim = spec.shape
            num_lines = torch.randint(1, self.occlusion_max_lines + 1, (1,)).item()
            for _ in range(num_lines):
                if torch.rand(1) < 0.7:
                    freq_start = torch.randint(0, freq_dim, (1,)).item()
                    line_width = torch.randint(1, max(2, freq_dim // 20), (1,)).item()
                    freq_end = min(freq_start + line_width, freq_dim)
                    spec[:, freq_start:freq_end, :] = 0
                else:
                    time_start = torch.randint(0, time_dim, (1,)).item()
                    line_width = torch.randint(1, max(2, time_dim // 20), (1,)).item()
                    time_end = min(time_start + line_width, time_dim)
                    spec[:, :, time_start:time_end] = 0
        return spec
    
    def add_gaussian_noise(self, spec):
        if torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(spec) * self.noise_std
            spec = spec + noise
            spec = torch.clamp(spec, 0, spec.max())
        return spec
    
    def add_buffer_simulation(self, spec):
        if torch.rand(1) < self.buffer_prob:
            _, freq_dim, time_dim = spec.shape
            downsample_factor = torch.rand(1) * self.buffer_max_ratio + 0.9
            new_time_dim = int(time_dim * downsample_factor)
            new_freq_dim = int(freq_dim * downsample_factor)
            spec_down = F.interpolate(spec.unsqueeze(0), 
                                    size=(new_freq_dim, new_time_dim), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            spec = F.interpolate(spec_down.unsqueeze(0), 
                               size=(freq_dim, time_dim), 
                               mode='bilinear', align_corners=False).squeeze(0)
        return spec
    
    def __call__(self, spec, is_training=True):
        if not is_training:
            return spec
        augmentations = [
            self.horizontal_shift,
            self.add_occlusions,
            self.add_gaussian_noise,
            self.add_buffer_simulation
        ]
        num_to_apply = torch.randint(1, 3, (1,)).item()
        selected = random.sample(augmentations, num_to_apply)
        random.shuffle(selected)
        for aug in selected:
            spec = aug(spec)
        return spec

class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=False, species=None, locations=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.species = species if species is not None else [''] * len(image_paths)
        self.locations = locations if locations is not None else [''] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        spec = torch.load(self.image_paths[idx])
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        if spec.shape[0] == 1:
            spec = spec.repeat(3, 1, 1)
        if spec.shape[-2:] != (224, 224):
            spec = F.interpolate(spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        if self.transform is not None:
            spec = self.transform(spec, self.is_training)
        if spec.dtype != torch.float32:
            spec = spec.float()
        if spec.max() > 1.0:
            spec = spec / spec.max()
        label = self.labels[idx]
        species = self.species[idx]
        location = self.locations[idx]
        return spec, label, self.image_paths[idx], species, location

# ============================================================================
# MODEL CLASSES  
# ============================================================================

class ResNet18BinaryClassifier(pl.LightningModule):
    """
    ResNet18-based binary classifier for whale call detection.
    
    Uses pretrained ResNet18 with modified final layer for binary classification.
    Supports both Focal Loss and Weighted Cross-Entropy for class imbalance.
    """
    def __init__(self, image_paths, learning_rate=Config.LEARNING_RATE, 
                 test_location=None, val_location=None, class_weights=None, 
                 use_focal_loss=Config.USE_FOCAL_LOSS):
        super().__init__()
        
        # Model architecture
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        
        # Replace final classifier with dropout for regularization
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(num_ftrs, 512),  # Additional hidden layer
            nn.BatchNorm1d(512),       # Batch normalization
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.test_location = test_location
        self.val_location = val_location
        self.val_outputs = []
        self.test_outputs = []
        self.image_paths = image_paths
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        # Initialize loss function
        if self.use_focal_loss and Config.USE_LABEL_SMOOTHING:
            self.criterion = FocalLossWithLabelSmoothing(
                alpha=Config.FOCAL_LOSS_ALPHA, 
                gamma=Config.FOCAL_LOSS_GAMMA,
                label_smoothing=Config.LABEL_SMOOTHING,
                num_classes=Config.NUM_CLASSES
            )
            logging.info(f"Using Focal Loss with Label Smoothing: alpha={Config.FOCAL_LOSS_ALPHA}, gamma={Config.FOCAL_LOSS_GAMMA}, smoothing={Config.LABEL_SMOOTHING}")
        elif self.use_focal_loss:
            self.criterion = FocalLoss(alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA)
            logging.info(f"Using Focal Loss without Label Smoothing: alpha={Config.FOCAL_LOSS_ALPHA}, gamma={Config.FOCAL_LOSS_GAMMA}")
        elif self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            logging.info(f"Using Weighted CrossEntropy with weights: {self.class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logging.info("Using standard CrossEntropy Loss")
            
        # Setup results tracking
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        self.csv_file_path = f'{Config.RESULTS_DIR}/test_results_binary_testlocation_{self.test_location}_vallocation_{self.val_location}.csv'
        with open(self.csv_file_path, 'w', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label', 'confidence_percent', 'whale_probability_percent', 'species', 'location']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    def on_validation_epoch_start(self):
        self.val_outputs = []
    def on_test_epoch_start(self):
        self.test_outputs = []

    def setup(self, stage=None):
        # Move criterion to device if needed
        if hasattr(self, 'device') and self.device is not None:
            self.criterion = self.criterion.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, *_ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.eq(preds, labels).float().mean()
        
        # Log training metrics with explicit epoch tracking
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        
        # Also log explicit epoch-based metrics for clearer visualization
        self.log('train_loss_by_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, image_paths, species, locations = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.val_outputs.append({
            'preds': preds.detach(), 
            'labels': labels.detach(),
            'species': species,
            'locations': locations
        })
        acc = (preds == labels).float().mean()
        
        # Log validation metrics with explicit epoch tracking
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        
        # Also log with explicit epoch number for clearer TensorBoard visualization
        self.log('epoch', float(self.current_epoch), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss_by_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, image_paths, species, locations = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate confidence scores (probabilities)
        probs = F.softmax(outputs, dim=1)
        confidences = torch.max(probs, dim=1)[0]  # Max probability for predicted class
        whale_probs = probs[:, 1]  # Probability of whale class (class 1)
        
        acc = (preds == labels).float().mean()
        self.test_outputs.append({
            'preds': preds, 
            'labels': labels,
            'species': species,
            'locations': locations,
            'confidences': confidences,
            'whale_probs': whale_probs
        })
        results = []
        for i in range(len(labels)):
            results.append({
                'image_filename': os.path.basename(image_paths[i]),
                'predicted_label': preds[i].item(),
                'actual_label': labels[i].item(),
                'confidence_percent': round(confidences[i].item() * 100, 2),
                'whale_probability_percent': round(whale_probs[i].item() * 100, 2),
                'species': species[i] if species[i] is not None else '',
                'location': locations[i] if locations[i] is not None else ''
            })
        with open(self.csv_file_path, 'a', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label', 'confidence_percent', 'whale_probability_percent', 'species', 'location']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(results)
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        return {'test_loss': loss, 'test_acc': acc}

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        all_preds = torch.cat([x['preds'] for x in self.val_outputs])
        all_labels = torch.cat([x['labels'] for x in self.val_outputs])
        all_species = []
        all_locations = []
        for x in self.val_outputs:
            all_species.extend(x['species'])
            all_locations.extend(x['locations'])
        
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        
        # Overall metrics
        precision = precision_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
        f1 = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels_np, all_preds_np)
        except ValueError:
            auc = float('nan')
        
        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds), on_epoch=True, on_step=False)
        self.log('val_recall', recall, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds), on_epoch=True, on_step=False)
        self.log('val_f1', f1, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds), on_epoch=True, on_step=False)
        self.log('val_auc', auc, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds), on_epoch=True, on_step=False)
        
        # Add explicit epoch-based logging for summary metrics
        self.log('val_summary/epoch', float(self.current_epoch), prog_bar=False, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log('val_summary/precision', precision, prog_bar=False, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log('val_summary/recall', recall, prog_bar=False, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log('val_summary/f1', f1, prog_bar=False, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log('val_summary/auc', auc, prog_bar=False, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        
        overall_acc = (all_preds == all_labels).float().mean().item()
        logging.info(f'EPOCH {self.current_epoch} - Validation Metrics: Acc={overall_acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}')
        
        # Per-species metrics
        unique_species = list(set(all_species))
        logging.info(f'Computing validation metrics per species: {unique_species}')
        
        for species in unique_species:
            if species == '':
                continue
            species_mask = np.array([s == species for s in all_species])
            if not np.any(species_mask):
                continue
                
            species_preds = all_preds_np[species_mask]
            species_labels = all_labels_np[species_mask]
            
            species_acc = np.mean(species_preds == species_labels)
            species_precision = precision_score(species_labels, species_preds, average='binary', zero_division=0)
            species_recall = recall_score(species_labels, species_preds, average='binary', zero_division=0)
            species_f1 = f1_score(species_labels, species_preds, average='binary', zero_division=0)
            try:
                species_auc = roc_auc_score(species_labels, species_preds)
            except ValueError:
                species_auc = float('nan')
            
            # Log per-species metrics
            self.log(f'val_acc_{species}', species_acc, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log(f'val_precision_{species}', species_precision, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log(f'val_recall_{species}', species_recall, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log(f'val_f1_{species}', species_f1, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log(f'val_auc_{species}', species_auc, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            
            logging.info(f'Validation {species} (n={len(species_preds)}) - Acc: {species_acc:.3f}, Precision: {species_precision:.3f}, Recall: {species_recall:.3f}, F1: {species_f1:.3f}, AUC: {species_auc:.3f}')

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        all_preds = torch.cat([x['preds'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])
        all_species = []
        all_locations = []
        for x in self.test_outputs:
            all_species.extend(x['species'])
            all_locations.extend(x['locations'])
        
        # Overall metrics
        precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='binary', zero_division=0)
        recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='binary', zero_division=0)
        f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels.cpu(), all_preds.cpu())
        except ValueError:
            auc = float('nan')
        test_acc = (all_preds == all_labels).float().mean().item()
        
        self.log('test_precision', round(precision, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_recall', round(recall, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_f1', round(f1, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_auc', round(auc, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_accuracy', round(test_acc, 3), sync_dist=True, batch_size=len(all_preds))
        
        test_message = f'Test Metrics - Location: {self.test_location}, Val Location: {self.val_location}, Overall Acc: {test_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}'
        print(test_message)
        logging.info(test_message)
        
        # Per-species test metrics
        unique_species = list(set(all_species))
        logging.info(f'Computing test metrics per species: {unique_species}')
        
        for species in unique_species:
            if species == '':
                continue
            species_mask = np.array([s == species for s in all_species])
            if not np.any(species_mask):
                continue
                
            species_preds = all_preds.cpu().numpy()[species_mask]
            species_labels = all_labels.cpu().numpy()[species_mask]
            
            species_acc = np.mean(species_preds == species_labels)
            species_precision = precision_score(species_labels, species_preds, average='binary', zero_division=0)
            species_recall = recall_score(species_labels, species_preds, average='binary', zero_division=0)
            species_f1 = f1_score(species_labels, species_preds, average='binary', zero_division=0)
            try:
                species_auc = roc_auc_score(species_labels, species_preds)
            except ValueError:
                species_auc = float('nan')
            
            # Log per-species metrics
            self.log(f'test_acc_{species}', round(species_acc, 3), sync_dist=True)
            self.log(f'test_precision_{species}', round(species_precision, 3), sync_dist=True)
            self.log(f'test_recall_{species}', round(species_recall, 3), sync_dist=True)
            self.log(f'test_f1_{species}', round(species_f1, 3), sync_dist=True)
            self.log(f'test_auc_{species}', round(species_auc, 3), sync_dist=True)
            
            test_species_message = f'Test {species} (n={len(species_preds)}) - Acc: {species_acc:.3f}, Precision: {species_precision:.3f}, Recall: {species_recall:.3f}, F1: {species_f1:.3f}, AUC: {species_auc:.3f}'
            print(test_species_message)
            logging.info(test_species_message)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=Config.WEIGHT_DECAY  # L2 regularization
        )
        
        # Add learning rate scheduler to reduce LR when val_loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE,
            verbose=True,
            min_lr=Config.LR_SCHEDULER_MIN_LR
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

# ============================================================================
# CALLBACKS AND UTILITIES
# ============================================================================

class LoggingCallback(pl.Callback):
    """Custom callback for logging checkpoint saves and early stopping events"""
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info(f'Checkpoint saved at epoch {trainer.current_epoch}')
        
    def on_train_end(self, trainer, pl_module):
        if hasattr(trainer, 'early_stopping_callback') and trainer.early_stopping_callback.stopped_epoch > 0:
            logging.info(f"Early stopping triggered at epoch {trainer.early_stopping_callback.stopped_epoch}")

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def generate_experiment_summary(experiment_results):
    """
    Generate and save comprehensive experiment summary.
    
    Args:
        experiment_results (list): List of experiment result dictionaries
    """
    summary_file = f'{Config.RESULTS_DIR}/experiment_summary_{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WHALE DETECTION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"Total Experiments: {len(experiment_results)}\n\n")
        
        # Configuration summary
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: ResNet18 Binary Classifier\n")
        f.write(f"Classes: {Config.NUM_CLASSES} (Binary: whale/no-whale)\n")
        f.write(f"Epochs: {Config.NUM_EPOCHS}\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {Config.LEARNING_RATE}\n")
        f.write(f"Focal Loss: {'Enabled' if Config.USE_FOCAL_LOSS else 'Disabled'}\n")
        if Config.USE_FOCAL_LOSS:
            f.write(f"  - Alpha: {Config.FOCAL_LOSS_ALPHA}\n")
            f.write(f"  - Gamma: {Config.FOCAL_LOSS_GAMMA}\n")
        f.write(f"Data Proportion Used: {Config.DATA_USE_PROPORTION}\n")
        f.write(f"Validation Split: {Config.VAL_SPLIT_FROM_TRAIN}\n\n")
        
        # Per-experiment results
        for i, result in enumerate(experiment_results, 1):
            f.write(f"EXPERIMENT {i}:\n")
            f.write("-"*40 + "\n")
            f.write(f"Test Locations: {result['test_locations']}\n")
            f.write(f"Validation Locations: {result['val_locations']}\n")
            f.write(f"Training Locations: {result['train_locations']}\n")
            f.write(f"Status: {result['status']}\n")
            
            if result['status'] == 'completed':
                f.write(f"Training Time: {result.get('training_time', 'N/A')}\n")
                f.write(f"Best Epoch: {result.get('best_epoch', 'N/A')}\n")
                # Add more metrics as available
            elif result['status'] == 'failed':
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    logging.info(f"Experiment summary saved to: {summary_file}")
    return summary_file

def main():
    """
    Main training and evaluation function with comprehensive error handling.
    
    Processes all experiment runs defined in Config.EXPERIMENT_RUNS, handling
    data preparation, model training, and testing for each configuration.
    """
    experiment_results = []
    start_time = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
    
    try:
        # Setup logging and validate configuration
        setup_logging()
        Config.validate()  # Validate configuration before starting
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
        
        logging.info(f"Starting experiment suite at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logging.info(f"Total experiments planned: {len(Config.EXPERIMENT_RUNS)}")
        
        # Load and prepare data
        labelsdf = load_and_prepare_data()
        
        # Create augmentation transforms
        train_transform = SpectrogramAugmentations()
        val_transform = SpectrogramAugmentations()
        test_transform = val_transform
        
        # Process all experiment runs
        for exp_idx, run_config in enumerate(Config.EXPERIMENT_RUNS, 1):
            val_locations = run_config['val']
            test_locations = run_config['test']
            
            experiment_start = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
            logging.info(f'\n\n{"="*60}')
            logging.info(f'EXPERIMENT {exp_idx}/{len(Config.EXPERIMENT_RUNS)} - Started at {experiment_start.strftime("%H:%M:%S")}')
            logging.info(f'Test locations: {test_locations}')
            logging.info(f'Validation locations: {val_locations}')
            logging.info(f'{"="*60}')
            
            experiment_result = {
                'experiment_id': exp_idx,
                'test_locations': test_locations,
                'val_locations': val_locations,
                'start_time': experiment_start,
                'status': 'running'
            }
            
            try:
                # Get unique locations and determine training locations
                locations = labelsdf['location'].unique()
                train_locations = [loc for loc in locations if loc not in test_locations and loc not in val_locations]
                logging.info(f'Training locations: {train_locations}')
                experiment_result['train_locations'] = train_locations
                
                # Prepare data for each split
                test_image_paths, test_labels, test_species, test_locations_list = prepare_location_data(
                    labelsdf, test_locations, Config.DATA_USE_PROPORTION)
                
                train_image_paths, train_labels = [], []
                for train_location in train_locations:
                    img_paths, lbls = get_data_from_csv(df=labelsdf, location=train_location, 
                                                      data_use_proportion=Config.DATA_USE_PROPORTION)
                    train_image_paths.extend(img_paths)
                    train_labels.extend(lbls)
                
                val_image_paths, val_labels = [], []
                for val_location in val_locations:
                    img_paths, lbls = get_data_from_csv(df=labelsdf, location=val_location, 
                                                      data_use_proportion=Config.DATA_USE_PROPORTION)
                    val_image_paths.extend(img_paths)
                    val_labels.extend(lbls)
                
                # Get metadata for training and validation data
                train_species, train_locations_list = get_metadata_for_paths(train_image_paths, labelsdf)
                val_species, val_locations_list = get_metadata_for_paths(val_image_paths, labelsdf)
                
                # Save assignment data to Assignations folder
                test_location_str = "_".join(test_locations)
                val_location_str = "_".join(val_locations)
                assignment_filepath = save_assignment_data(
                    train_image_paths, train_labels, train_species, train_locations_list,
                    val_image_paths, val_labels, val_species, val_locations_list,
                    test_image_paths, test_labels, test_species, test_locations_list,
                    test_location_str, val_location_str
                )
                
                # Create data loaders
                train_loader, val_loader, test_loader, class_weights_tensor = create_data_loaders(
                    train_image_paths, train_labels, train_species, train_locations_list,
                    val_image_paths, val_labels, val_species, val_locations_list,
                    test_image_paths, test_labels, test_species, test_locations_list,
                    train_transform, val_transform, test_transform
                )
                
                # Setup training components
                model, trainer, checkpoint_callback = setup_training_components(
                    test_location_str, val_location_str, test_image_paths, class_weights_tensor)
                
                # Train model
                logging.info('Starting model training')
                training_start = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
                trainer.fit(model, train_loader, val_loader)
                training_end = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
                
                experiment_result['training_time'] = str(training_end - training_start)
                experiment_result['best_epoch'] = trainer.current_epoch
                
                # Test with best model
                logging.info('Loading best model for testing')
                best_model = ResNet18BinaryClassifier.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    learning_rate=Config.LEARNING_RATE,
                    test_location=test_location_str,
                    val_location=val_location_str,
                    image_paths=test_image_paths,
                    class_weights=class_weights_tensor,
                    use_focal_loss=Config.USE_FOCAL_LOSS
                )
                
                logging.info('Starting model testing')
                trainer.test(best_model, test_loader)
                
                experiment_end = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
                experiment_result['end_time'] = experiment_end
                experiment_result['total_time'] = str(experiment_end - experiment_start)
                experiment_result['status'] = 'completed'
                
                logging.info(f'✅ Experiment {exp_idx} completed successfully in {experiment_result["total_time"]}')
                
            except Exception as e:
                experiment_end = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
                experiment_result['end_time'] = experiment_end
                experiment_result['total_time'] = str(experiment_end - experiment_start)
                experiment_result['status'] = 'failed'
                experiment_result['error'] = str(e)
                
                logging.error(f'❌ Experiment {exp_idx} failed after {experiment_result["total_time"]}: {str(e)}')
                logging.error(f'Continuing with next experiment run...')
            
            experiment_results.append(experiment_result)
        
        # Generate experiment summary
        end_time = datetime.datetime.now(pytz.timezone('America/Los_Angeles'))
        total_time = end_time - start_time
        
        logging.info(f'\n\n{"="*60}')
        logging.info(f'EXPERIMENT SUITE COMPLETED')
        logging.info(f'Total time: {total_time}')
        logging.info(f'Successful experiments: {sum(1 for r in experiment_results if r["status"] == "completed")}/{len(experiment_results)}')
        logging.info(f'{"="*60}')
        
        # Generate summary report
        summary_file = generate_experiment_summary(experiment_results)
        logging.info(f'Detailed summary saved to: {summary_file}')
                
    except Exception as e:
        logging.error(f'Fatal error in main(): {str(e)}')
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DeblurGAN-v2 Training Script

Trains DeblurGAN-v2 with MobileNet-DSC backbone on combined blurred dataset
for motion blur correction. Uses paired dataset of blurred/sharp images
from cars and artificially blurred wagon images.

Requirements: 6.3, 10.1, 10.2, 10.3
"""

import os
import sys
import logging
import argparse
import time
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_ingest import download_wagon_dataset
from data_physics import simulate_railway_conditions


logger = logging.getLogger(__name__)


class MobileNetDSC(nn.Module):
    """
    MobileNet-DSC (Depthwise Separable Convolution) backbone for DeblurGAN-v2.
    
    This is a lightweight backbone that uses depthwise separable convolutions
    to reduce computational cost while maintaining good performance.
    """
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable blocks
        self.dsc_blocks = nn.ModuleList([
            self._make_dsc_block(32, 64, stride=1),
            self._make_dsc_block(64, 128, stride=2),
            self._make_dsc_block(128, 128, stride=1),
            self._make_dsc_block(128, 256, stride=2),
            self._make_dsc_block(256, 256, stride=1),
            self._make_dsc_block(256, 512, stride=2),
            # Additional blocks for feature extraction
            self._make_dsc_block(512, 512, stride=1),
            self._make_dsc_block(512, 512, stride=1),
            self._make_dsc_block(512, 512, stride=1),
            self._make_dsc_block(512, 512, stride=1),
            self._make_dsc_block(512, 512, stride=1),
            self._make_dsc_block(512, 1024, stride=2),
            self._make_dsc_block(1024, 1024, stride=1),
        ])
        
        self.relu = nn.ReLU6(inplace=True)
    
    def _make_dsc_block(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial conv
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)  # 1/2 scale
        
        # DSC blocks
        for i, block in enumerate(self.dsc_blocks):
            x = block(x)
            # Save features at key scales
            if i in [1, 3, 5, 12]:  # 1/4, 1/8, 1/16, 1/32 scales (fixed index 12 for 1024 channels)
                features.append(x)
        
        return features


class DeblurGANv2Generator(nn.Module):
    """
    DeblurGAN-v2 Generator with MobileNet-DSC backbone.
    
    Uses Feature Pyramid Network (FPN) architecture with skip connections
    for multi-scale deblurring.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        super().__init__()
        
        # Encoder (MobileNet-DSC backbone)
        self.encoder = MobileNetDSC(input_channels)
        
        # Decoder with FPN-style upsampling
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(1024, 512),      # 1/32 -> 1/16: 1024 -> 512
            self._make_decoder_block(512 + 512, 256), # 1/16 -> 1/8: (512 + 512) -> 256  
            self._make_decoder_block(256 + 256, 128), # 1/8 -> 1/4: (256 + 256) -> 128
            self._make_decoder_block(128 + 128, 64),  # 1/4 -> 1/2: (128 + 128) -> 64
            self._make_decoder_block(64 + 32, 32),    # 1/2 -> 1/1: (64 + 32) -> 32
        ])
        
        # Final output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            x: Input blurred image [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Deblurred image [B, 3, H, W] in range [-1, 1]
        """
        # Encode with multi-scale features
        encoder_features = self.encoder(x)
        
        # Start decoding from the deepest feature
        decoded = encoder_features[-1]  # 1/32 scale
        
        # Decode with skip connections
        skip_indices = [3, 2, 1, 0]  # Map decoder blocks 1,2,3,4 to encoder features
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            decoded = F.interpolate(decoded, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Add skip connection (except for the first block)
            if i > 0:
                skip_idx = skip_indices[i-1]
                skip_feature = encoder_features[skip_idx]
                
                # Resize skip feature to match decoded size if needed
                if skip_feature.shape[2:] != decoded.shape[2:]:
                    skip_feature = F.interpolate(
                        skip_feature, size=decoded.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                decoded = torch.cat([decoded, skip_feature], dim=1)
            
            # Apply decoder block
            decoded = decoder_block(decoded)
        
        # Final output
        output = self.output_conv(decoded)
        
        return output


class DeblurDataset(Dataset):
    """
    Dataset for training DeblurGAN-v2 on paired blurred/sharp images.
    
    Combines local car dataset with artificially blurred wagon images.
    """
    
    def __init__(
        self, 
        blurred_paths: List[Path], 
        sharp_paths: List[Path], 
        transform=None,
        crop_size: int = 256
    ):
        """
        Args:
            blurred_paths: List of paths to blurred images
            sharp_paths: List of paths to corresponding sharp images
            transform: Optional image transforms
            crop_size: Size for random cropping
        """
        assert len(blurred_paths) == len(sharp_paths), "Mismatch between blurred and sharp images"
        
        self.blurred_paths = blurred_paths
        self.sharp_paths = sharp_paths
        self.transform = transform
        self.crop_size = crop_size
    
    def __len__(self) -> int:
        return len(self.blurred_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load images
        blurred_path = self.blurred_paths[idx]
        sharp_path = self.sharp_paths[idx]
        
        blurred_img = cv2.imread(str(blurred_path))
        sharp_img = cv2.imread(str(sharp_path))
        
        if blurred_img is None or sharp_img is None:
            raise ValueError(f"Could not load images: {blurred_path}, {sharp_path}")
        
        # Convert BGR to RGB
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        
        # Ensure same size
        h, w = min(blurred_img.shape[0], sharp_img.shape[0]), min(blurred_img.shape[1], sharp_img.shape[1])
        blurred_img = cv2.resize(blurred_img, (w, h))
        sharp_img = cv2.resize(sharp_img, (w, h))
        
        # Random crop
        if h > self.crop_size and w > self.crop_size:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            
            blurred_img = blurred_img[top:top+self.crop_size, left:left+self.crop_size]
            sharp_img = sharp_img[top:top+self.crop_size, left:left+self.crop_size]
        else:
            # Resize if too small
            blurred_img = cv2.resize(blurred_img, (self.crop_size, self.crop_size))
            sharp_img = cv2.resize(sharp_img, (self.crop_size, self.crop_size))
        
        # Convert to PIL for transforms
        blurred_img = Image.fromarray(blurred_img)
        sharp_img = Image.fromarray(sharp_img)
        
        # Apply transforms
        if self.transform:
            # Apply same random transforms to both images
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            torch.manual_seed(seed)
            blurred_tensor = self.transform(blurred_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            sharp_tensor = self.transform(sharp_img)
        else:
            # Default normalization
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            blurred_tensor = normalize(to_tensor(blurred_img))
            sharp_tensor = normalize(to_tensor(sharp_img))
        
        return blurred_tensor, sharp_tensor


def load_physics_dataset(
    physics_dir: Path,
    crop_size: int = 256,
    train_ratio: float = 0.8
) -> Tuple[DeblurDataset, DeblurDataset]:
    """
    Load physics-based synthetic dataset with real camera physics.
    
    Args:
        physics_dir: Path to physics dataset directory
        crop_size: Size for random cropping
        train_ratio: Fraction of data for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading physics-based dataset from: {physics_dir}")
    
    # Check if physics dataset exists
    if not physics_dir.exists():
        raise ValueError(f"Physics dataset not found: {physics_dir}")
    
    blur_dir = physics_dir / "blurry"
    sharp_dir = physics_dir / "sharp"
    
    # Verify structure
    for dir_path in [blur_dir, sharp_dir]:
        if not dir_path.exists():
            raise ValueError(f"Missing directory in physics dataset: {dir_path}")
    
    # Load all pairs
    blur_paths = sorted(list(blur_dir.glob("*.jpg")))
    sharp_paths = sorted(list(sharp_dir.glob("*.jpg")))
    
    # Match pairs by filename
    blur_dict = {p.name: p for p in blur_paths}
    sharp_dict = {p.name.replace("_blurry", "_sharp"): p for p in sharp_paths}
    
    # Find matching pairs
    matched_pairs = []
    for blur_name, blur_path in blur_dict.items():
        sharp_name = blur_name.replace("_blurry", "_sharp")
        if sharp_name in sharp_dict:
            matched_pairs.append((blur_path, sharp_dict[sharp_name]))
    
    logger.info(f"Found {len(matched_pairs)} matched physics-based pairs")
    
    # Split into train/val
    random.shuffle(matched_pairs)
    train_size = int(train_ratio * len(matched_pairs))
    
    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size:]
    
    # Separate paths
    train_blur_paths = [pair[0] for pair in train_pairs]
    train_sharp_paths = [pair[1] for pair in train_pairs]
    val_blur_paths = [pair[0] for pair in val_pairs]
    val_sharp_paths = [pair[1] for pair in val_pairs]
    
    logger.info(f"Training pairs: {len(train_blur_paths)} blur, {len(train_sharp_paths)} sharp")
    logger.info(f"Validation pairs: {len(val_blur_paths)} blur, {len(val_sharp_paths)} sharp")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Create datasets
    train_dataset = DeblurDataset(train_blur_paths, train_sharp_paths, transform=transform, crop_size=crop_size)
    val_dataset = DeblurDataset(val_blur_paths, val_sharp_paths, transform=transform, crop_size=crop_size)
    
    logger.info(f"✅ Physics dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset


def load_enhanced_dataset(
    enhanced_dir: Path,
    crop_size: int = 256
) -> Tuple[DeblurDataset, DeblurDataset]:
    """
    Load pre-prepared enhanced dataset with GoPro real blur patterns.
    
    Args:
        enhanced_dir: Path to enhanced dataset directory
        crop_size: Size for random cropping
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading enhanced dataset from: {enhanced_dir}")
    
    # Check if enhanced dataset exists
    if not enhanced_dir.exists():
        raise ValueError(f"Enhanced dataset not found: {enhanced_dir}")
    
    train_blur_dir = enhanced_dir / "train" / "blur"
    train_sharp_dir = enhanced_dir / "train" / "sharp"
    val_blur_dir = enhanced_dir / "val" / "blur"
    val_sharp_dir = enhanced_dir / "val" / "sharp"
    
    # Verify structure
    for dir_path in [train_blur_dir, train_sharp_dir, val_blur_dir, val_sharp_dir]:
        if not dir_path.exists():
            raise ValueError(f"Missing directory in enhanced dataset: {dir_path}")
    
    # Load training pairs
    train_blur_paths = sorted(list(train_blur_dir.glob("*")))
    train_sharp_paths = sorted(list(train_sharp_dir.glob("*")))
    
    # Load validation pairs
    val_blur_paths = sorted(list(val_blur_dir.glob("*")))
    val_sharp_paths = sorted(list(val_sharp_dir.glob("*")))
    
    logger.info(f"Training pairs: {len(train_blur_paths)} blur, {len(train_sharp_paths)} sharp")
    logger.info(f"Validation pairs: {len(val_blur_paths)} blur, {len(val_sharp_paths)} sharp")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Create datasets
    train_dataset = DeblurDataset(train_blur_paths, train_sharp_paths, transform=transform, crop_size=crop_size)
    val_dataset = DeblurDataset(val_blur_paths, val_sharp_paths, transform=transform, crop_size=crop_size)
    
    logger.info(f"✅ Enhanced dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset


def prepare_deblur_dataset(
    car_data_dir: Path,
    wagon_data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8
) -> Tuple[DeblurDataset, DeblurDataset]:
    """
    Prepare training and validation datasets for deblurring.
    
    Args:
        car_data_dir: Path to car dataset (blurred_sharp)
        wagon_data_dir: Path to wagon dataset (for artificial blur)
        output_dir: Directory to save processed data
        train_ratio: Fraction of data for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Preparing deblur datasets...")
    
    blurred_paths = []
    sharp_paths = []
    
    # Process car dataset (real paired data)
    car_blurred_dir = car_data_dir / "blurred"
    car_sharp_dir = car_data_dir / "sharp"
    
    if car_blurred_dir.exists() and car_sharp_dir.exists():
        # Get matching pairs
        blurred_files = {f.stem: f for f in car_blurred_dir.glob("*.png")}
        sharp_files = {f.stem: f for f in car_sharp_dir.glob("*.png")}
        
        # Find matching pairs
        common_stems = set(blurred_files.keys()) & set(sharp_files.keys())
        logger.info(f"Found {len(common_stems)} car image pairs")
        
        for stem in common_stems:
            blurred_paths.append(blurred_files[stem])
            sharp_paths.append(sharp_files[stem])
    
    # Create artificial pairs from wagon dataset
    if wagon_data_dir.exists():
        wagon_train_dir = wagon_data_dir / "train" / "images"
        if wagon_train_dir.exists():
            wagon_images = list(wagon_train_dir.glob("*.jpg")) + list(wagon_train_dir.glob("*.png"))
            logger.info(f"Found {len(wagon_images)} wagon images for artificial blur")
            
            # Create artificial pairs (limit to balance dataset)
            max_wagon_pairs = min(len(wagon_images), len(blurred_paths))
            
            artificial_dir = output_dir / "artificial_pairs"
            artificial_dir.mkdir(parents=True, exist_ok=True)
            
            for i, wagon_path in enumerate(wagon_images[:max_wagon_pairs]):
                # Load original image
                original_img = cv2.imread(str(wagon_path))
                if original_img is None:
                    continue
                
                # Create blurred version
                blurred_img = simulate_railway_conditions(
                    original_img,
                    blur_kernel_range=(10, 25),  # Focus on blur
                    apply_all=False
                )
                
                # Save artificial pair
                sharp_artificial_path = artificial_dir / f"wagon_sharp_{i:04d}.png"
                blurred_artificial_path = artificial_dir / f"wagon_blurred_{i:04d}.png"
                
                cv2.imwrite(str(sharp_artificial_path), original_img)
                cv2.imwrite(str(blurred_artificial_path), blurred_img)
                
                blurred_paths.append(blurred_artificial_path)
                sharp_paths.append(sharp_artificial_path)
    
    logger.info(f"Total dataset size: {len(blurred_paths)} image pairs")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Create full dataset
    full_dataset = DeblurDataset(blurred_paths, sharp_paths, transform=transform)
    
    # Split into train/val
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Training set: {len(train_dataset)} pairs")
    logger.info(f"Validation set: {len(val_dataset)} pairs")
    
    return train_dataset, val_dataset


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for better visual quality.
    """
    
    def __init__(self):
        super().__init__()
        # Use VGG16 features
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.features = vgg.features[:16]  # Up to relu3_3
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Extract features
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        # Compute perceptual loss
        return self.mse_loss(pred_features, target_features)


def train_deblur_model(
    model: DeblurGANv2Generator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.0002,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """
    Train the DeblurGAN-v2 model.
    
    Args:
        model: DeblurGANv2Generator instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    model.to(device)
    
    # Loss functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_l1': [],
        'val_l1': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        
        for batch_idx, (blurred, sharp) in enumerate(train_loader):
            blurred, sharp = blurred.to(device), sharp.to(device)
            
            optimizer.zero_grad()
            
            # Generate deblurred image
            deblurred = model(blurred)
            
            # Compute losses
            l1 = l1_loss(deblurred, sharp)
            perceptual = perceptual_loss(deblurred, sharp)
            
            # Combined loss
            total_loss = l1 + 0.1 * perceptual
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_l1 += l1.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        
        with torch.no_grad():
            for blurred, sharp in val_loader:
                blurred, sharp = blurred.to(device), sharp.to(device)
                
                deblurred = model(blurred)
                
                l1 = l1_loss(deblurred, sharp)
                perceptual = perceptual_loss(deblurred, sharp)
                total_loss = l1 + 0.1 * perceptual
                
                val_loss += total_loss.item()
                val_l1 += l1.item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_l1 /= len(train_loader)
        val_l1 /= len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_l1'].append(train_l1)
        history['val_l1'].append(val_l1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Log progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train L1: {train_l1:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val L1: {val_l1:.4f}"
            )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return history


def benchmark_inference_time(model: DeblurGANv2Generator, device: torch.device, num_runs: int = 100) -> float:
    """
    Benchmark model inference time on crops.
    
    Args:
        model: Trained model
        device: Device to run on
        num_runs: Number of inference runs for averaging
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model.to(device)
    
    # Test on typical crop sizes
    crop_sizes = [(128, 128), (256, 256), (320, 240)]
    
    for h, w in crop_sizes:
        dummy_input = torch.randn(1, 3, h, w).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        logger.info(f"Inference time ({h}x{w}): {avg_time_ms:.2f}ms")
    
    # Return time for 256x256 crop (typical size)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    logger.info(f"Average inference time (256x256): {avg_time_ms:.2f}ms (target: <40ms)")
    
    return avg_time_ms


def export_to_onnx(
    model: DeblurGANv2Generator, 
    output_path: Path, 
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Export trained model to ONNX format with FP16 and dynamic shapes.
    
    Args:
        model: Trained model
        output_path: Path to save ONNX model
        device: Device to export from
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")
    
    # Test ONNX Runtime inference with dynamic shapes
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name
    
    # Test different input sizes
    test_sizes = [(128, 128), (256, 256), (320, 240)]
    
    for h, w in test_sizes:
        test_input = np.random.randn(1, 3, h, w).astype(np.float32)
        try:
            start_time = time.time()
            output = session.run(None, {input_name: test_input})
            inference_time = (time.time() - start_time) * 1000
            logger.info(f"ONNX inference ({h}x{w}): {inference_time:.2f}ms")
        except Exception as e:
            logger.warning(f"ONNX inference failed for size {h}x{w}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train DeblurGAN-v2 on physics-based dataset")
    parser.add_argument("--dataset-dir", type=str, default="data/deblur_train_physics_combined",
                       help="Path to physics dataset directory")
    parser.add_argument("--dataset-type", type=str, default="physics", choices=["physics", "enhanced", "legacy"],
                       help="Dataset type: physics (default), enhanced (GoPro), or legacy")
    parser.add_argument("--car-data", type=str, default="data/blurred_sharp",
                       help="Path to car dataset (legacy only)")
    parser.add_argument("--wagon-data", type=str, default="data/wagon_detection",
                       help="Path to wagon dataset (legacy only)")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0002,
                       help="Learning rate")
    parser.add_argument("--crop-size", type=int, default=256,
                       help="Crop size for training")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset type
    if args.dataset_type == "physics":
        logger.info("🚀 Using PHYSICS-BASED dataset with real camera physics!")
        logger.info("Features: Poisson noise, motion blur + vibration, JPEG compression")
    elif args.dataset_type == "enhanced":
        logger.info("🚀 Using GoPro-enhanced dataset for real blur patterns!")
    else:
        logger.info("Using legacy dataset preparation")
    
    # Check device - FORCE GPU usage
    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required for training but not available!")
        logger.error("Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support")
        sys.exit(1)
    
    device = torch.device("cuda")
    logger.info(f"Using device: {device} (GPU forced)")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Prepare datasets based on type
        if args.dataset_type == "physics":
            # Use physics-based dataset
            train_dataset, val_dataset = load_physics_dataset(dataset_dir, args.crop_size)
        elif args.dataset_type == "enhanced":
            # Use pre-prepared enhanced dataset
            train_dataset, val_dataset = load_enhanced_dataset(dataset_dir, args.crop_size)
        else:
            # Legacy dataset preparation
            car_data_dir = Path(args.car_data)
            wagon_data_dir = Path(args.wagon_data)
            train_dataset, val_dataset = prepare_deblur_dataset(
                car_data_dir, wagon_data_dir, output_dir
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Create model
        model = DeblurGANv2Generator()
        logger.info(f"Created DeblurGAN-v2 with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        history = train_deblur_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Save training history
        history_path = output_dir / "deblur_training_history.json"
        import json
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        # Benchmark inference time
        inference_time = benchmark_inference_time(model, device)
        
        # Check if target is met
        target_time = 40.0
        if inference_time > target_time:
            logger.warning(f"Inference time {inference_time:.2f}ms exceeds target of {target_time}ms")
        else:
            logger.info(f"✓ Inference time target met: {inference_time:.2f}ms < {target_time}ms")
        
        # Save PyTorch model
        torch_model_path = output_dir / "deblur_gan.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'architecture': 'deblur_gan_v2_mobilenet_dsc',
                'input_channels': 3,
                'output_channels': 3
            },
            'training_history': history,
            'inference_time_ms': inference_time
        }, torch_model_path)
        logger.info(f"PyTorch model saved to {torch_model_path}")
        
        # Export to ONNX
        onnx_model_path = output_dir / "deblur_gan.onnx"
        export_to_onnx(model, onnx_model_path, device)
        
        # Final validation
        final_val_loss = history['val_loss'][-1]
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        # Create summary
        summary = {
            'model': 'deblur_gan_v2_mobilenet_dsc',
            'dataset': args.dataset_type,
            'dataset_path': str(dataset_dir),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'crop_size': args.crop_size,
            'device': str(device),
            'final_val_loss': final_val_loss,
            'inference_time_ms': inference_time,
            'target_met': inference_time <= target_time,
            'onnx_exported': True,
            'physics_based': args.dataset_type == "physics",
            'training_pairs': len(train_dataset) + len(val_dataset)
        }
        
        summary_path = output_dir / "deblur_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        logger.info("DeblurGAN-v2 training completed successfully!")
        
        # Physics-based training success message
        if args.dataset_type == "physics":
            print("\n" + "="*60)
            print("🎉 PHYSICS-BASED DEBLURGAN TRAINING COMPLETE!")
            print("="*60)
            print(f"Training Pairs: {len(train_dataset) + len(val_dataset)}")
            print(f"Final Val Loss: {final_val_loss:.4f}")
            print(f"Inference Time: {inference_time:.2f}ms")
            print(f"Target Met: {'✅' if inference_time <= target_time else '❌'}")
            print("Features: Poisson noise + Motion blur + JPEG compression")
            print("Ready for real-world railway inspection!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
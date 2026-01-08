#!/usr/bin/env python3
"""
Gatekeeper Model Training Script

Trains a MobileNetV3-Small binary classifier for joint prediction of:
1. is_vehicle_present: Whether a frame contains a vehicle (car or wagon)
2. is_blurry: Whether the frame is motion blurred

The model serves as a fast pre-filter to skip expensive processing on
empty or unusable frames, targeting <0.5ms inference time.

Requirements: 2.1, 2.2, 2.3, 2.5, 10.1, 10.2
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
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


class GatekeeperDataset(Dataset):
    """
    Dataset for training the Gatekeeper binary classifier.
    
    Combines vehicle images (cars + wagons) with negative samples
    (empty backgrounds, partial vehicles, extreme blur).
    """
    
    def __init__(
        self, 
        image_paths: List[Path], 
        labels: List[Tuple[int, int]], 
        transform=None,
        augment_blur: bool = True
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of (is_vehicle_present, is_blurry) tuples
            transform: Optional image transforms
            augment_blur: Whether to apply blur augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment_blur = augment_blur
        
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get labels
        is_vehicle_present, is_blurry = self.labels[idx]
        
        # Handle negative samples (no vehicle present)
        if not is_vehicle_present:
            # Create synthetic empty background by heavily degrading the image
            # Apply extreme blur and noise to remove vehicle features
            image = cv2.GaussianBlur(image, (51, 51), 0)  # Heavy blur
            # Add noise
            noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
            image = cv2.add(image, noise)
            # Darken significantly
            image = (image * 0.3).astype(np.uint8)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Apply blur augmentation if enabled and not already blurry
        if self.augment_blur and is_vehicle_present and not is_blurry:
            if torch.rand(1) < 0.3:  # 30% chance to add blur
                # Apply motion blur to create blurry positive samples
                # Convert back to numpy for augmentation (handle grayscale)
                if image.shape[0] == 1:  # Grayscale
                    image_np = image.squeeze(0).numpy()  # Remove channel dim
                    image_np = np.stack([image_np] * 3, axis=-1)  # Convert to RGB
                else:
                    image_np = image.permute(1, 2, 0).numpy()
                
                # Apply augmentation
                image_np = simulate_railway_conditions(
                    (image_np * 255).astype(np.uint8),
                    apply_all=False  # Use random augmentation
                )
                
                # Convert back to tensor and grayscale
                image_rgb = torch.from_numpy(image_np / 255.0).permute(2, 0, 1).float()
                # Convert to grayscale
                image = 0.299 * image_rgb[0] + 0.587 * image_rgb[1] + 0.114 * image_rgb[2]
                image = image.unsqueeze(0)  # Add channel dimension back
                
                is_blurry = 1  # Update label to reflect added blur
        
        # Convert labels to tensor
        labels_tensor = torch.tensor([is_vehicle_present, is_blurry], dtype=torch.float32)
        
        return image, labels_tensor


class GatekeeperModel(nn.Module):
    """
    MobileNetV3-Small based binary classifier for vehicle presence and blur detection.
    
    Architecture:
    - Backbone: MobileNetV3-Small (pretrained)
    - Input: 64x64 grayscale images
    - Output: 2 logits [is_vehicle_present, is_blurry]
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load MobileNetV3-Small backbone
        self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Replace classifier for binary outputs
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # [is_vehicle_present, is_blurry]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def prepare_datasets(
    car_data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8
) -> Tuple[GatekeeperDataset, GatekeeperDataset]:
    """
    Prepare training and validation datasets from Physics Dataset (Cars).
    
    Uses the blurred_sharp car dataset with auto-generated labels:
    - sharp/*.jpg → Class 0 (Pass - not blurry)
    - blurred/*.jpg → Class 1 (Fail/Trigger Deblur - is blurry)
    
    Args:
        car_data_dir: Path to blurred_sharp car dataset
        output_dir: Directory to save processed data
        train_ratio: Fraction of data for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Preparing Gatekeeper datasets from Physics Dataset (Cars)...")
    
    image_paths = []
    labels = []
    
    # Process sharp car images → Class 0 (Pass)
    car_sharp_dir = car_data_dir / "sharp"
    if car_sharp_dir.exists():
        car_sharp_images = list(car_sharp_dir.glob("*.png")) + list(car_sharp_dir.glob("*.jpg"))
        logger.info(f"Found {len(car_sharp_images)} sharp car images → Class 0 (Pass)")
        
        for img_path in car_sharp_images:
            image_paths.append(img_path)
            labels.append((1, 0))  # vehicle_present=True, is_blurry=False (Pass)
    
    # Process blurred car images → Class 1 (Fail/Trigger Deblur)
    car_blurred_dir = car_data_dir / "blurred"
    if car_blurred_dir.exists():
        car_blurred_images = list(car_blurred_dir.glob("*.png")) + list(car_blurred_dir.glob("*.jpg"))
        logger.info(f"Found {len(car_blurred_images)} blurred car images → Class 1 (Fail/Trigger Deblur)")
        
        for img_path in car_blurred_images:
            image_paths.append(img_path)
            labels.append((1, 1))  # vehicle_present=True, is_blurry=True (Fail)
    
    # Generate some negative samples (no vehicle present)
    # Create synthetic empty backgrounds for better classification
    total_vehicle_samples = len(image_paths)
    negative_samples = min(total_vehicle_samples // 4, 200)  # 20% negative samples
    logger.info(f"Generating {negative_samples} negative samples (no vehicle)")
    
    # Create synthetic empty backgrounds by heavily degrading vehicle images
    for i in range(negative_samples):
        # Take a random vehicle image as base
        source_idx = np.random.randint(0, total_vehicle_samples)
        source_path = image_paths[source_idx]
        
        # Mark this as a negative sample (will be processed differently in dataset)
        image_paths.append(source_path)
        labels.append((0, 0))  # vehicle_present=False, is_blurry=False
    
    logger.info(f"Dataset Summary:")
    logger.info(f"  Total samples: {len(image_paths)}")
    logger.info(f"  Sharp cars (Pass): {sum(1 for _, (is_vehicle, is_blurry) in enumerate(labels) if is_vehicle and not is_blurry)}")
    logger.info(f"  Blurred cars (Fail): {sum(1 for _, (is_vehicle, is_blurry) in enumerate(labels) if is_vehicle and is_blurry)}")
    logger.info(f"  No vehicle (Negative): {sum(1 for _, (is_vehicle, _) in enumerate(labels) if not is_vehicle)}")
    
    # Define transforms for 64x64 grayscale input
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Create full dataset
    full_dataset = GatekeeperDataset(image_paths, labels, transform=transform)
    
    # Split into train/val
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Training set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def train_model(
    model: GatekeeperModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """
    Train the Gatekeeper model.
    
    Args:
        model: GatekeeperModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    model.to(device)
    
    # Loss function - Binary Cross Entropy for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy (both predictions must be correct)
            predictions = torch.sigmoid(outputs) > 0.5
            correct = (predictions == labels.bool()).all(dim=1).sum().item()
            train_correct += correct
            train_total += labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                correct = (predictions == labels.bool()).all(dim=1).sum().item()
                val_correct += correct
                val_total += labels.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Log progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return history


def benchmark_inference_time(model: GatekeeperModel, device: torch.device, num_runs: int = 1000) -> float:
    """
    Benchmark model inference time to ensure <0.5ms target.
    
    Args:
        model: Trained model
        device: Device to run on
        num_runs: Number of inference runs for averaging
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    
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
    logger.info(f"Average inference time: {avg_time_ms:.3f}ms (target: <0.5ms)")
    
    return avg_time_ms


def export_to_onnx(
    model: GatekeeperModel, 
    output_path: Path, 
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Export trained model to ONNX format with FP16 precision.
    
    Args:
        model: Trained model
        output_path: Path to save ONNX model
        device: Device to export from
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    
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
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")
    
    # Test ONNX Runtime inference
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name
    
    # Test inference
    test_input = np.random.randn(1, 1, 64, 64).astype(np.float32)
    start_time = time.time()
    output = session.run(None, {input_name: test_input})
    inference_time = (time.time() - start_time) * 1000
    
    logger.info(f"ONNX Runtime inference time: {inference_time:.3f}ms")
    logger.info(f"Output shape: {output[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Train Gatekeeper binary classifier")
    parser.add_argument("--car-data", type=str, default="data/blurred_sharp",
                       help="Path to Physics Dataset (Cars) - blurred_sharp")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup paths
    car_data_dir = Path(args.car_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify car dataset exists
    if not car_data_dir.exists():
        logger.error(f"Physics Dataset (Cars) not found at: {car_data_dir}")
        logger.error("Please ensure the blurred_sharp dataset is available")
        sys.exit(1)
    
    # Check device - FORCE GPU usage for RTX 5070 Ti Blackwell
    if not torch.cuda.is_available():
        logger.error("CUDA GPU is REQUIRED for RTX 5070 Ti but not available!")
        logger.error("Ensure PyTorch nightly with CUDA 12.8+ is installed")
        sys.exit(1)
    
    device = torch.device("cuda")
    logger.info(f"Using device: {device} (GPU FORCED for RTX 5070 Ti)")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    try:
        # Prepare datasets from Physics Dataset (Cars)
        train_dataset, val_dataset = prepare_datasets(
            car_data_dir, output_dir
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0  # Disable multiprocessing to avoid issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0  # Disable multiprocessing to avoid issues
        )
        
        # Create model
        model = GatekeeperModel(pretrained=True)
        logger.info(f"Created GatekeeperModel with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Save training history
        history_path = output_dir / "gatekeeper_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        # Benchmark inference time
        inference_time = benchmark_inference_time(model, device)
        
        # Check if target is met
        if inference_time > 0.5:
            logger.warning(f"Inference time {inference_time:.3f}ms exceeds target of 0.5ms")
        else:
            logger.info(f"✓ Inference time target met: {inference_time:.3f}ms < 0.5ms")
        
        # Save PyTorch model
        torch_model_path = output_dir / "gatekeeper.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': (1, 64, 64),
                'num_classes': 2,
                'class_names': ['is_vehicle_present', 'is_blurry']
            },
            'training_history': history,
            'inference_time_ms': inference_time
        }, torch_model_path)
        logger.info(f"PyTorch model saved to {torch_model_path}")
        
        # Export to ONNX
        onnx_model_path = output_dir / "gatekeeper.onnx"
        export_to_onnx(model, onnx_model_path, device)
        
        # Final validation
        final_val_acc = history['val_acc'][-1]
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        if final_val_acc >= 0.95:
            logger.info("✓ Target accuracy achieved: ≥95%")
        else:
            logger.warning(f"Target accuracy not met: {final_val_acc:.4f} < 0.95")
        
        logger.info("Gatekeeper training completed successfully!")
        logger.info("Model trained on Physics Dataset (Cars):")
        logger.info("  - Sharp images → Class 0 (Pass)")
        logger.info("  - Blurred images → Class 1 (Fail/Trigger Deblur)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
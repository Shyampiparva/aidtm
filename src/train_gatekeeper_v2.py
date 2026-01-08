#!/usr/bin/env python3
"""
Gatekeeper V2: Upgraded Model Architecture

Addresses the underfitting issue by upgrading from MobileNetV3-Small to MobileNetV3-Large.
Uses domain-correct wagon data instead of car data.

Improvements:
1. MobileNetV3-Large (more capacity for complex "Night + Blur" patterns)
2. Domain-matched training data (wagons, not cars)
3. Advanced data augmentation
4. Improved training strategy
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
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import cv2
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


logger = logging.getLogger(__name__)


class GatekeeperV2Dataset(Dataset):
    """
    Dataset for Gatekeeper V2 using domain-correct wagon data.
    """
    
    def __init__(self, sharp_dir: Path, blurry_dir: Path, transform=None):
        """
        Args:
            sharp_dir: Directory containing sharp wagon images
            blurry_dir: Directory containing blurry wagon images
            transform: Optional image transforms
        """
        self.transform = transform
        
        # Load sharp images (Class 0)
        self.sharp_images = list(sharp_dir.glob("*.jpg")) + list(sharp_dir.glob("*.png"))
        self.sharp_labels = [0] * len(self.sharp_images)
        
        # Load blurry images (Class 1) 
        self.blurry_images = list(blurry_dir.glob("*.jpg")) + list(blurry_dir.glob("*.png"))
        self.blurry_labels = [1] * len(self.blurry_images)
        
        # Combine
        self.images = self.sharp_images + self.blurry_images
        self.labels = self.sharp_labels + self.blurry_labels
        
        logger.info(f"Dataset: {len(self.sharp_images)} sharp, {len(self.blurry_images)} blurry")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


class GatekeeperV2Model(nn.Module):
    """
    Gatekeeper V2 using MobileNetV3-Large for increased capacity.
    
    Architecture:
    - Backbone: MobileNetV3-Large (pretrained)
    - Input: 224x224 RGB images (larger than V1's 64x64)
    - Output: 2 classes [sharp, blurry]
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        
        # Load MobileNetV3-Large backbone
        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)
        
        # Replace classifier for binary classification
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def prepare_datasets(
    dataset_dir: Path,
    train_ratio: float = 0.8
) -> Tuple[GatekeeperV2Dataset, GatekeeperV2Dataset]:
    """
    Prepare training and validation datasets from improved gatekeeper data.
    
    Args:
        dataset_dir: Path to gatekeeper_improved dataset
        train_ratio: Fraction of data for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Preparing Gatekeeper V2 datasets from domain-correct data...")
    
    # Check if improved dataset exists
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Improved gatekeeper dataset not found at {dataset_dir}. "
            "Run src/augment_gatekeeper.py first."
        )
    
    # Define transforms for larger input size (224x224 vs 64x64)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GatekeeperV2Dataset(
        sharp_dir=train_dir / "sharp",
        blurry_dir=train_dir / "blurry",
        transform=train_transform
    )
    
    val_dataset = GatekeeperV2Dataset(
        sharp_dir=val_dir / "sharp", 
        blurry_dir=val_dir / "blurry",
        transform=val_transform
    )
    
    logger.info(f"Training set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def train_model(
    model: GatekeeperV2Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """
    Train the Gatekeeper V2 model.
    """
    model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
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
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
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
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
        logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return history


def benchmark_inference_time(model: GatekeeperV2Model, device: torch.device, num_runs: int = 1000) -> float:
    """
    Benchmark model inference time.
    """
    model.eval()
    model.to(device)
    
    # Create dummy input (224x224 vs 64x64 in V1)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
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
    logger.info(f"Average inference time: {avg_time_ms:.3f}ms")
    
    return avg_time_ms


def export_to_onnx(
    model: GatekeeperV2Model, 
    output_path: Path, 
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Export trained model to ONNX format.
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
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
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    start_time = time.time()
    output = session.run(None, {input_name: test_input})
    inference_time = (time.time() - start_time) * 1000
    
    logger.info(f"ONNX Runtime inference time: {inference_time:.3f}ms")
    logger.info(f"Output shape: {output[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Train Gatekeeper V2 with upgraded architecture")
    parser.add_argument("--dataset-dir", type=str, default="data/gatekeeper_improved",
                       help="Path to improved gatekeeper dataset")
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
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(dataset_dir)
        
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
            num_workers=0
        )
        
        # Create model
        model = GatekeeperV2Model(pretrained=True)
        logger.info(f"Created GatekeeperV2Model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Save training history
        history_path = output_dir / "gatekeeper_v2_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        # Benchmark inference time
        inference_time = benchmark_inference_time(model, device)
        
        # Save PyTorch model
        torch_model_path = output_dir / "gatekeeper_v2.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'architecture': 'mobilenet_v3_large',
                'input_size': (3, 224, 224),
                'num_classes': 2,
                'class_names': ['sharp', 'blurry']
            },
            'training_history': history,
            'inference_time_ms': inference_time
        }, torch_model_path)
        logger.info(f"PyTorch model saved to {torch_model_path}")
        
        # Export to ONNX
        onnx_model_path = output_dir / "gatekeeper_v2.onnx"
        export_to_onnx(model, onnx_model_path, device)
        
        # Final validation
        final_val_acc = history['val_acc'][-1]
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        # Check targets
        if final_val_acc >= 0.90:
            logger.info("🎉 SUCCESS: Gatekeeper V2 achieves ≥90% accuracy!")
        elif final_val_acc >= 0.85:
            logger.info("✅ GOOD: Gatekeeper V2 achieves ≥85% accuracy")
        else:
            logger.warning(f"⚠️  Target not met: {final_val_acc:.4f} < 0.85")
        
        logger.info("Gatekeeper V2 training completed successfully!")
        logger.info("Improvements over V1:")
        logger.info("  - MobileNetV3-Large (vs Small)")
        logger.info("  - Domain-correct wagon data (vs car data)")
        logger.info("  - 224x224 input (vs 64x64)")
        logger.info("  - Advanced augmentation")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
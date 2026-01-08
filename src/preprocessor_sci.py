#!/usr/bin/env python3
"""
SCI (Self-Calibrated Illumination) Preprocessor for Low-Light Enhancement.

Replaces Zero-DCE with faster SCI model (~0.5ms vs ~3ms).
Based on "Toward Fast, Flexible, and Robust Low-Light Image Enhancement" (CVPR 2022).

Repository: https://github.com/vis-opt-group/SCI
Paper: https://arxiv.org/abs/2204.10137
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Basic building block for SCI illumination estimation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic block."""
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.sigmoid(self.conv3(out))
        
        return out


class SCIModel(nn.Module):
    """
    SCI (Self-Calibrated Illumination) Model.
    
    Lightweight model for fast low-light image enhancement.
    Uses illumination estimation: Enhanced = Input / IlluminationMap
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # Single basic block for inference (weight-shared during training)
        self.illumination_estimator = BasicBlock(in_channels, out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: estimate illumination map.
        
        Args:
            x: Input low-light image [B, C, H, W] in range [0, 1]
            
        Returns:
            Illumination map [B, C, H, W] in range (0, 1]
        """
        # Estimate illumination map
        illu_map = self.illumination_estimator(x)
        
        # Ensure illumination map is in valid range (avoid division by zero)
        illu_map = torch.clamp(illu_map, min=0.01, max=1.0)
        
        return illu_map


class SCIPreprocessor:
    """
    SCI Preprocessor for fast low-light enhancement.
    
    Optimized for railway inspection with ~0.5ms inference time.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "cuda",
        target_size: int = 512,
        brightness_threshold: int = 50
    ):
        """
        Initialize SCI preprocessor.
        
        Args:
            model_path: Path to pretrained SCI model (.pth file)
            device: Device for inference ('cuda' or 'cpu')
            target_size: Target size for processing (512 for speed)
            brightness_threshold: Skip SCI if mean pixel > threshold (daytime optimization)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.brightness_threshold = brightness_threshold
        
        # Initialize model
        self.model = SCIModel()
        self.model.to(self.device)
        self.model.eval()
        
        # Load pretrained weights
        if model_path and Path(model_path).exists():
            self._load_pretrained_weights(model_path)
        else:
            logger.warning("No pretrained weights provided, using initialized weights")
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        logger.info(f"SCI Preprocessor initialized on {self.device}")
        logger.info(f"Target size: {target_size}, Brightness threshold: {brightness_threshold}")
    
    def _load_pretrained_weights(self, model_path: str):
        """Load pretrained SCI weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load weights with flexible key matching
            model_dict = self.model.state_dict()
            
            # Filter and match keys
            filtered_dict = {}
            for k, v in state_dict.items():
                # Remove module prefix if present
                key = k.replace('module.', '') if k.startswith('module.') else k
                
                if key in model_dict and v.shape == model_dict[key].shape:
                    filtered_dict[key] = v
            
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict, strict=False)
            
            logger.info(f"✅ Loaded SCI weights from {model_path}")
            logger.info(f"   Matched {len(filtered_dict)}/{len(model_dict)} parameters")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.warning("Using initialized weights instead")
    
    def should_enhance(self, image: np.ndarray) -> bool:
        """
        Check if image needs low-light enhancement.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            True if enhancement needed, False for daytime images
        """
        # Calculate mean brightness
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = np.mean(gray)
        
        # Skip enhancement for bright images (daytime optimization)
        return mean_brightness <= self.brightness_threshold
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess image for SCI model.
        
        Args:
            image: Input image [H, W, C] in BGR format
            
        Returns:
            Tuple of (preprocessed_tensor, metadata)
        """
        original_shape = image.shape[:2]  # H, W
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize for faster processing
        if max(original_shape) > self.target_size:
            scale = self.target_size / max(original_shape)
            new_h, new_w = int(original_shape[0] * scale), int(original_shape[1] * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        if isinstance(image_rgb, np.ndarray):
            image_rgb = Image.fromarray(image_rgb)
        
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        metadata = {
            'original_shape': original_shape,
            'processed_shape': tensor.shape[2:],  # H, W
            'scale_factor': self.target_size / max(original_shape) if max(original_shape) > self.target_size else 1.0
        }
        
        return tensor, metadata
    
    def postprocess_result(self, enhanced_tensor: torch.Tensor, metadata: dict) -> np.ndarray:
        """
        Postprocess enhanced result back to original format.
        
        Args:
            enhanced_tensor: Enhanced image tensor [1, C, H, W]
            metadata: Preprocessing metadata
            
        Returns:
            Enhanced image [H, W, C] in BGR format
        """
        # Convert tensor to numpy
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced = np.transpose(enhanced, (1, 2, 0))  # CHW -> HWC
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        # Resize back to original size if needed
        if enhanced.shape[:2] != metadata['original_shape']:
            enhanced = cv2.resize(enhanced, 
                                (metadata['original_shape'][1], metadata['original_shape'][0]),
                                interpolation=cv2.INTER_LINEAR)
        
        # Convert RGB back to BGR
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def enhance_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Enhance low-light image using SCI.
        
        Args:
            image: Input image [H, W, C] in BGR format
            
        Returns:
            Tuple of (enhanced_image, processing_info)
        """
        start_time = time.time()
        
        # Check if enhancement is needed
        if not self.should_enhance(image):
            processing_info = {
                'enhanced': False,
                'reason': 'bright_image',
                'mean_brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                'processing_time_ms': 0.0
            }
            return image.copy(), processing_info
        
        # Preprocess
        input_tensor, metadata = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            # Estimate illumination map
            illu_map = self.model(input_tensor)
            
            # Enhance: Enhanced = Input / IlluminationMap
            enhanced_tensor = input_tensor / illu_map
            
            # Clamp to valid range
            enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
        
        # Postprocess
        enhanced_image = self.postprocess_result(enhanced_tensor, metadata)
        
        processing_time = (time.time() - start_time) * 1000
        
        processing_info = {
            'enhanced': True,
            'reason': 'low_light_detected',
            'mean_brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
            'processing_time_ms': processing_time,
            'input_shape': image.shape,
            'processed_shape': input_tensor.shape[2:],
            'scale_factor': metadata['scale_factor']
        }
        
        return enhanced_image, processing_info
    
    def benchmark_performance(self, num_runs: int = 100) -> dict:
        """
        Benchmark SCI performance.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking SCI performance ({num_runs} runs)...")
        
        # Create test image
        test_image = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)  # Dark image
        
        # Warmup
        for _ in range(10):
            _, _ = self.enhance_image(test_image)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _, _ = self.enhance_image(test_image)
            times.append((time.time() - start_time) * 1000)
        
        stats = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps_capability': 1000 / np.mean(times),
            'target_met': np.mean(times) < 1.0,  # Target: <1ms
            'num_runs': num_runs
        }
        
        logger.info(f"SCI Performance: {stats['mean_time_ms']:.2f}±{stats['std_time_ms']:.2f}ms")
        logger.info(f"FPS Capability: {stats['fps_capability']:.1f}")
        logger.info(f"Target (<1ms): {'✅' if stats['target_met'] else '❌'}")
        
        return stats


def download_pretrained_weights(model_name: str = "medium") -> str:
    """
    Download pretrained SCI weights.
    
    Args:
        model_name: Model variant ('medium' or 'difficult')
        
    Returns:
        Path to downloaded weights
    """
    weights_dir = Path("models/sci_weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: These URLs are placeholders - in practice, you'd get them from the official repo
    urls = {
        "medium": "https://github.com/vis-opt-group/SCI/releases/download/v1.0/medium.pth",
        "difficult": "https://github.com/vis-opt-group/SCI/releases/download/v1.0/difficult.pth"
    }
    
    if model_name not in urls:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(urls.keys())}")
    
    weight_path = weights_dir / f"sci_{model_name}.pth"
    
    if not weight_path.exists():
        logger.info(f"Downloading SCI {model_name} weights...")
        try:
            urllib.request.urlretrieve(urls[model_name], weight_path)
            logger.info(f"✅ Downloaded weights to {weight_path}")
        except Exception as e:
            logger.warning(f"Failed to download weights: {e}")
            logger.warning("Using initialized weights instead")
            return None
    
    return str(weight_path)


def create_sci_preprocessor(
    model_variant: str = "medium",
    device: str = "cuda",
    target_size: int = 512,
    brightness_threshold: int = 50
) -> SCIPreprocessor:
    """
    Create SCI preprocessor with pretrained weights.
    
    Args:
        model_variant: Model variant ('medium' or 'difficult')
        device: Device for inference
        target_size: Target processing size
        brightness_threshold: Brightness threshold for daytime skip
        
    Returns:
        Configured SCI preprocessor
    """
    # Try to download pretrained weights
    weight_path = download_pretrained_weights(model_variant)
    
    # Create preprocessor
    preprocessor = SCIPreprocessor(
        model_path=weight_path,
        device=device,
        target_size=target_size,
        brightness_threshold=brightness_threshold
    )
    
    return preprocessor


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("🌙 SCI (Self-Calibrated Illumination) Preprocessor Demo")
    print("="*60)
    
    # Create preprocessor
    sci = create_sci_preprocessor(model_variant="medium")
    
    # Benchmark performance
    stats = sci.benchmark_performance(num_runs=50)
    
    print(f"\\n📊 Performance Results:")
    print(f"   Average Time: {stats['mean_time_ms']:.2f}ms")
    print(f"   FPS Capability: {stats['fps_capability']:.1f}")
    print(f"   Target Met (<1ms): {'✅' if stats['target_met'] else '❌'}")
    print(f"   vs Zero-DCE (~3ms): {3/stats['mean_time_ms']:.1f}x faster")
    
    print("\\n🚂 Ready for Railway Inspection Integration!")
    print("="*60)
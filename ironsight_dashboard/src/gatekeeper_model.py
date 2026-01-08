"""
Gatekeeper Model Implementation

MobileNetV3-Small binary classifier for dual prediction [is_wagon_present, is_blurry].
Target: <0.5ms inference on 64x64 grayscale thumbnails.
"""

import time
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path

# Optional torch imports for model training/loading
try:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    TORCH_AVAILABLE = True
except Exception:  # Catch all exceptions including OSError for DLL issues
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class MobileNetV3SmallGatekeeper(nn.Module):
        """
        MobileNetV3-Small based Gatekeeper for dual binary classification.
        
        Outputs two sigmoid-activated values:
        - is_wagon_present: probability that a wagon is visible
        - is_blurry: probability that the image is blurry
        """
        
        def __init__(self, num_classes: int = 2, pretrained: bool = True):
            """
            Initialize MobileNetV3-Small Gatekeeper.
            
            Args:
                num_classes: Number of output classes (default 2 for dual prediction)
                pretrained: Whether to use pretrained ImageNet weights
            """
            super().__init__()
            
            # Load MobileNetV3-Small backbone
            if pretrained:
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                self.backbone = mobilenet_v3_small(weights=weights)
            else:
                self.backbone = mobilenet_v3_small(weights=None)
            
            # Modify first conv layer to accept 1-channel grayscale input
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=1,  # Grayscale input
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize new conv layer with averaged weights from pretrained
            if pretrained:
                with torch.no_grad():
                    # Average RGB weights to create grayscale weights
                    self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            # Replace classifier head for dual binary classification
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Sequential(
                nn.Linear(in_features, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label binary classification
            )
            
            self.num_classes = num_classes
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, 1, 64, 64)
                
            Returns:
                Output tensor of shape (batch, 2) with sigmoid probabilities
            """
            return self.backbone(x)
        
        def export_to_onnx(
            self, 
            output_path: str, 
            use_fp16: bool = True,
            opset_version: int = 14
        ) -> bool:
            """
            Export model to ONNX format with optional FP16 precision.
            
            Args:
                output_path: Path to save ONNX model
                use_fp16: Whether to use FP16 precision
                opset_version: ONNX opset version
                
            Returns:
                True if export successful, False otherwise
            """
            try:
                self.eval()
                
                # Create dummy input
                dummy_input = torch.randn(1, 1, 64, 64)
                
                # Convert to FP16 if requested
                if use_fp16 and torch.cuda.is_available():
                    self.half()
                    dummy_input = dummy_input.half().cuda()
                    self.cuda()
                
                # Export to ONNX
                torch.onnx.export(
                    self,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                logger.info(f"Model exported to ONNX: {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
                return False
else:
    # Dummy class when torch is not available
    class MobileNetV3SmallGatekeeper:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Cannot create MobileNetV3SmallGatekeeper.")


class GatekeeperModel:
    """
    Gatekeeper binary classifier for pre-filtering frames.
    
    Performs dual prediction:
    - is_wagon_present: Whether a wagon is visible in the frame
    - is_blurry: Whether the frame is too blurry for processing
    
    Target performance: <0.5ms inference on 64x64 grayscale thumbnails
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the Gatekeeper model.
        
        Args:
            model_path: Path to ONNX or PyTorch model file. If None, uses mock model.
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        self.target_latency_ms = 0.5
        self.input_size = (64, 64)
        self.threshold = 0.5  # Binary classification threshold
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        # Model reference
        self.pytorch_model = None
        self.session = None
        
        # Load model
        if model_path and Path(model_path).exists():
            if model_path.endswith('.onnx'):
                self._load_onnx_model(model_path)
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                self._load_pytorch_model(model_path)
            else:
                logger.warning(f"Unknown model format: {model_path}. Using mock model.")
                self._create_mock_model()
        else:
            logger.warning("No model path provided or file not found. Creating mock model.")
            self._create_mock_model()
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model for inference."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Cannot load PyTorch model.")
            self._create_mock_model()
            return
            
        try:
            self.pytorch_model = MobileNetV3SmallGatekeeper(num_classes=2, pretrained=False)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.pytorch_model.load_state_dict(checkpoint)
            
            self.pytorch_model.eval()
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.pytorch_model = self.pytorch_model.cuda()
            
            self.model_type = "pytorch"
            logger.info(f"Loaded PyTorch Gatekeeper model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self._create_mock_model()
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model for optimized inference."""
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']  # Use CPU for compatibility
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.model_type = "onnx"
            logger.info(f"Loaded ONNX Gatekeeper model from {model_path}")
        except ImportError:
            logger.error("ONNX Runtime not available. Using mock model.")
            self._create_mock_model()
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock model for testing when real model is unavailable."""
        self.model_type = "mock"
        logger.info("Using mock Gatekeeper model")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to 64x64 grayscale thumbnail.
        
        Args:
            frame: Input frame (H, W, C) or (H, W)
            
        Returns:
            Preprocessed thumbnail (64, 64) grayscale
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Resize to 64x64
        thumbnail = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        thumbnail = thumbnail.astype(np.float32) / 255.0
        
        return thumbnail
    
    def predict(self, thumbnail: np.ndarray) -> Tuple[bool, bool]:
        """
        Joint prediction of wagon presence and blur.
        
        Args:
            thumbnail: 64x64 grayscale image (normalized to [0, 1])
            
        Returns:
            (is_wagon_present, is_blurry) tuple of booleans
        """
        start_time = time.time()
        
        try:
            if self.model_type == "onnx":
                predictions = self._predict_onnx(thumbnail)
            elif self.model_type == "pytorch":
                predictions = self._predict_pytorch(thumbnail)
            else:  # mock
                predictions = self._predict_mock(thumbnail)
            
            # Convert probabilities to boolean decisions
            is_wagon_present = predictions[0] > self.threshold
            is_blurry = predictions[1] > self.threshold
            
        except Exception as e:
            logger.error(f"Gatekeeper prediction failed: {e}")
            # Safe defaults: assume wagon present and not blurry
            is_wagon_present, is_blurry = True, False
        
        # Track performance
        inference_time_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time_ms)
        self.total_inferences += 1
        
        # Log performance violations
        if inference_time_ms > self.target_latency_ms:
            logger.warning(
                f"Gatekeeper latency violation: {inference_time_ms:.3f}ms "
                f"(target: {self.target_latency_ms}ms)"
            )
        
        return is_wagon_present, is_blurry
    
    def _predict_pytorch(self, thumbnail: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        # Prepare input
        input_tensor = torch.from_numpy(thumbnail).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            output = self.pytorch_model(input_tensor)
        
        # Convert to numpy
        predictions = output.cpu().numpy()[0]  # Shape: (2,)
        
        return predictions
    
    def _predict_onnx(self, thumbnail: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        # Prepare input
        input_tensor = thumbnail.reshape(1, 1, 64, 64).astype(np.float32)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_tensor})
        
        return result[0][0]  # Shape: (2,)
    
    def _predict_mock(self, thumbnail: np.ndarray) -> np.ndarray:
        """Mock prediction for testing."""
        # Simple heuristics for mock predictions
        mean_intensity = np.mean(thumbnail)
        variance = np.var(thumbnail)
        
        # Mock wagon detection: higher mean intensity suggests wagon presence
        wagon_prob = min(0.9, mean_intensity * 2.0)
        
        # Mock blur detection: lower variance suggests blur
        blur_prob = max(0.1, 1.0 - variance * 10.0)
        
        return np.array([wagon_prob, blur_prob], dtype=np.float32)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                'total_inferences': 0,
                'avg_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'violations': 0,
                'violation_rate': 0.0
            }
        
        violations = sum(1 for t in self.inference_times if t > self.target_latency_ms)
        
        return {
            'total_inferences': self.total_inferences,
            'avg_latency_ms': np.mean(self.inference_times),
            'max_latency_ms': np.max(self.inference_times),
            'violations': violations,
            'violation_rate': violations / len(self.inference_times),
            'model_type': self.model_type
        }


def create_gatekeeper_model(
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> GatekeeperModel:
    """
    Factory function to create a Gatekeeper model.
    
    Args:
        model_path: Path to model file (ONNX or PyTorch)
        device: Device to run inference on
        
    Returns:
        Initialized GatekeeperModel instance
    """
    return GatekeeperModel(model_path=model_path, device=device)


if __name__ == "__main__":
    # Example usage and testing
    
    # Create gatekeeper model
    gatekeeper = create_gatekeeper_model()
    
    # Test with random image
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    thumbnail = gatekeeper.preprocess(test_frame)
    
    # Predict
    is_wagon, is_blurry = gatekeeper.predict(thumbnail)
    
    print(f"Wagon present: {is_wagon}")
    print(f"Is blurry: {is_blurry}")
    print(f"Performance stats: {gatekeeper.get_performance_stats()}")
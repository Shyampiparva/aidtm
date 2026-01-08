"""
Model Optimizer for IronSight Command Center.

This module provides FP16 optimization for model inference without ONNX export.
Instead of converting models to ONNX, we load .pt models directly and use
FP16 acceleration via model(source, device='cuda', half=True).

Key Features:
- Direct .pt model loading using YOLO('model.pt')
- FP16 acceleration for faster inference
- Dynamic input shape support
- Performance validation and benchmarking
- Memory-efficient model management

Requirements: 13.1, 13.2, 13.4
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization."""
    use_fp16: bool = True
    device: str = "cuda"
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    target_latency_ms: float = 20.0


@dataclass
class OptimizedModelInfo:
    """Information about an optimized model."""
    model_name: str
    model_path: str
    is_loaded: bool = False
    uses_fp16: bool = False
    device: str = "cpu"
    load_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of model inference."""
    success: bool
    output: Any = None
    inference_time_ms: float = 0.0
    used_fp16: bool = False
    error_message: Optional[str] = None


class ModelOptimizer:
    """
    Optimizes model loading and inference using FP16 acceleration.
    
    Instead of ONNX export, this class:
    1. Loads .pt models directly using YOLO('model.pt') or torch.load()
    2. Enables FP16 acceleration via half=True parameter
    3. Validates performance meets latency targets
    4. Tracks memory usage and inference times
    """
    
    def __init__(self, config: Optional[ModelOptimizationConfig] = None):
        """
        Initialize ModelOptimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or ModelOptimizationConfig()
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, OptimizedModelInfo] = {}
        
        # Check CUDA availability
        self._check_cuda_availability()
        
        logger.info(f"ModelOptimizer initialized with FP16={self.config.use_fp16}, device={self.config.device}")
    
    def _check_cuda_availability(self):
        """Check if CUDA is available and update device accordingly."""
        # Skip torch import if device is already set to CPU
        if self.config.device == "cpu":
            self.config.use_fp16 = False
            logger.info("Device set to CPU, skipping CUDA check")
            return
            
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.config.device = "cpu"
                self.config.use_fp16 = False  # FP16 not beneficial on CPU
            else:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            self.config.device = "cpu"
            self.config.use_fp16 = False
        except OSError as e:
            # Handle DLL loading issues on Windows
            logger.warning(f"PyTorch DLL loading failed: {e}, falling back to CPU")
            self.config.device = "cpu"
            self.config.use_fp16 = False
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}, falling back to CPU")
            self.config.device = "cpu"
            self.config.use_fp16 = False
    
    def load_yolo_model(
        self,
        model_name: str,
        model_path: str,
        use_fp16: Optional[bool] = None
    ) -> OptimizedModelInfo:
        """
        Load a YOLO model with FP16 optimization.
        
        Uses YOLO('model.pt') for direct loading and enables FP16 via half=True.
        
        Args:
            model_name: Identifier for the model
            model_path: Path to .pt model file
            use_fp16: Override FP16 setting (uses config default if None)
            
        Returns:
            OptimizedModelInfo with loading results
        """
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.use_fp16
        
        info = OptimizedModelInfo(
            model_name=model_name,
            model_path=model_path,
            device=self.config.device,
            uses_fp16=use_fp16
        )
        
        start_time = time.time()
        
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                info.error_message = f"Model file not found: {model_path}"
                logger.error(info.error_message)
                return info
            
            # Load YOLO model directly
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Move to device
            if self.config.device == "cuda":
                model.to("cuda")
            
            # Store model
            self.loaded_models[model_name] = model
            
            info.is_loaded = True
            info.load_time_ms = (time.time() - start_time) * 1000
            
            # Estimate memory usage
            info.memory_usage_mb = self._estimate_model_memory(model)
            
            logger.info(
                f"Loaded YOLO model '{model_name}' from {model_path} "
                f"(FP16={use_fp16}, device={self.config.device}, "
                f"load_time={info.load_time_ms:.1f}ms)"
            )
            
            # Warmup and benchmark
            self._warmup_model(model_name, model)
            info.avg_inference_time_ms = self._benchmark_model(model_name, model, use_fp16)
            
        except ImportError as e:
            info.error_message = f"Ultralytics not installed: {e}"
            logger.error(info.error_message)
        except Exception as e:
            info.error_message = f"Failed to load model: {e}"
            logger.error(info.error_message)
        
        self.model_info[model_name] = info
        return info
    
    def load_pytorch_model(
        self,
        model_name: str,
        model_path: str,
        model_class: Any = None,
        use_fp16: Optional[bool] = None
    ) -> OptimizedModelInfo:
        """
        Load a PyTorch model with FP16 optimization.
        
        Args:
            model_name: Identifier for the model
            model_path: Path to .pt/.pth model file
            model_class: Optional model class for instantiation
            use_fp16: Override FP16 setting
            
        Returns:
            OptimizedModelInfo with loading results
        """
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.use_fp16
        
        info = OptimizedModelInfo(
            model_name=model_name,
            model_path=model_path,
            device=self.config.device,
            uses_fp16=use_fp16
        )
        
        start_time = time.time()
        
        try:
            import torch
            
            # Check if model file exists
            if not Path(model_path).exists():
                info.error_message = f"Model file not found: {model_path}"
                logger.error(info.error_message)
                return info
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)
            
            # Handle different checkpoint formats
            if model_class is not None:
                model = model_class()
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'params' in checkpoint:
                    model.load_state_dict(checkpoint['params'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Assume checkpoint is the model itself
                model = checkpoint
            
            # Move to device
            if hasattr(model, 'to'):
                model = model.to(self.config.device)
            
            # Enable FP16 if requested
            if use_fp16 and self.config.device == "cuda" and hasattr(model, 'half'):
                model = model.half()
            
            # Set to eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Store model
            self.loaded_models[model_name] = model
            
            info.is_loaded = True
            info.load_time_ms = (time.time() - start_time) * 1000
            info.memory_usage_mb = self._estimate_model_memory(model)
            
            logger.info(
                f"Loaded PyTorch model '{model_name}' from {model_path} "
                f"(FP16={use_fp16}, device={self.config.device})"
            )
            
        except ImportError as e:
            info.error_message = f"PyTorch not installed: {e}"
            logger.error(info.error_message)
        except Exception as e:
            info.error_message = f"Failed to load model: {e}"
            logger.error(info.error_message)
        
        self.model_info[model_name] = info
        return info
    
    def run_yolo_inference(
        self,
        model_name: str,
        source: Any,
        use_fp16: Optional[bool] = None,
        conf: float = 0.5,
        **kwargs
    ) -> InferenceResult:
        """
        Run YOLO inference with FP16 acceleration.
        
        Uses model(source, device='cuda', half=True) for optimized inference.
        
        Args:
            model_name: Name of loaded model
            source: Input image/video/path
            use_fp16: Override FP16 setting
            conf: Confidence threshold
            **kwargs: Additional arguments for YOLO inference
            
        Returns:
            InferenceResult with detection output
        """
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.use_fp16
        
        result = InferenceResult(success=False, used_fp16=use_fp16)
        
        if model_name not in self.loaded_models:
            result.error_message = f"Model '{model_name}' not loaded"
            return result
        
        model = self.loaded_models[model_name]
        
        start_time = time.time()
        
        try:
            # Run inference with FP16 acceleration
            # Key optimization: half=True enables FP16 inference
            output = model(
                source,
                device=self.config.device,
                half=use_fp16 and self.config.device == "cuda",
                conf=conf,
                verbose=False,
                **kwargs
            )
            
            result.success = True
            result.output = output
            result.inference_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            result.error_message = f"Inference failed: {e}"
            logger.error(result.error_message)
        
        return result
    
    def run_pytorch_inference(
        self,
        model_name: str,
        input_tensor: Any,
        use_fp16: Optional[bool] = None
    ) -> InferenceResult:
        """
        Run PyTorch model inference with FP16 acceleration.
        
        Args:
            model_name: Name of loaded model
            input_tensor: Input tensor
            use_fp16: Override FP16 setting
            
        Returns:
            InferenceResult with model output
        """
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.use_fp16
        
        result = InferenceResult(success=False, used_fp16=use_fp16)
        
        if model_name not in self.loaded_models:
            result.error_message = f"Model '{model_name}' not loaded"
            return result
        
        model = self.loaded_models[model_name]
        
        start_time = time.time()
        
        try:
            import torch
            
            # Ensure input is on correct device
            if hasattr(input_tensor, 'to'):
                input_tensor = input_tensor.to(self.config.device)
            
            # Convert to FP16 if needed
            if use_fp16 and self.config.device == "cuda":
                if hasattr(input_tensor, 'half'):
                    input_tensor = input_tensor.half()
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
            
            result.success = True
            result.output = output
            result.inference_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            result.error_message = f"Inference failed: {e}"
            logger.error(result.error_message)
        
        return result
    
    def _warmup_model(self, model_name: str, model: Any):
        """Warmup model with dummy inference."""
        try:
            # Create dummy input
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            for _ in range(self.config.warmup_iterations):
                if hasattr(model, '__call__'):
                    model(dummy_input, verbose=False)
            
            logger.debug(f"Warmed up model '{model_name}'")
        except Exception as e:
            logger.warning(f"Warmup failed for '{model_name}': {e}")
    
    def _benchmark_model(
        self,
        model_name: str,
        model: Any,
        use_fp16: bool
    ) -> float:
        """Benchmark model inference time."""
        try:
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            times = []
            
            for _ in range(self.config.benchmark_iterations):
                start = time.time()
                if hasattr(model, '__call__'):
                    model(
                        dummy_input,
                        device=self.config.device,
                        half=use_fp16 and self.config.device == "cuda",
                        verbose=False
                    )
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times)
            logger.info(f"Benchmark '{model_name}': {avg_time:.2f}ms avg (FP16={use_fp16})")
            return avg_time
            
        except Exception as e:
            logger.warning(f"Benchmark failed for '{model_name}': {e}")
            return 0.0
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        try:
            import torch
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_bytes = torch.cuda.memory_allocated()
                return memory_bytes / (1024 * 1024)
            
            # Fallback: estimate from parameters
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                # Assume FP32 (4 bytes per param)
                return (total_params * 4) / (1024 * 1024)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_model_info(self, model_name: str) -> Optional[OptimizedModelInfo]:
        """Get information about a loaded model."""
        return self.model_info.get(model_name)
    
    def get_all_model_info(self) -> Dict[str, OptimizedModelInfo]:
        """Get information about all loaded models."""
        return self.model_info.copy()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.loaded_models
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.model_info:
                self.model_info[model_name].is_loaded = False
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"Unloaded model '{model_name}'")
            return True
        return False
    
    def validate_performance(
        self,
        model_name: str,
        target_latency_ms: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate that model meets performance targets.
        
        Args:
            model_name: Name of model to validate
            target_latency_ms: Target latency (uses config default if None)
            
        Returns:
            Tuple of (meets_target, message)
        """
        target = target_latency_ms or self.config.target_latency_ms
        
        info = self.model_info.get(model_name)
        if not info:
            return False, f"Model '{model_name}' not found"
        
        if not info.is_loaded:
            return False, f"Model '{model_name}' not loaded"
        
        if info.avg_inference_time_ms <= target:
            return True, f"Model '{model_name}' meets target: {info.avg_inference_time_ms:.2f}ms <= {target}ms"
        else:
            return False, f"Model '{model_name}' exceeds target: {info.avg_inference_time_ms:.2f}ms > {target}ms"


def create_model_optimizer(
    use_fp16: bool = True,
    device: str = "cuda"
) -> ModelOptimizer:
    """
    Factory function to create a ModelOptimizer.
    
    Args:
        use_fp16: Enable FP16 acceleration
        device: Device for inference ('cuda' or 'cpu')
        
    Returns:
        Configured ModelOptimizer instance
    """
    config = ModelOptimizationConfig(
        use_fp16=use_fp16,
        device=device
    )
    return ModelOptimizer(config)


# Convenience functions for common model types

def load_yolo_with_fp16(
    model_path: str,
    model_name: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[Any, OptimizedModelInfo]:
    """
    Load a YOLO model with FP16 optimization enabled.
    
    This is the recommended way to load YOLO models for inference.
    Uses model(source, device='cuda', half=True) for FP16 acceleration.
    
    Args:
        model_path: Path to .pt model file
        model_name: Optional name for the model
        device: Device for inference
        
    Returns:
        Tuple of (model, info)
    """
    optimizer = create_model_optimizer(use_fp16=True, device=device)
    name = model_name or Path(model_path).stem
    info = optimizer.load_yolo_model(name, model_path)
    
    if info.is_loaded:
        return optimizer.loaded_models[name], info
    else:
        return None, info


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Model Optimizer Demo")
    print("=" * 60)
    
    # Create optimizer
    optimizer = create_model_optimizer(use_fp16=True, device="cuda")
    
    # Example: Load YOLO model (if available)
    yolo_path = "yolo_sideview_damage_obb.pt"
    if Path(yolo_path).exists():
        info = optimizer.load_yolo_model("sideview_damage", yolo_path)
        print(f"\nLoaded: {info.model_name}")
        print(f"  FP16: {info.uses_fp16}")
        print(f"  Device: {info.device}")
        print(f"  Load time: {info.load_time_ms:.1f}ms")
        print(f"  Avg inference: {info.avg_inference_time_ms:.1f}ms")
        
        # Validate performance
        meets_target, msg = optimizer.validate_performance("sideview_damage")
        print(f"  Performance: {msg}")
    else:
        print(f"\nModel file not found: {yolo_path}")
        print("This is expected in demo mode.")
    
    print("\n✅ Model Optimizer ready for IronSight Command Center!")

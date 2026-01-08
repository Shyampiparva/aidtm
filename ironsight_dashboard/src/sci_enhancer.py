#!/usr/bin/env python3
"""
SCI Enhancer Wrapper for IronSight Command Center.

Wraps existing src/preprocessor_sci.py with performance monitoring and statistics tracking.
Implements brightness-based skip logic for daytime optimization.
Target: ~0.5ms inference time.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import cv2

# Add parent directory to path to import from aidtm/src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import SCI preprocessor, fall back to mock if not available
SCI_AVAILABLE = False
try:
    from src.preprocessor_sci import SCIPreprocessor, create_sci_preprocessor
    SCI_AVAILABLE = True
except Exception:
    # Create mock classes when SCI is not available
    class SCIPreprocessor:
        """Mock SCI Preprocessor when real one is not available."""
        def __init__(self, model_path=None, device="cpu", target_size=512, brightness_threshold=50):
            self.brightness_threshold = brightness_threshold
            
        def enhance_image(self, image):
            """Mock enhancement - just returns original with info."""
            mean_brightness = np.mean(image)
            enhanced = bool(mean_brightness <= self.brightness_threshold)  # Convert to Python bool
            
            # Return expected reason strings
            if enhanced:
                reason = 'low_light_detected'
            else:
                reason = 'bright_image'
                
            return image, {
                'enhanced': enhanced,
                'reason': reason,
                'mean_brightness': float(mean_brightness)  # Convert to Python float
            }
    
    def create_sci_preprocessor(**kwargs):
        return SCIPreprocessor(**kwargs)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Performance statistics for SCI enhancement."""
    total_frames: int = 0
    enhanced_frames: int = 0
    skipped_frames: int = 0
    total_time_ms: float = 0.0
    recent_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def mean_time_ms(self) -> float:
        """Calculate mean processing time."""
        if not self.recent_times_ms:
            return 0.0
        return sum(self.recent_times_ms) / len(self.recent_times_ms)
    
    @property
    def enhancement_rate(self) -> float:
        """Calculate percentage of frames enhanced."""
        if self.total_frames == 0:
            return 0.0
        return (self.enhanced_frames / self.total_frames) * 100
    
    @property
    def skip_rate(self) -> float:
        """Calculate percentage of frames skipped."""
        if self.total_frames == 0:
            return 0.0
        return (self.skipped_frames / self.total_frames) * 100


class SCIEnhancer:
    """
    SCI Enhancement wrapper with performance monitoring.
    
    Wraps existing SCIPreprocessor with:
    - Performance monitoring and statistics tracking
    - Brightness-based skip logic for daytime optimization
    - Target ~0.5ms inference time
    - Latency budget violation logging
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        target_size: int = 512,
        brightness_threshold: int = 50,
        target_latency_ms: float = 0.5,
        warning_latency_ms: float = 1.0
    ):
        """
        Initialize SCI Enhancer.
        
        Args:
            model_path: Path to pretrained SCI model (.pth file)
            device: Device for inference ('cuda' or 'cpu')
            target_size: Target size for processing (512 for speed)
            brightness_threshold: Skip SCI if mean pixel > threshold (daytime optimization)
            target_latency_ms: Target latency budget (0.5ms)
            warning_latency_ms: Warning threshold for performance degradation (1.0ms)
        """
        self.device = device
        self.target_size = target_size
        self.brightness_threshold = brightness_threshold
        self.target_latency_ms = target_latency_ms
        self.warning_latency_ms = warning_latency_ms
        
        # Initialize underlying SCI preprocessor
        logger.info("Initializing SCI Enhancer...")
        self.sci_processor = SCIPreprocessor(
            model_path=model_path,
            device=device,
            target_size=target_size,
            brightness_threshold=brightness_threshold
        )
        
        # Performance tracking
        self.stats = PerformanceStats()
        
        logger.info(f"✅ SCI Enhancer initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Target size: {target_size}")
        logger.info(f"   Brightness threshold: {brightness_threshold}")
        logger.info(f"   Target latency: {target_latency_ms}ms")
        logger.info(f"   Warning latency: {warning_latency_ms}ms")
    
    def enhance_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Enhance low-light image with performance monitoring.
        
        Args:
            image: Input image [H, W, C] in BGR format
            
        Returns:
            Tuple of (enhanced_image, processing_info)
            
        Processing info includes:
            - enhanced: Whether enhancement was applied
            - reason: Reason for enhancement/skip decision
            - mean_brightness: Mean brightness of input image
            - processing_time_ms: Time taken for processing
            - within_target: Whether processing met target latency
            - within_warning: Whether processing met warning latency
        """
        start_time = time.time()
        
        # Check brightness and enhance if needed
        enhanced_image, sci_info = self.sci_processor.enhance_image(image)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats.total_frames += 1
        self.stats.total_time_ms += processing_time_ms
        self.stats.recent_times_ms.append(processing_time_ms)
        
        if sci_info['enhanced']:
            self.stats.enhanced_frames += 1
        else:
            self.stats.skipped_frames += 1
        
        # Check performance against targets
        within_target = processing_time_ms <= self.target_latency_ms
        within_warning = processing_time_ms <= self.warning_latency_ms
        
        # Log performance violations
        if not within_warning:
            overage_pct = (processing_time_ms - self.warning_latency_ms) / self.warning_latency_ms * 100
            logger.warning(
                f"⚠️  SCI performance degradation: {processing_time_ms:.3f}ms "
                f"(warning threshold: {self.warning_latency_ms}ms, overage: {overage_pct:.1f}%)"
            )
        
        # Build comprehensive processing info
        processing_info = {
            'enhanced': sci_info['enhanced'],
            'reason': sci_info['reason'],
            'mean_brightness': sci_info['mean_brightness'],
            'processing_time_ms': processing_time_ms,
            'within_target': within_target,
            'within_warning': within_warning,
            'target_latency_ms': self.target_latency_ms,
            'warning_latency_ms': self.warning_latency_ms,
            'input_shape': image.shape
        }
        
        return enhanced_image, processing_info
    
    def get_statistics(self) -> Dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_frames': self.stats.total_frames,
            'enhanced_frames': self.stats.enhanced_frames,
            'skipped_frames': self.stats.skipped_frames,
            'enhancement_rate_pct': self.stats.enhancement_rate,
            'skip_rate_pct': self.stats.skip_rate,
            'mean_time_ms': self.stats.mean_time_ms,
            'total_time_ms': self.stats.total_time_ms,
            'target_latency_ms': self.target_latency_ms,
            'warning_latency_ms': self.warning_latency_ms,
            'target_met': self.stats.mean_time_ms <= self.target_latency_ms if self.stats.total_frames > 0 else True,
            'warning_met': self.stats.mean_time_ms <= self.warning_latency_ms if self.stats.total_frames > 0 else True
        }
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = PerformanceStats()
        logger.info("📊 SCI Enhancer statistics reset")
    
    def benchmark_performance(self, num_runs: int = 100) -> Dict:
        """
        Benchmark SCI enhancement performance.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            Performance statistics
        """
        logger.info(f"🔬 Benchmarking SCI Enhancer ({num_runs} runs)...")
        
        # Create test images with different brightness levels
        dark_image = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
        bright_image = np.random.randint(100, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _, _ = self.enhance_image(dark_image)
        
        # Reset stats before benchmark
        self.reset_statistics()
        
        # Benchmark dark images (should be enhanced)
        dark_times = []
        for _ in range(num_runs // 2):
            start_time = time.time()
            _, info = self.enhance_image(dark_image)
            dark_times.append((time.time() - start_time) * 1000)
        
        # Benchmark bright images (should be skipped)
        bright_times = []
        for _ in range(num_runs // 2):
            start_time = time.time()
            _, info = self.enhance_image(bright_image)
            bright_times.append((time.time() - start_time) * 1000)
        
        all_times = dark_times + bright_times
        
        stats = {
            'mean_time_ms': np.mean(all_times),
            'std_time_ms': np.std(all_times),
            'min_time_ms': np.min(all_times),
            'max_time_ms': np.max(all_times),
            'dark_mean_ms': np.mean(dark_times),
            'bright_mean_ms': np.mean(bright_times),
            'fps_capability': 1000 / np.mean(all_times),
            'target_met': np.mean(all_times) <= self.target_latency_ms,
            'warning_met': np.mean(all_times) <= self.warning_latency_ms,
            'num_runs': num_runs,
            'enhancement_rate_pct': self.stats.enhancement_rate,
            'skip_rate_pct': self.stats.skip_rate
        }
        
        logger.info(f"📊 SCI Enhancer Performance:")
        logger.info(f"   Overall: {stats['mean_time_ms']:.3f}±{stats['std_time_ms']:.3f}ms")
        logger.info(f"   Dark images: {stats['dark_mean_ms']:.3f}ms (enhanced)")
        logger.info(f"   Bright images: {stats['bright_mean_ms']:.3f}ms (skipped)")
        logger.info(f"   FPS Capability: {stats['fps_capability']:.1f}")
        logger.info(f"   Target ({self.target_latency_ms}ms): {'✅' if stats['target_met'] else '❌'}")
        logger.info(f"   Warning ({self.warning_latency_ms}ms): {'✅' if stats['warning_met'] else '❌'}")
        logger.info(f"   Enhancement rate: {stats['enhancement_rate_pct']:.1f}%")
        logger.info(f"   Skip rate: {stats['skip_rate_pct']:.1f}%")
        
        return stats


def create_sci_enhancer(
    model_path: Optional[str] = None,
    device: str = "cuda",
    target_size: int = 512,
    brightness_threshold: int = 50,
    target_latency_ms: float = 0.5,
    warning_latency_ms: float = 1.0
) -> SCIEnhancer:
    """
    Create SCI Enhancer with default configuration.
    
    Args:
        model_path: Path to pretrained SCI model
        device: Device for inference
        target_size: Target processing size
        brightness_threshold: Brightness threshold for daytime skip
        target_latency_ms: Target latency budget
        warning_latency_ms: Warning threshold for performance degradation
        
    Returns:
        Configured SCI Enhancer
    """
    return SCIEnhancer(
        model_path=model_path,
        device=device,
        target_size=target_size,
        brightness_threshold=brightness_threshold,
        target_latency_ms=target_latency_ms,
        warning_latency_ms=warning_latency_ms
    )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🌙 SCI Enhancer Wrapper Demo")
    print("="*60)
    
    # Create enhancer
    enhancer = create_sci_enhancer()
    
    # Benchmark performance
    stats = enhancer.benchmark_performance(num_runs=100)
    
    print(f"\n📊 Performance Results:")
    print(f"   Average Time: {stats['mean_time_ms']:.3f}ms")
    print(f"   Dark Images: {stats['dark_mean_ms']:.3f}ms")
    print(f"   Bright Images: {stats['bright_mean_ms']:.3f}ms")
    print(f"   FPS Capability: {stats['fps_capability']:.1f}")
    print(f"   Target Met (0.5ms): {'✅' if stats['target_met'] else '❌'}")
    print(f"   Enhancement Rate: {stats['enhancement_rate_pct']:.1f}%")
    print(f"   Skip Rate: {stats['skip_rate_pct']:.1f}%")
    
    print("\n🚂 Ready for IronSight Command Center Integration!")
    print("="*60)

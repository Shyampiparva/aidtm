"""
Mock NAFNet Implementation

This module provides a mock NAFNet implementation that provides basic deblurring
functionality when the real NAFNet cannot be loaded due to dependency issues.
"""

import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)


class MockNAFNet:
    """
    Mock NAFNet implementation that provides basic deblurring using traditional methods.
    
    This is used as a fallback when BasicSR cannot be loaded due to compatibility issues.
    It provides reasonable deblurring results using classical image processing techniques.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize mock NAFNet.
        
        Args:
            model_path: Path to model (ignored in mock implementation)
            device: Device (ignored in mock implementation)
        """
        self.model_path = model_path
        self.device = device
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Mock model loading - always succeeds.
        
        Returns:
            True (always successful)
        """
        logger.info(f"Loading mock NAFNet (real model unavailable)")
        logger.info(f"Using enhanced unsharp masking for deblurring")
        self.is_loaded = True
        return True
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using mock deblurring algorithm.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Processed image as numpy array (H, W, 3) in BGR format
        """
        if not self.is_loaded:
            logger.warning("Mock model not loaded")
            return image
        
        try:
            return self._enhanced_deblur(image)
        except Exception as e:
            logger.error(f"Mock deblurring failed: {e}")
            return image
    
    def _enhanced_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced deblurring using multiple classical techniques.
        
        This combines several techniques:
        1. Unsharp masking for edge enhancement
        2. Bilateral filtering for noise reduction
        3. Contrast enhancement
        4. Sharpening kernel
        
        Args:
            image: Input blurry image
            
        Returns:
            Deblurred image
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Step 1: Bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(
            (img_float * 255).astype(np.uint8), 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        ).astype(np.float32) / 255.0
        
        # Step 2: Create unsharp mask
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
        unsharp_mask = bilateral - gaussian
        
        # Step 3: Apply unsharp masking with adaptive strength
        sharpened = bilateral + 1.5 * unsharp_mask
        
        # Step 4: Apply sharpening kernel
        kernel = np.array([
            [-0.1, -0.1, -0.1],
            [-0.1,  1.8, -0.1],
            [-0.1, -0.1, -0.1]
        ])
        kernel_sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        # Step 5: Blend original and processed
        result = 0.3 * img_float + 0.7 * kernel_sharpened
        
        # Step 6: Enhance contrast slightly
        result = np.power(result, 0.9)
        
        # Step 7: Apply edge enhancement
        gray = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(edges) / 255.0
        
        # Enhance edges in all channels
        for i in range(3):
            result[:, :, i] += 0.1 * edges
        
        # Clip and convert back to uint8
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result


class MockNAFNetLoader:
    """
    Mock NAFNet loader that mimics the interface of the real loader.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize mock loader.
        
        Args:
            model_path: Path to model file
            device: Device to use
        """
        self.model_path = model_path
        self.device = device
        self.model = MockNAFNet(model_path, device)
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the mock model.
        
        Returns:
            True if successful
        """
        success = self.model.load_model()
        self.is_loaded = success
        return success
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through mock NAFNet.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        return self.model.process_image(image)


def create_mock_nafnet(model_path: str, device: str = "cuda") -> MockNAFNetLoader:
    """
    Create and load a mock NAFNet model.
    
    Args:
        model_path: Path to model (ignored)
        device: Device (ignored)
        
    Returns:
        Loaded MockNAFNetLoader instance
    """
    loader = MockNAFNetLoader(model_path, device)
    loader.load_model()
    return loader


if __name__ == "__main__":
    # Test the mock implementation
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 Testing Mock NAFNet Implementation")
    print("="*60)
    
    # Create test image
    test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # Add blur
    blurred = cv2.GaussianBlur(test_image, (15, 15), 5)
    
    # Test mock NAFNet
    try:
        mock_nafnet = create_mock_nafnet("dummy_path.pth", "cpu")
        print("✅ Mock NAFNet created successfully")
        
        # Process image
        result = mock_nafnet.process_image(blurred)
        print(f"✅ Image processing successful: {blurred.shape} -> {result.shape}")
        
        # Calculate sharpness improvement
        def calculate_sharpness(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        
        original_sharpness = calculate_sharpness(blurred)
        result_sharpness = calculate_sharpness(result)
        improvement = ((result_sharpness - original_sharpness) / original_sharpness) * 100
        
        print(f"✅ Sharpness improvement: {improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ Mock NAFNet test failed: {e}")
    
    print("="*60)
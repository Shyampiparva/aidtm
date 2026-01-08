#!/usr/bin/env python3
"""
Restoration Lab - Interactive Image Restoration Testing Interface.

This module implements the Restoration Lab tab for the IronSight Command Center.
It provides an interactive interface for testing NAFNet image restoration with
before/after comparison visualization.

Key Features:
- File upload for JPG/PNG images
- Split-screen before/after comparison using streamlit-image-comparison
- Slider for comparison mix adjustment
- Processing time and quality metrics display
- Integration with CropFirstNAFNet for deblurring
- Error handling with automatic fallbacks

Requirements: 8.1, 8.2, 8.3, 8.5, 1.4, 11.5
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Import error handler for processing error handling
try:
    from error_handler import (
        ErrorHandler, create_error_handler, Result,
        ErrorCategory, RecoveryAction
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False
    ErrorHandler = None  # Set to None for type hints
    logger.warning("Error handler not available - running without error handling")


@dataclass
class RestorationResult:
    """Result of image restoration processing."""
    original_image: np.ndarray
    restored_image: np.ndarray
    processing_time_ms: float
    quality_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality metrics for restoration comparison."""
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index (simplified)
    sharpness_before: float  # Laplacian variance before
    sharpness_after: float  # Laplacian variance after
    sharpness_improvement: float  # Percentage improvement


class RestorationLab:
    """
    Interactive restoration interface for testing NAFNet image deblurring.
    
    This class provides:
    - File upload handling for JPG/PNG images
    - Integration with CropFirstNAFNet for deblurring
    - Quality metrics calculation (PSNR, sharpness)
    - Processing time tracking
    - Error handling with automatic fallbacks
    
    Requirements:
    - 8.1: File upload interface for blurry images (JPG, PNG formats)
    - 8.2: Split-screen comparison using streamlit-image-comparison
    - 8.3: Show [Raw Blurry Crop] vs [NAFNet Restored Crop]
    - 8.5: Display processing time and quality metrics
    - 1.4, 11.5: Error handling with fallbacks
    """
    
    def __init__(
        self, 
        nafnet_model_path: Optional[str] = None, 
        device: str = "cuda",
        error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize Restoration Lab.
        
        Args:
            nafnet_model_path: Path to NAFNet model checkpoint
            device: Device for inference ('cuda' or 'cpu')
            error_handler: Error handler instance (optional)
        """
        self.nafnet_model_path = nafnet_model_path
        self.device = device
        self.nafnet = None
        self.model_loaded = False
        
        # Initialize error handling
        self.error_handler = error_handler
        if ERROR_HANDLER_AVAILABLE and error_handler is None:
            self.error_handler = create_error_handler()
        
        # Error tracking
        self._consecutive_errors = 0
        self._max_retries = 2
        
        # Try to load NAFNet model
        self._load_nafnet()
        
        logger.info(f"RestorationLab initialized (model_loaded={self.model_loaded})")

    def _load_nafnet(self) -> bool:
        """Load NAFNet model for deblurring with error handling."""
        try:
            try:
                from crop_first_nafnet import CropFirstNAFNet
            except ImportError as e:
                logger.warning(f"Failed to import NAFNet module: {e}")
                logger.warning("This is likely due to BasicSR/torchvision compatibility issues.")
                logger.warning("Using mock restoration instead.")
                return False
            
            # Determine model path
            model_path = self.nafnet_model_path
            if model_path is None:
                # Try common locations
                possible_paths = [
                    Path(__file__).parent.parent.parent / "NAFNet-REDS-width64.pth",
                    Path("NAFNet-REDS-width64.pth"),
                    Path("models/NAFNet-REDS-width64.pth"),
                ]
                for p in possible_paths:
                    if p.exists():
                        model_path = str(p)
                        break
            
            if model_path and Path(model_path).exists():
                self.nafnet = CropFirstNAFNet(
                    model_path=model_path,
                    device=self.device
                )
                self.model_loaded = self.nafnet.load_model()
                logger.info(f"NAFNet model loaded: {self.model_loaded}")
            else:
                logger.warning("NAFNet model file not found, using mock restoration")
                self.model_loaded = False
                
                # Record error if handler available
                if self.error_handler and ERROR_HANDLER_AVAILABLE:
                    self.error_handler.record_error(
                        FileNotFoundError("NAFNet model file not found"),
                        stage_name="nafnet_loading",
                        recovery_action=RecoveryAction.USE_FALLBACK,
                        recovery_successful=True,
                        context={"fallback": "mock_restoration"}
                    )
                
        except Exception as e:
            logger.error(f"Failed to load NAFNet: {e}")
            self.model_loaded = False
            
            # Record error if handler available
            if self.error_handler and ERROR_HANDLER_AVAILABLE:
                self.error_handler.record_error(
                    e,
                    stage_name="nafnet_loading",
                    recovery_action=RecoveryAction.USE_FALLBACK,
                    recovery_successful=True,
                    context={"fallback": "mock_restoration"}
                )
            
        return self.model_loaded

    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load image from uploaded bytes.
        
        Args:
            image_bytes: Raw image bytes from file upload
            
        Returns:
            Image as numpy array in BGR format, or None if loading fails
        """
        try:
            # Use PIL to load image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array (RGB)
            image_rgb = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Loaded image: {image_bgr.shape}")
            return image_bgr
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Higher values indicate sharper images.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Sharpness score (Laplacian variance)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)

    def calculate_psnr(self, original: np.ndarray, restored: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between original and restored images.
        
        Note: For deblurring, we compare the restored image against itself
        as a baseline since we don't have ground truth sharp images.
        This metric shows the noise level in the restoration.
        
        Args:
            original: Original blurry image
            restored: Restored image
            
        Returns:
            PSNR value in dB
        """
        try:
            # Ensure same size
            if original.shape != restored.shape:
                restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
            
            # Calculate MSE
            mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
            
            if mse == 0:
                return float('inf')
            
            # Calculate PSNR
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return float(psnr)
            
        except Exception as e:
            logger.error(f"PSNR calculation failed: {e}")
            return 0.0
    
    def calculate_ssim_simple(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate simplified Structural Similarity Index.
        
        This is a simplified version that compares local statistics.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM value between 0 and 1
        """
        try:
            # Convert to grayscale
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Constants for stability
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            # Calculate means
            mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # Calculate variances
            sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
            
            # Calculate SSIM
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0

    def restore_image(self, image: np.ndarray) -> RestorationResult:
        """
        Restore a blurry image using NAFNet with error handling.
        
        Includes automatic retry for transient failures and fallback
        to mock restoration if NAFNet fails.
        
        Args:
            image: Blurry image as numpy array in BGR format
            
        Returns:
            RestorationResult with original, restored, metrics
        """
        start_time = time.time()
        
        try:
            # Calculate pre-restoration sharpness
            sharpness_before = self.calculate_sharpness(image)
            
            # Perform restoration with retry logic
            restored = self._restore_with_retry(image)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Calculate post-restoration sharpness
            sharpness_after = self.calculate_sharpness(restored)
            
            # Calculate quality metrics
            psnr = self.calculate_psnr(image, restored)
            ssim = self.calculate_ssim_simple(image, restored)
            
            # Calculate improvement
            if sharpness_before > 0:
                improvement = ((sharpness_after - sharpness_before) / sharpness_before) * 100
            else:
                improvement = 0.0
            
            quality_metrics = {
                'psnr_db': psnr,
                'ssim': ssim,
                'sharpness_before': sharpness_before,
                'sharpness_after': sharpness_after,
                'sharpness_improvement_pct': improvement,
                'model_used': 'NAFNet' if self.model_loaded else 'Mock'
            }
            
            # Reset error counter on success
            self._consecutive_errors = 0
            
            logger.info(
                f"Restoration complete: {processing_time_ms:.1f}ms, "
                f"sharpness {sharpness_before:.1f} -> {sharpness_after:.1f} "
                f"({improvement:+.1f}%)"
            )
            
            return RestorationResult(
                original_image=image,
                restored_image=restored,
                processing_time_ms=processing_time_ms,
                quality_metrics=quality_metrics,
                success=True
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._consecutive_errors += 1
            logger.error(f"Restoration failed: {e}")
            
            # Record error if handler available
            if self.error_handler and ERROR_HANDLER_AVAILABLE:
                self.error_handler.record_error(
                    e,
                    stage_name="image_restoration",
                    recovery_action=RecoveryAction.USE_FALLBACK,
                    recovery_successful=False
                )
            
            return RestorationResult(
                original_image=image,
                restored_image=image,  # Return original on failure
                processing_time_ms=processing_time_ms,
                quality_metrics={},
                success=False,
                error_message=str(e)
            )
    
    def _restore_with_retry(self, image: np.ndarray) -> np.ndarray:
        """
        Perform restoration with automatic retry on failure.
        
        Args:
            image: Input image
            
        Returns:
            Restored image
        """
        last_error = None
        
        for attempt in range(self._max_retries + 1):
            try:
                if self.model_loaded and self.nafnet is not None:
                    # Use NAFNet for real restoration
                    return self.nafnet.nafnet_deblur(image)
                else:
                    # Mock restoration: apply unsharp mask for demo
                    return self._mock_restoration(image)
                    
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(f"Restoration attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All restoration attempts failed: {e}")
        
        # All retries failed, use mock restoration as fallback
        logger.warning("Using mock restoration as fallback after failures")
        return self._mock_restoration(image)

    def _mock_restoration(self, image: np.ndarray) -> np.ndarray:
        """
        Mock restoration using unsharp masking when NAFNet is not available.
        
        This provides a visual demonstration of the interface even without
        the actual model loaded.
        
        Args:
            image: Input blurry image
            
        Returns:
            Sharpened image
        """
        # Apply unsharp mask for basic sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def blend_images(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray, 
        mix_ratio: float
    ) -> np.ndarray:
        """
        Blend two images with a given mix ratio.
        
        Args:
            image1: First image (shown at mix_ratio=0)
            image2: Second image (shown at mix_ratio=1)
            mix_ratio: Blend ratio between 0 and 1
            
        Returns:
            Blended image
        """
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Blend images
        blended = cv2.addWeighted(
            image1, 1.0 - mix_ratio,
            image2, mix_ratio,
            0
        )
        
        return blended
    
    def get_model_status(self) -> Dict[str, any]:
        """
        Get current model status.
        
        Returns:
            Dictionary with model status information
        """
        return {
            'model_loaded': self.model_loaded,
            'model_type': 'NAFNet-Width64' if self.model_loaded else 'Mock (Unsharp Mask)',
            'device': self.device,
            'model_path': self.nafnet_model_path
        }


def create_restoration_lab(
    nafnet_model_path: Optional[str] = None,
    device: str = "cuda",
    error_handler: Optional[ErrorHandler] = None
) -> RestorationLab:
    """
    Factory function to create RestorationLab instance.
    
    Args:
        nafnet_model_path: Path to NAFNet model checkpoint
        device: Device for inference
        error_handler: Error handler instance (optional)
        
    Returns:
        Configured RestorationLab instance
    """
    return RestorationLab(
        nafnet_model_path=nafnet_model_path,
        device=device,
        error_handler=error_handler
    )


def render_restoration_lab_ui(st_module):
    """
    Render the Restoration Lab UI in Streamlit.
    
    This function provides the complete UI for the Restoration Lab tab,
    including file upload, before/after comparison, and metrics display.
    
    Args:
        st_module: Streamlit module (passed to avoid import issues)
    """
    st = st_module
    
    # Import streamlit-image-comparison
    try:
        from streamlit_image_comparison import image_comparison
        comparison_available = True
    except ImportError:
        comparison_available = False
        st.warning("streamlit-image-comparison not installed. Using basic comparison.")
    
    # Initialize restoration lab in session state
    if 'restoration_lab' not in st.session_state:
        st.session_state.restoration_lab = create_restoration_lab()
    
    lab = st.session_state.restoration_lab
    
    # Model status indicator
    status = lab.get_model_status()
    if status['model_loaded']:
        st.success(f"🟢 Model: {status['model_type']} on {status['device']}")
    else:
        st.warning(f"🟡 Model: {status['model_type']} (NAFNet not loaded)")
    
    # File upload section
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload a blurry image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a blurry image to test the NAFNet restoration"
    )

    if uploaded_file is not None:
        # Load the image
        image_bytes = uploaded_file.read()
        original_image = lab.load_image_from_bytes(image_bytes)
        
        if original_image is not None:
            # Process button
            col1, col2 = st.columns([1, 3])
            with col1:
                process_btn = st.button("🔄 Restore Image", type="primary")
            
            # Store result in session state
            if 'restoration_result' not in st.session_state:
                st.session_state.restoration_result = None
            
            if process_btn:
                with st.spinner("Processing image..."):
                    result = lab.restore_image(original_image)
                    st.session_state.restoration_result = result
            
            # Display results if available
            result = st.session_state.restoration_result
            
            if result is not None and result.success:
                # Convert images to RGB for display
                original_rgb = cv2.cvtColor(result.original_image, cv2.COLOR_BGR2RGB)
                restored_rgb = cv2.cvtColor(result.restored_image, cv2.COLOR_BGR2RGB)
                
                # Metrics row
                st.subheader("📊 Quality Metrics")
                metric_cols = st.columns(5)
                
                metrics = result.quality_metrics
                with metric_cols[0]:
                    st.metric("Processing Time", f"{result.processing_time_ms:.1f}ms")
                with metric_cols[1]:
                    st.metric("Model", metrics.get('model_used', 'Unknown'))
                with metric_cols[2]:
                    st.metric(
                        "Sharpness Before", 
                        f"{metrics.get('sharpness_before', 0):.1f}"
                    )
                with metric_cols[3]:
                    st.metric(
                        "Sharpness After", 
                        f"{metrics.get('sharpness_after', 0):.1f}",
                        delta=f"{metrics.get('sharpness_improvement_pct', 0):+.1f}%"
                    )
                with metric_cols[4]:
                    st.metric("SSIM", f"{metrics.get('ssim', 0):.3f}")
                
                # Before/After comparison
                st.subheader("🔍 Before / After Comparison")
                
                if comparison_available:
                    # Use streamlit-image-comparison for split view
                    image_comparison(
                        img1=Image.fromarray(original_rgb),
                        img2=Image.fromarray(restored_rgb),
                        label1="Raw Blurry Image",
                        label2="NAFNet Restored",
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )
                else:
                    # Fallback: side-by-side comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_rgb, caption="Raw Blurry Image", width="stretch")
                    with col2:
                        st.image(restored_rgb, caption="NAFNet Restored", width="stretch")

                # Mix slider for blended view
                st.subheader("🎚️ Blend Comparison")
                mix_ratio = st.slider(
                    "Mix Ratio (0 = Original, 1 = Restored)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Adjust to blend between original and restored image"
                )
                
                # Show blended image
                blended = lab.blend_images(
                    result.original_image, 
                    result.restored_image, 
                    mix_ratio
                )
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                st.image(blended_rgb, caption=f"Blended (Mix: {mix_ratio:.0%})", width="stretch")
                
            elif result is not None and not result.success:
                st.error(f"Restoration failed: {result.error_message}")
            else:
                # Show original image preview
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, caption="Uploaded Image (Click 'Restore Image' to process)", width="stretch")
        else:
            st.error("Failed to load image. Please try a different file.")
    else:
        # Show placeholder
        st.info("👆 Upload a blurry image to test the restoration capabilities.")
        
        # Show example usage
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload** a blurry image (JPG or PNG format)
            2. Click **Restore Image** to process with NAFNet
            3. View the **Before/After comparison** with the slider
            4. Check **Quality Metrics** for sharpness improvement
            5. Use the **Mix slider** to blend between original and restored
            
            **Note:** If NAFNet model is not loaded, a mock restoration 
            (unsharp mask) will be used for demonstration.
            """)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔬 Restoration Lab Module")
    print("="*60)
    
    # Create restoration lab
    lab = create_restoration_lab()
    
    # Print status
    status = lab.get_model_status()
    print(f"Model loaded: {status['model_loaded']}")
    print(f"Model type: {status['model_type']}")
    print(f"Device: {status['device']}")
    
    # Test with a synthetic blurry image
    print("\nTesting with synthetic blurry image...")
    test_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    # Add some blur
    test_image = cv2.GaussianBlur(test_image, (15, 15), 5)
    
    # Restore
    result = lab.restore_image(test_image)
    
    print(f"\nRestoration Result:")
    print(f"  Success: {result.success}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"  Quality metrics: {result.quality_metrics}")
    
    print("\n🚂 Ready for IronSight Command Center Integration!")
    print("="*60)

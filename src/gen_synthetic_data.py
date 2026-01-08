"""
Advanced Physics-Based Synthetic Data Generation for Railway Wagon Deblurring.

This module implements real-world camera physics degradation to bridge the sim2real gap:
1. Poisson Noise (Shot Noise) - simulates sensor noise at different ISO levels
2. 1D Motion Blur with Track Vibration - horizontal motion with slight vertical jitter
3. JPEG Compression Artifacts - teaches model to ignore blocky artifacts around text

The degradation pipeline: Sharp → Poisson → Motion Blur → JPEG → Blurry Output
This creates realistic training data that matches actual railway inspection conditions.
"""

import os
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image
import io
from tqdm import tqdm
import argparse


class AdvancedDegradation:
    """
    Advanced physics-based image degradation for realistic synthetic data generation.
    
    Implements three key degradation types:
    1. Poisson noise (shot noise) - varies with ISO sensitivity
    2. Motion blur with track vibration - horizontal + slight vertical jitter
    3. JPEG compression artifacts - simulates real camera compression
    """
    
    def __init__(
        self,
        poisson_scale_range: Tuple[float, float] = (0.8, 1.2),
        motion_blur_size_range: Tuple[int, int] = (10, 20),
        track_vibration_angle: float = 0.5,  # degrees
        jpeg_quality_range: Tuple[int, int] = (30, 90),
        seed: Optional[int] = None
    ):
        """
        Initialize advanced degradation parameters.
        
        Args:
            poisson_scale_range: Range for Poisson noise scaling (simulates ISO)
            motion_blur_size_range: Range for motion blur kernel size
            track_vibration_angle: Maximum vertical jitter angle in degrees
            jpeg_quality_range: Range for JPEG compression quality
            seed: Random seed for reproducibility
        """
        self.poisson_scale_range = poisson_scale_range
        self.motion_blur_size_range = motion_blur_size_range
        self.track_vibration_angle = track_vibration_angle
        self.jpeg_quality_range = jpeg_quality_range
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.logger = logging.getLogger(__name__)
    
    def apply_poisson_noise(self, image: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """
        Apply Poisson noise (shot noise) to simulate sensor noise at different ISO levels.
        
        Poisson noise is the fundamental noise in digital sensors caused by the quantum
        nature of light. Higher ISO = higher scale = more noise.
        
        Args:
            image: Input image (0-255, uint8)
            scale: Poisson scaling factor (None for random)
            
        Returns:
            Noisy image with Poisson noise applied
        """
        if scale is None:
            scale = random.uniform(*self.poisson_scale_range)
        
        # Convert to float and normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Apply Poisson noise: noise = poisson(image * scale) / scale
        # Scale up, apply Poisson, scale back down
        scaled_image = image_float * scale
        
        # Poisson noise (shot noise)
        noisy_scaled = np.random.poisson(scaled_image).astype(np.float32)
        noisy_image = noisy_scaled / scale
        
        # Clip to valid range and convert back to uint8
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        return (noisy_image * 255.0).astype(np.uint8)
    
    def apply_motion_blur_with_vibration(
        self, 
        image: np.ndarray, 
        kernel_size: Optional[int] = None,
        angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply 1D motion blur with track vibration simulation.
        
        Simulates horizontal motion blur from wagon movement plus slight vertical
        jitter from track vibration and camera shake.
        
        Args:
            image: Input image
            kernel_size: Motion blur kernel size (None for random)
            angle: Blur angle in degrees (None for random vibration)
            
        Returns:
            Motion blurred image
        """
        if kernel_size is None:
            kernel_size = random.randint(*self.motion_blur_size_range)
        
        if angle is None:
            # Primarily horizontal (0°) with slight vertical jitter
            angle = random.uniform(-self.track_vibration_angle, self.track_vibration_angle)
        
        # Create motion blur kernel
        kernel = self._create_motion_blur_kernel(kernel_size, angle)
        
        # Apply convolution
        if len(image.shape) == 3:
            # Color image - apply to each channel
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        else:
            # Grayscale image
            blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def _create_motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        Create motion blur kernel with specified size and angle.
        
        Args:
            size: Kernel size (length of motion blur)
            angle: Angle in degrees (0 = horizontal)
            
        Returns:
            Motion blur kernel
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create kernel
        kernel = np.zeros((size, size), dtype=np.float32)
        
        # Calculate line endpoints
        center = size // 2
        dx = int(np.cos(angle_rad) * center)
        dy = int(np.sin(angle_rad) * center)
        
        # Draw line in kernel using Bresenham's algorithm
        x0, y0 = center - dx, center - dy
        x1, y1 = center + dx, center + dy
        
        # Simple line drawing
        points = self._bresenham_line(x0, y0, x1, y1)
        for x, y in points:
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        
        # Normalize kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            # Fallback: single point at center
            kernel[center, center] = 1.0
        
        return kernel
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for drawing motion blur kernel."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def apply_jpeg_compression(
        self, 
        image: np.ndarray, 
        quality: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply JPEG compression artifacts by encoding/decoding in memory.
        
        This simulates real camera JPEG compression which creates blocky artifacts
        around text and edges. The model learns to ignore these artifacts.
        
        Args:
            image: Input image
            quality: JPEG quality (1-100, None for random)
            
        Returns:
            Image with JPEG compression artifacts
        """
        if quality is None:
            quality = random.randint(*self.jpeg_quality_range)
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L')
        
        # Encode to JPEG in memory
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        
        # Decode back from JPEG
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to numpy array
        return np.array(compressed_image)
    
    def apply_full_degradation(
        self, 
        image: np.ndarray,
        poisson_scale: Optional[float] = None,
        motion_kernel_size: Optional[int] = None,
        motion_angle: Optional[float] = None,
        jpeg_quality: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply full degradation pipeline: Sharp → Poisson → Motion Blur → JPEG → Blurry.
        
        Args:
            image: Input sharp image
            poisson_scale: Poisson noise scale (None for random)
            motion_kernel_size: Motion blur kernel size (None for random)
            motion_angle: Motion blur angle (None for random)
            jpeg_quality: JPEG quality (None for random)
            
        Returns:
            Tuple of (degraded_image, degradation_params)
        """
        # Record actual parameters used
        if poisson_scale is None:
            poisson_scale = random.uniform(*self.poisson_scale_range)
        if motion_kernel_size is None:
            motion_kernel_size = random.randint(*self.motion_blur_size_range)
        if motion_angle is None:
            motion_angle = random.uniform(-self.track_vibration_angle, self.track_vibration_angle)
        if jpeg_quality is None:
            jpeg_quality = random.randint(*self.jpeg_quality_range)
        
        params = {
            'poisson_scale': poisson_scale,
            'motion_kernel_size': motion_kernel_size,
            'motion_angle': motion_angle,
            'jpeg_quality': jpeg_quality
        }
        
        # Step 1: Apply Poisson noise (shot noise)
        degraded = self.apply_poisson_noise(image, poisson_scale)
        
        # Step 2: Apply motion blur with track vibration
        degraded = self.apply_motion_blur_with_vibration(
            degraded, motion_kernel_size, motion_angle
        )
        
        # Step 3: Apply JPEG compression artifacts
        degraded = self.apply_jpeg_compression(degraded, jpeg_quality)
        
        return degraded, params


class SyntheticDataGenerator:
    """
    Synthetic data generator for railway wagon deblurring training.
    
    Processes sharp Roboflow wagon images through physics-based degradation
    to create realistic blurry/sharp pairs for deblurring model training.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        degradation: Optional[AdvancedDegradation] = None,
        target_size: Optional[Tuple[int, int]] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            input_dir: Directory containing sharp input images
            output_dir: Directory to save blurry/sharp pairs
            degradation: AdvancedDegradation instance (None for default)
            target_size: Target image size (width, height) for resizing
            extensions: Valid image file extensions
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.degradation = degradation or AdvancedDegradation()
        self.target_size = target_size
        self.extensions = extensions
        
        # Create output directories
        self.blurry_dir = self.output_dir / "blurry"
        self.sharp_dir = self.output_dir / "sharp"
        self.blurry_dir.mkdir(parents=True, exist_ok=True)
        self.sharp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def find_input_images(self) -> List[Path]:
        """Find all valid image files in input directory."""
        image_files = []
        for ext in self.extensions:
            image_files.extend(self.input_dir.rglob(f"*{ext}"))
            image_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} input images")
        return sorted(image_files)
    
    def process_image(self, image_path: Path, output_prefix: str) -> bool:
        """
        Process a single image through the degradation pipeline.
        
        Args:
            image_path: Path to input sharp image
            output_prefix: Prefix for output filenames
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Failed to load image: {image_path}")
                return False
            
            # Convert BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if target size specified
            if self.target_size:
                image_rgb = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply degradation pipeline
            degraded_rgb, params = self.degradation.apply_full_degradation(image_rgb)
            
            # Convert back to BGR for saving
            degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
            sharp_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Save blurry and sharp versions
            blurry_path = self.blurry_dir / f"{output_prefix}_blurry.jpg"
            sharp_path = self.sharp_dir / f"{output_prefix}_sharp.jpg"
            
            cv2.imwrite(str(blurry_path), degraded_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(sharp_path), sharp_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Log degradation parameters
            self.logger.debug(f"Processed {image_path.name}: {params}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def generate_dataset(
        self, 
        num_augmentations: int = 1,
        max_images: Optional[int] = None
    ) -> dict:
        """
        Generate synthetic dataset from input images.
        
        Args:
            num_augmentations: Number of degraded versions per input image
            max_images: Maximum number of input images to process (None for all)
            
        Returns:
            Dictionary with generation statistics
        """
        input_images = self.find_input_images()
        
        if max_images:
            input_images = input_images[:max_images]
        
        total_pairs = len(input_images) * num_augmentations
        successful_pairs = 0
        
        self.logger.info(f"Generating {total_pairs} image pairs...")
        
        with tqdm(total=total_pairs, desc="Generating synthetic data") as pbar:
            for img_idx, image_path in enumerate(input_images):
                for aug_idx in range(num_augmentations):
                    # Create unique output prefix
                    output_prefix = f"{image_path.stem}_{img_idx:04d}_{aug_idx:02d}"
                    
                    # Process image
                    if self.process_image(image_path, output_prefix):
                        successful_pairs += 1
                    
                    pbar.update(1)
        
        stats = {
            'input_images': len(input_images),
            'augmentations_per_image': num_augmentations,
            'total_pairs_attempted': total_pairs,
            'successful_pairs': successful_pairs,
            'success_rate': successful_pairs / total_pairs if total_pairs > 0 else 0,
            'output_directory': str(self.output_dir)
        }
        
        self.logger.info(f"Dataset generation complete: {stats}")
        return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate physics-based synthetic data for railway wagon deblurring"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/wagon_detection",
        help="Directory containing sharp input images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/deblur_train_physics",
        help="Output directory for blurry/sharp pairs"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        help="Number of degraded versions per input image"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of input images to process"
    )
    parser.add_argument(
        "--target-size",
        type=str,
        default=None,
        help="Target image size as 'width,height' (e.g., '512,512')"
    )
    parser.add_argument(
        "--poisson-scale-min",
        type=float,
        default=0.8,
        help="Minimum Poisson noise scale (ISO simulation)"
    )
    parser.add_argument(
        "--poisson-scale-max",
        type=float,
        default=1.2,
        help="Maximum Poisson noise scale (ISO simulation)"
    )
    parser.add_argument(
        "--motion-blur-min",
        type=int,
        default=10,
        help="Minimum motion blur kernel size"
    )
    parser.add_argument(
        "--motion-blur-max",
        type=int,
        default=20,
        help="Maximum motion blur kernel size"
    )
    parser.add_argument(
        "--vibration-angle",
        type=float,
        default=0.5,
        help="Track vibration angle in degrees"
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=30,
        help="Minimum JPEG compression quality"
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=90,
        help="Maximum JPEG compression quality"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Parse target size
    target_size = None
    if args.target_size:
        try:
            width, height = map(int, args.target_size.split(','))
            target_size = (width, height)
        except ValueError:
            logger.error("Invalid target size format. Use 'width,height' (e.g., '512,512')")
            return
    
    # Create degradation instance
    degradation = AdvancedDegradation(
        poisson_scale_range=(args.poisson_scale_min, args.poisson_scale_max),
        motion_blur_size_range=(args.motion_blur_min, args.motion_blur_max),
        track_vibration_angle=args.vibration_angle,
        jpeg_quality_range=(args.jpeg_quality_min, args.jpeg_quality_max),
        seed=args.seed
    )
    
    # Create generator
    generator = SyntheticDataGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        degradation=degradation,
        target_size=target_size
    )
    
    # Generate dataset
    logger.info("Starting physics-based synthetic data generation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Augmentations per image: {args.num_augmentations}")
    logger.info(f"Degradation pipeline: Sharp → Poisson → Motion Blur → JPEG → Blurry")
    
    stats = generator.generate_dataset(
        num_augmentations=args.num_augmentations,
        max_images=args.max_images
    )
    
    logger.info("Generation complete!")
    logger.info(f"Success rate: {stats['success_rate']:.2%}")
    logger.info(f"Generated {stats['successful_pairs']} training pairs")
    logger.info(f"Output saved to: {stats['output_directory']}")
    
    # Print summary
    print("\n" + "="*60)
    print("PHYSICS-BASED SYNTHETIC DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Input Images:     {stats['input_images']}")
    print(f"Augmentations:    {stats['augmentations_per_image']}")
    print(f"Total Pairs:      {stats['successful_pairs']}")
    print(f"Success Rate:     {stats['success_rate']:.2%}")
    print(f"Output Directory: {stats['output_directory']}")
    print("\nDegradation Pipeline:")
    print("  1. Poisson Noise (Shot Noise) - simulates ISO sensitivity")
    print("  2. Motion Blur + Track Vibration - horizontal motion with jitter")
    print("  3. JPEG Compression - realistic camera compression artifacts")
    print("\nReady for DeblurGAN training!")


if __name__ == "__main__":
    main()
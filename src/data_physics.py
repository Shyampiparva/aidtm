"""
Physics-based data augmentation for railway inspection conditions.

This module simulates real-world railway inspection conditions including
low-light environments, sensor noise, and motion blur to create robust
training data that matches actual deployment scenarios.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import random
from PIL import Image, ImageFilter


logger = logging.getLogger(__name__)


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """
    Apply gamma correction to simulate low-light conditions.
    
    Gamma values between 0.3-0.5 simulate the darkness conditions
    typical in railway inspection scenarios.
    
    Args:
        image: Input image as numpy array
        gamma: Gamma correction factor (0.3-0.5 for darkness simulation)
        
    Returns:
        Gamma-corrected image
    """
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using lookup table
    return cv2.LUT(image, table)


def apply_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Apply Poisson noise to simulate sensor grain.
    
    Poisson noise models the statistical variation in photon detection
    that occurs in real camera sensors, especially in low-light conditions.
    
    Args:
        image: Input image as numpy array
        scale: Noise scaling factor (higher = more noise)
        
    Returns:
        Image with Poisson noise applied
    """
    # Convert to float for noise calculation
    image_float = image.astype(np.float32)
    
    # Generate Poisson noise
    # Scale image to appropriate range for Poisson distribution
    scaled = image_float * scale
    noisy = np.random.poisson(scaled) / scale
    
    # Clip to valid range and convert back to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def apply_horizontal_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Apply horizontal motion blur to simulate wagon movement.
    
    Creates a horizontal motion blur kernel to simulate the effect
    of wagons moving horizontally past the camera at high speed.
    
    Args:
        image: Input image as numpy array
        kernel_size: Size of the motion blur kernel (pixels)
        
    Returns:
        Motion-blurred image
    """
    # Create horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Apply motion blur
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def simulate_railway_conditions(
    image: np.ndarray,
    gamma_range: Tuple[float, float] = (0.3, 0.5),
    noise_scale_range: Tuple[float, float] = (0.8, 1.2),
    blur_kernel_range: Tuple[int, int] = (10, 20),
    apply_all: bool = True
) -> np.ndarray:
    """
    Apply comprehensive physics-based augmentation to simulate railway conditions.
    
    Combines gamma correction (darkness), Poisson noise (sensor grain),
    and horizontal motion blur to create realistic training conditions.
    
    Args:
        image: Input image as numpy array
        gamma_range: Range for gamma correction values
        noise_scale_range: Range for noise scaling factors
        blur_kernel_range: Range for motion blur kernel sizes
        apply_all: If True, apply all augmentations; if False, randomly select
        
    Returns:
        Augmented image with railway inspection conditions
        
    Requirements:
        - 4.1: Training data matching real-world conditions
        - 6.1: Training data for motion blur correction
    """
    result = image.copy()
    
    if apply_all:
        # Apply all augmentations with random parameters
        
        # 1. Gamma correction for darkness
        gamma = random.uniform(*gamma_range)
        result = apply_gamma_correction(result, gamma)
        
        # 2. Poisson noise for sensor grain
        noise_scale = random.uniform(*noise_scale_range)
        result = apply_poisson_noise(result, noise_scale)
        
        # 3. Horizontal motion blur
        blur_kernel = random.randint(*blur_kernel_range)
        result = apply_horizontal_motion_blur(result, blur_kernel)
        
    else:
        # Randomly select which augmentations to apply
        augmentations = []
        
        if random.random() > 0.3:  # 70% chance
            gamma = random.uniform(*gamma_range)
            augmentations.append(('gamma', gamma))
        
        if random.random() > 0.4:  # 60% chance
            noise_scale = random.uniform(*noise_scale_range)
            augmentations.append(('noise', noise_scale))
        
        if random.random() > 0.5:  # 50% chance
            blur_kernel = random.randint(*blur_kernel_range)
            augmentations.append(('blur', blur_kernel))
        
        # Apply selected augmentations
        for aug_type, param in augmentations:
            if aug_type == 'gamma':
                result = apply_gamma_correction(result, param)
            elif aug_type == 'noise':
                result = apply_poisson_noise(result, param)
            elif aug_type == 'blur':
                result = apply_horizontal_motion_blur(result, param)
    
    return result


def augment_dataset_images(
    input_dir: Path,
    output_dir: Path,
    target_count: int = 2000,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
) -> int:
    """
    Generate augmented images from input dataset to reach target count.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save augmented images
        target_count: Target number of augmented images to generate
        image_extensions: List of valid image file extensions
        
    Returns:
        Number of augmented images actually generated
        
    Requirements:
        - Generate 2000+ augmented images from Roboflow wagon dataset
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files in input directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"**/*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(image_files)} source images")
    logger.info(f"Generating {target_count} augmented images...")
    
    generated_count = 0
    
    for i in range(target_count):
        # Randomly select source image
        source_file = random.choice(image_files)
        
        try:
            # Load image
            image = cv2.imread(str(source_file))
            if image is None:
                logger.warning(f"Could not load image: {source_file}")
                continue
            
            # Apply physics-based augmentation
            augmented = simulate_railway_conditions(image, apply_all=False)
            
            # Generate output filename
            output_filename = f"aug_{i:06d}_{source_file.stem}.jpg"
            output_path = output_dir / output_filename
            
            # Save augmented image
            cv2.imwrite(str(output_path), augmented)
            generated_count += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{target_count} augmented images")
                
        except Exception as e:
            logger.error(f"Error processing {source_file}: {e}")
            continue
    
    logger.info(f"Successfully generated {generated_count} augmented images")
    return generated_count


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test individual augmentation functions
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    print("Testing physics-based augmentations...")
    
    # Test gamma correction
    gamma_corrected = apply_gamma_correction(test_image, gamma=0.4)
    print(f"Gamma correction: {gamma_corrected.shape}, dtype: {gamma_corrected.dtype}")
    
    # Test Poisson noise
    noisy = apply_poisson_noise(test_image, scale=1.0)
    print(f"Poisson noise: {noisy.shape}, dtype: {noisy.dtype}")
    
    # Test motion blur
    blurred = apply_horizontal_motion_blur(test_image, kernel_size=15)
    print(f"Motion blur: {blurred.shape}, dtype: {blurred.dtype}")
    
    # Test combined augmentation
    augmented = simulate_railway_conditions(test_image)
    print(f"Combined augmentation: {augmented.shape}, dtype: {augmented.dtype}")
    
    print("All augmentation tests passed!")
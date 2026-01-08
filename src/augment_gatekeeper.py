#!/usr/bin/env python3
"""
Domain-Correct Gatekeeper Data Preparation

Creates proper training data using WAGON images (not cars) with advanced augmentation.
Addresses the domain mismatch: Car ≠ Train by using actual wagon images.

Strategy:
- Source: Sharp wagons from data/wagon_detection/train/images (Class 0)
- Generate: Corresponding blurry versions with advanced augmentation (Class 1)
- Output: data/gatekeeper_improved/ with proper domain-matched data
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import random
import shutil
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AdvancedAugmentor:
    """
    Advanced augmentation for creating realistic blur conditions.
    Simulates real-world railway inspection challenges.
    """
    
    def __init__(self):
        self.augmentation_types = [
            'motion_blur',
            'gaussian_blur', 
            'iso_noise',
            'darkness',
            'combined'
        ]
    
    def apply_motion_blur(self, image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
        """Apply horizontal motion blur (railway wagon movement)."""
        if intensity == 'light':
            kernel_size = random.randint(5, 10)
        elif intensity == 'medium':
            kernel_size = random.randint(10, 20)
        else:  # heavy
            kernel_size = random.randint(20, 35)
        
        # Create horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def apply_gaussian_blur(self, image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
        """Apply Gaussian blur (focus error)."""
        if intensity == 'light':
            sigma = random.uniform(1.0, 2.5)
        elif intensity == 'medium':
            sigma = random.uniform(2.5, 5.0)
        else:  # heavy
            sigma = random.uniform(5.0, 8.0)
        
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def apply_iso_noise(self, image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
        """Apply ISO noise (night grain)."""
        if intensity == 'light':
            noise_level = random.uniform(10, 25)
        elif intensity == 'medium':
            noise_level = random.uniform(25, 50)
        else:  # heavy
            noise_level = random.uniform(50, 80)
        
        # Generate noise
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        
        # Add noise to image
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def apply_darkness(self, image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
        """Apply darkness (gamma correction for low-light)."""
        if intensity == 'light':
            gamma = random.uniform(0.6, 0.8)
        elif intensity == 'medium':
            gamma = random.uniform(0.4, 0.6)
        else:  # heavy
            gamma = random.uniform(0.2, 0.4)
        
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        dark = cv2.LUT(image, table)
        return dark
    
    def apply_combined_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply combination of augmentations for realistic conditions."""
        result = image.copy()
        
        # Randomly select 2-3 augmentations
        num_augs = random.randint(2, 3)
        selected_augs = random.sample(['motion_blur', 'gaussian_blur', 'iso_noise', 'darkness'], num_augs)
        
        for aug_type in selected_augs:
            intensity = random.choice(['light', 'medium', 'heavy'])
            
            if aug_type == 'motion_blur':
                result = self.apply_motion_blur(result, intensity)
            elif aug_type == 'gaussian_blur':
                result = self.apply_gaussian_blur(result, intensity)
            elif aug_type == 'iso_noise':
                result = self.apply_iso_noise(result, intensity)
            elif aug_type == 'darkness':
                result = self.apply_darkness(result, intensity)
        
        return result
    
    def augment_image(self, image: np.ndarray, aug_type: Optional[str] = None) -> np.ndarray:
        """
        Apply augmentation to create blurry version.
        
        Args:
            image: Input sharp image
            aug_type: Type of augmentation or None for random
            
        Returns:
            Augmented (blurry) image
        """
        if aug_type is None:
            aug_type = random.choice(self.augmentation_types)
        
        intensity = random.choice(['light', 'medium', 'heavy'])
        
        if aug_type == 'motion_blur':
            return self.apply_motion_blur(image, intensity)
        elif aug_type == 'gaussian_blur':
            return self.apply_gaussian_blur(image, intensity)
        elif aug_type == 'iso_noise':
            return self.apply_iso_noise(image, intensity)
        elif aug_type == 'darkness':
            return self.apply_darkness(image, intensity)
        elif aug_type == 'combined':
            return self.apply_combined_augmentation(image)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")


def load_sharp_wagon_images(wagon_dir: Path) -> List[Path]:
    """Load all sharp wagon images from Roboflow dataset."""
    train_images_dir = wagon_dir / "train" / "images"
    
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Wagon train images not found: {train_images_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(train_images_dir.glob(f"*{ext}"))
        image_files.extend(train_images_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} sharp wagon images")
    return image_files


def create_gatekeeper_dataset(
    wagon_dir: Path,
    output_dir: Path,
    augmentation_factor: int = 2,
    train_ratio: float = 0.8
) -> None:
    """
    Create improved gatekeeper dataset with domain-correct data.
    
    Args:
        wagon_dir: Path to wagon detection dataset
        output_dir: Output directory for improved dataset
        augmentation_factor: Number of blurry versions per sharp image
        train_ratio: Ratio of data for training vs validation
    """
    logger.info("Creating domain-correct gatekeeper dataset...")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / "sharp").mkdir(parents=True, exist_ok=True)
        (split_dir / "blurry").mkdir(parents=True, exist_ok=True)
    
    # Load sharp wagon images
    sharp_images = load_sharp_wagon_images(wagon_dir)
    
    if len(sharp_images) == 0:
        raise ValueError("No sharp wagon images found")
    
    # Shuffle and split
    random.shuffle(sharp_images)
    train_size = int(len(sharp_images) * train_ratio)
    train_images = sharp_images[:train_size]
    val_images = sharp_images[train_size:]
    
    logger.info(f"Split: {len(train_images)} train, {len(val_images)} validation")
    
    # Initialize augmentor
    augmentor = AdvancedAugmentor()
    
    # Process training images
    logger.info("Processing training images...")
    process_image_split(train_images, train_dir, augmentor, augmentation_factor, "train")
    
    # Process validation images
    logger.info("Processing validation images...")
    process_image_split(val_images, val_dir, augmentor, augmentation_factor, "val")
    
    # Create dataset summary
    create_dataset_summary(output_dir)
    
    logger.info(f"Domain-correct gatekeeper dataset created at: {output_dir}")


def process_image_split(
    image_paths: List[Path],
    split_dir: Path,
    augmentor: AdvancedAugmentor,
    augmentation_factor: int,
    split_name: str
) -> None:
    """Process images for a specific split (train/val)."""
    
    sharp_dir = split_dir / "sharp"
    blurry_dir = split_dir / "blurry"
    
    for i, img_path in enumerate(tqdm(image_paths, desc=f"Processing {split_name}")):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue
            
            # Save sharp version (Class 0)
            sharp_filename = f"sharp_{i:04d}_{img_path.stem}.jpg"
            sharp_output_path = sharp_dir / sharp_filename
            cv2.imwrite(str(sharp_output_path), image)
            
            # Generate blurry versions (Class 1)
            for aug_idx in range(augmentation_factor):
                # Apply different augmentation types
                aug_types = ['motion_blur', 'gaussian_blur', 'iso_noise', 'darkness', 'combined']
                aug_type = aug_types[aug_idx % len(aug_types)]
                
                blurry_image = augmentor.augment_image(image, aug_type)
                
                blurry_filename = f"blurry_{i:04d}_{aug_idx}_{aug_type}_{img_path.stem}.jpg"
                blurry_output_path = blurry_dir / blurry_filename
                cv2.imwrite(str(blurry_output_path), blurry_image)
        
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            continue


def create_dataset_summary(output_dir: Path) -> None:
    """Create summary of the generated dataset."""
    
    summary = {
        "dataset_type": "domain_correct_gatekeeper",
        "source": "wagon_detection_roboflow",
        "classes": {
            "0": "sharp_wagons",
            "1": "blurry_wagons"
        },
        "augmentation_types": [
            "motion_blur",
            "gaussian_blur", 
            "iso_noise",
            "darkness",
            "combined"
        ]
    }
    
    # Count files
    for split in ["train", "val"]:
        split_dir = output_dir / split
        if split_dir.exists():
            sharp_count = len(list((split_dir / "sharp").glob("*.jpg")))
            blurry_count = len(list((split_dir / "blurry").glob("*.jpg")))
            
            summary[f"{split}_sharp"] = sharp_count
            summary[f"{split}_blurry"] = blurry_count
            summary[f"{split}_total"] = sharp_count + blurry_count
    
    # Save summary
    import json
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Dataset Summary:")
    for key, value in summary.items():
        if isinstance(value, (int, str)):
            logger.info(f"  {key}: {value}")


def main():
    """Main function to create domain-correct gatekeeper dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create domain-correct gatekeeper dataset")
    parser.add_argument("--wagon-dir", type=str, default="data/wagon_detection",
                       help="Path to wagon detection dataset")
    parser.add_argument("--output-dir", type=str, default="data/gatekeeper_improved",
                       help="Output directory for improved dataset")
    parser.add_argument("--augmentation-factor", type=int, default=3,
                       help="Number of blurry versions per sharp image")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data for training")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("DOMAIN-CORRECT GATEKEEPER DATA PREPARATION")
    logger.info("=" * 60)
    
    try:
        wagon_dir = Path(args.wagon_dir)
        output_dir = Path(args.output_dir)
        
        # Verify wagon dataset exists
        if not wagon_dir.exists():
            logger.error(f"Wagon dataset not found: {wagon_dir}")
            return
        
        # Create improved dataset
        create_gatekeeper_dataset(
            wagon_dir=wagon_dir,
            output_dir=output_dir,
            augmentation_factor=args.augmentation_factor,
            train_ratio=args.train_ratio
        )
        
        logger.info("✅ Domain-correct gatekeeper dataset created successfully!")
        logger.info(f"📁 Location: {output_dir}")
        logger.info("➡️  Next: Run train_gatekeeper_v2.py with improved data")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
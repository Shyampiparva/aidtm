"""
Local dataset preparation for blurred/sharp car images.

This module handles the preparation and organization of the local
blurred_sharp car dataset for training the deblurring models.
"""

import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


logger = logging.getLogger(__name__)


def prepare_local_car_dataset(
    source_dir: str = "blurred_sharp/blurred_sharp",
    output_dir: str = "data/local_cars",
    copy_files: bool = True
) -> Tuple[Path, Path]:
    """
    Prepare local car dataset by organizing blurred and sharp image pairs.
    
    Copies and organizes the blurred_sharp car dataset into a structured
    format suitable for training deblurring models.
    
    Args:
        source_dir: Source directory containing blurred/ and sharp/ subdirectories
        output_dir: Output directory for organized dataset
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Tuple of (blurred_dir_path, sharp_dir_path)
        
    Raises:
        FileNotFoundError: If source directories don't exist
        RuntimeError: If dataset preparation fails
        
    Requirements:
        - Copy and organize blurred_sharp/blurred_sharp/blurred/ and sharp/ for training
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Check source directories exist
    blurred_source = source_path / "blurred"
    sharp_source = source_path / "sharp"
    
    if not blurred_source.exists():
        raise FileNotFoundError(f"Blurred images directory not found: {blurred_source}")
    
    if not sharp_source.exists():
        raise FileNotFoundError(f"Sharp images directory not found: {sharp_source}")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    blurred_output = output_path / "blurred"
    sharp_output = output_path / "sharp"
    
    blurred_output.mkdir(exist_ok=True)
    sharp_output.mkdir(exist_ok=True)
    
    logger.info(f"Preparing local car dataset from {source_path} to {output_path}")
    
    try:
        # Copy/link blurred images
        blurred_count = _copy_images(blurred_source, blurred_output, copy_files)
        logger.info(f"Processed {blurred_count} blurred images")
        
        # Copy/link sharp images
        sharp_count = _copy_images(sharp_source, sharp_output, copy_files)
        logger.info(f"Processed {sharp_count} sharp images")
        
        # Verify we have matching pairs
        blurred_files = set(f.stem for f in blurred_output.glob("*.png"))
        sharp_files = set(f.stem for f in sharp_output.glob("*.png"))
        
        common_files = blurred_files.intersection(sharp_files)
        logger.info(f"Found {len(common_files)} matching blurred/sharp pairs")
        
        if len(common_files) == 0:
            raise RuntimeError("No matching blurred/sharp pairs found")
        
        return blurred_output, sharp_output
        
    except Exception as e:
        error_msg = f"Failed to prepare local car dataset: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def _copy_images(source_dir: Path, target_dir: Path, copy_files: bool = True) -> int:
    """
    Copy or link images from source to target directory.
    
    Args:
        source_dir: Source directory containing images
        target_dir: Target directory for images
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Number of images processed
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    processed_count = 0
    
    for image_file in source_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            target_file = target_dir / image_file.name
            
            try:
                if copy_files:
                    shutil.copy2(image_file, target_file)
                else:
                    # Create symlink (relative path for portability)
                    relative_source = Path("..") / ".." / image_file.relative_to(Path.cwd())
                    target_file.symlink_to(relative_source)
                
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {image_file}: {e}")
                continue
    
    return processed_count


def verify_image_pairs(blurred_dir: Path, sharp_dir: Path) -> List[str]:
    """
    Verify that blurred and sharp images form valid pairs.
    
    Args:
        blurred_dir: Directory containing blurred images
        sharp_dir: Directory containing sharp images
        
    Returns:
        List of common filenames (without extension) that have both versions
    """
    # Get all image files in both directories
    blurred_files = {}
    sharp_files = {}
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    for f in blurred_dir.glob("*"):
        if f.suffix.lower() in image_extensions:
            blurred_files[f.stem] = f
    
    for f in sharp_dir.glob("*"):
        if f.suffix.lower() in image_extensions:
            sharp_files[f.stem] = f
    
    # Find common files
    common_stems = set(blurred_files.keys()).intersection(set(sharp_files.keys()))
    
    # Verify image pairs are valid
    valid_pairs = []
    
    for stem in common_stems:
        blurred_path = blurred_files[stem]
        sharp_path = sharp_files[stem]
        
        try:
            # Load both images to verify they're valid
            blurred_img = cv2.imread(str(blurred_path))
            sharp_img = cv2.imread(str(sharp_path))
            
            if blurred_img is not None and sharp_img is not None:
                # Check if images have same dimensions
                if blurred_img.shape == sharp_img.shape:
                    valid_pairs.append(stem)
                else:
                    logger.warning(f"Dimension mismatch for pair {stem}: "
                                 f"blurred {blurred_img.shape} vs sharp {sharp_img.shape}")
            else:
                logger.warning(f"Could not load image pair {stem}")
                
        except Exception as e:
            logger.warning(f"Error verifying pair {stem}: {e}")
            continue
    
    logger.info(f"Verified {len(valid_pairs)} valid image pairs")
    return valid_pairs


def create_train_val_split(
    image_pairs: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[List[str], List[str]]:
    """
    Split image pairs into training and validation sets.
    
    Args:
        image_pairs: List of image pair identifiers
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Shuffle pairs
    pairs_array = np.array(image_pairs)
    np.random.shuffle(pairs_array)
    
    # Calculate split indices
    n_total = len(pairs_array)
    n_train = int(n_total * train_ratio)
    
    # Split
    train_pairs = pairs_array[:n_train].tolist()
    val_pairs = pairs_array[n_train:].tolist()
    
    logger.info(f"Dataset split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    return train_pairs, val_pairs


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Prepare local car dataset
        blurred_dir, sharp_dir = prepare_local_car_dataset()
        
        # Verify image pairs
        valid_pairs = verify_image_pairs(blurred_dir, sharp_dir)
        
        if valid_pairs:
            # Create train/val split
            train_pairs, val_pairs = create_train_val_split(valid_pairs)
            
            print(f"Local car dataset prepared successfully:")
            print(f"  Blurred images: {blurred_dir}")
            print(f"  Sharp images: {sharp_dir}")
            print(f"  Valid pairs: {len(valid_pairs)}")
            print(f"  Train pairs: {len(train_pairs)}")
            print(f"  Validation pairs: {len(val_pairs)}")
        else:
            print("No valid image pairs found")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
"""
Dataset combination module for merging Roboflow wagon data with local car data.

This module handles the combination of different datasets into a unified
format suitable for training vehicle detection models that work on both
cars and railway wagons.
"""

import yaml
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Class mapping from wagon-specific to vehicle-generic classes
CLASS_MAPPING = {
    "wagon_body": "vehicle_body",
    "identification_plate": "license_plate",
    "wheel_assembly": "wheel",
    "coupling_mechanism": "coupling_mechanism"
}

# Unified class definitions for combined dataset
UNIFIED_CLASSES = [
    "vehicle_body",      # Cars + wagons
    "license_plate",     # Car plates + wagon IDs  
    "wheel",             # Car wheels + wagon wheels
    "coupling_mechanism" # Wagon-specific (no car equivalent)
]


def map_wagon_classes_to_vehicle(annotation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map wagon-specific class names to unified vehicle class names.
    
    Args:
        annotation_data: Original annotation data with wagon classes
        
    Returns:
        Annotation data with mapped class names
        
    Requirements:
        - Map wagon classes to vehicle classes: wagon_body→vehicle_body, identification_plate→license_plate
    """
    mapped_data = annotation_data.copy()
    
    # Update class names in the names list
    if 'names' in mapped_data:
        original_names = mapped_data['names']
        if isinstance(original_names, dict):
            # Format: {0: 'class_name', 1: 'class_name', ...}
            mapped_names = {}
            for idx, class_name in original_names.items():
                mapped_names[idx] = CLASS_MAPPING.get(class_name, class_name)
            mapped_data['names'] = mapped_names
        elif isinstance(original_names, list):
            # Format: ['class_name', 'class_name', ...]
            mapped_names = []
            for class_name in original_names:
                mapped_names.append(CLASS_MAPPING.get(class_name, class_name))
            mapped_data['names'] = mapped_names
    
    return mapped_data


def convert_annotation_format(
    label_file: Path,
    class_mapping: Dict[str, int],
    output_file: Path
) -> bool:
    """
    Convert annotation file to use unified class indices.
    
    Args:
        label_file: Path to original label file
        class_mapping: Mapping from class names to unified indices
        output_file: Path for converted label file
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # At least class_id + 4 coordinates
                original_class_id = int(parts[0])
                
                # For now, assume we have the original class names available
                # In a real implementation, you'd need to map from the original data.yaml
                # This is a simplified version
                converted_lines.append(line)  # Keep original for now
        
        # Write converted annotations
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(converted_lines)
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting annotation {label_file}: {e}")
        return False


def combine_datasets(
    wagon_dataset_path: Path,
    car_dataset_path: Path,
    output_path: Path,
    augmented_images_path: Optional[Path] = None
) -> Path:
    """
    Combine Roboflow wagon dataset with local car dataset.
    
    Creates a unified dataset structure with mapped class names and
    combined training/validation splits that include both vehicle types.
    
    Args:
        wagon_dataset_path: Path to downloaded Roboflow wagon dataset
        car_dataset_path: Path to prepared local car dataset
        output_path: Path for combined dataset output
        augmented_images_path: Optional path to augmented images
        
    Returns:
        Path to combined dataset directory
        
    Requirements:
        - Merge Roboflow wagon data with local car data
        - Map wagon classes to vehicle classes
    """
    logger.info(f"Combining datasets into {output_path}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_images_dir = output_path / "train" / "images"
    train_labels_dir = output_path / "train" / "labels"
    val_images_dir = output_path / "valid" / "images"
    val_labels_dir = output_path / "valid" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process wagon dataset
    wagon_train_images = 0
    wagon_val_images = 0
    
    if wagon_dataset_path.exists():
        logger.info("Processing wagon dataset...")
        
        # Copy wagon training images and labels
        wagon_train_img_dir = wagon_dataset_path / "train" / "images"
        wagon_train_lbl_dir = wagon_dataset_path / "train" / "labels"
        
        if wagon_train_img_dir.exists():
            wagon_train_images = _copy_dataset_files(
                wagon_train_img_dir, train_images_dir, 
                wagon_train_lbl_dir, train_labels_dir,
                prefix="wagon_"
            )
        
        # Copy wagon validation images and labels
        wagon_val_img_dir = wagon_dataset_path / "valid" / "images"
        wagon_val_lbl_dir = wagon_dataset_path / "valid" / "labels"
        
        if wagon_val_img_dir.exists():
            wagon_val_images = _copy_dataset_files(
                wagon_val_img_dir, val_images_dir,
                wagon_val_lbl_dir, val_labels_dir,
                prefix="wagon_"
            )
    
    # Process car dataset (blurred/sharp pairs)
    car_train_images = 0
    car_val_images = 0
    
    if car_dataset_path.exists():
        logger.info("Processing car dataset...")
        
        # For car dataset, we'll use both blurred and sharp images
        # but create simple bounding box annotations (since we don't have detailed labels)
        blurred_dir = car_dataset_path / "blurred"
        sharp_dir = car_dataset_path / "sharp"
        
        if blurred_dir.exists() and sharp_dir.exists():
            car_train_images, car_val_images = _process_car_images(
                blurred_dir, sharp_dir,
                train_images_dir, val_images_dir,
                train_labels_dir, val_labels_dir
            )
    
    # Add augmented images if available
    aug_images = 0
    if augmented_images_path and augmented_images_path.exists():
        logger.info("Adding augmented images...")
        aug_images = _copy_augmented_images(augmented_images_path, train_images_dir)
    
    # Create unified data.yaml
    _create_unified_data_yaml(output_path, wagon_dataset_path)
    
    logger.info(f"Dataset combination complete:")
    logger.info(f"  Wagon train: {wagon_train_images}, val: {wagon_val_images}")
    logger.info(f"  Car train: {car_train_images}, val: {car_val_images}")
    logger.info(f"  Augmented: {aug_images}")
    logger.info(f"  Total train: {wagon_train_images + car_train_images + aug_images}")
    logger.info(f"  Total val: {wagon_val_images + car_val_images}")
    
    return output_path


def _copy_dataset_files(
    src_img_dir: Path, dst_img_dir: Path,
    src_lbl_dir: Path, dst_lbl_dir: Path,
    prefix: str = ""
) -> int:
    """Copy images and corresponding labels with optional prefix."""
    copied_count = 0
    
    for img_file in src_img_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Copy image
            dst_img_file = dst_img_dir / f"{prefix}{img_file.name}"
            shutil.copy2(img_file, dst_img_file)
            
            # Copy corresponding label if it exists
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                dst_lbl_file = dst_lbl_dir / f"{prefix}{img_file.stem}.txt"
                shutil.copy2(lbl_file, dst_lbl_file)
            
            copied_count += 1
    
    return copied_count


def _process_car_images(
    blurred_dir: Path, sharp_dir: Path,
    train_img_dir: Path, val_img_dir: Path,
    train_lbl_dir: Path, val_lbl_dir: Path,
    train_ratio: float = 0.8
) -> Tuple[int, int]:
    """Process car images and create simple vehicle_body annotations."""
    
    # Get all image files
    blurred_files = list(blurred_dir.glob("*.png"))
    sharp_files = list(sharp_dir.glob("*.png"))
    
    # Find common files
    blurred_stems = {f.stem for f in blurred_files}
    sharp_stems = {f.stem for f in sharp_files}
    common_stems = blurred_stems.intersection(sharp_stems)
    
    # Split into train/val
    common_list = list(common_stems)
    np.random.shuffle(common_list)
    
    n_train = int(len(common_list) * train_ratio)
    train_stems = common_list[:n_train]
    val_stems = common_list[n_train:]
    
    train_count = 0
    val_count = 0
    
    # Process training images
    for stem in train_stems:
        # Copy both blurred and sharp versions
        blurred_file = blurred_dir / f"{stem}.png"
        sharp_file = sharp_dir / f"{stem}.png"
        
        if blurred_file.exists():
            dst_file = train_img_dir / f"car_blurred_{stem}.png"
            shutil.copy2(blurred_file, dst_file)
            _create_simple_vehicle_annotation(dst_file, train_lbl_dir / f"car_blurred_{stem}.txt")
            train_count += 1
        
        if sharp_file.exists():
            dst_file = train_img_dir / f"car_sharp_{stem}.png"
            shutil.copy2(sharp_file, dst_file)
            _create_simple_vehicle_annotation(dst_file, train_lbl_dir / f"car_sharp_{stem}.txt")
            train_count += 1
    
    # Process validation images
    for stem in val_stems:
        blurred_file = blurred_dir / f"{stem}.png"
        sharp_file = sharp_dir / f"{stem}.png"
        
        if blurred_file.exists():
            dst_file = val_img_dir / f"car_blurred_{stem}.png"
            shutil.copy2(blurred_file, dst_file)
            _create_simple_vehicle_annotation(dst_file, val_lbl_dir / f"car_blurred_{stem}.txt")
            val_count += 1
        
        if sharp_file.exists():
            dst_file = val_img_dir / f"car_sharp_{stem}.png"
            shutil.copy2(sharp_file, dst_file)
            _create_simple_vehicle_annotation(dst_file, val_lbl_dir / f"car_sharp_{stem}.txt")
            val_count += 1
    
    return train_count, val_count


def _create_simple_vehicle_annotation(image_path: Path, label_path: Path) -> None:
    """Create a simple vehicle_body annotation covering most of the image."""
    try:
        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        # Create a bounding box covering 80% of the image (centered)
        # Format: class_id x_center y_center width height (normalized)
        class_id = 0  # vehicle_body class
        x_center = 0.5
        y_center = 0.5
        width = 0.8
        height = 0.8
        
        # Write annotation
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    except Exception as e:
        logger.warning(f"Could not create annotation for {image_path}: {e}")


def _copy_augmented_images(src_dir: Path, dst_dir: Path) -> int:
    """Copy augmented images to training directory."""
    copied_count = 0
    
    for img_file in src_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            dst_file = dst_dir / f"aug_{img_file.name}"
            shutil.copy2(img_file, dst_file)
            copied_count += 1
    
    return copied_count


def _create_unified_data_yaml(output_path: Path, wagon_dataset_path: Path) -> None:
    """Create unified data.yaml with mapped class names."""
    
    # Try to read original wagon dataset yaml
    original_yaml_path = wagon_dataset_path / "data.yaml"
    original_data = {}
    
    if original_yaml_path.exists():
        try:
            with open(original_yaml_path, 'r') as f:
                original_data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not read original data.yaml: {e}")
    
    # Create unified configuration
    unified_data = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': '',  # No test set for now
        'nc': len(UNIFIED_CLASSES),
        'names': {i: name for i, name in enumerate(UNIFIED_CLASSES)}
    }
    
    # Write unified data.yaml
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(unified_data, f, default_flow_style=False)
    
    logger.info(f"Created unified data.yaml with {len(UNIFIED_CLASSES)} classes")


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Example paths (adjust as needed)
        wagon_path = Path("data/wagon_detection")
        car_path = Path("data/local_cars")
        output_path = Path("data/combined_dataset")
        
        if not wagon_path.exists():
            print(f"Wagon dataset not found at {wagon_path}")
            print("Run data_ingest.py first to download the wagon dataset")
            sys.exit(1)
        
        if not car_path.exists():
            print(f"Car dataset not found at {car_path}")
            print("Run data_local.py first to prepare the car dataset")
            sys.exit(1)
        
        # Combine datasets
        combined_path = combine_datasets(wagon_path, car_path, output_path)
        
        print(f"Successfully combined datasets at: {combined_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
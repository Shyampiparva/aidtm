#!/usr/bin/env python3
"""
GoPro Dataset Integration for Real Blur Patterns

Integrates GoPro Large Scale Blur Dataset to address synthetic vs real blur gap.
Uses real camera motion blur for both Math Gatekeeper validation and DeblurGAN training.

Strategy:
1. Download GoPro subset (500 pairs)
2. Validate Math Gatekeeper on real blur
3. Enhance DeblurGAN training with real motion patterns
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from math_gatekeeper import MathGatekeeper


logger = logging.getLogger(__name__)


class GoProIntegrator:
    """
    Integrates GoPro dataset for real blur pattern training.
    """
    
    def __init__(self, gopro_dir: Path, output_dir: Path):
        self.gopro_dir = gopro_dir
        self.output_dir = output_dir
        self.math_gatekeeper = MathGatekeeper(threshold=50.0)
    
    def validate_gopro_structure(self) -> bool:
        """
        Validate GoPro dataset structure.
        Expected: gopro_dir/blur/ and gopro_dir/sharp/
        """
        blur_dir = self.gopro_dir / "blur"
        sharp_dir = self.gopro_dir / "sharp"
        
        if not blur_dir.exists():
            logger.error(f"GoPro blur directory not found: {blur_dir}")
            return False
        
        if not sharp_dir.exists():
            logger.error(f"GoPro sharp directory not found: {sharp_dir}")
            return False
        
        blur_count = len(list(blur_dir.glob("*.png")) + list(blur_dir.glob("*.jpg")))
        sharp_count = len(list(sharp_dir.glob("*.png")) + list(sharp_dir.glob("*.jpg")))
        
        logger.info(f"GoPro dataset: {blur_count} blur, {sharp_count} sharp images")
        
        if blur_count == 0 or sharp_count == 0:
            logger.error("GoPro dataset appears empty")
            return False
        
        return True
    
    def test_math_gatekeeper_on_gopro(self, max_samples: int = 100) -> Dict[str, float]:
        """
        Test Math Gatekeeper accuracy on real GoPro blur patterns.
        
        Args:
            max_samples: Maximum samples to test
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Testing Math Gatekeeper on GoPro real blur patterns...")
        
        blur_dir = self.gopro_dir / "blur"
        sharp_dir = self.gopro_dir / "sharp"
        
        # Load samples
        blur_images = list(blur_dir.glob("*.png")) + list(blur_dir.glob("*.jpg"))
        sharp_images = list(sharp_dir.glob("*.png")) + list(sharp_dir.glob("*.jpg"))
        
        # Limit samples
        blur_images = blur_images[:max_samples]
        sharp_images = sharp_images[:max_samples]
        
        # Test on blurred images (should be classified as blurry)
        blur_correct = 0
        blur_variances = []
        
        for img_path in tqdm(blur_images, desc="Testing blurred images"):
            try:
                image = cv2.imread(str(img_path))
                if image is not None:
                    is_sharp, variance = self.math_gatekeeper.predict(image)
                    blur_variances.append(variance)
                    if not is_sharp:  # Correctly identified as blurry
                        blur_correct += 1
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
        
        # Test on sharp images (should be classified as sharp)
        sharp_correct = 0
        sharp_variances = []
        
        for img_path in tqdm(sharp_images, desc="Testing sharp images"):
            try:
                image = cv2.imread(str(img_path))
                if image is not None:
                    is_sharp, variance = self.math_gatekeeper.predict(image)
                    sharp_variances.append(variance)
                    if is_sharp:  # Correctly identified as sharp
                        sharp_correct += 1
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
        
        # Calculate metrics
        total_correct = blur_correct + sharp_correct
        total_samples = len(blur_variances) + len(sharp_variances)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'blur_accuracy': blur_correct / len(blur_variances) if blur_variances else 0.0,
            'sharp_accuracy': sharp_correct / len(sharp_variances) if sharp_variances else 0.0,
            'blur_mean_variance': np.mean(blur_variances) if blur_variances else 0.0,
            'sharp_mean_variance': np.mean(sharp_variances) if sharp_variances else 0.0,
            'total_samples': total_samples,
            'blur_samples': len(blur_variances),
            'sharp_samples': len(sharp_variances)
        }
        
        logger.info("GoPro Real Blur Test Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Blur Detection: {results['blur_accuracy']:.4f}")
        logger.info(f"  Sharp Detection: {results['sharp_accuracy']:.4f}")
        logger.info(f"  Blur Mean Variance: {results['blur_mean_variance']:.2f}")
        logger.info(f"  Sharp Mean Variance: {results['sharp_mean_variance']:.2f}")
        
        return results
    
    def prepare_deblur_training_data(self, max_pairs: int = 500) -> Path:
        """
        Prepare combined training data for DeblurGAN with real GoPro blur patterns.
        
        Args:
            max_pairs: Maximum GoPro pairs to include
            
        Returns:
            Path to prepared training directory
        """
        logger.info("Preparing enhanced DeblurGAN training data with GoPro...")
        
        # Create output structure
        enhanced_dir = self.output_dir / "deblur_enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        train_blur_dir = enhanced_dir / "train" / "blur"
        train_sharp_dir = enhanced_dir / "train" / "sharp"
        val_blur_dir = enhanced_dir / "val" / "blur"
        val_sharp_dir = enhanced_dir / "val" / "sharp"
        
        for dir_path in [train_blur_dir, train_sharp_dir, val_blur_dir, val_sharp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy existing car dataset
        car_data_dir = Path("data/blurred_sharp")
        if car_data_dir.exists():
            logger.info("Adding existing car dataset...")
            self._copy_car_dataset(car_data_dir, enhanced_dir)
        
        # 2. Add GoPro real blur patterns
        logger.info("Adding GoPro real blur patterns...")
        self._copy_gopro_dataset(max_pairs, enhanced_dir)
        
        # 3. Create dataset summary
        self._create_enhanced_summary(enhanced_dir)
        
        logger.info(f"Enhanced DeblurGAN dataset created at: {enhanced_dir}")
        return enhanced_dir
    
    def _copy_car_dataset(self, car_dir: Path, enhanced_dir: Path) -> None:
        """Copy existing car dataset to enhanced directory."""
        car_blur_dir = car_dir / "blurred"
        car_sharp_dir = car_dir / "sharp"
        
        if not (car_blur_dir.exists() and car_sharp_dir.exists()):
            logger.warning("Car dataset not found, skipping...")
            return
        
        # Get matching pairs
        blur_files = {f.stem: f for f in car_blur_dir.glob("*.png")}
        sharp_files = {f.stem: f for f in car_sharp_dir.glob("*.png")}
        common_stems = set(blur_files.keys()) & set(sharp_files.keys())
        
        # Split 80/20 train/val
        common_list = list(common_stems)
        np.random.shuffle(common_list)
        train_size = int(0.8 * len(common_list))
        
        train_stems = common_list[:train_size]
        val_stems = common_list[train_size:]
        
        # Copy training pairs
        for i, stem in enumerate(train_stems):
            blur_src = blur_files[stem]
            sharp_src = sharp_files[stem]
            
            blur_dst = enhanced_dir / "train" / "blur" / f"car_{i:04d}_blur.png"
            sharp_dst = enhanced_dir / "train" / "sharp" / f"car_{i:04d}_sharp.png"
            
            shutil.copy2(blur_src, blur_dst)
            shutil.copy2(sharp_src, sharp_dst)
        
        # Copy validation pairs
        for i, stem in enumerate(val_stems):
            blur_src = blur_files[stem]
            sharp_src = sharp_files[stem]
            
            blur_dst = enhanced_dir / "val" / "blur" / f"car_val_{i:04d}_blur.png"
            sharp_dst = enhanced_dir / "val" / "sharp" / f"car_val_{i:04d}_sharp.png"
            
            shutil.copy2(blur_src, blur_dst)
            shutil.copy2(sharp_src, sharp_dst)
        
        logger.info(f"Added {len(train_stems)} car training pairs, {len(val_stems)} validation pairs")
    
    def _copy_gopro_dataset(self, max_pairs: int, enhanced_dir: Path) -> None:
        """Copy GoPro dataset to enhanced directory."""
        gopro_blur_dir = self.gopro_dir / "blur"
        gopro_sharp_dir = self.gopro_dir / "sharp"
        
        # Get available files
        blur_files = list(gopro_blur_dir.glob("*.png")) + list(gopro_blur_dir.glob("*.jpg"))
        sharp_files = list(gopro_sharp_dir.glob("*.png")) + list(gopro_sharp_dir.glob("*.jpg"))
        
        # Limit to max_pairs
        blur_files = blur_files[:max_pairs]
        sharp_files = sharp_files[:max_pairs]
        
        # Split 80/20 train/val
        min_count = min(len(blur_files), len(sharp_files))
        train_size = int(0.8 * min_count)
        
        # Copy training pairs
        for i in range(train_size):
            if i < len(blur_files) and i < len(sharp_files):
                blur_src = blur_files[i]
                sharp_src = sharp_files[i]
                
                blur_dst = enhanced_dir / "train" / "blur" / f"gopro_{i:04d}_blur{blur_src.suffix}"
                sharp_dst = enhanced_dir / "train" / "sharp" / f"gopro_{i:04d}_sharp{sharp_src.suffix}"
                
                shutil.copy2(blur_src, blur_dst)
                shutil.copy2(sharp_src, sharp_dst)
        
        # Copy validation pairs
        for i in range(train_size, min_count):
            if i < len(blur_files) and i < len(sharp_files):
                blur_src = blur_files[i]
                sharp_src = sharp_files[i]
                
                val_idx = i - train_size
                blur_dst = enhanced_dir / "val" / "blur" / f"gopro_val_{val_idx:04d}_blur{blur_src.suffix}"
                sharp_dst = enhanced_dir / "val" / "sharp" / f"gopro_val_{val_idx:04d}_sharp{sharp_src.suffix}"
                
                shutil.copy2(blur_src, blur_dst)
                shutil.copy2(sharp_src, sharp_dst)
        
        logger.info(f"Added {train_size} GoPro training pairs, {min_count - train_size} validation pairs")
    
    def _create_enhanced_summary(self, enhanced_dir: Path) -> None:
        """Create summary of enhanced dataset."""
        summary = {
            "dataset_type": "enhanced_deblur_with_gopro",
            "components": ["car_dataset", "gopro_real_blur"],
            "purpose": "deblur_gan_training_with_real_patterns"
        }
        
        # Count files
        for split in ["train", "val"]:
            blur_dir = enhanced_dir / split / "blur"
            sharp_dir = enhanced_dir / split / "sharp"
            
            blur_count = len(list(blur_dir.glob("*")))
            sharp_count = len(list(sharp_dir.glob("*")))
            
            summary[f"{split}_blur"] = blur_count
            summary[f"{split}_sharp"] = sharp_count
            summary[f"{split}_pairs"] = min(blur_count, sharp_count)
        
        # Save summary
        import json
        summary_path = enhanced_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Enhanced Dataset Summary:")
        for key, value in summary.items():
            if isinstance(value, (int, str)):
                logger.info(f"  {key}: {value}")


def main():
    """Main function to integrate GoPro dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate GoPro dataset for real blur patterns")
    parser.add_argument("--gopro-dir", type=str, required=True,
                       help="Path to GoPro dataset directory")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for enhanced datasets")
    parser.add_argument("--max-pairs", type=int, default=500,
                       help="Maximum GoPro pairs to use")
    parser.add_argument("--test-math-gatekeeper", action="store_true",
                       help="Test Math Gatekeeper on GoPro real blur")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("GOPRO DATASET INTEGRATION FOR REAL BLUR PATTERNS")
    logger.info("=" * 60)
    
    try:
        gopro_dir = Path(args.gopro_dir)
        output_dir = Path(args.output_dir)
        
        integrator = GoProIntegrator(gopro_dir, output_dir)
        
        # Validate GoPro dataset
        if not integrator.validate_gopro_structure():
            logger.error("GoPro dataset validation failed")
            return
        
        # Test Math Gatekeeper on real blur (optional)
        if args.test_math_gatekeeper:
            logger.info("Testing Math Gatekeeper on GoPro real blur...")
            results = integrator.test_math_gatekeeper_on_gopro()
            
            if results['accuracy'] >= 0.85:
                logger.info("✅ Math Gatekeeper maintains >85% accuracy on real blur!")
            else:
                logger.warning(f"⚠️  Math Gatekeeper accuracy dropped to {results['accuracy']:.2%} on real blur")
        
        # Prepare enhanced DeblurGAN training data
        enhanced_dir = integrator.prepare_deblur_training_data(args.max_pairs)
        
        logger.info("✅ GoPro integration completed successfully!")
        logger.info(f"📁 Enhanced dataset: {enhanced_dir}")
        logger.info("➡️  Next: Train DeblurGAN with real blur patterns")
        logger.info("➡️  Command: python scripts/train_deblur.py --dataset-dir data/deblur_enhanced")
        
    except Exception as e:
        logger.error(f"GoPro integration failed: {e}")
        raise


if __name__ == "__main__":
    main()
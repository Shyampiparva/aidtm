#!/usr/bin/env python3
"""
Math Gatekeeper: Laplacian Variance Baseline Test

Tests if simple mathematical blur detection can outperform the 68% neural network.
If Laplacian variance achieves >85% accuracy, we ABANDON the neural approach.

Logic:
- Sharp images have high Laplacian variance (edges are crisp)
- Blurry images have low Laplacian variance (edges are smoothed)
- Find optimal threshold that separates sharp wagons from blurry cars
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def calculate_laplacian_variance(image_path: Path) -> float:
    """
    Calculate Laplacian variance for blur detection.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Laplacian variance (higher = sharper, lower = blurrier)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def load_sharp_wagon_samples(wagon_dir: Path, max_samples: int = 100) -> List[Path]:
    """Load sharp wagon images from Roboflow dataset."""
    train_images_dir = wagon_dir / "train" / "images"
    
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Wagon train images not found: {train_images_dir}")
    
    # Get all wagon images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(train_images_dir.glob(f"*{ext}"))
    
    # Limit to max_samples
    if len(image_files) > max_samples:
        image_files = image_files[:max_samples]
    
    logger.info(f"Loaded {len(image_files)} sharp wagon images")
    return image_files


def load_blurry_car_samples(car_dir: Path, max_samples: int = 100) -> List[Path]:
    """Load blurry car images from car dataset."""
    blurred_dir = car_dir / "blurred"
    
    if not blurred_dir.exists():
        raise FileNotFoundError(f"Car blurred images not found: {blurred_dir}")
    
    # Get all blurred car images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(blurred_dir.glob(f"*{ext}"))
    
    # Limit to max_samples
    if len(image_files) > max_samples:
        image_files = image_files[:max_samples]
    
    logger.info(f"Loaded {len(image_files)} blurry car images")
    return image_files


def test_laplacian_threshold(
    sharp_images: List[Path], 
    blurry_images: List[Path],
    threshold_range: Tuple[float, float] = (50, 500),
    num_thresholds: int = 50
) -> Tuple[float, float, List[float], List[float]]:
    """
    Test different Laplacian variance thresholds to find optimal separation.
    
    Args:
        sharp_images: List of sharp image paths
        blurry_images: List of blurry image paths
        threshold_range: Range of thresholds to test
        num_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (best_threshold, best_accuracy, thresholds, accuracies)
    """
    logger.info("Calculating Laplacian variances for sharp wagon images...")
    sharp_variances = []
    for img_path in sharp_images:
        try:
            variance = calculate_laplacian_variance(img_path)
            sharp_variances.append(variance)
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
    
    logger.info("Calculating Laplacian variances for blurry car images...")
    blurry_variances = []
    for img_path in blurry_images:
        try:
            variance = calculate_laplacian_variance(img_path)
            blurry_variances.append(variance)
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
    
    logger.info(f"Sharp wagon variances: mean={np.mean(sharp_variances):.2f}, std={np.std(sharp_variances):.2f}")
    logger.info(f"Blurry car variances: mean={np.mean(blurry_variances):.2f}, std={np.std(blurry_variances):.2f}")
    
    # Test different thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    accuracies = []
    
    best_accuracy = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        # Classify: variance > threshold = sharp, variance <= threshold = blurry
        sharp_correct = sum(1 for v in sharp_variances if v > threshold)
        blurry_correct = sum(1 for v in blurry_variances if v <= threshold)
        
        total_correct = sharp_correct + blurry_correct
        total_samples = len(sharp_variances) + len(blurry_variances)
        accuracy = total_correct / total_samples
        
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy, thresholds.tolist(), accuracies


def plot_results(
    sharp_variances: List[float],
    blurry_variances: List[float], 
    best_threshold: float,
    thresholds: List[float],
    accuracies: List[float]
) -> None:
    """Plot Laplacian variance distributions and accuracy curve."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Variance distributions
    ax1.hist(sharp_variances, bins=30, alpha=0.7, label='Sharp Wagons', color='green')
    ax1.hist(blurry_variances, bins=30, alpha=0.7, label='Blurry Cars', color='red')
    ax1.axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Best Threshold: {best_threshold:.1f}')
    ax1.set_xlabel('Laplacian Variance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Laplacian Variance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs threshold
    ax2.plot(thresholds, accuracies, 'b-', linewidth=2)
    ax2.axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best: {best_threshold:.1f}')
    ax2.axhline(0.85, color='orange', linestyle=':', linewidth=2, label='85% Target')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Laplacian Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplacian_gatekeeper_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Analysis plot saved to: laplacian_gatekeeper_analysis.png")


def main():
    """Main function to test Math Gatekeeper."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("MATH GATEKEEPER: LAPLACIAN VARIANCE BASELINE TEST")
    logger.info("=" * 60)
    
    # Paths
    wagon_dir = Path("data/wagon_detection")
    car_dir = Path("data/blurred_sharp")
    
    try:
        # Load samples
        logger.info("Loading sharp wagon samples (Roboflow)...")
        sharp_images = load_sharp_wagon_samples(wagon_dir, max_samples=100)
        
        logger.info("Loading blurry car samples (Car dataset)...")
        blurry_images = load_blurry_car_samples(car_dir, max_samples=100)
        
        if len(sharp_images) < 50 or len(blurry_images) < 50:
            logger.error("Insufficient samples for reliable testing")
            return
        
        # Calculate variances for plotting
        logger.info("Calculating variances for visualization...")
        sharp_variances = []
        for img_path in sharp_images:
            try:
                variance = calculate_laplacian_variance(img_path)
                sharp_variances.append(variance)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
        
        blurry_variances = []
        for img_path in blurry_images:
            try:
                variance = calculate_laplacian_variance(img_path)
                blurry_variances.append(variance)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
        
        # Test thresholds
        logger.info("Testing Laplacian variance thresholds...")
        best_threshold, best_accuracy, thresholds, accuracies = test_laplacian_threshold(
            sharp_images, blurry_images
        )
        
        # Results
        logger.info("=" * 60)
        logger.info("MATH GATEKEEPER RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Threshold: {best_threshold:.2f}")
        logger.info(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        logger.info(f"Neural Network Baseline: 68%")
        
        # Decision
        if best_accuracy >= 0.85:
            logger.info("🎉 SUCCESS: Math Gatekeeper achieves >85% accuracy!")
            logger.info("🚫 RECOMMENDATION: ABANDON Neural Network approach")
            logger.info("✅ USE: Simple Laplacian variance with threshold {:.2f}".format(best_threshold))
        else:
            logger.warning("❌ FAILURE: Math Gatekeeper below 85% threshold")
            logger.info("➡️  NEXT STEP: Proceed to Domain-Correct Data (Step 2)")
            logger.info("➡️  NEXT STEP: Upgrade Model Architecture (Step 3)")
        
        # Generate analysis plot
        plot_results(sharp_variances, blurry_variances, best_threshold, thresholds, accuracies)
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("DETAILED STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Sharp Wagons (n={len(sharp_variances)}):")
        logger.info(f"  Mean: {np.mean(sharp_variances):.2f}")
        logger.info(f"  Std:  {np.std(sharp_variances):.2f}")
        logger.info(f"  Min:  {np.min(sharp_variances):.2f}")
        logger.info(f"  Max:  {np.max(sharp_variances):.2f}")
        
        logger.info(f"Blurry Cars (n={len(blurry_variances)}):")
        logger.info(f"  Mean: {np.mean(blurry_variances):.2f}")
        logger.info(f"  Std:  {np.std(blurry_variances):.2f}")
        logger.info(f"  Min:  {np.min(blurry_variances):.2f}")
        logger.info(f"  Max:  {np.max(blurry_variances):.2f}")
        
        # Separation analysis
        separation = np.mean(sharp_variances) - np.mean(blurry_variances)
        logger.info(f"Mean Separation: {separation:.2f}")
        
        if separation > 0:
            logger.info("✅ Sharp wagons have higher variance (expected)")
        else:
            logger.warning("⚠️  Unexpected: Blurry cars have higher variance")
        
    except Exception as e:
        logger.error(f"Math Gatekeeper test failed: {e}")
        raise


if __name__ == "__main__":
    main()
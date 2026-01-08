#!/usr/bin/env python3
"""
Checkpoint Verification Script

Verifies that all datasets and models are ready for the Iron-Sight system.
This script checks the completion status of tasks 2.1-3.4 before proceeding
to the next phase of implementation.

Task 4: Checkpoint - Data and Models Ready
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml


logger = logging.getLogger(__name__)


class CheckpointVerifier:
    """Verifies the completion status of data and model preparation tasks."""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.models_dir = base_dir / "models"
        self.config_dir = base_dir / "config"
        self.blurred_sharp_dir = base_dir / "data" / "blurred_sharp"
        
        # Expected model files
        self.expected_models = [
            "gatekeeper.onnx",
            "yolov8n_obb.onnx", 
            "zero_dce.onnx",
            "deblur_gan.onnx"
        ]
        
        # Expected dataset directories
        self.expected_datasets = [
            "wagon_detection",
            "dataset_train",
            "combined_dataset"
        ]
    
    def verify_directory_structure(self) -> Dict[str, bool]:
        """Verify that required directories exist."""
        logger.info("Checking directory structure...")
        
        results = {}
        required_dirs = [
            self.data_dir,
            self.models_dir,
            self.config_dir,
            self.blurred_sharp_dir
        ]
        
        for dir_path in required_dirs:
            exists = dir_path.exists() and dir_path.is_dir()
            results[str(dir_path)] = exists
            
            if exists:
                logger.info(f"✓ Directory exists: {dir_path}")
            else:
                logger.error(f"✗ Missing directory: {dir_path}")
        
        return results
    
    def verify_blurred_sharp_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """Verify the blurred_sharp car dataset is available."""
        logger.info("Checking blurred_sharp car dataset...")
        
        stats = {"blurred": 0, "sharp": 0}
        
        if not self.blurred_sharp_dir.exists():
            logger.error(f"✗ Blurred_sharp dataset not found at: {self.blurred_sharp_dir}")
            return False, stats
        
        # Check blurred images
        blurred_dir = self.blurred_sharp_dir / "blurred"
        if blurred_dir.exists():
            blurred_images = list(blurred_dir.glob("*.png")) + list(blurred_dir.glob("*.jpg"))
            stats["blurred"] = len(blurred_images)
            logger.info(f"✓ Found {stats['blurred']} blurred car images")
        else:
            logger.error(f"✗ Missing blurred directory: {blurred_dir}")
            return False, stats
        
        # Check sharp images
        sharp_dir = self.blurred_sharp_dir / "sharp"
        if sharp_dir.exists():
            sharp_images = list(sharp_dir.glob("*.png")) + list(sharp_dir.glob("*.jpg"))
            stats["sharp"] = len(sharp_images)
            logger.info(f"✓ Found {stats['sharp']} sharp car images")
        else:
            logger.error(f"✗ Missing sharp directory: {sharp_dir}")
            return False, stats
        
        # Check if we have a reasonable number of images
        min_images = 100
        if stats["blurred"] < min_images or stats["sharp"] < min_images:
            logger.warning(f"⚠ Low image count (need >{min_images} each): blurred={stats['blurred']}, sharp={stats['sharp']}")
            return False, stats
        
        logger.info(f"✓ Blurred_sharp dataset verified: {stats['blurred']} blurred, {stats['sharp']} sharp")
        return True, stats
    
    def verify_roboflow_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """Verify the Roboflow wagon dataset has been downloaded."""
        logger.info("Checking Roboflow wagon dataset...")
        
        stats = {"train": 0, "valid": 0}
        wagon_dir = self.data_dir / "wagon_detection"
        
        if not wagon_dir.exists():
            logger.error(f"✗ Wagon dataset not found at: {wagon_dir}")
            logger.info("  Run: python scripts/train_gatekeeper.py --download-data")
            return False, stats
        
        # Check data.yaml
        data_yaml = wagon_dir / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"✗ Missing data.yaml: {data_yaml}")
            return False, stats
        
        # Verify dataset structure
        for split in ["train", "valid"]:
            split_dir = wagon_dir / split
            if split_dir.exists():
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                
                if images_dir.exists() and labels_dir.exists():
                    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                    labels = list(labels_dir.glob("*.txt"))
                    stats[split] = len(images)
                    
                    logger.info(f"✓ {split}: {len(images)} images, {len(labels)} labels")
                else:
                    logger.error(f"✗ Missing {split} images/labels directories")
                    return False, stats
            else:
                logger.error(f"✗ Missing {split} directory: {split_dir}")
                return False, stats
        
        # Check if we have reasonable data
        if stats["train"] < 50 or stats["valid"] < 10:
            logger.warning(f"⚠ Low wagon dataset count: train={stats['train']}, valid={stats['valid']}")
            return False, stats
        
        logger.info(f"✓ Roboflow wagon dataset verified: {stats['train']} train, {stats['valid']} valid")
        return True, stats
    
    def verify_augmented_dataset(self) -> Tuple[bool, int]:
        """Verify augmented dataset has been created."""
        logger.info("Checking augmented dataset...")
        
        augmented_dir = self.data_dir / "dataset_train"
        
        if not augmented_dir.exists():
            logger.error(f"✗ Augmented dataset not found at: {augmented_dir}")
            logger.info("  Run data physics augmentation scripts")
            return False, 0
        
        # Count augmented images
        augmented_images = list(augmented_dir.glob("*.jpg")) + list(augmented_dir.glob("*.png"))
        count = len(augmented_images)
        
        if count < 1000:
            logger.warning(f"⚠ Low augmented image count: {count} (target: 2000+)")
            return False, count
        
        logger.info(f"✓ Augmented dataset verified: {count} images")
        return True, count
    
    def verify_combined_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """Verify combined dataset has been created."""
        logger.info("Checking combined dataset...")
        
        stats = {"train": 0, "valid": 0}
        combined_dir = self.data_dir / "combined_dataset"
        
        if not combined_dir.exists():
            logger.error(f"✗ Combined dataset not found at: {combined_dir}")
            logger.info("  Run data combination scripts")
            return False, stats
        
        # Check data.yaml
        data_yaml = combined_dir / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"✗ Missing combined data.yaml: {data_yaml}")
            return False, stats
        
        # Verify unified classes
        try:
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            expected_classes = ["vehicle_body", "license_plate", "wheel", "coupling_mechanism"]
            if 'names' in config:
                actual_classes = list(config['names'].values()) if isinstance(config['names'], dict) else config['names']
                if set(actual_classes) != set(expected_classes):
                    logger.error(f"✗ Incorrect classes in combined dataset: {actual_classes}")
                    return False, stats
                logger.info(f"✓ Unified classes verified: {actual_classes}")
        except Exception as e:
            logger.error(f"✗ Error reading combined data.yaml: {e}")
            return False, stats
        
        # Check dataset splits
        for split in ["train", "valid"]:
            split_dir = combined_dir / split / "images"
            if split_dir.exists():
                images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                stats[split] = len(images)
                logger.info(f"✓ Combined {split}: {stats[split]} images")
            else:
                logger.error(f"✗ Missing combined {split} directory: {split_dir}")
                return False, stats
        
        if stats["train"] < 100 or stats["valid"] < 20:
            logger.warning(f"⚠ Low combined dataset count: train={stats['train']}, valid={stats['valid']}")
            return False, stats
        
        logger.info(f"✓ Combined dataset verified: {stats['train']} train, {stats['valid']} valid")
        return True, stats
    
    def verify_config_files(self) -> Tuple[bool, List[str]]:
        """Verify configuration files exist."""
        logger.info("Checking configuration files...")
        
        missing_configs = []
        expected_configs = [
            "vehicle_detection.yaml"
        ]
        
        for config_file in expected_configs:
            config_path = self.config_dir / config_file
            if config_path.exists():
                logger.info(f"✓ Config file exists: {config_file}")
            else:
                logger.error(f"✗ Missing config file: {config_file}")
                missing_configs.append(config_file)
        
        return len(missing_configs) == 0, missing_configs
    
    def verify_trained_models(self) -> Tuple[bool, Dict[str, bool]]:
        """Verify all required ONNX models exist."""
        logger.info("Checking trained models...")
        
        model_status = {}
        
        for model_file in self.expected_models:
            model_path = self.models_dir / model_file
            exists = model_path.exists() and model_path.is_file()
            model_status[model_file] = exists
            
            if exists:
                # Check file size (should be > 1MB for real models)
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Model exists: {model_file} ({size_mb:.1f} MB)")
            else:
                logger.error(f"✗ Missing model: {model_file}")
        
        all_models_exist = all(model_status.values())
        return all_models_exist, model_status
    
    def verify_model_training_logs(self) -> Dict[str, bool]:
        """Check for training logs and summaries."""
        logger.info("Checking training logs...")
        
        log_status = {}
        expected_logs = [
            "gatekeeper_training_history.json",
            "yolo_training_summary.json", 
            "zero_dce_summary.json",
            "deblur_training_summary.json"
        ]
        
        for log_file in expected_logs:
            log_path = self.models_dir / log_file
            exists = log_path.exists()
            log_status[log_file] = exists
            
            if exists:
                logger.info(f"✓ Training log exists: {log_file}")
            else:
                logger.warning(f"⚠ Missing training log: {log_file}")
        
        return log_status
    
    def run_full_verification(self) -> Dict[str, any]:
        """Run complete checkpoint verification."""
        logger.info("=" * 60)
        logger.info("IRON-SIGHT CHECKPOINT VERIFICATION")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Directory structure
        results['directories'] = self.verify_directory_structure()
        
        # 2. Datasets
        results['blurred_sharp_ok'], results['blurred_sharp_stats'] = self.verify_blurred_sharp_dataset()
        results['roboflow_ok'], results['roboflow_stats'] = self.verify_roboflow_dataset()
        results['augmented_ok'], results['augmented_count'] = self.verify_augmented_dataset()
        results['combined_ok'], results['combined_stats'] = self.verify_combined_dataset()
        
        # 3. Configuration
        results['config_ok'], results['missing_configs'] = self.verify_config_files()
        
        # 4. Models
        results['models_ok'], results['model_status'] = self.verify_trained_models()
        results['training_logs'] = self.verify_model_training_logs()
        
        # Overall status
        datasets_ready = (
            results['blurred_sharp_ok'] and 
            results['roboflow_ok'] and 
            results['augmented_ok'] and 
            results['combined_ok']
        )
        
        models_ready = results['models_ok']
        config_ready = results['config_ok']
        
        results['datasets_ready'] = datasets_ready
        results['models_ready'] = models_ready
        results['config_ready'] = config_ready
        results['checkpoint_passed'] = datasets_ready and models_ready and config_ready
        
        # Summary
        logger.info("=" * 60)
        logger.info("CHECKPOINT SUMMARY")
        logger.info("=" * 60)
        
        status_symbol = "✓" if datasets_ready else "✗"
        logger.info(f"{status_symbol} Datasets Ready: {datasets_ready}")
        
        status_symbol = "✓" if models_ready else "✗"
        logger.info(f"{status_symbol} Models Ready: {models_ready}")
        
        status_symbol = "✓" if config_ready else "✗"
        logger.info(f"{status_symbol} Config Ready: {config_ready}")
        
        logger.info("-" * 60)
        
        if results['checkpoint_passed']:
            logger.info("🎉 CHECKPOINT PASSED - Ready to proceed to task 5!")
        else:
            logger.error("❌ CHECKPOINT FAILED - Complete missing tasks before proceeding")
            
            # Provide specific guidance
            if not datasets_ready:
                logger.info("\n📋 TO FIX DATASETS:")
                if not results['roboflow_ok']:
                    logger.info("  1. Set ROBOFLOW_KEY environment variable")
                    logger.info("  2. Run: python scripts/train_gatekeeper.py --download-data")
                if not results['augmented_ok']:
                    logger.info("  3. Run physics-based augmentation scripts")
                if not results['combined_ok']:
                    logger.info("  4. Run dataset combination scripts")
            
            if not models_ready:
                logger.info("\n🤖 TO FIX MODELS:")
                logger.info("  1. Run: python scripts/train_gatekeeper.py")
                logger.info("  2. Run: python scripts/train_yolo.py")
                logger.info("  3. Run: python scripts/prepare_zero_dce.py")
                logger.info("  4. Run: python scripts/train_deblur.py")
        
        return results


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Iron-Sight checkpoint status")
    parser.add_argument("--base-dir", type=str, default=".",
                       help="Base directory for verification")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run verification
        verifier = CheckpointVerifier(Path(args.base_dir))
        results = verifier.run_full_verification()
        
        # Exit with appropriate code
        if results['checkpoint_passed']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
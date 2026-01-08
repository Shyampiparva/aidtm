#!/usr/bin/env python3
"""
Iron-Sight Dataset Mapping Script

Maps and verifies the hybrid multi-dataset strategy:
- Physics Dataset (Cars): data/BLURRED_sharp/ - Unlabeled blurred/sharp pairs
- Structure Dataset (Wagons): data/wagon_detection/ - Fully labeled YOLOv8-OBB

This implements the "Split-Brain" approach where different models train on
different datasets optimized for their specific tasks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)


class DatasetMapper:
    """Maps and verifies the hybrid dataset structure for Iron-Sight."""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir
        
        # Physics Dataset (Cars) - Unlabeled
        self.physics_dataset = {
            "name": "Physics Dataset (Cars)",
            "location": base_dir / "data" / "blurred_sharp",
            "type": "Unlabeled Pairs",
            "purpose": "DeblurGAN + Gatekeeper + Zero-DCE Training",
            "folders": ["blurred", "sharp"]
        }
        
        # Structure Dataset (Wagons) - Labeled
        self.structure_dataset = {
            "name": "Structure Dataset (Wagons)", 
            "location": base_dir / "data" / "wagon_detection",
            "type": "YOLOv8-OBB Labeled",
            "purpose": "YOLO Detection Training",
            "files": ["data.yaml"]
        }
    
    def verify_physics_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """
        Verify the Physics Dataset (Cars) structure.
        
        Returns:
            Tuple of (is_valid, stats)
        """
        logger.info("🔍 Verifying Physics Dataset (Cars)...")
        
        stats = {"blurred": 0, "sharp": 0}
        dataset_path = self.physics_dataset["location"]
        
        if not dataset_path.exists():
            logger.error(f"❌ Physics dataset not found: {dataset_path}")
            return False, stats
        
        # Check blurred folder
        blurred_dir = dataset_path / "blurred"
        if not blurred_dir.exists():
            logger.error(f"❌ Missing blurred folder: {blurred_dir}")
            return False, stats
        
        blurred_images = list(blurred_dir.glob("*.png")) + list(blurred_dir.glob("*.jpg"))
        stats["blurred"] = len(blurred_images)
        
        # Check sharp folder
        sharp_dir = dataset_path / "sharp"
        if not sharp_dir.exists():
            logger.error(f"❌ Missing sharp folder: {sharp_dir}")
            return False, stats
        
        sharp_images = list(sharp_dir.glob("*.png")) + list(sharp_dir.glob("*.jpg"))
        stats["sharp"] = len(sharp_images)
        
        # Verify we have sufficient data
        min_images = 100
        if stats["blurred"] < min_images or stats["sharp"] < min_images:
            logger.warning(f"⚠️ Low image count: blurred={stats['blurred']}, sharp={stats['sharp']}")
            return False, stats
        
        logger.info(f"✅ Physics Dataset verified:")
        logger.info(f"   📁 Blurred: {stats['blurred']} images")
        logger.info(f"   📁 Sharp: {stats['sharp']} images")
        logger.info(f"   🎯 Purpose: DeblurGAN + Gatekeeper + Zero-DCE")
        
        return True, stats
    
    def verify_structure_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """
        Verify the Structure Dataset (Wagons) structure.
        
        Returns:
            Tuple of (is_valid, stats)
        """
        logger.info("🔍 Verifying Structure Dataset (Wagons)...")
        
        stats = {"train": 0, "valid": 0, "classes": 0}
        dataset_path = self.structure_dataset["location"]
        
        if not dataset_path.exists():
            logger.error(f"❌ Structure dataset not found: {dataset_path}")
            return False, stats
        
        # Check data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"❌ Missing data.yaml: {data_yaml}")
            return False, stats
        
        # Parse data.yaml
        try:
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'names' in config:
                if isinstance(config['names'], dict):
                    stats["classes"] = len(config['names'])
                    class_names = list(config['names'].values())
                elif isinstance(config['names'], list):
                    stats["classes"] = len(config['names'])
                    class_names = config['names']
                else:
                    class_names = []
                
                logger.info(f"   📋 Classes: {class_names}")
            
        except Exception as e:
            logger.error(f"❌ Error reading data.yaml: {e}")
            return False, stats
        
        # Check train/valid splits
        for split in ["train", "valid"]:
            split_dir = dataset_path / split
            if split_dir.exists():
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                
                if images_dir.exists() and labels_dir.exists():
                    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                    labels = list(labels_dir.glob("*.txt"))
                    stats[split] = len(images)
                    
                    logger.info(f"   📁 {split.capitalize()}: {len(images)} images, {len(labels)} labels")
                else:
                    logger.error(f"❌ Missing {split} images/labels directories")
                    return False, stats
            else:
                logger.warning(f"⚠️ Missing {split} directory: {split_dir}")
        
        # Verify sufficient data
        if stats["train"] < 50:
            logger.warning(f"⚠️ Low training data: {stats['train']} images")
            return False, stats
        
        logger.info(f"✅ Structure Dataset verified:")
        logger.info(f"   📊 Train: {stats['train']} images")
        logger.info(f"   📊 Valid: {stats['valid']} images") 
        logger.info(f"   📊 Classes: {stats['classes']}")
        logger.info(f"   🎯 Purpose: YOLOv8-OBB Detection")
        
        return True, stats
    
    def generate_training_strategy(self) -> Dict[str, Dict]:
        """
        Generate the Split-Brain training strategy mapping.
        
        Returns:
            Dictionary with training strategy for each model
        """
        strategy = {
            "gatekeeper": {
                "model": "MobileNetV3-Small",
                "dataset": "Physics Dataset (Cars)",
                "source": "data/BLURRED_sharp/",
                "labels": "Auto-generated (sharp=Pass, blurred=Fail)",
                "purpose": "Binary classification: is_blurry",
                "script": "scripts/train_gatekeeper.py"
            },
            "deblur": {
                "model": "DeblurGAN-v2",
                "dataset": "Physics Dataset (Cars)",
                "input": "data/BLURRED_sharp/blurred/",
                "target": "data/BLURRED_sharp/sharp/",
                "purpose": "Blur removal: blurry → sharp",
                "script": "scripts/train_deblur.py"
            },
            "detector": {
                "model": "YOLOv8-OBB",
                "dataset": "Structure Dataset (Wagons)",
                "source": "data/wagon_detection/data.yaml",
                "purpose": "Object detection: wagon_body, text_plate",
                "script": "scripts/train_yolo.py"
            },
            "enhancer": {
                "model": "Zero-DCE++",
                "dataset": "Physics Dataset (Cars)",
                "source": "data/BLURRED_sharp/blurred/",
                "purpose": "Low-light enhancement (unsupervised)",
                "script": "scripts/prepare_zero_dce.py"
            }
        }
        
        return strategy
    
    def print_training_strategy(self, strategy: Dict[str, Dict]) -> None:
        """Print the complete training strategy."""
        logger.info("🧠 SPLIT-BRAIN TRAINING STRATEGY:")
        logger.info("=" * 60)
        
        for model_name, config in strategy.items():
            logger.info(f"🤖 {model_name.upper()}:")
            logger.info(f"   Model: {config['model']}")
            logger.info(f"   Dataset: {config['dataset']}")
            if 'source' in config:
                logger.info(f"   Source: {config['source']}")
            if 'input' in config:
                logger.info(f"   Input: {config['input']}")
            if 'target' in config:
                logger.info(f"   Target: {config['target']}")
            if 'labels' in config:
                logger.info(f"   Labels: {config['labels']}")
            logger.info(f"   Purpose: {config['purpose']}")
            logger.info(f"   Script: {config['script']}")
            logger.info("")
    
    def run_mapping(self) -> bool:
        """
        Run complete dataset mapping and verification.
        
        Returns:
            True if all datasets are properly mapped
        """
        logger.info("🚀 IRON-SIGHT DATASET MAPPING")
        logger.info("=" * 60)
        
        # Verify Physics Dataset
        physics_ok, physics_stats = self.verify_physics_dataset()
        
        # Verify Structure Dataset  
        structure_ok, structure_stats = self.verify_structure_dataset()
        
        # Generate and print strategy
        if physics_ok and structure_ok:
            strategy = self.generate_training_strategy()
            self.print_training_strategy(strategy)
            
            logger.info("✅ DATASETS MAPPED: Physics (Cars) & Structure (Wagons)")
            logger.info("🎯 Ready for Split-Brain Training!")
            return True
        else:
            logger.error("❌ Dataset mapping failed - fix issues above")
            return False


def main():
    """Main mapping function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        mapper = DatasetMapper()
        success = mapper.run_mapping()
        
        if success:
            print("\n🎉 SUCCESS: Datasets mapped successfully!")
            print("📋 Next steps:")
            print("   1. Run: python scripts/train_gatekeeper.py")
            print("   2. Run: python scripts/train_deblur.py") 
            print("   3. Run: python scripts/train_yolo.py")
            print("   4. Run: python scripts/prepare_zero_dce.py")
            sys.exit(0)
        else:
            print("\n❌ FAILED: Fix dataset issues above")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Mapping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
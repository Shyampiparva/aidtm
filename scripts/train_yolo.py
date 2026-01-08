#!/usr/bin/env python3
"""
YOLOv8-OBB Training Script

Fine-tunes YOLOv8-OBB nano model on combined vehicle dataset (cars + wagons)
for oriented bounding box detection. Supports unified classes:
- vehicle_body (cars + wagons)
- license_plate (car plates + wagon IDs)  
- wheel (car wheels + wagon assemblies)
- coupling_mechanism (wagon-specific)

Requirements: 5.1, 5.2, 5.3, 5.7, 10.1, 10.2
"""

import os
import sys
import logging
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_ingest import download_wagon_dataset, verify_dataset_structure
from data_physics import simulate_railway_conditions


logger = logging.getLogger(__name__)


def prepare_combined_dataset(
    wagon_data_dir: Path,
    car_data_dir: Path,
    output_dir: Path,
    config_path: Path
) -> Path:
    """
    Prepare combined dataset from wagon and car data for YOLO training.
    
    Args:
        wagon_data_dir: Path to Roboflow wagon dataset
        car_data_dir: Path to car dataset (blurred_sharp)
        output_dir: Output directory for combined dataset
        config_path: Path to vehicle_detection.yaml config
        
    Returns:
        Path to prepared dataset directory
    """
    logger.info("Preparing combined dataset for YOLO training...")
    
    combined_dir = output_dir / "combined_dataset"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ["train", "val"]:
        (combined_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (combined_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Class mapping from config
    class_mapping = {
        "wagon_body": 0,           # -> vehicle_body
        "identification_plate": 1,  # -> license_plate
        "wheel_assembly": 2,       # -> wheel
        "coupling_mechanism": 3    # -> coupling_mechanism
    }
    
    # Process wagon dataset
    if wagon_data_dir.exists() and verify_dataset_structure(wagon_data_dir):
        logger.info("Processing Roboflow wagon dataset...")
        
        # Copy wagon images and convert labels
        for split in ["train", "valid"]:
            wagon_split_dir = wagon_data_dir / split
            if not wagon_split_dir.exists():
                continue
                
            target_split = "train" if split == "train" else "val"
            
            # Process images
            wagon_images_dir = wagon_split_dir / "images"
            wagon_labels_dir = wagon_split_dir / "labels"
            
            if wagon_images_dir.exists():
                wagon_images = list(wagon_images_dir.glob("*.jpg")) + list(wagon_images_dir.glob("*.png"))
                logger.info(f"Found {len(wagon_images)} wagon images in {split}")
                
                for img_path in wagon_images:
                    # Copy image with physics-based augmentation
                    target_img_path = combined_dir / target_split / "images" / f"wagon_{img_path.name}"
                    
                    # Apply railway conditions augmentation
                    import cv2
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        # Apply augmentation to some images
                        if np.random.random() < 0.3:  # 30% augmentation rate
                            image = simulate_railway_conditions(
                                image,
                                apply_all=False  # Use random augmentation
                            )
                        cv2.imwrite(str(target_img_path), image)
                    
                    # Convert and copy label
                    label_path = wagon_labels_dir / f"{img_path.stem}.txt"
                    target_label_path = combined_dir / target_split / "labels" / f"wagon_{img_path.stem}.txt"
                    
                    if label_path.exists():
                        convert_wagon_labels(label_path, target_label_path, class_mapping)
    
    # Process car dataset (create synthetic annotations)
    logger.info("Processing car dataset...")
    
    car_sharp_dir = car_data_dir / "sharp"
    car_blurred_dir = car_data_dir / "blurred"
    
    # Process sharp car images (80% train, 20% val)
    if car_sharp_dir.exists():
        car_images = list(car_sharp_dir.glob("*.png")) + list(car_sharp_dir.glob("*.jpg"))
        logger.info(f"Found {len(car_images)} sharp car images")
        
        # Split into train/val
        np.random.shuffle(car_images)
        train_split = int(0.8 * len(car_images))
        
        train_images = car_images[:train_split]
        val_images = car_images[train_split:]
        
        # Process training images
        for img_path in train_images:
            target_img_path = combined_dir / "train" / "images" / f"car_sharp_{img_path.name}"
            target_label_path = combined_dir / "train" / "labels" / f"car_sharp_{img_path.stem}.txt"
            
            # Copy image
            import shutil
            shutil.copy2(img_path, target_img_path)
            
            # Create synthetic label (assume whole image is vehicle_body)
            create_synthetic_car_label(target_label_path, "vehicle_body")
        
        # Process validation images
        for img_path in val_images:
            target_img_path = combined_dir / "val" / "images" / f"car_sharp_{img_path.name}"
            target_label_path = combined_dir / "val" / "labels" / f"car_sharp_{img_path.stem}.txt"
            
            # Copy image
            shutil.copy2(img_path, target_img_path)
            
            # Create synthetic label
            create_synthetic_car_label(target_label_path, "vehicle_body")
    
    # Process blurred car images
    if car_blurred_dir.exists():
        car_blurred_images = list(car_blurred_dir.glob("*.png")) + list(car_blurred_dir.glob("*.jpg"))
        logger.info(f"Found {len(car_blurred_images)} blurred car images")
        
        # Split into train/val
        np.random.shuffle(car_blurred_images)
        train_split = int(0.8 * len(car_blurred_images))
        
        train_images = car_blurred_images[:train_split]
        val_images = car_blurred_images[train_split:]
        
        # Process training images
        for img_path in train_images:
            target_img_path = combined_dir / "train" / "images" / f"car_blurred_{img_path.name}"
            target_label_path = combined_dir / "train" / "labels" / f"car_blurred_{img_path.stem}.txt"
            
            # Copy image
            shutil.copy2(img_path, target_img_path)
            
            # Create synthetic label
            create_synthetic_car_label(target_label_path, "vehicle_body")
        
        # Process validation images
        for img_path in val_images:
            target_img_path = combined_dir / "val" / "images" / f"car_blurred_{img_path.name}"
            target_label_path = combined_dir / "val" / "labels" / f"car_blurred_{img_path.stem}.txt"
            
            # Copy image
            shutil.copy2(img_path, target_img_path)
            
            # Create synthetic label
            create_synthetic_car_label(target_label_path, "vehicle_body")
    
    # Create dataset YAML file
    dataset_yaml = {
        'path': str(combined_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 4,
        'names': {
            0: 'vehicle_body',
            1: 'license_plate', 
            2: 'wheel',
            3: 'coupling_mechanism'
        }
    }
    
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    logger.info(f"Combined dataset prepared at: {combined_dir}")
    logger.info(f"Dataset configuration saved to: {yaml_path}")
    
    return combined_dir


def convert_wagon_labels(
    input_label_path: Path, 
    output_label_path: Path, 
    class_mapping: Dict[str, int]
) -> None:
    """
    Convert wagon dataset labels to unified class format.
    
    Args:
        input_label_path: Path to original wagon label file
        output_label_path: Path to save converted label
        class_mapping: Mapping from original to unified classes
    """
    if not input_label_path.exists():
        return
    
    with open(input_label_path, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:  # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
            original_class = int(parts[0])
            
            # Map to unified classes (assuming original order matches our mapping)
            class_names = ["wagon_body", "wheel_assembly", "coupling_mechanism", "identification_plate"]
            if original_class < len(class_names):
                original_class_name = class_names[original_class]
                if original_class_name in class_mapping:
                    new_class = class_mapping[original_class_name]
                    parts[0] = str(new_class)
                    converted_lines.append(' '.join(parts) + '\n')
    
    with open(output_label_path, 'w') as f:
        f.writelines(converted_lines)


def create_synthetic_car_label(label_path: Path, class_name: str) -> None:
    """
    Create synthetic label for car images (assume whole image is vehicle).
    
    Args:
        label_path: Path to save label file
        class_name: Class name for the label
    """
    # Map class name to ID
    class_id_map = {
        'vehicle_body': 0,
        'license_plate': 1,
        'wheel': 2,
        'coupling_mechanism': 3
    }
    
    class_id = class_id_map.get(class_name, 0)
    
    # Create bounding box covering most of the image (OBB format)
    # Format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
    bbox = f"{class_id} 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    
    with open(label_path, 'w') as f:
        f.write(bbox)


def train_yolo_model(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: Optional[str] = None
) -> YOLO:
    """
    Train YOLOv8-OBB model on combined dataset.
    
    Args:
        dataset_path: Path to dataset directory containing data.yaml
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
        device: Device to train on (auto-detect if None)
        
    Returns:
        Trained YOLO model
    """
    logger.info("Starting YOLOv8-OBB training...")
    
    # Load pre-trained YOLOv8n-OBB model
    model = YOLO('yolov8n-obb.pt')
    
    # Configure training parameters
    train_args = {
        'data': str(dataset_path / 'data.yaml'),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': image_size,
        'project': str(output_dir),
        'name': 'yolov8n_obb_vehicle',
        'save': True,
        'save_period': 10,
        'cache': False,  # Disable caching for large datasets
        'device': device,
        'workers': 8,
        'patience': 20,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    # Start training
    results = model.train(**train_args)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {results.save_dir}")
    
    return model


def validate_model_performance(model: YOLO, dataset_path: Path) -> Dict[str, float]:
    """
    Validate model performance and check if targets are met.
    
    Args:
        model: Trained YOLO model
        dataset_path: Path to dataset for validation
        
    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating model performance...")
    
    # Run validation
    results = model.val(
        data=str(dataset_path / 'data.yaml'),
        split='val',
        save_json=True,
        save_hybrid=True
    )
    
    # Extract metrics
    metrics = {
        'map50': results.box.map50,
        'map50_95': results.box.map,
        'precision': results.box.p.mean(),
        'recall': results.box.r.mean(),
        'f1': results.box.f1.mean()
    }
    
    logger.info("Validation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Check targets
    target_map50 = 0.85
    if metrics['map50'] >= target_map50:
        logger.info(f"✓ mAP@50 target met: {metrics['map50']:.4f} >= {target_map50}")
    else:
        logger.warning(f"mAP@50 target not met: {metrics['map50']:.4f} < {target_map50}")
    
    return metrics


def benchmark_inference_time(model: YOLO, image_size: int = 640, num_runs: int = 100) -> float:
    """
    Benchmark model inference time.
    
    Args:
        model: Trained YOLO model
        image_size: Input image size
        num_runs: Number of runs for averaging
        
    Returns:
        Average inference time in milliseconds
    """
    logger.info("Benchmarking inference time...")
    
    # Create dummy input
    dummy_input = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=False)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(dummy_input, verbose=False)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    logger.info(f"Average inference time: {avg_time_ms:.2f}ms (target: <25ms)")
    
    return avg_time_ms


def export_to_onnx(
    model: YOLO, 
    output_path: Path, 
    image_size: int = 640,
    half: bool = True,
    dynamic: bool = True
) -> None:
    """
    Export YOLO model to ONNX format with FP16 and dynamic shapes.
    
    Args:
        model: Trained YOLO model
        output_path: Path to save ONNX model
        image_size: Input image size
        half: Use FP16 precision
        dynamic: Enable dynamic input shapes
    """
    logger.info("Exporting model to ONNX...")
    
    # Export to ONNX
    onnx_path = model.export(
        format='onnx',
        imgsz=image_size,
        half=half,
        dynamic=dynamic,
        opset=11,
        simplify=True,
        optimize=True
    )
    
    # Move to desired location
    import shutil
    shutil.move(onnx_path, output_path)
    
    logger.info(f"Model exported to ONNX: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")
    
    # Test ONNX Runtime inference
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    logger.info(f"ONNX input shape: {input_shape}")
    
    # Test inference with dynamic shapes
    for test_size in [320, 640, 1280]:
        test_input = np.random.randn(1, 3, test_size, test_size).astype(np.float32)
        try:
            start_time = time.time()
            output = session.run(None, {input_name: test_input})
            inference_time = (time.time() - start_time) * 1000
            logger.info(f"ONNX inference ({test_size}x{test_size}): {inference_time:.2f}ms")
        except Exception as e:
            logger.warning(f"ONNX inference failed for size {test_size}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-OBB on combined vehicle dataset")
    parser.add_argument("--wagon-data", type=str, default="data/wagon_detection",
                       help="Path to Roboflow wagon dataset")
    parser.add_argument("--car-data", type=str, default="data/blurred_sharp",
                       help="Path to car dataset")
    parser.add_argument("--config", type=str, default="config/vehicle_detection.yaml",
                       help="Path to dataset configuration")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--image-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to train on (auto-detect if not specified)")
    parser.add_argument("--download-data", action="store_true",
                       help="Download Roboflow dataset if not present")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup paths
    wagon_data_dir = Path(args.wagon_data)
    car_data_dir = Path(args.car_data)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download wagon dataset if requested
    if args.download_data or not wagon_data_dir.exists():
        logger.info("Downloading Roboflow wagon dataset...")
        try:
            wagon_data_dir = download_wagon_dataset(output_dir=str(wagon_data_dir))
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Continuing with existing data if available...")
    
    # Check device - FORCE GPU usage
    if args.device is None:
        if not torch.cuda.is_available():
            logger.error("CUDA GPU is required for training but not available!")
            logger.error("Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support")
            sys.exit(1)
        device = "cuda"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA requested but not available!")
            sys.exit(1)
    
    logger.info(f"Using device: {device} (GPU forced)")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Prepare combined dataset
        dataset_dir = prepare_combined_dataset(
            wagon_data_dir, car_data_dir, output_dir, config_path
        )
        
        # Train model
        model = train_yolo_model(
            dataset_dir,
            output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=device
        )
        
        # Validate performance
        metrics = validate_model_performance(model, dataset_dir)
        
        # Benchmark inference time
        inference_time = benchmark_inference_time(model, args.image_size)
        
        # Check inference time target
        if inference_time > 25.0:
            logger.warning(f"Inference time {inference_time:.2f}ms exceeds target of 25ms")
        else:
            logger.info(f"✓ Inference time target met: {inference_time:.2f}ms < 25ms")
        
        # Export to ONNX
        onnx_model_path = output_dir / "yolov8n_obb.onnx"
        export_to_onnx(
            model, 
            onnx_model_path, 
            image_size=args.image_size,
            half=True,
            dynamic=True
        )
        
        # Save training summary
        summary = {
            'model': 'yolov8n-obb',
            'dataset': 'combined_vehicle_dataset',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'device': device,
            'metrics': metrics,
            'inference_time_ms': inference_time,
            'onnx_exported': True,
            'targets_met': {
                'map50_target': metrics['map50'] >= 0.85,
                'inference_time_target': inference_time <= 25.0
            }
        }
        
        summary_path = output_dir / "yolo_training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        logger.info("YOLOv8-OBB training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
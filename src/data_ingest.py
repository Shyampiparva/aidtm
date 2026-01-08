"""
Data ingestion module for downloading and preparing datasets.

This module handles downloading the Roboflow wagon detection dataset
and preparing it for training the Iron-Sight inspection system.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from roboflow import Roboflow


logger = logging.getLogger(__name__)


def download_wagon_dataset(
    api_key: Optional[str] = None,
    project_id: str = "vishakha-singh/wagon-detection-eh2ov",
    version: int = 1,
    format_type: str = "yolov8-obb",
    output_dir: str = "data/wagon_detection"
) -> Path:
    """
    Download wagon detection dataset from Roboflow.
    
    Downloads the specified Roboflow dataset in YOLOv8-OBB format for training
    wagon detection models. The dataset contains annotations for wagon_body,
    wheel_assembly, coupling_mechanism, and identification_plate classes.
    
    Args:
        api_key: Roboflow API key. If None, will try to get from ROBOFLOW_KEY env var
        project_id: Roboflow project identifier in format "workspace/project"
        version: Dataset version to download
        format_type: Export format (default: "yolov8-obb" for oriented bounding boxes)
        output_dir: Local directory to store downloaded dataset
        
    Returns:
        Path to the downloaded dataset directory
        
    Raises:
        ValueError: If API key is not provided and not found in environment
        RuntimeError: If dataset download fails
        
    Requirements:
        - 5.2: Training data for wagon_body, wheel_assembly, coupling_mechanism, identification_plate
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_KEY")
        if api_key is None:
            raise ValueError(
                "Roboflow API key not provided. Set ROBOFLOW_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading wagon dataset from {project_id} to {output_path}")
    
    try:
        # Initialize Roboflow client
        rf = Roboflow(api_key=api_key)
        
        # Get project and version
        workspace, project_name = project_id.split("/")
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version)
        
        # Download dataset in specified format
        dataset_path = dataset.download(
            format=format_type,
            location=str(output_path)
        )
        
        downloaded_path = Path(dataset_path)
        logger.info(f"Successfully downloaded wagon dataset to {downloaded_path}")
        
        # Verify expected classes are present
        expected_classes = {
            "wagon_body", 
            "wheel_assembly", 
            "coupling_mechanism", 
            "identification_plate"
        }
        
        # Check if data.yaml exists and contains expected classes
        data_yaml_path = downloaded_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                content = f.read()
                logger.info("Dataset classes found in data.yaml")
        else:
            logger.warning("data.yaml not found in downloaded dataset")
        
        return downloaded_path
        
    except Exception as e:
        error_msg = f"Failed to download wagon dataset: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def verify_dataset_structure(dataset_path: Path) -> bool:
    """
    Verify that the downloaded dataset has the expected structure.
    
    Args:
        dataset_path: Path to the downloaded dataset
        
    Returns:
        True if dataset structure is valid, False otherwise
    """
    required_dirs = ["train", "valid"]
    required_files = ["data.yaml"]
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Missing required directory: {dir_path}")
            return False
        
        # Check for images and labels subdirectories
        images_dir = dir_path / "images"
        labels_dir = dir_path / "labels"
        
        if not images_dir.exists():
            logger.error(f"Missing images directory: {images_dir}")
            return False
            
        if not labels_dir.exists():
            logger.error(f"Missing labels directory: {labels_dir}")
            return False
    
    # Check required files
    for file_name in required_files:
        file_path = dataset_path / file_name
        if not file_path.exists():
            logger.error(f"Missing required file: {file_path}")
            return False
    
    logger.info("Dataset structure verification passed")
    return True


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        dataset_path = download_wagon_dataset()
        if verify_dataset_structure(dataset_path):
            print(f"Dataset successfully downloaded and verified at: {dataset_path}")
        else:
            print("Dataset structure verification failed")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
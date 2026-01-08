#!/usr/bin/env python3
"""
Visual Validation of Trained DeblurGAN Model

Creates before/after comparison grids to validate deblurring performance
on both car and wagon images. Focus on readability of wagon numbers.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from train_deblur import DeblurGANv2Generator


logger = logging.getLogger(__name__)


class DeblurVisualizer:
    """
    Visual validation tool for DeblurGAN model.
    """
    
    def __init__(self, model_path: Path, use_onnx: bool = True):
        """
        Initialize visualizer with trained model.
        
        Args:
            model_path: Path to model file (.onnx or .pth)
            use_onnx: Whether to use ONNX runtime (faster) or PyTorch
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if use_onnx and model_path.suffix == '.onnx':
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # [-1,1] -> [0,1]
            transforms.ToPILImage()
        ])
    
    def _load_onnx_model(self):
        """Load ONNX model for inference."""
        logger.info(f"Loading ONNX model: {self.model_path}")
        
        # Configure ONNX Runtime for GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        self.onnx_session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
        
        logger.info(f"ONNX model loaded with providers: {self.onnx_session.get_providers()}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model for inference."""
        logger.info(f"Loading PyTorch model: {self.model_path}")
        
        # Load model
        if self.model_path.suffix == '.pth':
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            self.model = DeblurGANv2Generator()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"PyTorch model loaded on {self.device}")
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from cv2)
            target_size: Target size for processing
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Resize while maintaining aspect ratio
        original_size = pil_image.size
        pil_image = pil_image.resize(target_size, Image.LANCZOS)
        
        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension
        
        return tensor, original_size
    
    def postprocess_output(self, output_tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to image.
        
        Args:
            output_tensor: Model output tensor
            original_size: Original image size for resizing
            
        Returns:
            Output image (BGR format)
        """
        # Remove batch dimension and clamp to [-1, 1]
        output_tensor = output_tensor.squeeze(0).clamp(-1, 1)
        
        # Convert to PIL
        pil_output = self.inverse_transform(output_tensor)
        
        # Resize back to original size
        pil_output = pil_output.resize(original_size, Image.LANCZOS)
        
        # Convert to BGR numpy array
        output_rgb = np.array(pil_output)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        
        return output_bgr
    
    def deblur_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur a single image.
        
        Args:
            image: Input blurred image (BGR format)
            
        Returns:
            Deblurred image (BGR format)
        """
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)
        
        # Inference
        if self.use_onnx:
            # ONNX inference
            input_np = input_tensor.cpu().numpy()
            output_np = self.onnx_session.run([self.output_name], {self.input_name: input_np})[0]
            output_tensor = torch.from_numpy(output_np)
        else:
            # PyTorch inference
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            output_tensor = output_tensor.cpu()
        
        # Postprocess
        deblurred_image = self.postprocess_output(output_tensor, original_size)
        
        return deblurred_image
    
    def find_test_images(self, data_dirs: List[Path], max_images: int = 5) -> List[Path]:
        """
        Find test images from multiple directories.
        
        Args:
            data_dirs: List of directories to search
            max_images: Maximum number of images to return
            
        Returns:
            List of image paths
        """
        image_paths = []
        
        for data_dir in data_dirs:
            if not data_dir.exists():
                logger.warning(f"Directory not found: {data_dir}")
                continue
            
            # Find image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                image_paths.extend(list(data_dir.glob(ext)))
                image_paths.extend(list(data_dir.glob(ext.upper())))
        
        # Shuffle and limit
        random.shuffle(image_paths)
        selected_paths = image_paths[:max_images]
        
        logger.info(f"Found {len(image_paths)} total images, selected {len(selected_paths)}")
        
        return selected_paths
    
    def create_comparison_grid(
        self, 
        image_paths: List[Path], 
        output_path: Path,
        grid_title: str = "DeblurGAN Validation Results"
    ) -> None:
        """
        Create before/after comparison grid.
        
        Args:
            image_paths: List of input image paths
            output_path: Path to save grid image
            grid_title: Title for the grid
        """
        if not image_paths:
            logger.error("No images provided for grid creation")
            return
        
        logger.info(f"Creating comparison grid with {len(image_paths)} images...")
        
        # Process images
        processed_pairs = []
        
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {img_path.name}")
            
            try:
                # Load original image
                original = cv2.imread(str(img_path))
                if original is None:
                    logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Deblur image
                deblurred = self.deblur_image(original)
                
                # Resize for grid (maintain aspect ratio)
                target_height = 300
                h, w = original.shape[:2]
                aspect_ratio = w / h
                target_width = int(target_height * aspect_ratio)
                
                original_resized = cv2.resize(original, (target_width, target_height))
                deblurred_resized = cv2.resize(deblurred, (target_width, target_height))
                
                processed_pairs.append({
                    'original': original_resized,
                    'deblurred': deblurred_resized,
                    'name': img_path.stem,
                    'width': target_width,
                    'height': target_height
                })
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue
        
        if not processed_pairs:
            logger.error("No images were successfully processed")
            return
        
        # Create grid
        self._create_grid_image(processed_pairs, output_path, grid_title)
    
    def _create_grid_image(
        self, 
        processed_pairs: List[dict], 
        output_path: Path, 
        grid_title: str
    ) -> None:
        """Create the actual grid image."""
        
        # Calculate grid dimensions
        max_width = max(pair['width'] for pair in processed_pairs)
        row_height = max(pair['height'] for pair in processed_pairs)
        
        # Grid layout: 2 columns (original, deblurred), N rows
        cols = 2
        rows = len(processed_pairs)
        
        # Add padding and labels
        padding = 20
        label_height = 30
        title_height = 50
        
        grid_width = cols * max_width + (cols + 1) * padding
        grid_height = title_height + rows * (row_height + label_height) + (rows + 1) * padding
        
        # Create grid canvas
        grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Convert to PIL for text rendering
        grid_pil = Image.fromarray(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(grid_pil)
        
        # Try to load a font
        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_label = ImageFont.truetype("arial.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), grid_title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 10), grid_title, fill=(0, 0, 0), font=font_title)
        
        # Draw column headers
        header_y = title_height - 25
        draw.text((padding + max_width//4, header_y), "Original (Blurred)", fill=(0, 0, 0), font=font_label)
        draw.text((padding + max_width + padding + max_width//4, header_y), "AI Restored", fill=(0, 0, 0), font=font_label)
        
        # Convert back to OpenCV format
        grid_image = cv2.cvtColor(np.array(grid_pil), cv2.COLOR_RGB2BGR)
        
        # Place images in grid
        for i, pair in enumerate(processed_pairs):
            row = i
            
            # Calculate positions
            y_start = title_height + row * (row_height + label_height + padding) + padding
            
            # Original image (left column)
            x_original = padding
            original_img = pair['original']
            h, w = original_img.shape[:2]
            
            # Center image in cell
            x_offset = (max_width - w) // 2
            y_offset = 0
            
            grid_image[y_start + y_offset:y_start + y_offset + h, 
                      x_original + x_offset:x_original + x_offset + w] = original_img
            
            # Deblurred image (right column)
            x_deblurred = padding + max_width + padding
            deblurred_img = pair['deblurred']
            
            grid_image[y_start + y_offset:y_start + y_offset + h, 
                      x_deblurred + x_offset:x_deblurred + x_offset + w] = deblurred_img
            
            # Add image name label
            label_y = y_start + row_height + 5
            cv2.putText(grid_image, pair['name'], (x_original, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save grid
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grid_image)
        
        logger.info(f"Comparison grid saved to: {output_path}")
        logger.info(f"Grid dimensions: {grid_width}x{grid_height}")
        logger.info(f"Images processed: {len(processed_pairs)}")


def main():
    """Main function for visual validation."""
    parser = argparse.ArgumentParser(description="Visual validation of DeblurGAN model")
    parser.add_argument("--model-path", type=str, default="models/deblur_gan.onnx",
                       help="Path to trained model (.onnx or .pth)")
    parser.add_argument("--use-pytorch", action="store_true",
                       help="Use PyTorch instead of ONNX (slower but more compatible)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for validation results")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of images to test per category")
    parser.add_argument("--test-cars", action="store_true", default=True,
                       help="Test on car images")
    parser.add_argument("--test-wagons", action="store_true", default=True,
                       help="Test on wagon images")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("DEBLURGAN VISUAL VALIDATION")
    logger.info("=" * 60)
    
    try:
        # Setup paths
        model_path = Path(args.model_path)
        output_dir = Path(args.output_dir)
        
        # Check model exists
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            
            # Try alternative paths
            alternatives = [
                Path("models/deblur_gan.pth"),
                Path("models/deblur_gan.onnx")
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    logger.info(f"Using alternative model: {alt_path}")
                    model_path = alt_path
                    break
            else:
                logger.error("No trained model found. Please train DeblurGAN first.")
                return
        
        # Initialize visualizer
        use_onnx = not args.use_pytorch and model_path.suffix == '.onnx'
        visualizer = DeblurVisualizer(model_path, use_onnx=use_onnx)
        
        # Test on different datasets
        if args.test_cars:
            logger.info("Testing on car images...")
            
            # Car dataset paths
            car_dirs = [
                Path("data/blurred_sharp/blurred"),
                Path("data/blurred_sharp/sharp"),  # Test on sharp images too
            ]
            
            car_images = visualizer.find_test_images(car_dirs, args.num_images)
            
            if car_images:
                car_output = output_dir / "car_validation_grid.jpg"
                visualizer.create_comparison_grid(
                    car_images, 
                    car_output,
                    "DeblurGAN Validation - Car Images"
                )
            else:
                logger.warning("No car images found for testing")
        
        if args.test_wagons:
            logger.info("Testing on wagon images...")
            
            # Wagon dataset paths
            wagon_dirs = [
                Path("data/wagon_detection/train/images"),
                Path("data/wagon_detection/valid/images"),
                Path("models/artificial_pairs"),  # Artificially blurred wagons
            ]
            
            wagon_images = visualizer.find_test_images(wagon_dirs, args.num_images)
            
            if wagon_images:
                wagon_output = output_dir / "wagon_validation_grid.jpg"
                visualizer.create_comparison_grid(
                    wagon_images, 
                    wagon_output,
                    "DeblurGAN Validation - Wagon Images (Number Readability Test)"
                )
            else:
                logger.warning("No wagon images found for testing")
        
        # Create combined test if both datasets available
        if args.test_cars and args.test_wagons:
            logger.info("Creating combined validation grid...")
            
            # Mix car and wagon images
            all_dirs = [
                Path("data/blurred_sharp/blurred"),
                Path("data/wagon_detection/train/images"),
                Path("models/artificial_pairs"),
            ]
            
            mixed_images = visualizer.find_test_images(all_dirs, args.num_images * 2)
            
            if mixed_images:
                combined_output = output_dir / "validation_grid.jpg"
                visualizer.create_comparison_grid(
                    mixed_images, 
                    combined_output,
                    "DeblurGAN Validation - Mixed Dataset (Cars + Wagons)"
                )
        
        logger.info("✅ Visual validation completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("🔍 Check the grid images to evaluate:")
        logger.info("   - Overall deblurring quality")
        logger.info("   - Wagon number readability")
        logger.info("   - Edge preservation")
        logger.info("   - Artifact reduction")
        
    except Exception as e:
        logger.error(f"Visual validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
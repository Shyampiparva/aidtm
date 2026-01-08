#!/usr/bin/env python3
"""
Demo Data Generator for IronSight Command Center.

This module generates sample data for demonstrating the dashboard:
- Sample wagon images with various blur levels for Restoration Lab
- Synthetic video frames for Mission Control
- Test images with simulated defects

Requirements: Demo preparation (Task 16.2)
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - some demo data generation will be limited")


@dataclass
class DemoImageConfig:
    """Configuration for demo image generation."""
    width: int = 640
    height: int = 480
    blur_levels: List[int] = None  # Kernel sizes for blur
    noise_levels: List[float] = None  # Noise standard deviations
    brightness_levels: List[float] = None  # Brightness multipliers
    
    def __post_init__(self):
        if self.blur_levels is None:
            self.blur_levels = [0, 5, 11, 21, 31]  # No blur to heavy blur
        if self.noise_levels is None:
            self.noise_levels = [0, 10, 25, 50]
        if self.brightness_levels is None:
            self.brightness_levels = [0.3, 0.5, 0.7, 1.0, 1.3]


class DemoDataGenerator:
    """
    Generates demo data for IronSight Command Center testing.
    
    Provides:
    - Sample wagon images with various degradations
    - Synthetic video frames
    - Test images with simulated defects
    """
    
    # Wagon serial number patterns
    SERIAL_PATTERNS = [
        "ABC-{:04d}",
        "XYZ-{:04d}",
        "RW-{:05d}",
        "WAGON-{:03d}",
        "FR-{:06d}",
    ]
    
    # Damage type descriptions
    DAMAGE_TYPES = [
        "rust_spot",
        "dent",
        "scratch",
        "corrosion",
        "hole",
        "crack",
        "wear",
        "paint_damage",
    ]
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[DemoImageConfig] = None
    ):
        """
        Initialize demo data generator.
        
        Args:
            output_dir: Directory to save generated data
            config: Image generation configuration
        """
        self.output_dir = output_dir or Path(__file__).parent / "generated"
        self.config = config or DemoImageConfig()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "restoration_lab").mkdir(exist_ok=True)
        (self.output_dir / "mission_control").mkdir(exist_ok=True)
        (self.output_dir / "semantic_search").mkdir(exist_ok=True)
        
        logger.info(f"DemoDataGenerator initialized, output: {self.output_dir}")
    
    def generate_wagon_image(
        self,
        serial_number: Optional[str] = None,
        blur_level: int = 0,
        noise_level: float = 0,
        brightness: float = 1.0,
        add_defects: bool = False
    ) -> np.ndarray:
        """
        Generate a synthetic wagon image.
        
        Args:
            serial_number: Serial number to render on image
            blur_level: Gaussian blur kernel size (0 = no blur)
            noise_level: Gaussian noise standard deviation
            brightness: Brightness multiplier
            add_defects: Whether to add simulated defects
            
        Returns:
            Generated image as numpy array (H, W, 3) BGR
        """
        if not CV2_AVAILABLE:
            # Return simple random image if OpenCV not available
            return np.random.randint(
                50, 200, 
                (self.config.height, self.config.width, 3), 
                dtype=np.uint8
            )
        
        # Create base wagon image
        image = self._create_base_wagon_image()
        
        # Add serial number plate
        if serial_number is None:
            pattern = random.choice(self.SERIAL_PATTERNS)
            serial_number = pattern.format(random.randint(1, 9999))
        
        image = self._add_serial_number_plate(image, serial_number)
        
        # Add defects if requested
        if add_defects:
            image = self._add_simulated_defects(image)
        
        # Apply degradations
        image = self._apply_brightness(image, brightness)
        image = self._apply_blur(image, blur_level)
        image = self._apply_noise(image, noise_level)
        
        return image
    
    def _create_base_wagon_image(self) -> np.ndarray:
        """Create base wagon image with structure."""
        h, w = self.config.height, self.config.width
        
        # Create gray metallic background
        image = np.full((h, w, 3), [80, 85, 90], dtype=np.uint8)
        
        # Add wagon body (darker rectangle)
        body_margin = 50
        cv2.rectangle(
            image,
            (body_margin, body_margin),
            (w - body_margin, h - body_margin),
            (60, 65, 70),
            -1
        )
        
        # Add horizontal lines (wagon panels)
        for y in range(body_margin + 40, h - body_margin, 60):
            cv2.line(image, (body_margin, y), (w - body_margin, y), (50, 55, 60), 2)
        
        # Add vertical lines (wagon sections)
        for x in range(body_margin + 80, w - body_margin, 120):
            cv2.line(image, (x, body_margin), (x, h - body_margin), (50, 55, 60), 2)
        
        # Add rivets
        for y in range(body_margin + 20, h - body_margin, 40):
            for x in range(body_margin + 20, w - body_margin, 40):
                cv2.circle(image, (x, y), 3, (40, 45, 50), -1)
        
        # Add wheels at bottom
        wheel_y = h - 30
        wheel_radius = 25
        for x in [100, 250, w - 250, w - 100]:
            cv2.circle(image, (x, wheel_y), wheel_radius, (30, 30, 30), -1)
            cv2.circle(image, (x, wheel_y), wheel_radius - 5, (50, 50, 50), 2)
        
        return image
    
    def _add_serial_number_plate(
        self, 
        image: np.ndarray, 
        serial_number: str
    ) -> np.ndarray:
        """Add serial number plate to wagon image."""
        h, w = image.shape[:2]
        
        # Plate position (upper right area)
        plate_x = w - 200
        plate_y = 80
        plate_w = 150
        plate_h = 50
        
        # Draw plate background (white/cream)
        cv2.rectangle(
            image,
            (plate_x, plate_y),
            (plate_x + plate_w, plate_y + plate_h),
            (200, 210, 220),
            -1
        )
        
        # Draw plate border
        cv2.rectangle(
            image,
            (plate_x, plate_y),
            (plate_x + plate_w, plate_y + plate_h),
            (40, 40, 40),
            2
        )
        
        # Draw serial number text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for centering
        (text_w, text_h), _ = cv2.getTextSize(serial_number, font, font_scale, thickness)
        
        text_x = plate_x + (plate_w - text_w) // 2
        text_y = plate_y + (plate_h + text_h) // 2
        
        cv2.putText(
            image, serial_number, (text_x, text_y),
            font, font_scale, (20, 20, 20), thickness
        )
        
        return image
    
    def _add_simulated_defects(self, image: np.ndarray) -> np.ndarray:
        """Add simulated defects to wagon image."""
        h, w = image.shape[:2]
        
        # Add 1-3 random defects
        num_defects = random.randint(1, 3)
        
        for _ in range(num_defects):
            defect_type = random.choice(self.DAMAGE_TYPES)
            
            # Random position (avoiding edges and plate area)
            x = random.randint(100, w - 200)
            y = random.randint(100, h - 100)
            
            if defect_type in ["rust_spot", "corrosion"]:
                # Orange/brown irregular shape
                radius = random.randint(10, 30)
                color = (30, 80, 150)  # BGR - rusty orange
                cv2.circle(image, (x, y), radius, color, -1)
                # Add texture
                for _ in range(20):
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    if dx*dx + dy*dy < radius*radius:
                        cv2.circle(image, (x + dx, y + dy), 2, (20, 60, 120), -1)
            
            elif defect_type in ["dent"]:
                # Darker ellipse
                axes = (random.randint(15, 40), random.randint(10, 25))
                angle = random.randint(0, 180)
                cv2.ellipse(image, (x, y), axes, angle, 0, 360, (40, 45, 50), -1)
            
            elif defect_type in ["scratch"]:
                # Line scratch
                length = random.randint(30, 80)
                angle = random.randint(0, 180)
                dx = int(length * np.cos(np.radians(angle)))
                dy = int(length * np.sin(np.radians(angle)))
                cv2.line(image, (x, y), (x + dx, y + dy), (100, 100, 100), 2)
            
            elif defect_type in ["hole"]:
                # Dark circle
                radius = random.randint(5, 15)
                cv2.circle(image, (x, y), radius, (20, 20, 20), -1)
        
        return image
    
    def _apply_brightness(self, image: np.ndarray, brightness: float) -> np.ndarray:
        """Apply brightness adjustment."""
        if brightness == 1.0:
            return image
        
        adjusted = image.astype(np.float32) * brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def _apply_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur."""
        if kernel_size <= 0:
            return image
        
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _apply_noise(self, image: np.ndarray, std_dev: float) -> np.ndarray:
        """Apply Gaussian noise."""
        if std_dev <= 0:
            return image
        
        noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def generate_restoration_lab_samples(
        self, 
        num_samples: int = 10
    ) -> List[Dict]:
        """
        Generate sample images for Restoration Lab testing.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of dicts with image info and paths
        """
        samples = []
        output_dir = self.output_dir / "restoration_lab"
        
        for i in range(num_samples):
            # Vary blur levels
            blur_level = random.choice(self.config.blur_levels)
            noise_level = random.choice([0, 10, 20])
            brightness = random.choice([0.7, 0.85, 1.0])
            
            # Generate image
            serial = f"DEMO-{i+1:04d}"
            image = self.generate_wagon_image(
                serial_number=serial,
                blur_level=blur_level,
                noise_level=noise_level,
                brightness=brightness,
                add_defects=random.random() > 0.5
            )
            
            # Save image
            filename = f"wagon_sample_{i+1:03d}_blur{blur_level}.png"
            filepath = output_dir / filename
            
            if CV2_AVAILABLE:
                cv2.imwrite(str(filepath), image)
            
            samples.append({
                "filename": filename,
                "filepath": str(filepath),
                "serial_number": serial,
                "blur_level": blur_level,
                "noise_level": noise_level,
                "brightness": brightness,
                "has_defects": random.random() > 0.5
            })
            
            logger.debug(f"Generated restoration sample: {filename}")
        
        logger.info(f"Generated {len(samples)} restoration lab samples")
        return samples
    
    def generate_mission_control_frames(
        self, 
        num_frames: int = 30,
        fps: int = 10
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Generate synthetic video frames for Mission Control demo.
        
        Args:
            num_frames: Number of frames to generate
            fps: Target frames per second
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        frames = []
        
        # Generate frames with slight variations
        base_serial = f"DEMO-{random.randint(1000, 9999)}"
        
        for i in range(num_frames):
            # Slight variations in each frame
            blur = random.choice([0, 3, 5])
            noise = random.choice([0, 5, 10])
            
            frame = self.generate_wagon_image(
                serial_number=base_serial,
                blur_level=blur,
                noise_level=noise,
                brightness=random.uniform(0.9, 1.1),
                add_defects=True
            )
            
            frames.append(frame)
        
        metadata = {
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "serial_number": base_serial,
            "resolution": (self.config.width, self.config.height)
        }
        
        logger.info(f"Generated {num_frames} mission control frames")
        return frames, metadata
    
    def save_demo_video(
        self, 
        frames: List[np.ndarray], 
        filename: str = "demo_video.mp4",
        fps: int = 10
    ) -> Optional[Path]:
        """
        Save frames as video file.
        
        Args:
            frames: List of frames
            filename: Output filename
            fps: Frames per second
            
        Returns:
            Path to saved video or None if failed
        """
        if not CV2_AVAILABLE or not frames:
            return None
        
        output_path = self.output_dir / "mission_control" / filename
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        
        logger.info(f"Saved demo video: {output_path}")
        return output_path


def create_demo_data_generator(
    output_dir: Optional[Path] = None
) -> DemoDataGenerator:
    """Factory function to create DemoDataGenerator."""
    return DemoDataGenerator(output_dir=output_dir)


def generate_all_demo_data(output_dir: Optional[Path] = None) -> Dict:
    """
    Generate all demo data for IronSight Command Center.
    
    Args:
        output_dir: Output directory for generated data
        
    Returns:
        Dict with paths and metadata for all generated data
    """
    generator = create_demo_data_generator(output_dir)
    
    result = {
        "restoration_lab_samples": [],
        "mission_control_video": None,
        "output_dir": str(generator.output_dir)
    }
    
    # Generate restoration lab samples
    result["restoration_lab_samples"] = generator.generate_restoration_lab_samples(10)
    
    # Generate mission control video
    frames, metadata = generator.generate_mission_control_frames(30, 10)
    video_path = generator.save_demo_video(frames, "demo_wagon_video.mp4", 10)
    result["mission_control_video"] = {
        "path": str(video_path) if video_path else None,
        "metadata": metadata
    }
    
    logger.info(f"Generated all demo data in {generator.output_dir}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎬 Demo Data Generator")
    print("=" * 60)
    
    result = generate_all_demo_data()
    
    print(f"\nOutput directory: {result['output_dir']}")
    print(f"Restoration samples: {len(result['restoration_lab_samples'])}")
    print(f"Mission control video: {result['mission_control_video']['path']}")
    
    print("\n✅ Demo data generation complete!")

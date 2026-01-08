"""
Multi-YOLO Detection System for IronSight Command Center.

This module manages 3 specialized YOLO models and merges their detections:
- sideview_damage_obb: Detects damage on wagon sides
- structure_obb: Detects structural components
- wagon_number_obb: Detects wagon identification plates

Implements oriented bounding box (OBB) support for handling angled wagons.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection with oriented bounding box support."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, width, height] or [x1, y1, x2, y2, x3, y3, x4, y4] for OBB
    is_obb: bool = False  # True if oriented bounding box
    detection_id: str = ""
    model_source: str = ""  # Which YOLO model produced this detection
    
    def __post_init__(self):
        """Generate unique detection ID if not provided."""
        if not self.detection_id:
            import uuid
            self.detection_id = str(uuid.uuid4())[:8]


@dataclass
class DetectionResult:
    """Merged detection result from all 3 YOLO models."""
    sideview_detections: List[Detection] = field(default_factory=list)
    structure_detections: List[Detection] = field(default_factory=list)
    wagon_number_detections: List[Detection] = field(default_factory=list)
    combined_json: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_status: Dict[str, bool] = field(default_factory=dict)
    
    def get_all_detections(self) -> List[Detection]:
        """Get all detections from all models."""
        return (self.sideview_detections + 
                self.structure_detections + 
                self.wagon_number_detections)


class MockYOLOModel:
    """Mock YOLO model for demonstration purposes."""
    
    def __init__(self, model_name: str, class_names: List[str]):
        self.model_name = model_name
        self.class_names = class_names
        self.is_loaded = True
        logger.info(f"Initialized mock YOLO model: {model_name}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """
        Generate mock detections for demonstration.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of Detection objects
        """
        h, w = image.shape[:2]
        detections = []
        
        # Generate 1-3 mock detections
        num_detections = np.random.randint(1, 4)
        
        for i in range(num_detections):
            class_name = np.random.choice(self.class_names)
            confidence = np.random.uniform(conf_threshold, 1.0)
            
            # Generate random bounding box
            if "obb" in self.model_name.lower():
                # Oriented bounding box (8 coordinates)
                cx, cy = np.random.uniform(0.2, 0.8, 2)
                width, height = np.random.uniform(0.1, 0.3, 2)
                angle = np.random.uniform(0, 360)
                
                # Convert to 4 corner points (simplified OBB)
                bbox = self._generate_obb_points(cx * w, cy * h, 
                                                 width * w, height * h, 
                                                 angle)
                is_obb = True
            else:
                # Regular bounding box [x, y, width, height]
                x = np.random.uniform(0.1, 0.7) * w
                y = np.random.uniform(0.1, 0.7) * h
                box_w = np.random.uniform(0.1, 0.3) * w
                box_h = np.random.uniform(0.1, 0.3) * h
                bbox = [x, y, box_w, box_h]
                is_obb = False
            
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                is_obb=is_obb,
                model_source=self.model_name
            )
            detections.append(detection)
        
        return detections
    
    def _generate_obb_points(self, cx: float, cy: float, 
                            width: float, height: float, 
                            angle: float) -> List[float]:
        """Generate 4 corner points for oriented bounding box."""
        import math
        
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Half dimensions
        hw = width / 2
        hh = height / 2
        
        # Four corners (relative to center)
        corners = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh)
        ]
        
        # Rotate and translate
        points = []
        for dx, dy in corners:
            x = cx + dx * cos_a - dy * sin_a
            y = cy + dx * sin_a + dy * cos_a
            points.extend([x, y])
        
        return points


class RealYOLOModel:
    """
    Real YOLO model wrapper with FP16 optimization.
    
    Loads .pt models directly using YOLO('model.pt') and enables
    FP16 acceleration via model(source, device='cuda', half=True).
    """
    
    def __init__(self, model_path: str, model_name: str, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize real YOLO model with FP16 support.
        
        Args:
            model_path: Path to .pt model file
            model_name: Name identifier for the model
            device: Device for inference ('cuda' or 'cpu')
            use_fp16: Enable FP16 acceleration (half=True)
        """
        from ultralytics import YOLO
        
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.is_loaded = False
        
        # Load model directly from .pt file
        self.model = YOLO(model_path)
        self.is_loaded = True
        
        logger.info(f"Loaded real YOLO model: {model_name} (FP16={self.use_fp16}, device={device})")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """
        Run detection with FP16 acceleration.
        
        Uses model(source, device='cuda', half=True) for optimized inference.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of Detection objects
        """
        # Run inference with FP16 optimization
        # Key: half=True enables FP16 acceleration on CUDA
        results = self.model(
            image,
            device=self.device,
            half=self.use_fp16,
            conf=conf_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            # Handle OBB (oriented bounding box) results
            if hasattr(result, 'obb') and result.obb is not None:
                for i, box in enumerate(result.obb.xyxyxyxy):
                    # OBB format: 4 corner points (x1,y1,x2,y2,x3,y3,x4,y4)
                    points = box.cpu().numpy().flatten().tolist()
                    conf = float(result.obb.conf[i].cpu().numpy())
                    cls_id = int(result.obb.cls[i].cpu().numpy())
                    cls_name = result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                    
                    detection = Detection(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=points,
                        is_obb=True,
                        model_source=self.model_name
                    )
                    detections.append(detection)
            
            # Handle regular bounding box results
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for i, box in enumerate(result.boxes.xyxy):
                    # Regular format: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    cls_id = int(result.boxes.cls[i].cpu().numpy())
                    cls_name = result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                    
                    # Convert to [x, y, width, height] format
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    
                    detection = Detection(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                        is_obb=False,
                        model_source=self.model_name
                    )
                    detections.append(detection)
        
        return detections


class MultiYOLODetector:
    """
    Manages 3 specialized YOLO models and merges their detections.
    
    Models:
    - sideview_damage_obb: Detects damage on wagon sides (dents, holes, rust)
    - structure_obb: Detects structural components (doors, wheels, couplers)
    - wagon_number_obb: Detects identification plates and numbers
    
    Supports FP16 acceleration via model(source, device='cuda', half=True).
    """
    
    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize multi-YOLO detector with FP16 support.
        
        Args:
            model_paths: Dict mapping model names to file paths
            device: Device for inference ('cuda' or 'cpu')
            use_fp16: Enable FP16 acceleration (half=True)
        """
        self.model_paths = model_paths or {}
        self.device = device
        self.use_fp16 = use_fp16
        self.combined_latency_budget_ms = 20.0
        
        # Initialize models
        self.sideview_model = self._load_model(
            "sideview_damage_obb",
            ["dent", "hole", "rust", "scratch", "corrosion"]
        )
        
        self.structure_model = self._load_model(
            "structure_obb",
            ["door", "wheel", "coupler", "brake", "undercarriage", "roof"]
        )
        
        self.wagon_number_model = self._load_model(
            "wagon_number_obb",
            ["identification_plate", "wagon_number", "serial_number"]
        )
        
        self.model_status = {
            "sideview_damage_obb": self.sideview_model is not None,
            "structure_obb": self.structure_model is not None,
            "wagon_number_obb": self.wagon_number_model is not None
        }
        
        logger.info(f"MultiYOLODetector initialized (FP16={use_fp16}, device={device})")
        logger.info(f"Model status: {self.model_status}")
    
    def _load_model(self, model_name: str, class_names: List[str]):
        """
        Load a YOLO model with FP16 optimization.
        
        Loads .pt models directly using YOLO('model.pt').
        Falls back to mock model if real model not available.
        
        Args:
            model_name: Name of the model
            class_names: List of class names this model can detect
            
        Returns:
            RealYOLOModel or MockYOLOModel instance
        """
        try:
            model_path = self.model_paths.get(model_name)
            
            # Try to load real model if path exists
            if model_path and Path(model_path).exists():
                try:
                    return RealYOLOModel(
                        model_path=model_path,
                        model_name=model_name,
                        device=self.device,
                        use_fp16=self.use_fp16
                    )
                except ImportError:
                    logger.warning(f"Ultralytics not installed, using mock for {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load real model {model_name}: {e}, using mock")
            
            # Use mock model for demonstration
            return MockYOLOModel(model_name, class_names)
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None
    
    def detect_all(self, image: np.ndarray, conf_threshold: float = 0.5) -> DetectionResult:
        """
        Run all 3 YOLO models and merge results.
        
        Args:
            image: Input image as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            DetectionResult with merged detections and metadata
        """
        start_time = time.time()
        
        # Run all models
        sideview_detections = []
        structure_detections = []
        wagon_number_detections = []
        
        if self.sideview_model:
            try:
                sideview_detections = self.sideview_model.detect(image, conf_threshold)
            except Exception as e:
                logger.error(f"Sideview model detection failed: {e}")
                self.model_status["sideview_damage_obb"] = False
        
        if self.structure_model:
            try:
                structure_detections = self.structure_model.detect(image, conf_threshold)
            except Exception as e:
                logger.error(f"Structure model detection failed: {e}")
                self.model_status["structure_obb"] = False
        
        if self.wagon_number_model:
            try:
                wagon_number_detections = self.wagon_number_model.detect(image, conf_threshold)
            except Exception as e:
                logger.error(f"Wagon number model detection failed: {e}")
                self.model_status["wagon_number_obb"] = False
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Merge detections into single JSON
        combined_json = self.merge_detections(
            sideview_detections,
            structure_detections,
            wagon_number_detections
        )
        
        # Create result
        result = DetectionResult(
            sideview_detections=sideview_detections,
            structure_detections=structure_detections,
            wagon_number_detections=wagon_number_detections,
            combined_json=combined_json,
            processing_time_ms=processing_time_ms,
            model_status=self.model_status.copy()
        )
        
        # Check latency budget
        if processing_time_ms > self.combined_latency_budget_ms:
            logger.warning(
                f"YOLO detection exceeded latency budget: "
                f"{processing_time_ms:.2f}ms > {self.combined_latency_budget_ms}ms"
            )
        
        return result
    
    def merge_detections(
        self,
        sideview: List[Detection],
        structure: List[Detection],
        wagon_number: List[Detection]
    ) -> Dict[str, Any]:
        """
        Merge detections from all 3 models into unified JSON format.
        
        Args:
            sideview: Detections from sideview damage model
            structure: Detections from structure model
            wagon_number: Detections from wagon number model
            
        Returns:
            Merged detection dictionary
        """
        merged = {
            "timestamp": time.time(),
            "total_detections": len(sideview) + len(structure) + len(wagon_number),
            "detections_by_model": {
                "sideview_damage_obb": self._detections_to_dict(sideview),
                "structure_obb": self._detections_to_dict(structure),
                "wagon_number_obb": self._detections_to_dict(wagon_number)
            },
            "all_detections": self._detections_to_dict(
                sideview + structure + wagon_number
            ),
            "model_status": self.model_status
        }
        
        return merged
    
    def _detections_to_dict(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """Convert list of Detection objects to list of dictionaries."""
        return [
            {
                "detection_id": det.detection_id,
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "bbox": [float(x) for x in det.bbox],
                "is_obb": det.is_obb,
                "model_source": det.model_source
            }
            for det in detections
        ]
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw detection overlays on image with different colors per model.
        
        Args:
            image: Input image
            detection_result: Detection result to visualize
            colors: Optional color mapping for each model type
            
        Returns:
            Image with detection overlays
        """
        import cv2
        
        # Default colors (BGR format)
        if colors is None:
            colors = {
                "sideview_damage_obb": (0, 0, 255),    # Red for damage
                "structure_obb": (0, 255, 0),          # Green for structure
                "wagon_number_obb": (255, 255, 0)      # Cyan for numbers
            }
        
        vis_image = image.copy()
        
        # Draw each model's detections
        for model_name, detections in [
            ("sideview_damage_obb", detection_result.sideview_detections),
            ("structure_obb", detection_result.structure_detections),
            ("wagon_number_obb", detection_result.wagon_number_detections)
        ]:
            color = colors.get(model_name, (255, 255, 255))
            
            for det in detections:
                if det.is_obb:
                    # Draw oriented bounding box
                    points = np.array(det.bbox).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(vis_image, [points], True, color, 2)
                else:
                    # Draw regular bounding box
                    x, y, w, h = det.bbox
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{det.class_name} {det.confidence:.2f}"
                label_pos = (int(det.bbox[0]), int(det.bbox[1]) - 10)
                cv2.putText(
                    vis_image, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        return vis_image
    
    def get_identification_plate_detections(
        self,
        detection_result: DetectionResult
    ) -> List[Detection]:
        """
        Extract only identification_plate detections for NAFNet processing.
        
        Args:
            detection_result: Detection result from detect_all()
            
        Returns:
            List of identification_plate detections
        """
        plates = []
        for det in detection_result.wagon_number_detections:
            if det.class_name in ["identification_plate", "serial_number"]:
                plates.append(det)
        return plates

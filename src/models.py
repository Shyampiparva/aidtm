"""
Data models for the Iron-Sight railway inspection system.
Contains configuration and result dataclasses for the pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class PipelineConfig:
    """Configuration for the inspection pipeline."""
    video_source: str
    model_dir: str = "models/"
    queue_maxsize: int = 2
    gatekeeper_timeout_ms: float = 1.0
    enhancement_timeout_ms: float = 15.0
    detection_timeout_ms: float = 25.0
    deblur_timeout_ms: float = 40.0
    ocr_timeout_ms: float = 50.0
    total_timeout_ms: float = 100.0
    db_connection_string: str = ""
    fallback_db_path: str = "fallback.db"


@dataclass
class Detection:
    """Detection result from YOLO model with oriented bounding box."""
    x: float           # Center x
    y: float           # Center y
    width: float       # Box width
    height: float      # Box height
    angle: float       # Rotation angle (OBB)
    confidence: float  # Detection confidence
    class_id: int      # Class index
    class_name: str    # Class name


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    is_valid_wagon_id: bool
    raw_results: List[Tuple[str, float]]  # All detected text with confidence
    fallback_used: bool = False  # Whether SmolVLM fallback was used
    forensic_processing_time_ms: int = 0  # Time spent in forensic analysis


@dataclass
class InspectionResult:
    """Complete inspection result for a frame."""
    frame_id: int
    timestamp: datetime
    wagon_id: Optional[str]
    detection_confidence: float
    ocr_confidence: float
    blur_score: float
    enhancement_applied: bool
    deblur_applied: bool
    processing_time_ms: int
    spectral_channel: str
    bounding_box: Dict[str, Any]  # JSON-serializable OBB
    wagon_angle: float
    fallback_ocr_used: bool = False  # Whether SmolVLM fallback was triggered
    damage_assessment: Optional[str] = None  # Damage description if applicable
    
    def to_db_row(self) -> Dict[str, Any]:
        """Convert to database row format."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "wagon_id": self.wagon_id,
            "detection_confidence": self.detection_confidence,
            "ocr_confidence": self.ocr_confidence,
            "blur_score": self.blur_score,
            "enhancement_applied": self.enhancement_applied,
            "deblur_applied": self.deblur_applied,
            "processing_time_ms": self.processing_time_ms,
            "spectral_channel": self.spectral_channel,
            "bounding_box": self.bounding_box,
            "wagon_angle": self.wagon_angle,
            "fallback_ocr_used": self.fallback_ocr_used,
            "damage_assessment": self.damage_assessment,
        }


@dataclass
class SpectralChannels:
    """Spectral channel decomposition results."""
    red: np.ndarray       # For OCR path
    saturation: np.ndarray  # For damage detection path
    original: np.ndarray   # Original BGR frame


@dataclass
class LatencyMetrics:
    """Latency tracking for each pipeline stage."""
    gatekeeper_ms: float
    enhancement_ms: float
    detection_ms: float
    crop_ms: float
    deblur_ms: float
    ocr_ms: float
    total_ms: float
    
    def exceeds_budget(self, budget_ms: float = 100.0) -> bool:
        """Check if total latency exceeds budget."""
        return self.total_ms > budget_ms
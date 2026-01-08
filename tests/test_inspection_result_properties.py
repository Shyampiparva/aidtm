"""
Property-based tests for InspectionResult field completeness.
Tests Property 17: Inspection Logging Completeness
"""

from datetime import datetime
from typing import Dict, Any, Optional
import pytest
from hypothesis import given, strategies as st, settings
from src.models import InspectionResult


# Feature: iron-sight-inspection, Property 17: Inspection Logging Completeness
@settings(max_examples=100)
@given(
    frame_id=st.integers(min_value=0, max_value=1000000),
    wagon_id=st.one_of(st.none(), st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=10, max_size=10)),
    detection_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ocr_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    blur_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    enhancement_applied=st.booleans(),
    deblur_applied=st.booleans(),
    processing_time_ms=st.integers(min_value=1, max_value=1000),
    spectral_channel=st.sampled_from(["red", "saturation", "original"]),
    wagon_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
)
def test_inspection_result_field_completeness(
    frame_id: int,
    wagon_id: Optional[str],
    detection_confidence: float,
    ocr_confidence: float,
    blur_score: float,
    enhancement_applied: bool,
    deblur_applied: bool,
    processing_time_ms: int,
    spectral_channel: str,
    wagon_angle: float,
) -> None:
    """
    For any detected wagon, the logged record SHALL contain all required fields:
    frame_id, timestamp, wagon_id, detection_confidence, ocr_confidence, blur_score,
    enhancement_applied, deblur_applied, processing_time_ms, spectral_channel,
    bounding_box, wagon_angle.
    **Validates: Requirements 8.1, 8.2**
    """
    # Create a sample bounding box
    bounding_box = {
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 30.0,
        "angle": wagon_angle
    }
    
    # Create InspectionResult with generated data
    result = InspectionResult(
        frame_id=frame_id,
        timestamp=datetime.now(),
        wagon_id=wagon_id,
        detection_confidence=detection_confidence,
        ocr_confidence=ocr_confidence,
        blur_score=blur_score,
        enhancement_applied=enhancement_applied,
        deblur_applied=deblur_applied,
        processing_time_ms=processing_time_ms,
        spectral_channel=spectral_channel,
        bounding_box=bounding_box,
        wagon_angle=wagon_angle,
    )
    
    # Convert to database row format
    db_row = result.to_db_row()
    
    # Verify all required fields are present
    required_fields = {
        "frame_id",
        "timestamp", 
        "wagon_id",
        "detection_confidence",
        "ocr_confidence",
        "blur_score",
        "enhancement_applied",
        "deblur_applied",
        "processing_time_ms",
        "spectral_channel",
        "bounding_box",
        "wagon_angle",
    }
    
    # Assert all required fields are present in the database row
    assert set(db_row.keys()) == required_fields
    
    # Assert field values match the original InspectionResult
    assert db_row["frame_id"] == result.frame_id
    assert db_row["timestamp"] == result.timestamp
    assert db_row["wagon_id"] == result.wagon_id
    assert db_row["detection_confidence"] == result.detection_confidence
    assert db_row["ocr_confidence"] == result.ocr_confidence
    assert db_row["blur_score"] == result.blur_score
    assert db_row["enhancement_applied"] == result.enhancement_applied
    assert db_row["deblur_applied"] == result.deblur_applied
    assert db_row["processing_time_ms"] == result.processing_time_ms
    assert db_row["spectral_channel"] == result.spectral_channel
    assert db_row["bounding_box"] == result.bounding_box
    assert db_row["wagon_angle"] == result.wagon_angle
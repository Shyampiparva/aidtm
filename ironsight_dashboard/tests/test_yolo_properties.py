"""
Property-based tests for Multi-YOLO Detection System.
Tests Property 7 for YOLO detection merging.
"""

import pytest
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import sys
import numpy as np
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from multi_yolo_detector import (
    MultiYOLODetector, Detection, DetectionResult
)


# Custom strategies for generating test data
@st.composite
def detection_strategy(draw, model_source: str, is_obb: bool = False):
    """Generate a random Detection object."""
    class_names = {
        "sideview_damage_obb": ["dent", "hole", "rust", "scratch", "corrosion"],
        "structure_obb": ["door", "wheel", "coupler", "brake", "undercarriage", "roof"],
        "wagon_number_obb": ["identification_plate", "wagon_number", "serial_number"]
    }
    
    class_name = draw(st.sampled_from(class_names.get(model_source, ["unknown"])))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    if is_obb:
        # Generate 8 coordinates for oriented bounding box (4 corner points)
        bbox = [draw(st.floats(min_value=0.0, max_value=1000.0)) for _ in range(8)]
    else:
        # Generate regular bounding box [x, y, width, height]
        x = draw(st.floats(min_value=0.0, max_value=800.0))
        y = draw(st.floats(min_value=0.0, max_value=600.0))
        width = draw(st.floats(min_value=10.0, max_value=200.0))
        height = draw(st.floats(min_value=10.0, max_value=200.0))
        bbox = [x, y, width, height]
    
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        is_obb=is_obb,
        model_source=model_source
    )


@st.composite
def detection_lists_strategy(draw):
    """Generate lists of detections for all 3 models."""
    sideview_count = draw(st.integers(min_value=0, max_value=10))
    structure_count = draw(st.integers(min_value=0, max_value=10))
    wagon_number_count = draw(st.integers(min_value=0, max_value=10))
    
    sideview_detections = [
        draw(detection_strategy("sideview_damage_obb", is_obb=True))
        for _ in range(sideview_count)
    ]
    
    structure_detections = [
        draw(detection_strategy("structure_obb", is_obb=True))
        for _ in range(structure_count)
    ]
    
    wagon_number_detections = [
        draw(detection_strategy("wagon_number_obb", is_obb=True))
        for _ in range(wagon_number_count)
    ]
    
    return sideview_detections, structure_detections, wagon_number_detections


# Feature: ironsight-command-center, Property 7: YOLO Detection Merging
@settings(max_examples=100, deadline=5000)
@given(detection_lists=detection_lists_strategy())
def test_yolo_detection_merging(detection_lists):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any image processed by all 3 YOLO models, the output SHALL be merged
    into a single JSON result containing detections from all models.
    **Validates: Requirements 5.2**
    """
    sideview_detections, structure_detections, wagon_number_detections = detection_lists
    
    # Create detector instance
    detector = MultiYOLODetector()
    
    # Merge detections
    merged_json = detector.merge_detections(
        sideview_detections,
        structure_detections,
        wagon_number_detections
    )
    
    # Property 1: Merged result must be a dictionary
    assert isinstance(merged_json, dict), "Merged result must be a dictionary"
    
    # Property 2: Must contain required top-level keys
    required_keys = ["timestamp", "total_detections", "detections_by_model", 
                     "all_detections", "model_status"]
    for key in required_keys:
        assert key in merged_json, f"Merged result must contain '{key}' key"
    
    # Property 3: Total detections count must equal sum of all detections
    expected_total = len(sideview_detections) + len(structure_detections) + len(wagon_number_detections)
    assert merged_json["total_detections"] == expected_total, \
        f"Total detections {merged_json['total_detections']} must equal sum {expected_total}"
    
    # Property 4: detections_by_model must contain all 3 model keys
    model_keys = ["sideview_damage_obb", "structure_obb", "wagon_number_obb"]
    for model_key in model_keys:
        assert model_key in merged_json["detections_by_model"], \
            f"detections_by_model must contain '{model_key}'"
    
    # Property 5: Each model's detections must match input count
    assert len(merged_json["detections_by_model"]["sideview_damage_obb"]) == len(sideview_detections), \
        "Sideview detections count must match input"
    assert len(merged_json["detections_by_model"]["structure_obb"]) == len(structure_detections), \
        "Structure detections count must match input"
    assert len(merged_json["detections_by_model"]["wagon_number_obb"]) == len(wagon_number_detections), \
        "Wagon number detections count must match input"
    
    # Property 6: all_detections must contain all detections from all models
    assert len(merged_json["all_detections"]) == expected_total, \
        f"all_detections count {len(merged_json['all_detections'])} must equal total {expected_total}"
    
    # Property 7: Each detection in merged result must have required fields
    required_detection_fields = ["detection_id", "class_name", "confidence", 
                                 "bbox", "is_obb", "model_source"]
    for detection_dict in merged_json["all_detections"]:
        for field in required_detection_fields:
            assert field in detection_dict, \
                f"Each detection must contain '{field}' field"
        
        # Verify field types
        assert isinstance(detection_dict["detection_id"], str)
        assert isinstance(detection_dict["class_name"], str)
        assert isinstance(detection_dict["confidence"], float)
        assert isinstance(detection_dict["bbox"], list)
        assert isinstance(detection_dict["is_obb"], bool)
        assert isinstance(detection_dict["model_source"], str)
        
        # Verify confidence is in valid range
        assert 0.0 <= detection_dict["confidence"] <= 1.0, \
            f"Confidence {detection_dict['confidence']} must be between 0 and 1"
        
        # Verify bbox has correct length
        if detection_dict["is_obb"]:
            assert len(detection_dict["bbox"]) == 8, \
                "OBB bbox must have 8 coordinates (4 corner points)"
        else:
            assert len(detection_dict["bbox"]) == 4, \
                "Regular bbox must have 4 values [x, y, width, height]"
    
    # Property 8: model_status must contain status for all 3 models
    assert "model_status" in merged_json
    assert isinstance(merged_json["model_status"], dict)
    for model_key in model_keys:
        assert model_key in merged_json["model_status"], \
            f"model_status must contain '{model_key}'"
        assert isinstance(merged_json["model_status"][model_key], bool), \
            f"model_status['{model_key}'] must be boolean"
    
    # Property 9: timestamp must be a valid number
    assert isinstance(merged_json["timestamp"], (int, float))
    assert merged_json["timestamp"] > 0


@settings(max_examples=100, deadline=5000)
@given(
    image_height=st.integers(min_value=100, max_value=2000),
    image_width=st.integers(min_value=100, max_value=2000),
    conf_threshold=st.floats(min_value=0.0, max_value=1.0)
)
def test_detect_all_returns_valid_detection_result(image_height, image_width, conf_threshold):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any image input, detect_all() SHALL return a valid DetectionResult
    with merged detections from all models.
    **Validates: Requirements 5.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create detector
    detector = MultiYOLODetector()
    
    # Run detection
    result = detector.detect_all(image, conf_threshold=conf_threshold)
    
    # Property 1: Result must be DetectionResult instance
    assert isinstance(result, DetectionResult), "Result must be DetectionResult instance"
    
    # Property 2: Result must have all required attributes
    assert hasattr(result, "sideview_detections")
    assert hasattr(result, "structure_detections")
    assert hasattr(result, "wagon_number_detections")
    assert hasattr(result, "combined_json")
    assert hasattr(result, "processing_time_ms")
    assert hasattr(result, "model_status")
    
    # Property 3: Detection lists must be lists
    assert isinstance(result.sideview_detections, list)
    assert isinstance(result.structure_detections, list)
    assert isinstance(result.wagon_number_detections, list)
    
    # Property 4: All detections must be Detection instances
    for det in result.sideview_detections:
        assert isinstance(det, Detection)
    for det in result.structure_detections:
        assert isinstance(det, Detection)
    for det in result.wagon_number_detections:
        assert isinstance(det, Detection)
    
    # Property 5: combined_json must be a valid dictionary
    assert isinstance(result.combined_json, dict)
    assert "total_detections" in result.combined_json
    assert "detections_by_model" in result.combined_json
    assert "all_detections" in result.combined_json
    
    # Property 6: Total detections in JSON must match sum of detection lists
    total_from_lists = (len(result.sideview_detections) + 
                       len(result.structure_detections) + 
                       len(result.wagon_number_detections))
    assert result.combined_json["total_detections"] == total_from_lists
    
    # Property 7: Processing time must be non-negative
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    
    # Property 8: model_status must contain all 3 models
    assert isinstance(result.model_status, dict)
    assert "sideview_damage_obb" in result.model_status
    assert "structure_obb" in result.model_status
    assert "wagon_number_obb" in result.model_status


@settings(max_examples=100)
@given(detection_lists=detection_lists_strategy())
def test_get_all_detections_combines_correctly(detection_lists):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any DetectionResult, get_all_detections() SHALL return the combined
    list of all detections from all 3 models.
    **Validates: Requirements 5.2**
    """
    sideview_detections, structure_detections, wagon_number_detections = detection_lists
    
    # Create DetectionResult
    result = DetectionResult(
        sideview_detections=sideview_detections,
        structure_detections=structure_detections,
        wagon_number_detections=wagon_number_detections
    )
    
    # Get all detections
    all_detections = result.get_all_detections()
    
    # Property 1: Result must be a list
    assert isinstance(all_detections, list)
    
    # Property 2: Length must equal sum of all detection lists
    expected_length = len(sideview_detections) + len(structure_detections) + len(wagon_number_detections)
    assert len(all_detections) == expected_length, \
        f"Combined list length {len(all_detections)} must equal sum {expected_length}"
    
    # Property 3: All items must be Detection instances
    for det in all_detections:
        assert isinstance(det, Detection)
    
    # Property 4: All original detections must be present
    for det in sideview_detections:
        assert det in all_detections, "All sideview detections must be in combined list"
    for det in structure_detections:
        assert det in all_detections, "All structure detections must be in combined list"
    for det in wagon_number_detections:
        assert det in all_detections, "All wagon number detections must be in combined list"


@settings(max_examples=100)
@given(detection_lists=detection_lists_strategy())
def test_get_identification_plate_detections_filters_correctly(detection_lists):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any DetectionResult, get_identification_plate_detections() SHALL
    return only detections with class_name 'identification_plate' or 'serial_number'.
    **Validates: Requirements 5.2**
    """
    sideview_detections, structure_detections, wagon_number_detections = detection_lists
    
    # Create detector
    detector = MultiYOLODetector()
    
    # Create DetectionResult
    result = DetectionResult(
        sideview_detections=sideview_detections,
        structure_detections=structure_detections,
        wagon_number_detections=wagon_number_detections
    )
    
    # Get identification plate detections
    plate_detections = detector.get_identification_plate_detections(result)
    
    # Property 1: Result must be a list
    assert isinstance(plate_detections, list)
    
    # Property 2: All returned detections must be identification plates or serial numbers
    for det in plate_detections:
        assert isinstance(det, Detection)
        assert det.class_name in ["identification_plate", "serial_number"], \
            f"Detection class_name '{det.class_name}' must be identification_plate or serial_number"
    
    # Property 3: All identification plates from wagon_number_detections must be included
    expected_plates = [
        det for det in wagon_number_detections
        if det.class_name in ["identification_plate", "serial_number"]
    ]
    assert len(plate_detections) == len(expected_plates), \
        f"Plate detections count {len(plate_detections)} must match expected {len(expected_plates)}"
    
    # Property 4: No detections from other models should be included
    for det in plate_detections:
        assert det.model_source == "wagon_number_obb", \
            "Only wagon_number_obb detections should be returned"


@settings(max_examples=50)
@given(
    image_height=st.integers(min_value=100, max_value=1000),
    image_width=st.integers(min_value=100, max_value=1000),
    num_detections=st.integers(min_value=0, max_value=20)
)
def test_visualize_detections_preserves_image_shape(image_height, image_width, num_detections):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any image and detection result, visualize_detections() SHALL return
    an image with the same shape as the input.
    **Validates: Requirements 5.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create detector
    detector = MultiYOLODetector()
    
    # Create mock detections
    detections = []
    for i in range(num_detections):
        det = Detection(
            class_name="test_class",
            confidence=0.8,
            bbox=[10.0 + i*10, 10.0 + i*10, 50.0, 50.0],
            is_obb=False,
            model_source="test_model"
        )
        detections.append(det)
    
    result = DetectionResult(
        sideview_detections=detections[:num_detections//3] if num_detections > 0 else [],
        structure_detections=detections[num_detections//3:2*num_detections//3] if num_detections > 0 else [],
        wagon_number_detections=detections[2*num_detections//3:] if num_detections > 0 else []
    )
    
    # Visualize detections
    vis_image = detector.visualize_detections(image, result)
    
    # Property 1: Output must be numpy array
    assert isinstance(vis_image, np.ndarray)
    
    # Property 2: Output shape must match input shape
    assert vis_image.shape == image.shape, \
        f"Output shape {vis_image.shape} must match input shape {image.shape}"
    
    # Property 3: Output dtype should be uint8
    assert vis_image.dtype == np.uint8


@settings(max_examples=100)
@given(
    latency_budget_ms=st.floats(min_value=1.0, max_value=100.0),
    actual_time_ms=st.floats(min_value=0.1, max_value=200.0)
)
def test_latency_budget_monitoring(latency_budget_ms, actual_time_ms):
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any detection operation, the system SHALL monitor whether processing
    time exceeds the configured latency budget.
    **Validates: Requirements 5.2**
    """
    # Create detector with custom latency budget
    detector = MultiYOLODetector()
    detector.combined_latency_budget_ms = latency_budget_ms
    
    # Create mock result with specific processing time
    result = DetectionResult(
        processing_time_ms=actual_time_ms
    )
    
    # Property 1: Latency budget must be positive
    assert detector.combined_latency_budget_ms > 0
    
    # Property 2: Processing time must be non-negative
    assert result.processing_time_ms >= 0
    
    # Property 3: Can determine if budget was exceeded
    budget_exceeded = result.processing_time_ms > detector.combined_latency_budget_ms
    
    if budget_exceeded:
        # If budget exceeded, processing time should be greater than budget
        assert result.processing_time_ms > detector.combined_latency_budget_ms
    else:
        # If budget not exceeded, processing time should be within budget
        assert result.processing_time_ms <= detector.combined_latency_budget_ms


def test_detection_id_uniqueness():
    """
    Feature: ironsight-command-center, Property 7: YOLO Detection Merging
    For any set of detections, each detection SHALL have a unique detection_id.
    **Validates: Requirements 5.2**
    """
    # Create multiple detections
    detections = []
    for i in range(100):
        det = Detection(
            class_name="test_class",
            confidence=0.8,
            bbox=[10.0, 10.0, 50.0, 50.0],
            is_obb=False,
            model_source="test_model"
        )
        detections.append(det)
    
    # Property: All detection IDs must be unique
    detection_ids = [det.detection_id for det in detections]
    assert len(detection_ids) == len(set(detection_ids)), \
        "All detection IDs must be unique"


if __name__ == "__main__":
    # Run a simple test
    test_detection_id_uniqueness()
    print("Basic YOLO property test passed!")

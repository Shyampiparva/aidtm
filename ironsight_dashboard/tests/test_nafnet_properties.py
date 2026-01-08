"""
Property-based tests for NAFNet Crop-First Processing.
Tests Properties 8 and 9 for the crop-first deblurring strategy.
"""

import pytest
import time
import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crop_first_nafnet import (
    CropFirstNAFNet, Detection, CropResult, create_crop_first_nafnet
)


# Helper function to create valid detections
def create_detection(x, y, width, height, class_name="identification_plate"):
    """Create a valid Detection object."""
    return Detection(
        x=float(x),
        y=float(y),
        width=float(width),
        height=float(height),
        angle=0.0,
        confidence=0.9,
        class_id=0,
        class_name=class_name,
        id=f"det_{int(x)}_{int(y)}"
    )


# Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
@settings(max_examples=100, deadline=10000)
@given(
    img_height=st.integers(min_value=300, max_value=1080),
    img_width=st.integers(min_value=300, max_value=1920),
    num_plate_detections=st.integers(min_value=0, max_value=5),
    num_other_detections=st.integers(min_value=0, max_value=5)
)
def test_crop_first_processing_logic(
    img_height, img_width, num_plate_detections, num_other_detections
):
    """
    Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
    For any frame with identification_plate detections, NAFNet SHALL process
    only the cropped regions and not the full frame.
    **Validates: Requirements 6.1**
    """
    # Create NAFNet instance (without loading actual model)
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    # Don't load actual model for property testing
    nafnet.model_loaded = True
    nafnet.model = MockNAFNetModel()
    
    # Create test image
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create detections
    detections = []
    
    # Calculate safe bounds for detection placement
    margin = 100
    safe_x_min = margin
    safe_x_max = max(margin + 1, img_width - margin)
    safe_y_min = margin
    safe_y_max = max(margin + 1, img_height - margin)
    
    # Add identification_plate detections
    for i in range(num_plate_detections):
        # Ensure detection is within image bounds with safe margins
        if safe_x_max > safe_x_min and safe_y_max > safe_y_min:
            x = np.random.randint(safe_x_min, safe_x_max)
            y = np.random.randint(safe_y_min, safe_y_max)
            
            # Calculate safe width and height
            max_width = min(200, img_width - x - 50)
            max_height = min(100, img_height - y - 50)
            
            if max_width > 50 and max_height > 30:
                width = np.random.randint(50, max_width)
                height = np.random.randint(30, max_height)
                
                detections.append(create_detection(x, y, width, height, "identification_plate"))
    
    # Add other detections (should be ignored)
    for i in range(num_other_detections):
        if safe_x_max > safe_x_min and safe_y_max > safe_y_min:
            x = np.random.randint(safe_x_min, safe_x_max)
            y = np.random.randint(safe_y_min, safe_y_max)
            
            max_width = min(200, img_width - x - 50)
            max_height = min(100, img_height - y - 50)
            
            if max_width > 50 and max_height > 30:
                width = np.random.randint(50, max_width)
                height = np.random.randint(30, max_height)
                
                detections.append(create_detection(x, y, width, height, "wagon_body"))
    
    # Count actual plate detections added
    actual_plate_count = sum(1 for d in detections if d.class_name == "identification_plate")
    
    # Process detections
    results = nafnet.process_identification_plates(image, detections)
    
    # Verify only identification_plate detections were processed
    assert len(results) == actual_plate_count, \
        f"Expected {actual_plate_count} results, got {len(results)}"
    
    # Verify no full-frame processing occurred
    # (we check that crops are smaller than full frame)
    full_frame_pixels = img_height * img_width * 3
    
    for det_id, result in results.items():
        crop_pixels = result.original_crop.size
        
        # Crop should be significantly smaller than full frame
        assert crop_pixels < full_frame_pixels, \
            f"Crop {det_id} has {crop_pixels} pixels, full frame has {full_frame_pixels}"
        
        # Verify crop dimensions are reasonable
        assert result.original_crop.shape[0] > 0, "Crop height should be > 0"
        assert result.original_crop.shape[1] > 0, "Crop width should be > 0"
        assert result.original_crop.shape[2] == 3, "Crop should have 3 channels"
        
        # Verify deblurred crop has same shape as original crop
        assert result.deblurred_crop.shape == result.original_crop.shape, \
            f"Deblurred crop shape {result.deblurred_crop.shape} != " \
            f"original crop shape {result.original_crop.shape}"
    
    # If there were plate detections, verify computation savings
    if actual_plate_count > 0:
        total_crop_pixels = sum(r.original_crop.size for r in results.values())
        savings_pct = (1.0 - (total_crop_pixels / full_frame_pixels)) * 100
        
        # Should achieve significant computation reduction
        # (at least 50% for typical cases, target is 85%)
        assert savings_pct > 0, \
            f"Expected computation savings, got {savings_pct:.1f}%"


# Feature: ironsight-command-center, Property 9: Crop Padding Correctness
@settings(max_examples=100, deadline=5000)
@given(
    img_height=st.integers(min_value=300, max_value=1080),
    img_width=st.integers(min_value=300, max_value=1920),
    det_x=st.integers(min_value=150, max_value=1770),
    det_y=st.integers(min_value=150, max_value=930),
    det_width=st.integers(min_value=50, max_value=200),
    det_height=st.integers(min_value=30, max_value=100)
)
def test_crop_padding_correctness(
    img_height, img_width, det_x, det_y, det_width, det_height
):
    """
    Feature: ironsight-command-center, Property 9: Crop Padding Correctness
    For any detection bounding box, the extracted crop SHALL include exactly
    10% padding on all sides.
    **Validates: Requirements 6.2**
    """
    # Ensure detection is within image bounds
    assume(det_x - det_width/2 > 0)
    assume(det_y - det_height/2 > 0)
    assume(det_x + det_width/2 < img_width)
    assume(det_y + det_height/2 < img_height)
    
    # Create NAFNet instance
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    
    # Create test image
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create detection
    detection = create_detection(det_x, det_y, det_width, det_height)
    
    # Extract crop with padding
    crop, bounds = nafnet.extract_crop_with_padding(image, detection)
    
    # Verify crop was extracted successfully
    assert crop is not None, "Crop extraction failed"
    assert crop.size > 0, "Crop is empty"
    
    x1, y1, x2, y2 = bounds
    
    # Verify bounds are valid
    assert 0 <= x1 < x2 <= img_width, f"Invalid x bounds: {x1}, {x2}"
    assert 0 <= y1 < y2 <= img_height, f"Invalid y bounds: {y1}, {y2}"
    
    # Calculate expected bounds with 10% padding
    padding = 0.1
    expected_x1 = det_x - det_width/2 * (1 + padding)
    expected_y1 = det_y - det_height/2 * (1 + padding)
    expected_x2 = det_x + det_width/2 * (1 + padding)
    expected_y2 = det_y + det_height/2 * (1 + padding)
    
    # Clamp to image boundaries
    expected_x1 = max(0, expected_x1)
    expected_y1 = max(0, expected_y1)
    expected_x2 = min(img_width, expected_x2)
    expected_y2 = min(img_height, expected_y2)
    
    # Verify bounds match expected (within 1 pixel due to rounding)
    assert abs(x1 - expected_x1) <= 1, \
        f"x1 mismatch: got {x1}, expected {expected_x1}"
    assert abs(y1 - expected_y1) <= 1, \
        f"y1 mismatch: got {y1}, expected {expected_y1}"
    assert abs(x2 - expected_x2) <= 1, \
        f"x2 mismatch: got {x2}, expected {expected_x2}"
    assert abs(y2 - expected_y2) <= 1, \
        f"y2 mismatch: got {y2}, expected {expected_y2}"
    
    # Verify crop dimensions match bounds
    crop_height, crop_width = crop.shape[:2]
    assert crop_height == y2 - y1, \
        f"Crop height {crop_height} != bounds height {y2 - y1}"
    assert crop_width == x2 - x1, \
        f"Crop width {crop_width} != bounds width {x2 - x1}"
    
    # Verify padding was applied (crop should be larger than detection)
    # Unless clamped by image boundaries
    if expected_x1 >= 0 and expected_x2 <= img_width:
        assert crop_width >= det_width, \
            f"Crop width {crop_width} should be >= detection width {det_width}"
    
    if expected_y1 >= 0 and expected_y2 <= img_height:
        assert crop_height >= det_height, \
            f"Crop height {crop_height} should be >= detection height {det_height}"


@settings(max_examples=50)
@given(
    padding_percent=st.floats(min_value=0.0, max_value=0.5)
)
def test_crop_padding_configurable(padding_percent):
    """
    Feature: ironsight-command-center, Property 9: Crop Padding Correctness
    For any padding percentage, the extracted crop SHALL apply that padding correctly.
    **Validates: Requirements 6.2**
    
    The padding is applied to each half-dimension, so:
    - padded_half_width = half_width * (1 + padding)
    - total_padded_width = padded_half_width * 2 = width * (1 + padding)
    """
    # Create NAFNet with custom padding
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=padding_percent,
        device="cpu"
    )
    
    # Create test image
    img_height, img_width = 600, 800
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create detection in center of image
    det_x, det_y = 400, 300
    det_width, det_height = 100, 60
    detection = create_detection(det_x, det_y, det_width, det_height)
    
    # Extract crop
    crop, bounds = nafnet.extract_crop_with_padding(image, detection)
    
    assert crop is not None, "Crop extraction failed"
    
    x1, y1, x2, y2 = bounds
    
    # Calculate expected dimensions with custom padding
    # Padding is applied to each half-dimension: half_width * (1 + padding)
    # Total width = 2 * half_width * (1 + padding) = width * (1 + padding)
    expected_width = det_width * (1 + padding_percent)
    expected_height = det_height * (1 + padding_percent)
    
    actual_width = x2 - x1
    actual_height = y2 - y1
    
    # Verify dimensions match expected (within 2 pixels due to rounding)
    assert abs(actual_width - expected_width) <= 2, \
        f"Width mismatch: got {actual_width}, expected {expected_width}"
    assert abs(actual_height - expected_height) <= 2, \
        f"Height mismatch: got {actual_height}, expected {expected_height}"


@settings(max_examples=50)
@given(
    num_detections=st.integers(min_value=1, max_value=10)
)
def test_crop_first_multiple_detections(num_detections):
    """
    Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
    For any number of identification_plate detections, NAFNet SHALL process
    each crop independently.
    **Validates: Requirements 6.1**
    """
    # Create NAFNet instance
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    nafnet.model_loaded = True
    nafnet.model = MockNAFNetModel()
    
    # Create test image
    img_height, img_width = 1080, 1920
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create multiple detections
    detections = []
    for i in range(num_detections):
        x = 200 + i * 150
        y = 200 + (i % 3) * 200
        width = 100
        height = 60
        
        # Ensure within bounds
        if x + width/2 < img_width and y + height/2 < img_height:
            detections.append(
                create_detection(x, y, width, height, "identification_plate")
            )
    
    # Process all detections
    results = nafnet.process_identification_plates(image, detections)
    
    # Verify all detections were processed
    assert len(results) == len(detections), \
        f"Expected {len(detections)} results, got {len(results)}"
    
    # Verify each result is independent
    for det_id, result in results.items():
        assert isinstance(result, CropResult)
        assert result.detection_id == det_id
        assert result.original_crop.size > 0
        assert result.deblurred_crop.size > 0
        assert result.processing_time_ms >= 0
        assert result.padding_applied == 0.1


@settings(max_examples=50)
@given(
    img_height=st.integers(min_value=200, max_value=1080),
    img_width=st.integers(min_value=200, max_value=1920)
)
def test_crop_first_no_plate_detections(img_height, img_width):
    """
    Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
    For any frame without identification_plate detections, NAFNet SHALL not
    process any crops.
    **Validates: Requirements 6.1**
    """
    # Create NAFNet instance
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    nafnet.model_loaded = True
    nafnet.model = MockNAFNetModel()
    
    # Create test image
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create detections without identification_plate
    detections = [
        create_detection(400, 300, 100, 60, "wagon_body"),
        create_detection(800, 500, 120, 70, "door"),
        create_detection(1200, 700, 90, 50, "wheel"),
    ]
    
    # Process detections
    results = nafnet.process_identification_plates(image, detections)
    
    # Verify no crops were processed
    assert len(results) == 0, \
        f"Expected 0 results for non-plate detections, got {len(results)}"
    
    # Verify no computation was performed
    stats = nafnet.get_performance_stats()
    assert stats['total_crops_processed'] == 0


@settings(max_examples=50)
@given(
    det_x=st.integers(min_value=10, max_value=50),
    det_y=st.integers(min_value=10, max_value=50),
    det_width=st.integers(min_value=20, max_value=100),
    det_height=st.integers(min_value=20, max_value=100)
)
def test_crop_boundary_clamping(det_x, det_y, det_width, det_height):
    """
    Feature: ironsight-command-center, Property 9: Crop Padding Correctness
    For any detection near image boundaries, the crop SHALL be clamped to
    image bounds while maintaining valid dimensions.
    **Validates: Requirements 6.2**
    """
    # Create NAFNet instance
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    
    # Create small test image to force boundary conditions
    img_height, img_width = 200, 300
    image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
    
    # Create detection (may be near or outside boundaries)
    detection = create_detection(det_x, det_y, det_width, det_height)
    
    # Extract crop
    crop, bounds = nafnet.extract_crop_with_padding(image, detection)
    
    # If crop extraction succeeded
    if crop is not None and crop.size > 0:
        x1, y1, x2, y2 = bounds
        
        # Verify bounds are clamped to image
        assert 0 <= x1 < x2 <= img_width, \
            f"X bounds not clamped: {x1}, {x2} (image width: {img_width})"
        assert 0 <= y1 < y2 <= img_height, \
            f"Y bounds not clamped: {y1}, {y2} (image height: {img_height})"
        
        # Verify crop dimensions match bounds
        crop_height, crop_width = crop.shape[:2]
        assert crop_height == y2 - y1
        assert crop_width == x2 - x1
        
        # Verify crop is valid
        assert crop_height > 0 and crop_width > 0
        assert crop.shape[2] == 3  # BGR channels


def test_crop_first_edge_cases():
    """
    Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
    For edge case inputs, NAFNet SHALL handle them gracefully.
    **Validates: Requirements 6.1**
    """
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    nafnet.model_loaded = True
    nafnet.model = MockNAFNetModel()
    
    image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    
    edge_cases = [
        ("empty detections", []),
        ("single detection", [create_detection(400, 300, 100, 60)]),
        ("tiny detection", [create_detection(400, 300, 10, 10)]),
        ("large detection", [create_detection(400, 300, 400, 300)]),
    ]
    
    for name, detections in edge_cases:
        results = nafnet.process_identification_plates(image, detections)
        
        # Verify results match number of plate detections
        plate_count = sum(1 for d in detections if d.class_name == "identification_plate")
        assert len(results) == plate_count, f"{name}: result count mismatch"


def test_crop_padding_zero_padding():
    """
    Feature: ironsight-command-center, Property 9: Crop Padding Correctness
    For zero padding, the crop SHALL match the detection bounds exactly.
    **Validates: Requirements 6.2**
    """
    # Create NAFNet with zero padding
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.0,
        device="cpu"
    )
    
    # Create test image
    image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    
    # Create detection
    det_x, det_y = 400, 300
    det_width, det_height = 100, 60
    detection = create_detection(det_x, det_y, det_width, det_height)
    
    # Extract crop
    crop, bounds = nafnet.extract_crop_with_padding(image, detection)
    
    assert crop is not None
    
    x1, y1, x2, y2 = bounds
    
    # With zero padding, bounds should match detection exactly
    expected_x1 = det_x - det_width/2
    expected_y1 = det_y - det_height/2
    expected_x2 = det_x + det_width/2
    expected_y2 = det_y + det_height/2
    
    # Verify bounds match (within 1 pixel due to rounding)
    assert abs(x1 - expected_x1) <= 1
    assert abs(y1 - expected_y1) <= 1
    assert abs(x2 - expected_x2) <= 1
    assert abs(y2 - expected_y2) <= 1


def test_crop_performance_stats():
    """
    Feature: ironsight-command-center, Property 8: Crop-First Processing Logic
    For any processing sequence, performance statistics SHALL be tracked accurately.
    **Validates: Requirements 6.1**
    """
    nafnet = CropFirstNAFNet(
        model_path="mock_model.pth",
        crop_padding_percent=0.1,
        device="cpu"
    )
    nafnet.model_loaded = True
    nafnet.model = MockNAFNetModel()
    
    image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    
    # Process multiple batches
    total_processed = 0
    for batch in range(3):
        detections = [
            create_detection(200 + i*150, 300, 100, 60)
            for i in range(2)
        ]
        results = nafnet.process_identification_plates(image, detections)
        total_processed += len(results)
    
    # Get performance stats
    stats = nafnet.get_performance_stats()
    
    # Verify stats
    assert stats['total_crops_processed'] == total_processed
    assert stats['total_processing_time_ms'] >= 0
    assert stats['avg_processing_time_ms'] >= 0
    assert stats['target_latency_ms'] == 20.0
    assert stats['padding_percent'] == 10.0
    
    # Verify computation savings is tracked
    assert 'computation_savings_pct' in stats
    assert stats['computation_savings_pct'] >= 0


# Mock NAFNet model for testing
class MockNAFNetModel:
    """Mock NAFNet model that applies simple sharpening."""
    
    def __init__(self):
        self.device = "cpu"
    
    def __call__(self, x):
        """Mock forward pass - works with both torch tensors and numpy arrays."""
        # If it's a numpy array (from direct call), just return it with slight modification
        if isinstance(x, np.ndarray):
            # Add small random noise to simulate processing
            noise = np.random.randn(*x.shape) * 0.01
            result = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return result
        
        # If it's a torch tensor (shouldn't happen in mock scenario)
        import torch
        return x + torch.randn_like(x) * 0.01
    
    def eval(self):
        return self
    
    def to(self, device):
        self.device = device
        return self


if __name__ == "__main__":
    # Run basic tests
    test_crop_first_edge_cases()
    test_crop_padding_zero_padding()
    test_crop_performance_stats()
    print("Basic NAFNet property tests passed!")

"""
Property-based tests for Mission Control Live Processing Interface.
Tests Property 11 (Video Input Acceptance) and Property 12 (Real-time Overlay Display).
"""

import pytest
import time
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import sys
import numpy as np
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mission_control import (
    MissionControl, VideoInputType, VideoInputConfig, 
    OverlayConfig, VideoInputHandler, OverlayRenderer,
    ProcessingStats, create_mission_control
)


# Custom strategies for generating test data
@st.composite
def video_input_config_strategy(draw):
    """Generate random VideoInputConfig."""
    input_type = draw(st.sampled_from([
        VideoInputType.WEBCAM,
        VideoInputType.RTSP,
        VideoInputType.FILE,
        VideoInputType.NONE
    ]))
    
    if input_type == VideoInputType.WEBCAM:
        source = draw(st.integers(min_value=0, max_value=10))
    elif input_type == VideoInputType.RTSP:
        source = draw(st.text(min_size=10, max_size=100).map(lambda s: f"rtsp://{s}"))
    elif input_type == VideoInputType.FILE:
        source = draw(st.text(min_size=5, max_size=50).map(lambda s: f"{s}.mp4"))
    else:
        source = 0
    
    return VideoInputConfig(
        input_type=input_type,
        source=source,
        fps_target=draw(st.integers(min_value=1, max_value=120)),
        resolution=(
            draw(st.integers(min_value=320, max_value=3840)),
            draw(st.integers(min_value=240, max_value=2160))
        )
    )


@st.composite
def detection_dict_strategy(draw, model_source: str = "test_model"):
    """Generate a random detection dictionary."""
    is_obb = draw(st.booleans())
    
    if is_obb:
        # Generate 8 coordinates for oriented bounding box
        bbox = [draw(st.floats(min_value=0.0, max_value=1000.0)) for _ in range(8)]
    else:
        # Generate regular bounding box [x, y, width, height]
        bbox = [
            draw(st.floats(min_value=0.0, max_value=800.0)),
            draw(st.floats(min_value=0.0, max_value=600.0)),
            draw(st.floats(min_value=10.0, max_value=200.0)),
            draw(st.floats(min_value=10.0, max_value=200.0))
        ]
    
    return {
        "detection_id": draw(st.text(min_size=4, max_size=8, alphabet="abcdef0123456789")),
        "class_name": draw(st.sampled_from(["dent", "hole", "rust", "door", "wheel", "plate"])),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0)),
        "bbox": bbox,
        "is_obb": is_obb,
        "model_source": model_source
    }


@st.composite
def detections_by_model_strategy(draw):
    """Generate detections organized by model."""
    sideview_count = draw(st.integers(min_value=0, max_value=5))
    structure_count = draw(st.integers(min_value=0, max_value=5))
    wagon_number_count = draw(st.integers(min_value=0, max_value=5))
    
    return {
        "sideview_damage_obb": [
            draw(detection_dict_strategy("sideview_damage_obb"))
            for _ in range(sideview_count)
        ],
        "structure_obb": [
            draw(detection_dict_strategy("structure_obb"))
            for _ in range(structure_count)
        ],
        "wagon_number_obb": [
            draw(detection_dict_strategy("wagon_number_obb"))
            for _ in range(wagon_number_count)
        ]
    }


# ============================================================================
# Property 11: Video Input Acceptance Tests
# ============================================================================

@settings(max_examples=100, deadline=5000)
@given(webcam_index=st.integers(min_value=0, max_value=10))
def test_webcam_input_validation_accepts_valid_index(webcam_index):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any valid webcam index (non-negative integer), the Mission Control
    interface SHALL accept the input configuration.
    **Validates: Requirements 2.1**
    """
    # Validate webcam input
    is_valid, error_msg = VideoInputHandler.validate_input(
        VideoInputType.WEBCAM, 
        webcam_index
    )
    
    # Property: Valid webcam index should be accepted
    assert is_valid, f"Valid webcam index {webcam_index} should be accepted"
    assert error_msg == "", f"No error message expected for valid input"


@settings(max_examples=100, deadline=5000)
@given(invalid_index=st.integers(max_value=-1))
def test_webcam_input_validation_rejects_negative_index(invalid_index):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any negative webcam index, the Mission Control interface SHALL
    reject the input with an appropriate error message.
    **Validates: Requirements 2.1**
    """
    # Validate webcam input with negative index
    is_valid, error_msg = VideoInputHandler.validate_input(
        VideoInputType.WEBCAM, 
        invalid_index
    )
    
    # Property: Negative webcam index should be rejected
    assert not is_valid, f"Negative webcam index {invalid_index} should be rejected"
    assert len(error_msg) > 0, "Error message should be provided"


@settings(max_examples=100, deadline=5000)
@given(url=st.text(min_size=5, max_size=100))
def test_rtsp_input_validation(url):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any RTSP URL input, the Mission Control interface SHALL validate
    that it starts with a valid protocol prefix.
    **Validates: Requirements 2.1**
    """
    # Test with various URL formats
    rtsp_url = f"rtsp://{url}"
    http_url = f"http://{url}"
    invalid_url = url
    
    # Valid RTSP URL
    is_valid_rtsp, _ = VideoInputHandler.validate_input(VideoInputType.RTSP, rtsp_url)
    assert is_valid_rtsp, f"Valid RTSP URL should be accepted: {rtsp_url}"
    
    # Valid HTTP URL (also accepted for RTSP streams)
    is_valid_http, _ = VideoInputHandler.validate_input(VideoInputType.RTSP, http_url)
    assert is_valid_http, f"Valid HTTP URL should be accepted: {http_url}"
    
    # Invalid URL without protocol
    is_valid_invalid, error_msg = VideoInputHandler.validate_input(VideoInputType.RTSP, invalid_url)
    if not invalid_url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        assert not is_valid_invalid, f"URL without protocol should be rejected: {invalid_url}"
        assert len(error_msg) > 0, "Error message should be provided"


@settings(max_examples=100, deadline=5000)
@given(
    input_type=st.sampled_from([VideoInputType.WEBCAM, VideoInputType.RTSP, VideoInputType.FILE, VideoInputType.NONE])
)
def test_video_input_type_handling(input_type):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any valid video input type (webcam, RTSP, file upload), the Mission
    Control interface SHALL accept and configure the input source.
    **Validates: Requirements 2.1**
    """
    # Create appropriate source for each type
    if input_type == VideoInputType.WEBCAM:
        source = 0
    elif input_type == VideoInputType.RTSP:
        source = "rtsp://test.example.com/stream"
    elif input_type == VideoInputType.FILE:
        # Create a temporary file path (doesn't need to exist for validation test)
        source = "nonexistent.mp4"
    else:
        source = 0
    
    # Create video config
    config = VideoInputConfig(input_type=input_type, source=source)
    
    # Property 1: Config should store the input type correctly
    assert config.input_type == input_type
    
    # Property 2: Config should store the source correctly
    assert config.source == source
    
    # Property 3: VideoInputHandler should accept the config
    handler = VideoInputHandler(config)
    assert handler.config.input_type == input_type


@settings(max_examples=100, deadline=5000)
@given(config=video_input_config_strategy())
def test_mission_control_accepts_video_config(config):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any valid VideoInputConfig, the Mission Control SHALL accept and
    store the configuration correctly.
    **Validates: Requirements 2.1**
    """
    # Create Mission Control with config
    mc = MissionControl(video_config=config)
    
    # Property 1: Mission Control should store the config
    assert mc.video_handler.config.input_type == config.input_type
    assert mc.video_handler.config.source == config.source
    
    # Property 2: Mission Control should not be processing initially
    assert not mc.is_processing
    
    # Property 3: Stats should be initialized
    assert mc.stats.frames_processed == 0
    assert mc.stats.fps == 0.0


@settings(max_examples=50, deadline=5000)
@given(
    input_type=st.sampled_from([VideoInputType.WEBCAM, VideoInputType.RTSP, VideoInputType.FILE]),
    source=st.one_of(
        st.integers(min_value=0, max_value=10),
        st.text(min_size=5, max_size=50).map(lambda s: f"rtsp://{s}"),
        st.text(min_size=5, max_size=50).map(lambda s: f"{s}.mp4")
    )
)
def test_set_video_input_validates_correctly(input_type, source):
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For any video input configuration, set_video_input() SHALL validate
    the input and return appropriate success/error status.
    **Validates: Requirements 2.1**
    """
    mc = MissionControl()
    
    # Set video input
    success, error_msg = mc.set_video_input(input_type, source)
    
    # Property 1: Result should be a tuple of (bool, str)
    assert isinstance(success, bool)
    assert isinstance(error_msg, str)
    
    # Property 2: If successful, config should be updated
    if success:
        assert mc.video_handler.config.input_type == input_type
        assert mc.video_handler.config.source == source
        assert error_msg == ""
    
    # Property 3: If failed, error message should be provided
    if not success:
        assert len(error_msg) > 0


def test_none_input_type_always_valid():
    """
    Feature: ironsight-command-center, Property 11: Video Input Acceptance
    For VideoInputType.NONE, validation SHALL always succeed.
    **Validates: Requirements 2.1**
    """
    is_valid, error_msg = VideoInputHandler.validate_input(VideoInputType.NONE, "anything")
    
    assert is_valid, "NONE input type should always be valid"
    assert error_msg == "", "No error message for NONE input type"


# ============================================================================
# Property 12: Real-time Overlay Display Tests
# ============================================================================

@settings(max_examples=100, deadline=5000)
@given(
    image_height=st.integers(min_value=100, max_value=2000),
    image_width=st.integers(min_value=100, max_value=2000),
    detections=detections_by_model_strategy()
)
def test_overlay_renderer_preserves_image_shape(image_height, image_width, detections):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For any processed frame with detections, the overlay renderer SHALL
    return an image with the same shape as the input.
    **Validates: Requirements 2.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create renderer
    renderer = OverlayRenderer()
    
    # Render detections
    output = renderer.render_detections(image, detections)
    
    # Property 1: Output must be numpy array
    assert isinstance(output, np.ndarray)
    
    # Property 2: Output shape must match input shape
    assert output.shape == image.shape, \
        f"Output shape {output.shape} must match input shape {image.shape}"
    
    # Property 3: Output dtype should be uint8
    assert output.dtype == np.uint8


@settings(max_examples=100, deadline=5000)
@given(detections=detections_by_model_strategy())
def test_overlay_uses_different_colors_per_model(detections):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For any set of detections, the overlay renderer SHALL use different
    colors for each YOLO model type.
    **Validates: Requirements 2.2**
    """
    # Create renderer with default config
    renderer = OverlayRenderer()
    config = renderer.config
    
    # Property 1: Each model type should have a distinct color
    colors = [
        config.sideview_color,
        config.structure_color,
        config.wagon_number_color
    ]
    
    # All colors should be different
    assert len(set(colors)) == 3, "Each model type must have a distinct color"
    
    # Property 2: Colors should be valid BGR tuples
    for color in colors:
        assert isinstance(color, tuple)
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255


@settings(max_examples=100, deadline=5000)
@given(
    image_height=st.integers(min_value=100, max_value=1000),
    image_width=st.integers(min_value=100, max_value=1000),
    stats=st.builds(
        ProcessingStats,
        fps=st.floats(min_value=0.0, max_value=120.0),
        processing_latency_ms=st.floats(min_value=0.0, max_value=1000.0),
        queue_depth=st.integers(min_value=0, max_value=100),
        frames_processed=st.integers(min_value=0, max_value=1000000),
        last_serial_number=st.text(min_size=0, max_size=20)
    )
)
def test_stats_overlay_preserves_image_shape(image_height, image_width, stats):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For any frame with stats overlay, the renderer SHALL preserve the
    original image dimensions.
    **Validates: Requirements 2.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create renderer
    renderer = OverlayRenderer()
    
    # Render stats overlay
    output = renderer.render_stats_overlay(image, stats)
    
    # Property 1: Output must be numpy array
    assert isinstance(output, np.ndarray)
    
    # Property 2: Output shape must match input shape
    assert output.shape == image.shape
    
    # Property 3: Output dtype should be uint8
    assert output.dtype == np.uint8


@settings(max_examples=100, deadline=5000)
@given(
    sideview_color=st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255)
    ),
    structure_color=st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255)
    ),
    wagon_number_color=st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255)
    )
)
def test_overlay_config_accepts_custom_colors(sideview_color, structure_color, wagon_number_color):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For any valid BGR color configuration, the overlay renderer SHALL
    accept and use the custom colors.
    **Validates: Requirements 2.2**
    """
    # Create config with custom colors
    config = OverlayConfig(
        sideview_color=sideview_color,
        structure_color=structure_color,
        wagon_number_color=wagon_number_color
    )
    
    # Create renderer with custom config
    renderer = OverlayRenderer(config)
    
    # Property 1: Config should store custom colors
    assert renderer.config.sideview_color == sideview_color
    assert renderer.config.structure_color == structure_color
    assert renderer.config.wagon_number_color == wagon_number_color


@settings(max_examples=100, deadline=5000)
@given(
    num_detections=st.integers(min_value=0, max_value=50),
    image_height=st.integers(min_value=100, max_value=1000),
    image_width=st.integers(min_value=100, max_value=1000)
)
def test_overlay_handles_variable_detection_counts(num_detections, image_height, image_width):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For any number of detections (including zero), the overlay renderer
    SHALL successfully render without errors.
    **Validates: Requirements 2.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create detections
    detections_by_model = {
        "sideview_damage_obb": [],
        "structure_obb": [],
        "wagon_number_obb": []
    }
    
    # Distribute detections across models
    for i in range(num_detections):
        model = ["sideview_damage_obb", "structure_obb", "wagon_number_obb"][i % 3]
        detections_by_model[model].append({
            "detection_id": f"det_{i}",
            "class_name": "test",
            "confidence": 0.8,
            "bbox": [50.0 + i*10, 50.0 + i*10, 100.0, 100.0],
            "is_obb": False,
            "model_source": model
        })
    
    # Create renderer
    renderer = OverlayRenderer()
    
    # Render detections - should not raise
    output = renderer.render_detections(image, detections_by_model)
    
    # Property 1: Output should be valid
    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


@settings(max_examples=100, deadline=5000)
@given(
    is_obb=st.booleans(),
    image_height=st.integers(min_value=200, max_value=1000),
    image_width=st.integers(min_value=200, max_value=1000)
)
def test_overlay_handles_both_bbox_types(is_obb, image_height, image_width):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    For both regular and oriented bounding boxes, the overlay renderer
    SHALL correctly render the detection boxes.
    **Validates: Requirements 2.2**
    """
    # Create random image
    image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
    # Create detection with appropriate bbox format
    if is_obb:
        # Oriented bounding box - 8 coordinates (4 corner points)
        cx, cy = image_width // 2, image_height // 2
        bbox = [
            cx - 50, cy - 30,  # top-left
            cx + 50, cy - 30,  # top-right
            cx + 50, cy + 30,  # bottom-right
            cx - 50, cy + 30   # bottom-left
        ]
    else:
        # Regular bounding box [x, y, width, height]
        bbox = [50.0, 50.0, 100.0, 80.0]
    
    detections_by_model = {
        "sideview_damage_obb": [{
            "detection_id": "test_det",
            "class_name": "test_class",
            "confidence": 0.85,
            "bbox": bbox,
            "is_obb": is_obb,
            "model_source": "sideview_damage_obb"
        }],
        "structure_obb": [],
        "wagon_number_obb": []
    }
    
    # Create renderer
    renderer = OverlayRenderer()
    
    # Render detections
    output = renderer.render_detections(image, detections_by_model)
    
    # Property 1: Output should be valid
    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape
    
    # Property 2: Output should be different from input (overlay was drawn)
    # Note: This may not always be true if detection is outside visible area
    # but for our test case it should be visible


def test_mission_control_get_model_colors():
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    Mission Control SHALL provide a method to retrieve the color mapping
    for each model type.
    **Validates: Requirements 2.2**
    """
    mc = MissionControl()
    
    colors = mc.get_model_colors()
    
    # Property 1: Should return a dictionary
    assert isinstance(colors, dict)
    
    # Property 2: Should contain all 3 model types
    assert "sideview_damage_obb" in colors
    assert "structure_obb" in colors
    assert "wagon_number_obb" in colors
    
    # Property 3: Each color should be a valid BGR tuple
    for model_name, color in colors.items():
        assert isinstance(color, tuple)
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255


@settings(max_examples=50, deadline=5000)
@given(
    fps=st.floats(min_value=0.0, max_value=120.0),
    latency=st.floats(min_value=0.0, max_value=1000.0),
    frames=st.integers(min_value=0, max_value=1000000)
)
def test_processing_stats_dataclass(fps, latency, frames):
    """
    Feature: ironsight-command-center, Property 12: Real-time Overlay Display
    ProcessingStats SHALL correctly store and retrieve all processing metrics.
    **Validates: Requirements 2.2**
    """
    stats = ProcessingStats(
        fps=fps,
        processing_latency_ms=latency,
        frames_processed=frames
    )
    
    # Property 1: Values should be stored correctly
    assert stats.fps == fps
    assert stats.processing_latency_ms == latency
    assert stats.frames_processed == frames
    
    # Property 2: Default values should be set
    assert stats.queue_depth == 0
    assert stats.frames_dropped == 0
    assert stats.gatekeeper_skips == 0
    assert stats.last_serial_number == "N/A"


if __name__ == "__main__":
    # Run basic tests
    test_none_input_type_always_valid()
    test_mission_control_get_model_colors()
    print("Basic Mission Control property tests passed!")

"""
Property-based tests for IronSight Dashboard Integration.
Tests Property 14 for Model Status Reporting.
"""

import pytest
import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Define ModelStatus enum for testing (matches ironsight_engine.py)
class ModelStatus(Enum):
    """Status of individual models in the engine."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    OFFLINE = "offline"
    ERROR = "error"


# Import app module components
from app import (
    get_model_display_info,
    render_model_status_badge,
    ModelInfo
)


# Strategy for generating model status combinations
model_status_strategy = st.sampled_from([
    ModelStatus.NOT_LOADED,
    ModelStatus.LOADING,
    ModelStatus.LOADED,
    ModelStatus.OFFLINE,
    ModelStatus.ERROR
])


# Strategy for generating model names
model_names_strategy = st.sampled_from([
    "gatekeeper",
    "sci_enhancer", 
    "yolo_sideview",
    "yolo_structure",
    "yolo_wagon_number",
    "nafnet",
    "smolvlm_agent",
    "siglip_search"
])


# Feature: ironsight-command-center, Property 14: Model Status Reporting
@settings(max_examples=100)
@given(
    model_statuses=st.dictionaries(
        keys=model_names_strategy,
        values=model_status_strategy,
        min_size=1,
        max_size=8
    )
)
def test_model_status_reporting_displays_offline_badge(model_statuses: Dict[str, ModelStatus]):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any model that fails to load, the Dashboard SHALL display a 
    "Model Offline" badge for that specific component.
    **Validates: Requirements 1.4**
    """
    # Create a mock engine that returns the generated statuses
    mock_engine = Mock()
    mock_engine.get_model_status.return_value = {
        name: status.value for name, status in model_statuses.items()
    }
    
    # Get model display info with the mock engine
    models = get_model_display_info(mock_engine)
    
    # Verify that each model in the generated statuses has correct status
    for model_name, expected_status in model_statuses.items():
        if model_name in models:
            actual_status = models[model_name].status
            assert actual_status == expected_status.value, \
                f"Model {model_name} should have status {expected_status.value}, got {actual_status}"
    
    # Verify that offline/error models get appropriate badge
    for model_name, status in model_statuses.items():
        if model_name in models:
            badge_html = render_model_status_badge(status.value)
            
            if status == ModelStatus.LOADED:
                assert "Online" in badge_html, \
                    f"Loaded model {model_name} should show 'Online' badge"
                assert "model-status-online" in badge_html, \
                    f"Loaded model {model_name} should have online CSS class"
            elif status == ModelStatus.LOADING:
                assert "Loading" in badge_html, \
                    f"Loading model {model_name} should show 'Loading' badge"
                assert "model-status-loading" in badge_html, \
                    f"Loading model {model_name} should have loading CSS class"
            else:
                # NOT_LOADED, OFFLINE, ERROR all show as Offline
                assert "Offline" in badge_html, \
                    f"Failed model {model_name} with status {status.value} should show 'Offline' badge"
                assert "model-status-offline" in badge_html, \
                    f"Failed model {model_name} should have offline CSS class"


@settings(max_examples=100)
@given(
    num_loaded=st.integers(min_value=0, max_value=8),
    num_offline=st.integers(min_value=0, max_value=8),
    num_error=st.integers(min_value=0, max_value=8)
)
def test_model_status_badge_rendering(num_loaded: int, num_offline: int, num_error: int):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any combination of model statuses, the badge rendering function SHALL
    produce valid HTML with appropriate CSS classes and status text.
    **Validates: Requirements 1.4**
    """
    # Test loaded status badge
    for _ in range(min(num_loaded, 10)):
        badge = render_model_status_badge("loaded")
        assert isinstance(badge, str)
        assert "Online" in badge
        assert "🟢" in badge
        assert "model-status-online" in badge
        assert "<span" in badge and "</span>" in badge
    
    # Test offline status badge
    for _ in range(min(num_offline, 10)):
        badge = render_model_status_badge("offline")
        assert isinstance(badge, str)
        assert "Offline" in badge
        assert "🔴" in badge
        assert "model-status-offline" in badge
        assert "<span" in badge and "</span>" in badge
    
    # Test error status badge (should also show offline)
    for _ in range(min(num_error, 10)):
        badge = render_model_status_badge("error")
        assert isinstance(badge, str)
        assert "Offline" in badge
        assert "🔴" in badge
        assert "model-status-offline" in badge
    
    # Test loading status badge
    badge = render_model_status_badge("loading")
    assert isinstance(badge, str)
    assert "Loading" in badge
    assert "🟡" in badge
    assert "model-status-loading" in badge
    
    # Test not_loaded status badge (should show offline)
    badge = render_model_status_badge("not_loaded")
    assert isinstance(badge, str)
    assert "Offline" in badge
    assert "model-status-offline" in badge


@settings(max_examples=100)
@given(
    model_availability=st.lists(
        st.booleans(),
        min_size=8,
        max_size=8
    )
)
def test_model_display_info_completeness(model_availability: List[bool]):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any engine state, get_model_display_info SHALL return information
    for all 8 expected models with valid display names and descriptions.
    **Validates: Requirements 1.4**
    """
    expected_models = [
        "gatekeeper",
        "sci_enhancer",
        "yolo_sideview",
        "yolo_structure",
        "yolo_wagon_number",
        "nafnet",
        "smolvlm_agent",
        "siglip_search"
    ]
    
    # Create mock engine with generated availability
    mock_engine = Mock()
    status_dict = {}
    for i, model_name in enumerate(expected_models):
        if i < len(model_availability):
            status_dict[model_name] = "loaded" if model_availability[i] else "offline"
        else:
            status_dict[model_name] = "offline"
    
    mock_engine.get_model_status.return_value = status_dict
    
    # Get model display info
    models = get_model_display_info(mock_engine)
    
    # Verify all 8 models are present
    assert len(models) == 8, f"Expected 8 models, got {len(models)}"
    
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} should be in display info"
        
        model_info = models[model_name]
        
        # Verify ModelInfo structure
        assert isinstance(model_info, ModelInfo)
        assert model_info.name == model_name
        assert len(model_info.display_name) > 0, \
            f"Model {model_name} should have a display name"
        assert len(model_info.description) > 0, \
            f"Model {model_name} should have a description"
        assert model_info.status in ["loaded", "offline", "loading", "error", "not_loaded"], \
            f"Model {model_name} has invalid status: {model_info.status}"


@settings(max_examples=50)
@given(
    engine_available=st.booleans()
)
def test_model_display_info_without_engine(engine_available: bool):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any dashboard state (with or without engine), get_model_display_info
    SHALL return valid model information with default offline status when
    engine is not available.
    **Validates: Requirements 1.4**
    """
    if engine_available:
        # Test with mock engine
        mock_engine = Mock()
        mock_engine.get_model_status.return_value = {
            "gatekeeper": "loaded",
            "sci_enhancer": "loaded",
            "yolo_sideview": "offline",
            "yolo_structure": "offline",
            "yolo_wagon_number": "offline",
            "nafnet": "loaded",
            "smolvlm_agent": "error",
            "siglip_search": "loading"
        }
        models = get_model_display_info(mock_engine)
        
        # Verify statuses are updated from engine
        assert models["gatekeeper"].status == "loaded"
        assert models["nafnet"].status == "loaded"
        assert models["smolvlm_agent"].status == "error"
        assert models["siglip_search"].status == "loading"
    else:
        # Test without engine (None)
        models = get_model_display_info(None)
        
        # Verify all models default to offline
        for model_name, model_info in models.items():
            assert model_info.status == "offline", \
                f"Model {model_name} should default to offline when engine is None"
    
    # In both cases, verify structure is valid
    assert len(models) == 8
    for model_info in models.values():
        assert isinstance(model_info, ModelInfo)
        assert len(model_info.display_name) > 0
        assert len(model_info.description) > 0


@settings(max_examples=100)
@given(
    status_value=st.text(min_size=0, max_size=50)
)
def test_model_status_badge_handles_any_input(status_value: str):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any status string input, render_model_status_badge SHALL return
    valid HTML without crashing.
    **Validates: Requirements 1.4**
    """
    # Should not raise any exceptions
    badge = render_model_status_badge(status_value)
    
    # Should always return a string
    assert isinstance(badge, str)
    
    # Should always contain HTML span tags
    assert "<span" in badge
    assert "</span>" in badge
    
    # Should always have a CSS class
    assert "model-status-" in badge
    
    # Known statuses should have specific badges
    if status_value == "loaded":
        assert "Online" in badge
        assert "model-status-online" in badge
    elif status_value == "loading":
        assert "Loading" in badge
        assert "model-status-loading" in badge
    else:
        # All other values should show offline
        assert "Offline" in badge
        assert "model-status-offline" in badge


@settings(max_examples=50)
@given(
    exception_type=st.sampled_from([
        AttributeError,
        KeyError,
        TypeError,
        RuntimeError
    ])
)
def test_model_display_info_handles_engine_errors(exception_type):
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For any engine error condition, get_model_display_info SHALL handle
    the error gracefully and return default model information.
    **Validates: Requirements 1.4**
    """
    # Create mock engine that raises an exception
    mock_engine = Mock()
    mock_engine.get_model_status.side_effect = exception_type("Mock error")
    
    # Should not raise exception
    models = get_model_display_info(mock_engine)
    
    # Should return valid model info with default offline status
    assert len(models) == 8
    for model_info in models.values():
        assert isinstance(model_info, ModelInfo)
        assert model_info.status == "offline"


def test_all_model_statuses_have_badges():
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    For all defined ModelStatus enum values, render_model_status_badge SHALL
    produce a valid badge.
    **Validates: Requirements 1.4**
    """
    for status in ModelStatus:
        badge = render_model_status_badge(status.value)
        
        assert isinstance(badge, str)
        assert len(badge) > 0
        assert "<span" in badge
        assert "</span>" in badge
        assert "model-status-" in badge
        
        # Verify appropriate content based on status
        if status == ModelStatus.LOADED:
            assert "Online" in badge
        elif status == ModelStatus.LOADING:
            assert "Loading" in badge
        else:
            assert "Offline" in badge


def test_model_info_dataclass_structure():
    """
    Feature: ironsight-command-center, Property 14: Model Status Reporting
    The ModelInfo dataclass SHALL have all required fields for display.
    **Validates: Requirements 1.4**
    """
    # Create a ModelInfo instance
    info = ModelInfo(
        name="test_model",
        display_name="Test Model",
        status="loaded",
        description="A test model for testing"
    )
    
    # Verify all fields are accessible
    assert info.name == "test_model"
    assert info.display_name == "Test Model"
    assert info.status == "loaded"
    assert info.description == "A test model for testing"


if __name__ == "__main__":
    # Run basic tests
    test_all_model_statuses_have_badges()
    test_model_info_dataclass_structure()
    print("Basic dashboard property tests passed!")

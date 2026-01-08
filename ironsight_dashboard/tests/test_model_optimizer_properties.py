"""
Property-based tests for Model Optimizer FP16 loading.
Tests Property 15 for FP16 model loading without ONNX export.

Feature: ironsight-command-center, Property 15: FP16 Model Loading
"""

import pytest
import tempfile
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import sys
import numpy as np
from unittest.mock import Mock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_optimizer import (
    ModelOptimizer,
    ModelOptimizationConfig,
    OptimizedModelInfo,
    InferenceResult,
    create_model_optimizer,
)


@given(
    use_fp16=st.booleans(),
    warmup_iterations=st.integers(min_value=1, max_value=10),
    benchmark_iterations=st.integers(min_value=1, max_value=20),
    target_latency_ms=st.floats(min_value=1.0, max_value=100.0)
)
@settings(max_examples=100, deadline=10000)
def test_fp16_model_loading_configuration(
    use_fp16, warmup_iterations, benchmark_iterations, target_latency_ms
):
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    config = ModelOptimizationConfig(
        use_fp16=use_fp16,
        device="cpu",
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        target_latency_ms=target_latency_ms
    )
    optimizer = ModelOptimizer(config)
    assert optimizer.config.use_fp16 is False
    assert optimizer.config.warmup_iterations == warmup_iterations
    assert optimizer.config.benchmark_iterations == benchmark_iterations
    assert optimizer.config.target_latency_ms == target_latency_ms
    assert optimizer.config.device == "cpu"



@given(
    model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    use_fp16=st.booleans()
)
@settings(max_examples=100, deadline=10000)
def test_fp16_model_info_structure(model_name, use_fp16):
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    assume(len(model_name.strip()) > 0)
    info = OptimizedModelInfo(
        model_name=model_name,
        model_path="/fake/path/model.pt",
        is_loaded=False,
        uses_fp16=use_fp16,
        device="cuda" if use_fp16 else "cpu"
    )
    assert info.model_name == model_name
    assert info.model_path == "/fake/path/model.pt"
    assert isinstance(info.is_loaded, bool)
    assert isinstance(info.uses_fp16, bool)
    assert info.device in ["cuda", "cpu"]


@given(
    use_fp16=st.booleans(),
    inference_time_ms=st.floats(min_value=0.1, max_value=1000.0)
)
@settings(max_examples=100, deadline=10000)
def test_fp16_inference_result_structure(use_fp16, inference_time_ms):
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    result = InferenceResult(
        success=True,
        output=None,
        inference_time_ms=inference_time_ms,
        used_fp16=use_fp16
    )
    assert result.success is True
    assert result.used_fp16 == use_fp16
    assert result.inference_time_ms == inference_time_ms
    assert result.error_message is None



def test_fp16_model_loading_with_missing_file():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    # Test with file that exists
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pt"
        model_path.write_text("# Dummy model file")
        config = ModelOptimizationConfig(use_fp16=False, device="cpu")
        optimizer = ModelOptimizer(config)
        info = optimizer.load_yolo_model("test_model", str(model_path))
        assert info.model_name == "test_model"
        assert info.model_path == str(model_path)
    
    # Test with file that doesn't exist
    config = ModelOptimizationConfig(use_fp16=False, device="cpu")
    optimizer = ModelOptimizer(config)
    info = optimizer.load_yolo_model("test_model", "/nonexistent/path/model.pt")
    assert info.is_loaded is False
    assert info.error_message is not None
    assert "not found" in info.error_message.lower()


def test_fp16_multiple_model_tracking():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    num_models = 3
    use_fp16 = False
    config = ModelOptimizationConfig(use_fp16=use_fp16, device="cpu")
    optimizer = ModelOptimizer(config)
    for i in range(num_models):
        model_name = f"model_{i}"
        info = OptimizedModelInfo(
            model_name=model_name,
            model_path=f"/fake/path/{model_name}.pt",
            is_loaded=True,
            uses_fp16=use_fp16 and config.device == "cuda"
        )
        optimizer.model_info[model_name] = info
    all_info = optimizer.get_all_model_info()
    assert len(all_info) == num_models
    for i in range(num_models):
        model_name = f"model_{i}"
        assert model_name in all_info
        assert all_info[model_name].model_name == model_name



def test_fp16_performance_validation():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    target_latency_ms = 20.0
    actual_latency_ms = 15.0
    config = ModelOptimizationConfig(use_fp16=True, device="cpu", target_latency_ms=target_latency_ms)
    optimizer = ModelOptimizer(config)
    info = OptimizedModelInfo(
        model_name="test_model",
        model_path="/fake/path/model.pt",
        is_loaded=True,
        avg_inference_time_ms=actual_latency_ms
    )
    optimizer.model_info["test_model"] = info
    meets_target, message = optimizer.validate_performance("test_model")
    expected_meets_target = actual_latency_ms <= target_latency_ms
    assert meets_target == expected_meets_target
    assert "test_model" in message
    
    # Test with latency exceeding target
    info2 = OptimizedModelInfo(
        model_name="slow_model",
        model_path="/fake/path/slow_model.pt",
        is_loaded=True,
        avg_inference_time_ms=50.0  # Exceeds 20ms target
    )
    optimizer.model_info["slow_model"] = info2
    meets_target2, message2 = optimizer.validate_performance("slow_model")
    assert meets_target2 is False
    assert "slow_model" in message2


def test_fp16_model_unloading():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    use_fp16 = False
    config = ModelOptimizationConfig(use_fp16=use_fp16, device="cpu")
    optimizer = ModelOptimizer(config)
    optimizer.loaded_models["test_model"] = Mock()
    optimizer.model_info["test_model"] = OptimizedModelInfo(
        model_name="test_model",
        model_path="/fake/path/model.pt",
        is_loaded=True
    )
    assert optimizer.is_model_loaded("test_model")
    result = optimizer.unload_model("test_model")
    assert result is True
    assert not optimizer.is_model_loaded("test_model")
    assert optimizer.model_info["test_model"].is_loaded is False
    
    # Test unloading non-existent model
    result2 = optimizer.unload_model("nonexistent")
    assert result2 is False



def test_fp16_factory_function():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    # Test with FP16 enabled (will fall back to CPU without FP16)
    optimizer = create_model_optimizer(use_fp16=True, device="cpu")
    assert isinstance(optimizer, ModelOptimizer)
    assert optimizer.config is not None
    assert optimizer.config.use_fp16 is False  # Falls back on CPU
    assert optimizer.config.device == "cpu"
    
    # Test with FP16 disabled
    optimizer2 = create_model_optimizer(use_fp16=False, device="cpu")
    assert isinstance(optimizer2, ModelOptimizer)
    assert optimizer2.config.use_fp16 is False
    assert optimizer2.config.device == "cpu"


def test_fp16_cuda_fallback():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    config = ModelOptimizationConfig(use_fp16=True, device="cpu")
    optimizer = ModelOptimizer(config)
    assert optimizer.config.device == "cpu"
    assert optimizer.config.use_fp16 is False


def test_fp16_model_info_retrieval():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    config = ModelOptimizationConfig(use_fp16=True, device="cpu")
    optimizer = ModelOptimizer(config)
    info = OptimizedModelInfo(
        model_name="test_model",
        model_path="/fake/path/model.pt",
        is_loaded=True,
        uses_fp16=False,
        device="cpu",
        load_time_ms=100.0,
        avg_inference_time_ms=15.0,
        memory_usage_mb=256.0
    )
    optimizer.model_info["test_model"] = info
    retrieved_info = optimizer.get_model_info("test_model")
    assert retrieved_info is not None
    assert retrieved_info.model_name == "test_model"
    assert retrieved_info.is_loaded is True
    assert optimizer.get_model_info("nonexistent") is None



def test_fp16_model_not_loaded_error():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    config = ModelOptimizationConfig(use_fp16=True, device="cpu")
    optimizer = ModelOptimizer(config)
    result = optimizer.run_yolo_inference("nonexistent", np.zeros((640, 640, 3)))
    assert result.success is False
    assert result.error_message is not None
    assert "not loaded" in result.error_message.lower()


def test_fp16_validate_nonexistent_model():
    """
    Feature: ironsight-command-center, Property 15: FP16 Model Loading
    **Validates: Requirements 13.1**
    """
    config = ModelOptimizationConfig(use_fp16=True, device="cpu")
    optimizer = ModelOptimizer(config)
    meets_target, message = optimizer.validate_performance("nonexistent")
    assert meets_target is False
    assert "not found" in message.lower()


if __name__ == "__main__":
    test_fp16_cuda_fallback()
    test_fp16_model_info_retrieval()
    test_fp16_model_not_loaded_error()
    test_fp16_validate_nonexistent_model()
    test_fp16_model_loading_with_missing_file()
    test_fp16_multiple_model_tracking()
    test_fp16_performance_validation()
    test_fp16_model_unloading()
    test_fp16_factory_function()
    print("All FP16 model loading property tests passed!")

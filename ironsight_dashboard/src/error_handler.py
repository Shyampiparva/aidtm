"""
Error Handler for IronSight Command Center

Provides comprehensive error handling including:
- Model loading failure handling with mock model fallbacks
- Timeout handling for all AI processing stages
- User-friendly error messages and recovery suggestions
- Automatic retry logic for transient failures

Requirements: 1.4, 11.5
"""

import logging
import time
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Callable, TypeVar, Generic, 
    Tuple, Union
)
import numpy as np


T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    MODEL_LOADING = "model_loading"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    GPU = "gpu"
    INPUT_VALIDATION = "input_validation"
    PROCESSING = "processing"
    NETWORK = "network"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Possible recovery actions."""
    RETRY = "retry"
    USE_FALLBACK = "use_fallback"
    SKIP = "skip"
    ABORT = "abort"
    NOTIFY_USER = "notify_user"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    stage_name: str
    recovery_action: RecoveryAction
    recovery_successful: bool
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "stage_name": self.stage_name,
            "recovery_action": self.recovery_action.value,
            "recovery_successful": self.recovery_successful,
            "context": self.context
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    retryable_exceptions: Tuple[type, ...] = (Exception,)


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    timeout_ms: float
    fallback_value: Any = None
    raise_on_timeout: bool = False


class Result(Generic[T]):
    """Result wrapper for operations that may fail."""
    
    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[Exception] = None,
        is_fallback: bool = False
    ):
        self._value = value
        self._error = error
        self._is_fallback = is_fallback
    
    @property
    def is_success(self) -> bool:
        return self._error is None
    
    @property
    def is_failure(self) -> bool:
        return self._error is not None
    
    @property
    def is_fallback(self) -> bool:
        return self._is_fallback
    
    @property
    def value(self) -> T:
        if self._error:
            raise self._error
        return self._value
    
    @property
    def error(self) -> Optional[Exception]:
        return self._error
    
    def get_or_default(self, default: T) -> T:
        return self._value if self.is_success else default
    
    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        return cls(value=value)
    
    @classmethod
    def failure(cls, error: Exception) -> 'Result[T]':
        return cls(error=error)
    
    @classmethod
    def fallback(cls, value: T) -> 'Result[T]':
        return cls(value=value, is_fallback=True)


class ErrorHandler:
    """
    Comprehensive error handling system for IronSight.
    
    Features:
    - Model loading failure handling with mock fallbacks
    - Timeout handling for AI processing stages
    - User-friendly error messages
    - Automatic retry logic for transient failures
    """
    
    # User-friendly error messages
    ERROR_MESSAGES = {
        ErrorCategory.MODEL_LOADING: {
            "title": "Model Loading Failed",
            "description": "One or more AI models could not be loaded.",
            "suggestions": [
                "Check if model files exist in the expected location",
                "Verify you have sufficient GPU memory",
                "Try restarting the application",
                "Check the logs for detailed error information"
            ]
        },
        ErrorCategory.TIMEOUT: {
            "title": "Processing Timeout",
            "description": "AI processing took longer than expected.",
            "suggestions": [
                "Try processing a smaller image",
                "Check GPU utilization and temperature",
                "Consider reducing the number of active models",
                "Restart the application if the issue persists"
            ]
        },
        ErrorCategory.MEMORY: {
            "title": "Memory Error",
            "description": "Insufficient memory for processing.",
            "suggestions": [
                "Close other applications to free memory",
                "Reduce input image resolution",
                "Enable memory optimization in settings",
                "Consider using a system with more GPU memory"
            ]
        },
        ErrorCategory.GPU: {
            "title": "GPU Error",
            "description": "GPU processing encountered an error.",
            "suggestions": [
                "Check if CUDA is properly installed",
                "Verify GPU drivers are up to date",
                "Try switching to CPU mode",
                "Check GPU temperature and cooling"
            ]
        },
        ErrorCategory.INPUT_VALIDATION: {
            "title": "Invalid Input",
            "description": "The provided input is not valid.",
            "suggestions": [
                "Ensure the image format is supported (JPG, PNG)",
                "Check that the image is not corrupted",
                "Verify the image dimensions are reasonable",
                "Try with a different input file"
            ]
        },
        ErrorCategory.PROCESSING: {
            "title": "Processing Error",
            "description": "An error occurred during AI processing.",
            "suggestions": [
                "Try processing the image again",
                "Check if the input is valid",
                "Review the logs for specific error details",
                "Contact support if the issue persists"
            ]
        },
        ErrorCategory.NETWORK: {
            "title": "Network Error",
            "description": "Network communication failed.",
            "suggestions": [
                "Check your internet connection",
                "Verify the RTSP stream URL is correct",
                "Try reconnecting to the video source",
                "Check firewall settings"
            ]
        },
        ErrorCategory.UNKNOWN: {
            "title": "Unknown Error",
            "description": "An unexpected error occurred.",
            "suggestions": [
                "Try restarting the application",
                "Check the logs for error details",
                "Report this issue if it persists"
            ]
        }
    }
    
    def __init__(self, max_error_history: int = 1000):
        """
        Initialize error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorRecord] = []
        self.max_error_history = max_error_history
        self._lock = threading.Lock()
        
        # Fallback models registry
        self._fallback_models: Dict[str, Any] = {}
        
        # Error counts by category
        self._error_counts: Dict[ErrorCategory, int] = {
            cat: 0 for cat in ErrorCategory
        }
        
        # Thread pool for timeout handling
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("Error handler initialized")
    
    def classify_error(self, exception: Exception) -> ErrorCategory:
        """
        Classify an exception into an error category.
        
        Args:
            exception: The exception to classify
            
        Returns:
            ErrorCategory for the exception
        """
        exception_type = type(exception).__name__
        exception_msg = str(exception).lower()
        
        # Check for specific exception types
        if "cuda" in exception_msg or "gpu" in exception_msg:
            return ErrorCategory.GPU
        
        if "memory" in exception_msg or "oom" in exception_msg:
            return ErrorCategory.MEMORY
        
        if "timeout" in exception_msg or isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT
        
        if "load" in exception_msg or "model" in exception_msg:
            return ErrorCategory.MODEL_LOADING
        
        if "connection" in exception_msg or "network" in exception_msg:
            return ErrorCategory.NETWORK
        
        if "invalid" in exception_msg or "validation" in exception_msg:
            return ErrorCategory.INPUT_VALIDATION
        
        if exception_type in ("FileNotFoundError", "ModuleNotFoundError"):
            return ErrorCategory.MODEL_LOADING
        
        if exception_type in ("ValueError", "TypeError"):
            return ErrorCategory.INPUT_VALIDATION
        
        return ErrorCategory.UNKNOWN
    
    def get_user_message(self, category: ErrorCategory) -> Dict[str, Any]:
        """
        Get user-friendly error message for a category.
        
        Args:
            category: Error category
            
        Returns:
            Dictionary with title, description, and suggestions
        """
        return self.ERROR_MESSAGES.get(category, self.ERROR_MESSAGES[ErrorCategory.UNKNOWN])
    
    def record_error(
        self,
        exception: Exception,
        stage_name: str,
        recovery_action: RecoveryAction,
        recovery_successful: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """
        Record an error occurrence.
        
        Args:
            exception: The exception that occurred
            stage_name: Name of the processing stage
            recovery_action: Action taken to recover
            recovery_successful: Whether recovery was successful
            context: Additional context information
            
        Returns:
            ErrorRecord for the error
        """
        category = self.classify_error(exception)
        
        record = ErrorRecord(
            timestamp=datetime.now(),
            category=category,
            message=self.get_user_message(category)["title"],
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stage_name=stage_name,
            recovery_action=recovery_action,
            recovery_successful=recovery_successful,
            context=context or {}
        )
        
        with self._lock:
            self.error_history.append(record)
            self._error_counts[category] += 1
            
            # Trim history if needed
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]
        
        # Log the error
        self.logger.error(
            f"Error in {stage_name}: {type(exception).__name__}: {exception} "
            f"(Recovery: {recovery_action.value}, Success: {recovery_successful})"
        )
        
        return record
    
    def register_fallback_model(self, model_name: str, fallback: Any) -> None:
        """
        Register a fallback model for when the real model fails.
        
        Args:
            model_name: Name of the model
            fallback: Fallback model instance
        """
        self._fallback_models[model_name] = fallback
        self.logger.info(f"Registered fallback for model: {model_name}")
    
    def get_fallback_model(self, model_name: str) -> Optional[Any]:
        """
        Get fallback model for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Fallback model or None if not registered
        """
        return self._fallback_models.get(model_name)
    
    def with_retry(
        self,
        func: Callable[..., T],
        config: Optional[RetryConfig] = None,
        stage_name: str = "unknown"
    ) -> Callable[..., Result[T]]:
        """
        Wrap a function with retry logic.
        
        Args:
            func: Function to wrap
            config: Retry configuration
            stage_name: Name of the processing stage
            
        Returns:
            Wrapped function that returns Result
        """
        config = config or RetryConfig()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Result[T]:
            last_exception = None
            delay_ms = config.initial_delay_ms
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return Result.success(result)
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        self.logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {stage_name}: {e}"
                        )
                        time.sleep(delay_ms / 1000.0)
                        
                        if config.exponential_backoff:
                            delay_ms = min(
                                delay_ms * config.backoff_multiplier,
                                config.max_delay_ms
                            )
            
            # All retries failed
            self.record_error(
                last_exception,
                stage_name,
                RecoveryAction.RETRY,
                recovery_successful=False
            )
            return Result.failure(last_exception)
        
        return wrapper
    
    def with_timeout(
        self,
        func: Callable[..., T],
        config: TimeoutConfig,
        stage_name: str = "unknown"
    ) -> Callable[..., Result[T]]:
        """
        Wrap a function with timeout handling.
        
        Args:
            func: Function to wrap
            config: Timeout configuration
            stage_name: Name of the processing stage
            
        Returns:
            Wrapped function that returns Result
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Result[T]:
            future = self._executor.submit(func, *args, **kwargs)
            
            try:
                result = future.result(timeout=config.timeout_ms / 1000.0)
                return Result.success(result)
                
            except FuturesTimeoutError:
                future.cancel()
                
                error = TimeoutError(
                    f"{stage_name} timed out after {config.timeout_ms}ms"
                )
                
                self.record_error(
                    error,
                    stage_name,
                    RecoveryAction.USE_FALLBACK if config.fallback_value is not None else RecoveryAction.SKIP,
                    recovery_successful=config.fallback_value is not None
                )
                
                if config.fallback_value is not None:
                    return Result.fallback(config.fallback_value)
                
                if config.raise_on_timeout:
                    return Result.failure(error)
                
                return Result.failure(error)
                
            except Exception as e:
                self.record_error(
                    e,
                    stage_name,
                    RecoveryAction.ABORT,
                    recovery_successful=False
                )
                return Result.failure(e)
        
        return wrapper
    
    def with_fallback(
        self,
        func: Callable[..., T],
        fallback_func: Callable[..., T],
        stage_name: str = "unknown"
    ) -> Callable[..., Result[T]]:
        """
        Wrap a function with fallback on failure.
        
        Args:
            func: Primary function to call
            fallback_func: Fallback function if primary fails
            stage_name: Name of the processing stage
            
        Returns:
            Wrapped function that returns Result
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Result[T]:
            try:
                result = func(*args, **kwargs)
                return Result.success(result)
                
            except Exception as primary_error:
                self.logger.warning(
                    f"Primary function failed for {stage_name}, trying fallback: {primary_error}"
                )
                
                try:
                    fallback_result = fallback_func(*args, **kwargs)
                    
                    self.record_error(
                        primary_error,
                        stage_name,
                        RecoveryAction.USE_FALLBACK,
                        recovery_successful=True
                    )
                    
                    return Result.fallback(fallback_result)
                    
                except Exception as fallback_error:
                    self.record_error(
                        primary_error,
                        stage_name,
                        RecoveryAction.USE_FALLBACK,
                        recovery_successful=False,
                        context={"fallback_error": str(fallback_error)}
                    )
                    return Result.failure(primary_error)
        
        return wrapper
    
    def safe_execute(
        self,
        func: Callable[..., T],
        *args,
        stage_name: str = "unknown",
        timeout_ms: Optional[float] = None,
        fallback_value: Optional[T] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Result[T]:
        """
        Safely execute a function with all error handling features.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            stage_name: Name of the processing stage
            timeout_ms: Optional timeout in milliseconds
            fallback_value: Optional fallback value on failure
            retry_config: Optional retry configuration
            **kwargs: Keyword arguments for func
            
        Returns:
            Result wrapping the function output or error
        """
        # Apply retry wrapper if configured
        if retry_config:
            func = self.with_retry(func, retry_config, stage_name)
            # Now func returns Result, so we need to unwrap
            result = func(*args, **kwargs)
            if result.is_failure:
                if fallback_value is not None:
                    return Result.fallback(fallback_value)
                return result
            func = lambda *a, **k: result.value
        
        # Apply timeout wrapper if configured
        if timeout_ms:
            timeout_config = TimeoutConfig(
                timeout_ms=timeout_ms,
                fallback_value=fallback_value
            )
            wrapped = self.with_timeout(func, timeout_config, stage_name)
            return wrapped(*args, **kwargs)
        
        # Direct execution with basic error handling
        try:
            result = func(*args, **kwargs)
            return Result.success(result)
        except Exception as e:
            self.record_error(
                e,
                stage_name,
                RecoveryAction.USE_FALLBACK if fallback_value is not None else RecoveryAction.ABORT,
                recovery_successful=fallback_value is not None
            )
            if fallback_value is not None:
                return Result.fallback(fallback_value)
            return Result.failure(e)
    
    def get_error_counts(self) -> Dict[ErrorCategory, int]:
        """Get error counts by category."""
        with self._lock:
            return dict(self._error_counts)
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorRecord]:
        """Get most recent errors."""
        with self._lock:
            return self.error_history[-count:]
    
    def get_errors_by_stage(self, stage_name: str) -> List[ErrorRecord]:
        """Get errors for a specific stage."""
        with self._lock:
            return [e for e in self.error_history if e.stage_name == stage_name]
    
    def clear_history(self) -> None:
        """Clear error history."""
        with self._lock:
            self.error_history.clear()
            self._error_counts = {cat: 0 for cat in ErrorCategory}
        self.logger.info("Error history cleared")
    
    def shutdown(self) -> None:
        """Shutdown the error handler."""
        self._executor.shutdown(wait=False)
        self.logger.info("Error handler shutdown")


# ============================================================================
# Mock Models for Fallback
# ============================================================================

class MockGatekeeperFallback:
    """Fallback gatekeeper that always passes frames through."""
    
    def predict(self, thumbnail: np.ndarray) -> Tuple[bool, bool]:
        """Always return (True, False) - wagon present, not blurry."""
        return True, False


class MockYOLOFallback:
    """Fallback YOLO detector that returns empty detections."""
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Return empty detection list."""
        return []


class MockNAFNetFallback:
    """Fallback NAFNet that returns the input unchanged."""
    
    def deblur(self, crop: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return crop


class MockSCIFallback:
    """Fallback SCI enhancer that returns the input unchanged."""
    
    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return input unchanged with info dict."""
        return image, {"enhanced": False, "skipped": True, "reason": "fallback"}


class MockSmolVLMFallback:
    """Fallback SmolVLM that returns placeholder text."""
    
    def analyze(self, image: np.ndarray, prompt: str) -> str:
        """Return placeholder response."""
        return "[OCR unavailable - model offline]"


def create_error_handler() -> ErrorHandler:
    """
    Factory function to create a configured error handler with fallbacks.
    
    Returns:
        Configured ErrorHandler instance
    """
    handler = ErrorHandler()
    
    # Register default fallback models
    handler.register_fallback_model("gatekeeper", MockGatekeeperFallback())
    handler.register_fallback_model("yolo_sideview", MockYOLOFallback())
    handler.register_fallback_model("yolo_structure", MockYOLOFallback())
    handler.register_fallback_model("yolo_wagon_number", MockYOLOFallback())
    handler.register_fallback_model("nafnet", MockNAFNetFallback())
    handler.register_fallback_model("sci_enhancer", MockSCIFallback())
    handler.register_fallback_model("smolvlm", MockSmolVLMFallback())
    
    return handler


# ============================================================================
# Decorator versions for convenience
# ============================================================================

def with_error_handling(
    stage_name: str,
    timeout_ms: Optional[float] = None,
    fallback_value: Any = None,
    max_retries: int = 0
):
    """
    Decorator for adding error handling to functions.
    
    Args:
        stage_name: Name of the processing stage
        timeout_ms: Optional timeout in milliseconds
        fallback_value: Optional fallback value on failure
        max_retries: Number of retries (0 = no retry)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Result[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Result[T]:
            handler = ErrorHandler()
            
            retry_config = RetryConfig(max_retries=max_retries) if max_retries > 0 else None
            
            return handler.safe_execute(
                func,
                *args,
                stage_name=stage_name,
                timeout_ms=timeout_ms,
                fallback_value=fallback_value,
                retry_config=retry_config,
                **kwargs
            )
        
        return wrapper
    
    return decorator


# ============================================================================
# Model Loading Manager
# ============================================================================

class ModelLoadingManager:
    """
    Centralized manager for model loading with fallback support.
    
    Handles graceful degradation when models fail to load by providing
    mock model fallbacks that return safe defaults.
    
    Requirements: 1.4, 11.5
    """
    
    # Default timeout configurations for each model type (in milliseconds)
    DEFAULT_TIMEOUTS = {
        "gatekeeper": 5000.0,      # 5 seconds for loading
        "sci_enhancer": 10000.0,   # 10 seconds
        "yolo_sideview": 15000.0,  # 15 seconds
        "yolo_structure": 15000.0,
        "yolo_wagon_number": 15000.0,
        "nafnet": 20000.0,         # 20 seconds (larger model)
        "smolvlm": 60000.0,        # 60 seconds (VLM model)
        "siglip": 30000.0,         # 30 seconds
    }
    
    # Transient exceptions that should trigger retry
    TRANSIENT_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        OSError,  # Includes file system errors
    )
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize the model loading manager.
        
        Args:
            error_handler: Optional error handler instance
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = error_handler or create_error_handler()
        self._loaded_models: Dict[str, Any] = {}
        self._model_status: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def load_model(
        self,
        model_name: str,
        loader_func: Callable[[], Any],
        timeout_ms: Optional[float] = None,
        max_retries: int = 2
    ) -> Tuple[Optional[Any], bool]:
        """
        Load a model with error handling and fallback support.
        
        Args:
            model_name: Name of the model
            loader_func: Function that loads and returns the model
            timeout_ms: Optional timeout override
            max_retries: Number of retries for transient failures
            
        Returns:
            Tuple of (model_or_fallback, is_fallback)
        """
        timeout = timeout_ms or self.DEFAULT_TIMEOUTS.get(model_name, 30000.0)
        
        # Configure retry for transient failures
        retry_config = RetryConfig(
            max_retries=max_retries,
            initial_delay_ms=500.0,
            max_delay_ms=5000.0,
            exponential_backoff=True,
            retryable_exceptions=self.TRANSIENT_EXCEPTIONS
        )
        
        # Attempt to load the model
        result = self.error_handler.safe_execute(
            loader_func,
            stage_name=f"load_{model_name}",
            timeout_ms=timeout,
            retry_config=retry_config
        )
        
        with self._lock:
            if result.is_success:
                self._loaded_models[model_name] = result.value
                self._model_status[model_name] = "loaded"
                self.logger.info(f"Model {model_name} loaded successfully")
                return result.value, False
            else:
                # Use fallback model
                fallback = self.error_handler.get_fallback_model(model_name)
                if fallback:
                    self._loaded_models[model_name] = fallback
                    self._model_status[model_name] = "offline"
                    self.logger.warning(
                        f"Model {model_name} failed to load, using fallback. "
                        f"Error: {result.error}"
                    )
                    return fallback, True
                else:
                    self._model_status[model_name] = "error"
                    self.logger.error(
                        f"Model {model_name} failed to load and no fallback available. "
                        f"Error: {result.error}"
                    )
                    return None, False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        with self._lock:
            return self._loaded_models.get(model_name)
    
    def get_status(self, model_name: str) -> str:
        """Get the status of a model."""
        with self._lock:
            return self._model_status.get(model_name, "not_loaded")
    
    def get_all_status(self) -> Dict[str, str]:
        """Get status of all models."""
        with self._lock:
            return dict(self._model_status)
    
    def is_using_fallback(self, model_name: str) -> bool:
        """Check if a model is using its fallback."""
        with self._lock:
            return self._model_status.get(model_name) == "offline"


# ============================================================================
# Stage Timeout Manager
# ============================================================================

class StageTimeoutManager:
    """
    Manages timeout configurations for all AI processing stages.
    
    Provides centralized timeout handling with stage-specific configurations
    and automatic fallback behavior.
    
    Requirements: 11.5
    """
    
    # Default stage timeouts (in milliseconds)
    DEFAULT_STAGE_TIMEOUTS = {
        "gatekeeper": 0.5,          # 0.5ms for pre-filtering
        "sci_enhancer": 0.5,        # 0.5ms for enhancement
        "yolo_sideview": 7.0,       # 7ms per YOLO model
        "yolo_structure": 7.0,
        "yolo_wagon_number": 7.0,
        "yolo_combined": 20.0,      # 20ms for all YOLO models
        "nafnet": 20.0,             # 20ms for deblurring
        "smolvlm_ocr": 30000.0,     # 30 seconds for VLM OCR
        "smolvlm_damage": 30000.0,  # 30 seconds for damage assessment
        "siglip_embed": 100.0,      # 100ms for embedding generation
        "siglip_search": 500.0,     # 500ms for search
        "total_frame": 16.67,       # ~60 FPS target
    }
    
    def __init__(
        self,
        custom_timeouts: Optional[Dict[str, float]] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize the stage timeout manager.
        
        Args:
            custom_timeouts: Optional custom timeout overrides
            error_handler: Optional error handler instance
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = error_handler or create_error_handler()
        
        # Merge default and custom timeouts
        self.timeouts = {**self.DEFAULT_STAGE_TIMEOUTS}
        if custom_timeouts:
            self.timeouts.update(custom_timeouts)
    
    def get_timeout(self, stage_name: str) -> float:
        """Get timeout for a stage in milliseconds."""
        return self.timeouts.get(stage_name, 1000.0)  # Default 1 second
    
    def set_timeout(self, stage_name: str, timeout_ms: float) -> None:
        """Set timeout for a stage."""
        self.timeouts[stage_name] = timeout_ms
    
    def execute_with_timeout(
        self,
        stage_name: str,
        func: Callable[..., T],
        *args,
        fallback_value: Optional[T] = None,
        **kwargs
    ) -> Result[T]:
        """
        Execute a function with stage-specific timeout.
        
        Args:
            stage_name: Name of the processing stage
            func: Function to execute
            *args: Positional arguments
            fallback_value: Optional fallback value on timeout
            **kwargs: Keyword arguments
            
        Returns:
            Result wrapping the function output or error
        """
        timeout_ms = self.get_timeout(stage_name)
        
        return self.error_handler.safe_execute(
            func,
            *args,
            stage_name=stage_name,
            timeout_ms=timeout_ms,
            fallback_value=fallback_value,
            **kwargs
        )


# ============================================================================
# Transient Failure Detector
# ============================================================================

class TransientFailureDetector:
    """
    Detects transient failures that should trigger automatic retry.
    
    Analyzes exceptions to determine if they are likely transient
    (temporary) failures that may succeed on retry.
    
    Requirements: 11.5
    """
    
    # Exception types that are typically transient
    TRANSIENT_EXCEPTION_TYPES = {
        "ConnectionError",
        "TimeoutError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "BrokenPipeError",
        "TemporaryError",
        "ResourceExhaustedError",
    }
    
    # Keywords in error messages that indicate transient failures
    TRANSIENT_KEYWORDS = [
        "timeout",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "resource temporarily",
        "try again",
        "retry",
        "busy",
        "overloaded",
        "rate limit",
        "throttl",
        "temporary",
        "transient",
    ]
    
    # Keywords that indicate permanent failures (should not retry)
    PERMANENT_KEYWORDS = [
        "not found",
        "does not exist",
        "invalid",
        "permission denied",
        "access denied",
        "unauthorized",
        "forbidden",
        "corrupt",
        "malformed",
        "unsupported",
    ]
    
    @classmethod
    def is_transient(cls, exception: Exception) -> bool:
        """
        Determine if an exception represents a transient failure.
        
        Args:
            exception: The exception to analyze
            
        Returns:
            True if the failure is likely transient
        """
        exception_type = type(exception).__name__
        exception_msg = str(exception).lower()
        
        # Check exception type
        if exception_type in cls.TRANSIENT_EXCEPTION_TYPES:
            return True
        
        # Check for permanent failure keywords first
        for keyword in cls.PERMANENT_KEYWORDS:
            if keyword in exception_msg:
                return False
        
        # Check for transient failure keywords
        for keyword in cls.TRANSIENT_KEYWORDS:
            if keyword in exception_msg:
                return True
        
        # Check for specific exception base classes
        if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
            # OSError can be transient (e.g., too many open files)
            # but also permanent (e.g., file not found)
            if "no such file" in exception_msg or "not found" in exception_msg:
                return False
            return True
        
        return False
    
    @classmethod
    def get_retry_config(
        cls,
        exception: Exception,
        base_config: Optional[RetryConfig] = None
    ) -> Optional[RetryConfig]:
        """
        Get appropriate retry configuration for an exception.
        
        Args:
            exception: The exception that occurred
            base_config: Optional base configuration to modify
            
        Returns:
            RetryConfig if retry is appropriate, None otherwise
        """
        if not cls.is_transient(exception):
            return None
        
        config = base_config or RetryConfig()
        
        # Adjust retry parameters based on exception type
        exception_msg = str(exception).lower()
        
        if "rate limit" in exception_msg or "throttl" in exception_msg:
            # Rate limiting - use longer delays
            return RetryConfig(
                max_retries=config.max_retries,
                initial_delay_ms=2000.0,  # Start with 2 seconds
                max_delay_ms=30000.0,     # Up to 30 seconds
                exponential_backoff=True,
                backoff_multiplier=2.0
            )
        
        if "timeout" in exception_msg:
            # Timeout - moderate delays
            return RetryConfig(
                max_retries=config.max_retries,
                initial_delay_ms=500.0,
                max_delay_ms=5000.0,
                exponential_backoff=True,
                backoff_multiplier=1.5
            )
        
        # Default transient failure config
        return config


# ============================================================================
# UI Integration Helpers
# ============================================================================

class ErrorDisplayHelper:
    """
    Helper class for displaying errors in the UI.
    
    Provides formatted error messages, suggestions, and recovery
    options suitable for display in Streamlit or other UIs.
    
    Requirements: 11.5
    """
    
    @staticmethod
    def format_error_for_ui(error_record: ErrorRecord) -> Dict[str, Any]:
        """
        Format an error record for UI display.
        
        Args:
            error_record: The error record to format
            
        Returns:
            Dictionary with formatted error information
        """
        error_handler = ErrorHandler()
        user_message = error_handler.get_user_message(error_record.category)
        
        return {
            "title": user_message["title"],
            "description": user_message["description"],
            "suggestions": user_message["suggestions"],
            "technical_details": {
                "exception_type": error_record.exception_type,
                "exception_message": error_record.exception_message,
                "stage": error_record.stage_name,
                "timestamp": error_record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "recovery": {
                "action_taken": error_record.recovery_action.value,
                "successful": error_record.recovery_successful,
            },
            "severity": ErrorDisplayHelper._get_severity(error_record.category),
        }
    
    @staticmethod
    def format_errors_for_streamlit(
        error_handler: 'ErrorHandler',
        max_errors: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Format recent errors for Streamlit display.
        
        Args:
            error_handler: ErrorHandler instance
            max_errors: Maximum number of errors to return
            
        Returns:
            List of formatted error dictionaries
        """
        recent_errors = error_handler.get_recent_errors(max_errors)
        return [ErrorDisplayHelper.format_error_for_ui(e) for e in recent_errors]
    
    @staticmethod
    def get_error_summary_for_ui(error_handler: 'ErrorHandler') -> Dict[str, Any]:
        """
        Get a summary of errors suitable for UI display.
        
        Args:
            error_handler: ErrorHandler instance
            
        Returns:
            Dictionary with error summary information
        """
        counts = error_handler.get_error_counts()
        total_errors = sum(counts.values())
        
        # Get most common error category
        most_common = max(counts.items(), key=lambda x: x[1]) if counts else (ErrorCategory.UNKNOWN, 0)
        
        return {
            "total_errors": total_errors,
            "error_counts": {k.value: v for k, v in counts.items()},
            "most_common_category": most_common[0].value,
            "most_common_count": most_common[1],
            "has_critical_errors": counts.get(ErrorCategory.GPU, 0) > 0 or counts.get(ErrorCategory.MEMORY, 0) > 0,
            "fallback_models_active": any(
                error_handler.get_fallback_model(name) is not None
                for name in ["gatekeeper", "yolo_sideview", "yolo_structure", 
                            "yolo_wagon_number", "nafnet", "sci_enhancer", "smolvlm"]
            )
        }
    
    @staticmethod
    def _get_severity(category: ErrorCategory) -> str:
        """Get severity level for an error category."""
        severity_map = {
            ErrorCategory.MODEL_LOADING: "warning",
            ErrorCategory.TIMEOUT: "warning",
            ErrorCategory.MEMORY: "error",
            ErrorCategory.GPU: "error",
            ErrorCategory.INPUT_VALIDATION: "info",
            ErrorCategory.PROCESSING: "warning",
            ErrorCategory.NETWORK: "warning",
            ErrorCategory.UNKNOWN: "error",
        }
        return severity_map.get(category, "warning")
    
    @staticmethod
    def get_recovery_suggestion(error_record: ErrorRecord) -> str:
        """
        Get a specific recovery suggestion based on the error.
        
        Args:
            error_record: The error record
            
        Returns:
            Recovery suggestion string
        """
        if error_record.recovery_successful:
            if error_record.recovery_action == RecoveryAction.USE_FALLBACK:
                return f"The system is using a fallback for {error_record.stage_name}. Some features may be limited."
            elif error_record.recovery_action == RecoveryAction.RETRY:
                return "The operation succeeded after retry."
            elif error_record.recovery_action == RecoveryAction.SKIP:
                return f"The {error_record.stage_name} stage was skipped. Processing continues with reduced functionality."
        else:
            if error_record.category == ErrorCategory.MODEL_LOADING:
                return f"Model '{error_record.stage_name}' is offline. Check if the model file exists and try restarting."
            elif error_record.category == ErrorCategory.TIMEOUT:
                return "Processing is taking too long. Try with a smaller input or check system resources."
            elif error_record.category == ErrorCategory.MEMORY:
                return "System is running low on memory. Close other applications or reduce input size."
            elif error_record.category == ErrorCategory.GPU:
                return "GPU error occurred. Check CUDA installation or try CPU mode."
        
        return "An error occurred. Check the logs for more details."
    
    @staticmethod
    def format_model_status_badge(status: str, model_name: str) -> Dict[str, str]:
        """
        Format model status for badge display.
        
        Args:
            status: Model status string
            model_name: Name of the model
            
        Returns:
            Dictionary with badge styling information
        """
        status_styles = {
            "loaded": {
                "color": "#90ee90",
                "background": "#1e5f2e",
                "icon": "🟢",
                "text": "Online",
            },
            "offline": {
                "color": "#ff9090",
                "background": "#5f1e1e",
                "icon": "🔴",
                "text": "Offline",
            },
            "loading": {
                "color": "#ffff90",
                "background": "#5f5f1e",
                "icon": "🟡",
                "text": "Loading",
            },
            "error": {
                "color": "#ff6060",
                "background": "#5f1e1e",
                "icon": "❌",
                "text": "Error",
            },
            "not_loaded": {
                "color": "#a0a0a0",
                "background": "#3a3a3a",
                "icon": "⚪",
                "text": "Not Loaded",
            },
        }
        
        style = status_styles.get(status, status_styles["not_loaded"])
        return {
            "model_name": model_name,
            "status": status,
            **style
        }


# ============================================================================
# Processing Stage Wrapper
# ============================================================================

class ProcessingStageWrapper:
    """
    Wrapper for AI processing stages with comprehensive error handling.
    
    Provides:
    - Automatic timeout handling
    - Retry logic for transient failures
    - Fallback to mock models on failure
    - Performance tracking integration
    
    Requirements: 1.4, 11.5
    """
    
    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        timeout_manager: Optional[StageTimeoutManager] = None
    ):
        """
        Initialize the processing stage wrapper.
        
        Args:
            error_handler: ErrorHandler instance
            timeout_manager: StageTimeoutManager instance
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = error_handler or create_error_handler()
        self.timeout_manager = timeout_manager or StageTimeoutManager(
            error_handler=self.error_handler
        )
        
        # Track which stages are using fallbacks
        self._fallback_active: Dict[str, bool] = {}
    
    def execute_stage(
        self,
        stage_name: str,
        func: Callable[..., T],
        *args,
        fallback_func: Optional[Callable[..., T]] = None,
        fallback_value: Optional[T] = None,
        max_retries: int = 2,
        **kwargs
    ) -> Result[T]:
        """
        Execute a processing stage with full error handling.
        
        Args:
            stage_name: Name of the processing stage
            func: Primary function to execute
            *args: Positional arguments for func
            fallback_func: Optional fallback function
            fallback_value: Optional fallback value
            max_retries: Maximum retry attempts for transient failures
            **kwargs: Keyword arguments for func
            
        Returns:
            Result wrapping the output or error
        """
        # Check if we should use fallback directly (previous failures)
        if self._fallback_active.get(stage_name, False) and fallback_func:
            try:
                result = fallback_func(*args, **kwargs)
                return Result.fallback(result)
            except Exception as e:
                self.logger.error(f"Fallback also failed for {stage_name}: {e}")
                if fallback_value is not None:
                    return Result.fallback(fallback_value)
                return Result.failure(e)
        
        # Configure retry for transient failures
        retry_config = RetryConfig(
            max_retries=max_retries,
            initial_delay_ms=100.0,
            max_delay_ms=2000.0,
            exponential_backoff=True
        )
        
        # Execute with timeout and retry
        result = self.timeout_manager.execute_with_timeout(
            stage_name,
            func,
            *args,
            fallback_value=fallback_value,
            **kwargs
        )
        
        # If primary failed, try fallback
        if result.is_failure:
            if fallback_func:
                try:
                    fallback_result = fallback_func(*args, **kwargs)
                    
                    # Record that we're using fallback
                    self._fallback_active[stage_name] = True
                    
                    self.error_handler.record_error(
                        result.error,
                        stage_name,
                        RecoveryAction.USE_FALLBACK,
                        recovery_successful=True
                    )
                    
                    return Result.fallback(fallback_result)
                    
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback failed for {stage_name}: {fallback_error}"
                    )
            
            # Use fallback value if available
            if fallback_value is not None:
                self._fallback_active[stage_name] = True
                return Result.fallback(fallback_value)
        
        return result
    
    def reset_fallback(self, stage_name: str) -> None:
        """Reset fallback status for a stage (try primary again)."""
        self._fallback_active[stage_name] = False
    
    def is_using_fallback(self, stage_name: str) -> bool:
        """Check if a stage is currently using its fallback."""
        return self._fallback_active.get(stage_name, False)
    
    def get_fallback_status(self) -> Dict[str, bool]:
        """Get fallback status for all stages."""
        return dict(self._fallback_active)


# ============================================================================
# Global Error Handler Instance
# ============================================================================

# Singleton instance for global access
_global_error_handler: Optional[ErrorHandler] = None
_global_timeout_manager: Optional[StageTimeoutManager] = None
_global_stage_wrapper: Optional[ProcessingStageWrapper] = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = create_error_handler()
    return _global_error_handler


def get_global_timeout_manager() -> StageTimeoutManager:
    """Get or create the global timeout manager instance."""
    global _global_timeout_manager
    if _global_timeout_manager is None:
        _global_timeout_manager = StageTimeoutManager(
            error_handler=get_global_error_handler()
        )
    return _global_timeout_manager


def get_global_stage_wrapper() -> ProcessingStageWrapper:
    """Get or create the global processing stage wrapper."""
    global _global_stage_wrapper
    if _global_stage_wrapper is None:
        _global_stage_wrapper = ProcessingStageWrapper(
            error_handler=get_global_error_handler(),
            timeout_manager=get_global_timeout_manager()
        )
    return _global_stage_wrapper


def reset_global_handlers() -> None:
    """Reset all global handler instances."""
    global _global_error_handler, _global_timeout_manager, _global_stage_wrapper
    
    if _global_error_handler:
        _global_error_handler.shutdown()
    
    _global_error_handler = None
    _global_timeout_manager = None
    _global_stage_wrapper = None

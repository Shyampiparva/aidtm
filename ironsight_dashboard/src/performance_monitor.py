"""
Performance Monitor for IronSight Command Center

Provides comprehensive performance tracking including:
- Latency budgets and violation logging
- GPU utilization and temperature monitoring
- Graceful degradation when models exceed latency budgets
- Incident logging for post-mortem analysis

Requirements: 11.1, 11.3, 11.4
"""

import logging
import time
import threading
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np


class PerformanceLevel(Enum):
    """Performance level indicators."""
    OPTIMAL = "optimal"       # Within budget
    DEGRADED = "degraded"     # Slightly over budget (1-2x)
    CRITICAL = "critical"     # Significantly over budget (>2x)
    FAILED = "failed"         # Processing failed


class IncidentSeverity(Enum):
    """Severity levels for performance incidents."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LatencyBudget:
    """Latency budget configuration for a processing stage."""
    stage_name: str
    budget_ms: float
    warning_threshold_ms: float = 0.0  # Auto-calculated if 0
    critical_threshold_ms: float = 0.0  # Auto-calculated if 0
    
    def __post_init__(self):
        if self.warning_threshold_ms == 0:
            self.warning_threshold_ms = self.budget_ms * 1.5
        if self.critical_threshold_ms == 0:
            self.critical_threshold_ms = self.budget_ms * 2.0


@dataclass
class PerformanceIncident:
    """Record of a performance incident for post-mortem analysis."""
    timestamp: datetime
    stage_name: str
    severity: IncidentSeverity
    actual_latency_ms: float
    budget_ms: float
    overage_percent: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage_name": self.stage_name,
            "severity": self.severity.value,
            "actual_latency_ms": self.actual_latency_ms,
            "budget_ms": self.budget_ms,
            "overage_percent": self.overage_percent,
            "message": self.message,
            "context": self.context
        }


@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    utilization_percent: float = 0.0
    power_watts: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""
    stage_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    violations_count: int = 0
    last_latency_ms: float = 0.0
    performance_level: PerformanceLevel = PerformanceLevel.OPTIMAL
    
    def update(self, latency_ms: float, budget: LatencyBudget) -> None:
        """Update metrics with new measurement."""
        self.call_count += 1
        self.total_time_ms += latency_ms
        self.min_time_ms = min(self.min_time_ms, latency_ms)
        self.max_time_ms = max(self.max_time_ms, latency_ms)
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.last_latency_ms = latency_ms
        
        # Update performance level
        if latency_ms <= budget.budget_ms:
            self.performance_level = PerformanceLevel.OPTIMAL
        elif latency_ms <= budget.warning_threshold_ms:
            self.performance_level = PerformanceLevel.DEGRADED
            self.violations_count += 1
        else:
            self.performance_level = PerformanceLevel.CRITICAL
            self.violations_count += 1


@dataclass
class SystemMetrics:
    """Overall system performance metrics."""
    fps: float = 0.0
    total_latency_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    queue_depth: int = 0
    gatekeeper_skips: int = 0
    ocr_fallbacks: int = 0
    gpu_metrics: GPUMetrics = field(default_factory=GPUMetrics)
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    performance_level: PerformanceLevel = PerformanceLevel.OPTIMAL


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for IronSight.
    
    Features:
    - Latency budget tracking per processing stage
    - GPU utilization and temperature monitoring
    - Graceful degradation recommendations
    - Incident logging for post-mortem analysis
    """
    
    # Default latency budgets (in milliseconds)
    DEFAULT_BUDGETS = {
        "gatekeeper": 0.5,
        "sci_enhancer": 0.5,
        "yolo_combined": 20.0,
        "yolo_sideview": 7.0,
        "yolo_structure": 7.0,
        "yolo_wagon_number": 7.0,
        "nafnet": 20.0,
        "smolvlm": 30000.0,  # 30 seconds for VLM
        "siglip": 100.0,
        "total_frame": 16.67,  # 60 FPS target
    }
    
    def __init__(
        self,
        budgets: Optional[Dict[str, float]] = None,
        incident_log_path: Optional[Path] = None,
        history_size: int = 1000,
        gpu_monitoring_interval: float = 1.0
    ):
        """
        Initialize performance monitor.
        
        Args:
            budgets: Custom latency budgets (stage_name -> budget_ms)
            incident_log_path: Path for incident log file
            history_size: Number of measurements to keep in history
            gpu_monitoring_interval: Seconds between GPU metric updates
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize latency budgets
        self.budgets: Dict[str, LatencyBudget] = {}
        budget_config = {**self.DEFAULT_BUDGETS, **(budgets or {})}
        for stage_name, budget_ms in budget_config.items():
            self.budgets[stage_name] = LatencyBudget(
                stage_name=stage_name,
                budget_ms=budget_ms
            )
        
        # Metrics storage
        self.system_metrics = SystemMetrics()
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.latency_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Incident tracking
        self.incidents: List[PerformanceIncident] = []
        self.incident_log_path = incident_log_path or Path("logs/performance_incidents.jsonl")
        self.incident_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # GPU monitoring
        self.gpu_monitoring_interval = gpu_monitoring_interval
        self._gpu_monitor_thread: Optional[threading.Thread] = None
        self._gpu_monitor_running = False
        self._gpu_lock = threading.Lock()
        
        # FPS calculation
        self._frame_times: deque = deque(maxlen=60)
        self._last_frame_time: float = 0.0
        
        # Degradation callbacks
        self._degradation_callbacks: List[Callable[[str, PerformanceLevel], None]] = []
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        
        self.logger.info("Performance monitor initialized")
    
    def start_gpu_monitoring(self) -> bool:
        """Start background GPU monitoring thread."""
        if self._gpu_monitor_running:
            return True
        
        try:
            # Test if GPU monitoring is available
            self._update_gpu_metrics()
            
            self._gpu_monitor_running = True
            self._gpu_monitor_thread = threading.Thread(
                target=self._gpu_monitor_loop,
                daemon=True
            )
            self._gpu_monitor_thread.start()
            self.logger.info("GPU monitoring started")
            return True
            
        except Exception as e:
            self.logger.warning(f"GPU monitoring not available: {e}")
            return False
    
    def stop_gpu_monitoring(self) -> None:
        """Stop GPU monitoring thread."""
        self._gpu_monitor_running = False
        if self._gpu_monitor_thread:
            self._gpu_monitor_thread.join(timeout=2.0)
            self._gpu_monitor_thread = None
        self.logger.info("GPU monitoring stopped")
    
    def _gpu_monitor_loop(self) -> None:
        """Background loop for GPU monitoring."""
        while self._gpu_monitor_running:
            try:
                self._update_gpu_metrics()
            except Exception as e:
                self.logger.debug(f"GPU metrics update failed: {e}")
            time.sleep(self.gpu_monitoring_interval)
    
    def _update_gpu_metrics(self) -> None:
        """Update GPU metrics from hardware."""
        metrics = GPUMetrics(timestamp=datetime.now())
        
        try:
            import torch
            if torch.cuda.is_available():
                # Memory metrics
                metrics.memory_used_mb = torch.cuda.memory_allocated() / 1024**2
                metrics.memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                metrics.memory_percent = (metrics.memory_used_mb / metrics.memory_total_mb) * 100
                
                # Try to get additional metrics via pynvml
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Temperature
                    metrics.temperature_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.utilization_percent = util.gpu
                    
                    # Power
                    try:
                        metrics.power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except pynvml.NVMLError:
                        pass
                        
                except ImportError:
                    pass
                except Exception:
                    pass
                    
        except ImportError:
            pass
        except Exception:
            pass
        
        with self._gpu_lock:
            self.system_metrics.gpu_metrics = metrics
    
    def record_latency(
        self,
        stage_name: str,
        latency_ms: float,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceLevel:
        """
        Record latency measurement for a processing stage.
        
        Args:
            stage_name: Name of the processing stage
            latency_ms: Measured latency in milliseconds
            context: Optional context for incident logging
            
        Returns:
            Performance level after this measurement
        """
        with self._metrics_lock:
            # Get or create budget
            if stage_name not in self.budgets:
                self.budgets[stage_name] = LatencyBudget(
                    stage_name=stage_name,
                    budget_ms=latency_ms * 2  # Default to 2x first measurement
                )
            
            budget = self.budgets[stage_name]
            
            # Get or create stage metrics
            if stage_name not in self.stage_metrics:
                self.stage_metrics[stage_name] = StageMetrics(stage_name=stage_name)
            
            stage = self.stage_metrics[stage_name]
            stage.update(latency_ms, budget)
            
            # Store in history
            self.latency_history[stage_name].append(latency_ms)
            
            # Check for violations and log incidents
            if latency_ms > budget.budget_ms:
                self._handle_violation(stage_name, latency_ms, budget, context)
            
            # Update system metrics
            self.system_metrics.stage_metrics[stage_name] = stage
            
            return stage.performance_level
    
    def _handle_violation(
        self,
        stage_name: str,
        latency_ms: float,
        budget: LatencyBudget,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Handle a latency budget violation."""
        overage_percent = ((latency_ms - budget.budget_ms) / budget.budget_ms) * 100
        
        # Determine severity
        if latency_ms <= budget.warning_threshold_ms:
            severity = IncidentSeverity.WARNING
        elif latency_ms <= budget.critical_threshold_ms:
            severity = IncidentSeverity.ERROR
        else:
            severity = IncidentSeverity.CRITICAL
        
        # Create incident
        incident = PerformanceIncident(
            timestamp=datetime.now(),
            stage_name=stage_name,
            severity=severity,
            actual_latency_ms=latency_ms,
            budget_ms=budget.budget_ms,
            overage_percent=overage_percent,
            message=f"{stage_name} exceeded budget: {latency_ms:.2f}ms > {budget.budget_ms:.2f}ms ({overage_percent:.1f}% over)",
            context=context or {}
        )
        
        # Store incident
        self.incidents.append(incident)
        
        # Log incident
        self._log_incident(incident)
        
        # Notify degradation callbacks
        level = PerformanceLevel.DEGRADED if severity == IncidentSeverity.WARNING else PerformanceLevel.CRITICAL
        for callback in self._degradation_callbacks:
            try:
                callback(stage_name, level)
            except Exception as e:
                self.logger.error(f"Degradation callback error: {e}")
    
    def _log_incident(self, incident: PerformanceIncident) -> None:
        """Log incident to file for post-mortem analysis."""
        try:
            with open(self.incident_log_path, 'a') as f:
                f.write(json.dumps(incident.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log incident: {e}")
        
        # Also log to standard logger
        log_method = {
            IncidentSeverity.INFO: self.logger.info,
            IncidentSeverity.WARNING: self.logger.warning,
            IncidentSeverity.ERROR: self.logger.error,
            IncidentSeverity.CRITICAL: self.logger.critical,
        }.get(incident.severity, self.logger.warning)
        
        log_method(incident.message)
    
    def record_frame(self, total_latency_ms: float) -> None:
        """
        Record a complete frame processing.
        
        Args:
            total_latency_ms: Total frame processing time
        """
        with self._metrics_lock:
            current_time = time.time()
            
            # Update frame count
            self.system_metrics.frames_processed += 1
            self.system_metrics.total_latency_ms = total_latency_ms
            
            # Calculate FPS
            if self._last_frame_time > 0:
                frame_interval = current_time - self._last_frame_time
                self._frame_times.append(frame_interval)
                
                if len(self._frame_times) > 0:
                    avg_interval = sum(self._frame_times) / len(self._frame_times)
                    self.system_metrics.fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
            
            self._last_frame_time = current_time
            
            # Record total frame latency
            self.record_latency("total_frame", total_latency_ms)
    
    def record_frame_dropped(self) -> None:
        """Record a dropped frame."""
        with self._metrics_lock:
            self.system_metrics.frames_dropped += 1
    
    def record_gatekeeper_skip(self) -> None:
        """Record a gatekeeper skip (frame filtered out)."""
        with self._metrics_lock:
            self.system_metrics.gatekeeper_skips += 1
    
    def record_ocr_fallback(self) -> None:
        """Record an OCR fallback to SmolVLM."""
        with self._metrics_lock:
            self.system_metrics.ocr_fallbacks += 1
    
    def update_queue_depth(self, depth: int) -> None:
        """Update current queue depth."""
        with self._metrics_lock:
            self.system_metrics.queue_depth = depth
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        with self._metrics_lock:
            return self.system_metrics
    
    def get_stage_metrics(self, stage_name: str) -> Optional[StageMetrics]:
        """Get metrics for a specific stage."""
        with self._metrics_lock:
            return self.stage_metrics.get(stage_name)
    
    def get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        with self._gpu_lock:
            return self.system_metrics.gpu_metrics
    
    def get_latency_history(self, stage_name: str) -> List[float]:
        """Get latency history for a stage."""
        with self._metrics_lock:
            return list(self.latency_history.get(stage_name, []))
    
    def get_recent_incidents(self, count: int = 10) -> List[PerformanceIncident]:
        """Get most recent incidents."""
        return self.incidents[-count:]
    
    def get_incidents_by_severity(
        self,
        severity: IncidentSeverity,
        since: Optional[datetime] = None
    ) -> List[PerformanceIncident]:
        """Get incidents filtered by severity."""
        incidents = [i for i in self.incidents if i.severity == severity]
        if since:
            incidents = [i for i in incidents if i.timestamp >= since]
        return incidents
    
    def register_degradation_callback(
        self,
        callback: Callable[[str, PerformanceLevel], None]
    ) -> None:
        """
        Register callback for performance degradation events.
        
        Args:
            callback: Function(stage_name, performance_level) to call on degradation
        """
        self._degradation_callbacks.append(callback)
    
    def get_degradation_recommendations(self) -> List[str]:
        """
        Get recommendations for handling current performance issues.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        with self._metrics_lock:
            # Check each stage
            for stage_name, metrics in self.stage_metrics.items():
                if metrics.performance_level == PerformanceLevel.CRITICAL:
                    if stage_name == "gatekeeper":
                        recommendations.append(
                            "Gatekeeper is critically slow. Consider reducing input resolution."
                        )
                    elif stage_name == "sci_enhancer":
                        recommendations.append(
                            "SCI enhancement is slow. Consider skipping for bright images."
                        )
                    elif "yolo" in stage_name:
                        recommendations.append(
                            f"{stage_name} is slow. Consider reducing detection confidence threshold."
                        )
                    elif stage_name == "nafnet":
                        recommendations.append(
                            "NAFNet deblurring is slow. Consider reducing crop size or skipping for sharp images."
                        )
                    elif stage_name == "total_frame":
                        recommendations.append(
                            "Total frame processing exceeds budget. Consider enabling frame skipping."
                        )
            
            # Check GPU metrics
            gpu = self.system_metrics.gpu_metrics
            if gpu.memory_percent > 90:
                recommendations.append(
                    f"GPU memory usage is high ({gpu.memory_percent:.1f}%). Consider reducing batch size."
                )
            if gpu.temperature_c > 80:
                recommendations.append(
                    f"GPU temperature is high ({gpu.temperature_c}°C). Consider reducing workload."
                )
        
        return recommendations
    
    def should_skip_processing(self, stage_name: str) -> bool:
        """
        Check if processing should be skipped for graceful degradation.
        
        Args:
            stage_name: Name of the processing stage
            
        Returns:
            True if processing should be skipped
        """
        with self._metrics_lock:
            metrics = self.stage_metrics.get(stage_name)
            if metrics and metrics.performance_level == PerformanceLevel.CRITICAL:
                # Skip if consistently critical (more than 50% violations)
                if metrics.call_count > 10:
                    violation_rate = metrics.violations_count / metrics.call_count
                    return violation_rate > 0.5
        return False
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for new sessions)."""
        with self._metrics_lock:
            self.system_metrics = SystemMetrics()
            self.stage_metrics.clear()
            self.latency_history.clear()
            self._frame_times.clear()
            self._last_frame_time = 0.0
        
        self.logger.info("Performance metrics reset")
    
    def export_metrics(self, filepath: Path) -> None:
        """Export current metrics to JSON file."""
        with self._metrics_lock:
            data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "fps": self.system_metrics.fps,
                    "total_latency_ms": self.system_metrics.total_latency_ms,
                    "frames_processed": self.system_metrics.frames_processed,
                    "frames_dropped": self.system_metrics.frames_dropped,
                    "queue_depth": self.system_metrics.queue_depth,
                    "gatekeeper_skips": self.system_metrics.gatekeeper_skips,
                    "ocr_fallbacks": self.system_metrics.ocr_fallbacks,
                },
                "gpu": asdict(self.system_metrics.gpu_metrics),
                "stages": {
                    name: asdict(metrics) 
                    for name, metrics in self.stage_metrics.items()
                },
                "incidents_count": len(self.incidents),
                "recommendations": self.get_degradation_recommendations()
            }
        
        # Fix datetime serialization
        data["gpu"]["timestamp"] = data["gpu"]["timestamp"].isoformat()
        for stage in data["stages"].values():
            stage["performance_level"] = stage["performance_level"].value
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")


class LatencyTimer:
    """Context manager for timing code blocks and recording to monitor."""
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        stage_name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.monitor = monitor
        self.stage_name = stage_name
        self.context = context
        self.start_time: float = 0.0
        self.latency_ms: float = 0.0
    
    def __enter__(self) -> 'LatencyTimer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.monitor.record_latency(self.stage_name, self.latency_ms, self.context)


def create_performance_monitor(
    budgets: Optional[Dict[str, float]] = None,
    incident_log_path: Optional[Path] = None,
    start_gpu_monitoring: bool = True
) -> PerformanceMonitor:
    """
    Factory function to create a configured performance monitor.
    
    Args:
        budgets: Custom latency budgets
        incident_log_path: Path for incident log
        start_gpu_monitoring: Whether to start GPU monitoring
        
    Returns:
        Configured PerformanceMonitor instance
    """
    monitor = PerformanceMonitor(
        budgets=budgets,
        incident_log_path=incident_log_path
    )
    
    if start_gpu_monitoring:
        monitor.start_gpu_monitoring()
    
    return monitor

"""
Mission Control - Live Processing Interface for IronSight Command Center.

This module implements the live processing interface with:
- Video input selection (webcam, RTSP stream, or uploaded video file)
- Real-time video display with oriented bounding box overlays
- Metric cards for latest serial number, FPS, and processing stats
- Different colored overlays for each YOLO model type
- Error handling with automatic fallbacks and recovery

Requirements: 2.1, 2.2, 2.3, 1.4, 11.5
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import error handler for processing stage error handling
try:
    from error_handler import (
        ErrorHandler, create_error_handler, Result,
        ProcessingStageWrapper, get_global_stage_wrapper,
        ErrorCategory, RecoveryAction
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False
    logger.warning("Error handler not available - running without error handling")


class VideoInputType(Enum):
    """Types of video input sources."""
    WEBCAM = "webcam"
    RTSP = "rtsp"
    FILE = "file"
    NONE = "none"


@dataclass
class VideoInputConfig:
    """Configuration for video input source."""
    input_type: VideoInputType = VideoInputType.NONE
    source: Union[str, int] = 0  # Webcam index, RTSP URL, or file path
    fps_target: int = 30
    resolution: Tuple[int, int] = (1280, 720)


@dataclass
class ProcessingStats:
    """Real-time processing statistics."""
    fps: float = 0.0
    processing_latency_ms: float = 0.0
    queue_depth: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    gatekeeper_skips: int = 0
    last_serial_number: str = "N/A"
    last_update_time: float = field(default_factory=time.time)
    
    # Per-model timing
    gatekeeper_time_ms: float = 0.0
    sci_time_ms: float = 0.0
    yolo_time_ms: float = 0.0
    nafnet_time_ms: float = 0.0
    smolvlm_time_ms: float = 0.0


@dataclass
class OverlayConfig:
    """Configuration for detection overlays."""
    # Colors in BGR format for OpenCV
    sideview_color: Tuple[int, int, int] = (0, 0, 255)      # Red for damage
    structure_color: Tuple[int, int, int] = (0, 255, 0)     # Green for structure
    wagon_number_color: Tuple[int, int, int] = (255, 255, 0)  # Cyan for numbers
    
    # Drawing settings
    line_thickness: int = 2
    font_scale: float = 0.5
    show_confidence: bool = True
    show_class_name: bool = True


class VideoInputHandler:
    """
    Handles video input from various sources.
    
    Supports:
    - Webcam (device index)
    - RTSP stream (URL)
    - Video file upload (file path)
    """
    
    def __init__(self, config: VideoInputConfig):
        """
        Initialize video input handler.
        
        Args:
            config: Video input configuration
        """
        self.config = config
        self.capture = None
        self.is_running = False
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """
        Start video capture from configured source.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            import cv2
            
            with self._lock:
                if self.config.input_type == VideoInputType.NONE:
                    logger.warning("No video input configured")
                    return False
                
                source = self.config.source
                
                if self.config.input_type == VideoInputType.WEBCAM:
                    # Webcam - source is device index
                    self.capture = cv2.VideoCapture(int(source))
                    # Test if camera is actually available
                    if not self.capture.isOpened():
                        logger.warning(f"Camera index {source} not available. No camera connected or camera in use by another application.")
                        self.capture.release()
                        return False
                elif self.config.input_type == VideoInputType.RTSP:
                    # RTSP stream - source is URL
                    self.capture = cv2.VideoCapture(str(source))
                elif self.config.input_type == VideoInputType.FILE:
                    # Video file - source is file path
                    if not Path(str(source)).exists():
                        logger.error(f"Video file not found: {source}")
                        return False
                    self.capture = cv2.VideoCapture(str(source))
                
                if not self.capture or not self.capture.isOpened():
                    logger.error(f"Failed to open video source: {source}")
                    return False
                
                # Set resolution if webcam
                if self.config.input_type == VideoInputType.WEBCAM:
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                
                self.is_running = True
                logger.info(f"Video capture started: {self.config.input_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False
    
    def stop(self) -> None:
        """Stop video capture and release resources."""
        with self._lock:
            self.is_running = False
            if self.capture:
                self.capture.release()
                self.capture = None
            logger.info("Video capture stopped")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the video source.
        
        Returns:
            Frame as numpy array (H, W, C) in BGR format, or None if failed
        """
        with self._lock:
            if not self.is_running or not self.capture:
                return None
            
            ret, frame = self.capture.read()
            if not ret:
                # For video files, loop back to start
                if self.config.input_type == VideoInputType.FILE:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.capture.read()
                
                if not ret:
                    return None
            
            return frame
    
    def get_frame_info(self) -> Dict[str, Any]:
        """Get information about the current video source."""
        info = {
            "input_type": self.config.input_type.value,
            "source": str(self.config.source),
            "is_running": self.is_running,
            "width": 0,
            "height": 0,
            "fps": 0
        }
        
        if self.capture and self.capture.isOpened():
            import cv2
            info["width"] = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = self.capture.get(cv2.CAP_PROP_FPS)
        
        return info
    
    @staticmethod
    def validate_input(input_type: VideoInputType, source: Union[str, int]) -> Tuple[bool, str]:
        """
        Validate video input configuration.
        
        Args:
            input_type: Type of video input
            source: Video source (index, URL, or path)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if input_type == VideoInputType.NONE:
            return True, ""
        
        if input_type == VideoInputType.WEBCAM:
            try:
                idx = int(source)
                if idx < 0:
                    return False, "Webcam index must be non-negative"
                return True, ""
            except (ValueError, TypeError):
                return False, "Webcam source must be an integer device index"
        
        if input_type == VideoInputType.RTSP:
            source_str = str(source)
            if not source_str.startswith(("rtsp://", "rtsps://", "http://", "https://")):
                return False, "RTSP source must be a valid URL"
            return True, ""
        
        if input_type == VideoInputType.FILE:
            path = Path(str(source))
            if not path.exists():
                return False, f"Video file not found: {source}"
            if path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return False, f"Unsupported video format: {path.suffix}"
            return True, ""
        
        return False, f"Unknown input type: {input_type}"


class OverlayRenderer:
    """
    Renders detection overlays on video frames.
    
    Supports:
    - Different colors for each YOLO model type
    - Oriented bounding boxes (OBB)
    - Regular bounding boxes
    - Class labels and confidence scores
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize overlay renderer.
        
        Args:
            config: Overlay configuration (uses defaults if None)
        """
        self.config = config or OverlayConfig()
    
    def render_detections(
        self,
        frame: np.ndarray,
        detections_by_model: Dict[str, List[Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Render detection overlays on frame.
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            detections_by_model: Dict mapping model name to list of detections
            
        Returns:
            Frame with overlays rendered
        """
        import cv2
        
        output = frame.copy()
        
        # Color mapping for each model
        color_map = {
            "sideview_damage_obb": self.config.sideview_color,
            "structure_obb": self.config.structure_color,
            "wagon_number_obb": self.config.wagon_number_color
        }
        
        for model_name, detections in detections_by_model.items():
            color = color_map.get(model_name, (255, 255, 255))
            
            for det in detections:
                output = self._draw_detection(output, det, color)
        
        return output
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw a single detection on the frame."""
        import cv2
        
        bbox = detection.get("bbox", [])
        is_obb = detection.get("is_obb", False)
        class_name = detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.0)
        
        if not bbox:
            return frame
        
        if is_obb and len(bbox) == 8:
            # Oriented bounding box - 4 corner points
            points = np.array(bbox).reshape(-1, 2).astype(np.int32)
            cv2.polylines(frame, [points], True, color, self.config.line_thickness)
            label_pos = (int(bbox[0]), int(bbox[1]) - 10)
        elif len(bbox) >= 4:
            # Regular bounding box [x, y, width, height]
            x, y, w, h = bbox[:4]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.line_thickness)
            label_pos = (x1, y1 - 10)
        else:
            return frame
        
        # Draw label
        if self.config.show_class_name or self.config.show_confidence:
            label_parts = []
            if self.config.show_class_name:
                label_parts.append(class_name)
            if self.config.show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Ensure label position is within frame
            label_pos = (max(0, label_pos[0]), max(20, label_pos[1]))
            
            cv2.putText(
                frame, label, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                color, self.config.line_thickness
            )
        
        return frame
    
    def render_stats_overlay(
        self,
        frame: np.ndarray,
        stats: ProcessingStats
    ) -> np.ndarray:
        """
        Render processing statistics overlay on frame.
        
        Args:
            frame: Input frame
            stats: Processing statistics
            
        Returns:
            Frame with stats overlay
        """
        import cv2
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Semi-transparent background for stats
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Stats text
        stats_lines = [
            f"FPS: {stats.fps:.1f}",
            f"Latency: {stats.processing_latency_ms:.1f}ms",
            f"Queue: {stats.queue_depth}",
            f"Frames: {stats.frames_processed}",
            f"Serial: {stats.last_serial_number}"
        ]
        
        y_offset = 30
        for line in stats_lines:
            cv2.putText(
                output, line, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 18
        
        return output


class MissionControl:
    """
    Main Mission Control interface for live processing.
    
    Coordinates:
    - Video input handling
    - Frame processing through IronSight Engine
    - Real-time overlay rendering
    - Statistics tracking and display
    - Error handling with automatic fallbacks
    """
    
    def __init__(
        self,
        engine: Optional[Any] = None,
        video_config: Optional[VideoInputConfig] = None,
        overlay_config: Optional[OverlayConfig] = None,
        error_handler: Optional[Any] = None
    ):
        """
        Initialize Mission Control.
        
        Args:
            engine: IronSight Engine instance (optional)
            video_config: Video input configuration
            overlay_config: Overlay rendering configuration
            error_handler: Error handler instance (optional)
        """
        self.engine = engine
        self.video_handler = VideoInputHandler(video_config or VideoInputConfig())
        self.overlay_renderer = OverlayRenderer(overlay_config)
        
        # Initialize error handling
        self.error_handler = error_handler
        self.stage_wrapper: Optional[ProcessingStageWrapper] = None
        if ERROR_HANDLER_AVAILABLE:
            if error_handler is None:
                self.error_handler = create_error_handler()
            try:
                self.stage_wrapper = get_global_stage_wrapper()
            except Exception:
                self.stage_wrapper = ProcessingStageWrapper(error_handler=self.error_handler)
        
        # Processing state
        self.is_processing = False
        self.stats = ProcessingStats()
        self._stats_lock = threading.Lock()
        
        # Error tracking
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10
        self._last_error_time = 0.0
        self._error_cooldown_seconds = 5.0
        
        # Frame queue for async processing
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._result_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # FPS calculation
        self._fps_timestamps: List[float] = []
        self._fps_window_size = 30
        
        logger.info("Mission Control initialized")
    
    def set_video_input(
        self,
        input_type: VideoInputType,
        source: Union[str, int]
    ) -> Tuple[bool, str]:
        """
        Configure video input source.
        
        Args:
            input_type: Type of video input
            source: Video source (index, URL, or path)
            
        Returns:
            Tuple of (success, error_message)
        """
        # Validate input
        is_valid, error_msg = VideoInputHandler.validate_input(input_type, source)
        if not is_valid:
            return False, error_msg
        
        # Stop current capture if running
        if self.video_handler.is_running:
            self.video_handler.stop()
        
        # Update configuration
        self.video_handler.config.input_type = input_type
        self.video_handler.config.source = source
        
        return True, ""
    
    def start_processing(self) -> bool:
        """
        Start live processing.
        
        Returns:
            True if started successfully
        """
        if self.is_processing:
            logger.warning("Processing already running")
            return True
        
        # Start video capture
        if not self.video_handler.start():
            logger.error("Failed to start video capture")
            return False
        
        self.is_processing = True
        self.stats = ProcessingStats()
        
        logger.info("Mission Control processing started")
        return True
    
    def stop_processing(self) -> None:
        """Stop live processing."""
        self.is_processing = False
        self.video_handler.stop()
        logger.info("Mission Control processing stopped")
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], ProcessingStats]:
        """
        Process a single frame and return result with overlays.
        
        Uses error handling with automatic fallbacks for robust processing.
        
        Returns:
            Tuple of (processed_frame, stats) or (None, stats) if no frame
        """
        start_time = time.time()
        
        # Read frame with error handling
        frame = self._read_frame_safe()
        if frame is None:
            return None, self.stats
        
        # Process through engine if available
        detections_by_model = {}
        serial_number = "N/A"
        
        if self.engine:
            result = self._process_engine_frame(frame)
            if result:
                # Extract detections from result
                if hasattr(result, 'detection_result') and result.detection_result:
                    dr = result.detection_result
                    if hasattr(dr, 'combined_json') and dr.combined_json:
                        detections_by_model = dr.combined_json.get('detections_by_model', {})
                
                # Extract serial number
                if hasattr(result, 'ocr_results') and result.ocr_results:
                    serial_number = list(result.ocr_results.values())[0] if result.ocr_results else "N/A"
                
                # Update timing stats
                with self._stats_lock:
                    if hasattr(result, 'gatekeeper_time_ms'):
                        self.stats.gatekeeper_time_ms = result.gatekeeper_time_ms
                    if hasattr(result, 'sci_time_ms'):
                        self.stats.sci_time_ms = result.sci_time_ms
                    if hasattr(result, 'yolo_time_ms'):
                        self.stats.yolo_time_ms = result.yolo_time_ms
                    if hasattr(result, 'nafnet_time_ms'):
                        self.stats.nafnet_time_ms = result.nafnet_time_ms
                    if hasattr(result, 'smolvlm_time_ms'):
                        self.stats.smolvlm_time_ms = result.smolvlm_time_ms
                
                # Reset error counter on success
                self._consecutive_errors = 0
        
        # Render overlays
        output_frame = self.overlay_renderer.render_detections(frame, detections_by_model)
        
        # Update stats
        processing_time_ms = (time.time() - start_time) * 1000
        self._update_stats(processing_time_ms, serial_number)
        
        # Render stats overlay
        output_frame = self.overlay_renderer.render_stats_overlay(output_frame, self.stats)
        
        return output_frame, self.stats
    
    def _read_frame_safe(self) -> Optional[np.ndarray]:
        """
        Read a frame with error handling.
        
        Returns:
            Frame or None if failed
        """
        try:
            frame = self.video_handler.read_frame()
            if frame is None:
                self._handle_frame_read_error("No frame available")
            return frame
        except Exception as e:
            self._handle_frame_read_error(str(e))
            return None
    
    def _process_engine_frame(self, frame: np.ndarray) -> Optional[Any]:
        """
        Process frame through engine with error handling.
        
        Args:
            frame: Input frame
            
        Returns:
            Processing result or None if failed
        """
        if self.stage_wrapper and ERROR_HANDLER_AVAILABLE:
            # Use stage wrapper for error handling with fallback
            # Define a wrapper function to pass frame as argument
            def process_func():
                return self.engine.process_single_frame(frame)
            
            result = self.stage_wrapper.execute_stage(
                stage_name="frame_processing",
                func=process_func,
                fallback_value=None,
                max_retries=1
            )
            
            if result.is_success or result.is_fallback:
                return result.value
            else:
                self._handle_processing_error(result.error)
                return None
        else:
            # Direct processing without error handler
            try:
                return self.engine.process_single_frame(frame)
            except Exception as e:
                self._handle_processing_error(e)
                return None
    
    def _handle_frame_read_error(self, error_msg: str) -> None:
        """Handle frame read errors with rate limiting."""
        current_time = time.time()
        
        # Rate limit error logging
        if current_time - self._last_error_time > self._error_cooldown_seconds:
            logger.warning(f"Frame read error: {error_msg}")
            self._last_error_time = current_time
        
        with self._stats_lock:
            self.stats.frames_dropped += 1
    
    def _handle_processing_error(self, error: Optional[Exception]) -> None:
        """Handle processing errors with consecutive error tracking."""
        self._consecutive_errors += 1
        current_time = time.time()
        
        # Rate limit error logging
        if current_time - self._last_error_time > self._error_cooldown_seconds:
            logger.error(f"Processing error: {error}")
            self._last_error_time = current_time
        
        # Record error if handler available
        if self.error_handler and ERROR_HANDLER_AVAILABLE and error:
            self.error_handler.record_error(
                error,
                stage_name="frame_processing",
                recovery_action=RecoveryAction.SKIP,
                recovery_successful=False
            )
        
        with self._stats_lock:
            self.stats.frames_dropped += 1
        
        # Check for too many consecutive errors
        if self._consecutive_errors >= self._max_consecutive_errors:
            logger.error(f"Too many consecutive errors ({self._consecutive_errors}), stopping processing")
            self.stop_processing()
    
    def get_error_count(self) -> int:
        """Get the number of consecutive errors."""
        return self._consecutive_errors
    
    def is_healthy(self) -> bool:
        """Check if processing is healthy (not too many errors)."""
        return self._consecutive_errors < self._max_consecutive_errors // 2
    
    def _update_stats(self, processing_time_ms: float, serial_number: str) -> None:
        """Update processing statistics."""
        current_time = time.time()
        
        with self._stats_lock:
            # Update frame count
            self.stats.frames_processed += 1
            self.stats.processing_latency_ms = processing_time_ms
            self.stats.last_serial_number = serial_number
            self.stats.last_update_time = current_time
            
            # Calculate FPS
            self._fps_timestamps.append(current_time)
            
            # Keep only recent timestamps
            while len(self._fps_timestamps) > self._fps_window_size:
                self._fps_timestamps.pop(0)
            
            if len(self._fps_timestamps) >= 2:
                time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                if time_span > 0:
                    self.stats.fps = (len(self._fps_timestamps) - 1) / time_span
    
    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        with self._stats_lock:
            return ProcessingStats(
                fps=self.stats.fps,
                processing_latency_ms=self.stats.processing_latency_ms,
                queue_depth=self.stats.queue_depth,
                frames_processed=self.stats.frames_processed,
                frames_dropped=self.stats.frames_dropped,
                gatekeeper_skips=self.stats.gatekeeper_skips,
                last_serial_number=self.stats.last_serial_number,
                last_update_time=self.stats.last_update_time,
                gatekeeper_time_ms=self.stats.gatekeeper_time_ms,
                sci_time_ms=self.stats.sci_time_ms,
                yolo_time_ms=self.stats.yolo_time_ms,
                nafnet_time_ms=self.stats.nafnet_time_ms,
                smolvlm_time_ms=self.stats.smolvlm_time_ms
            )
    
    def get_model_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get color mapping for each model type."""
        return {
            "sideview_damage_obb": self.overlay_renderer.config.sideview_color,
            "structure_obb": self.overlay_renderer.config.structure_color,
            "wagon_number_obb": self.overlay_renderer.config.wagon_number_color
        }


def create_mission_control(
    engine: Optional[Any] = None,
    input_type: VideoInputType = VideoInputType.NONE,
    source: Union[str, int] = 0,
    error_handler: Optional[Any] = None
) -> MissionControl:
    """
    Factory function to create Mission Control instance.
    
    Args:
        engine: IronSight Engine instance
        input_type: Video input type
        source: Video source
        error_handler: Error handler instance (optional)
        
    Returns:
        Configured MissionControl instance
    """
    video_config = VideoInputConfig(
        input_type=input_type,
        source=source
    )
    
    return MissionControl(
        engine=engine,
        video_config=video_config,
        error_handler=error_handler
    )

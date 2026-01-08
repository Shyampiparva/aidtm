"""
Iron-Sight Railway Inspection Pipeline Core.

This module implements the main IronSightEngine with two-stream processing:
- Stream A: Fast visualization lane (60 FPS, <5ms latency)
- Stream B: AI inspection lane with SmolVLM fallback OCR

Key features:
- Gatekeeper pre-filtering to skip empty frames
- SCI (Self-Calibrated Illumination) for fast low-light enhancement (~0.5ms)
- Conditional deblurring based on blur detection
- SmolVLM 2 fallback for OCR when PaddleOCR fails
- Damage assessment using SmolVLM for detected defects
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
import cv2

# Import models and components
from .models import (
    PipelineConfig, Detection, OCRResult, InspectionResult,
    SpectralChannels, LatencyMetrics
)
from .agent_forensic import get_forensic_agent, ForensicResult
from .semantic_search import get_search_engine
from .preprocessor_sci import create_sci_preprocessor


@dataclass
class ProcessingState:
    """Current state of the processing pipeline."""
    frames_processed: int = 0
    frames_dropped: int = 0
    avg_fps: float = 0.0
    current_latency_ms: float = 0.0
    queue_depth: int = 0
    gatekeeper_skips: int = 0
    ocr_fallbacks: int = 0
    damage_assessments: int = 0


class IronSightEngine:
    """
    Main orchestrator for the Iron-Sight railway inspection system.
    
    Implements a two-stream architecture:
    - Stream A: Fast visualization (capture → resize → dashboard)
    - Stream B: AI inspection (gatekeeper → enhance → detect → OCR → log)
    
    Features SmolVLM 2 fallback OCR and damage assessment capabilities.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the Iron-Sight engine.
        
        Args:
            config: Pipeline configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Threading components
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.queue_maxsize)
        self.result_queue: queue.Queue[InspectionResult] = queue.Queue()
        self.forensic_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=20)
        
        # Thread management
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Video capture
        self.video_capture: Optional[cv2.VideoCapture] = None
        
        # AI Models (loaded lazily)
        self.models_loaded = False
        self.gatekeeper = None
        self.sci_preprocessor = None  # SCI for low-light enhancement
        self.yolo_detector = None
        self.deblur_gan = None
        self.paddle_ocr = None
        
        # Forensic agent
        self.forensic_agent = get_forensic_agent()
        
        # Semantic search engine
        self.search_engine = get_search_engine()
        
        # Image storage for search
        self.crop_storage_dir = Path("data/wagon_crops")
        self.crop_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.state = ProcessingState()
        self.last_fps_update = time.time()
        self.frame_times: List[float] = []
        
        # Dashboard callbacks
        self.dashboard_callbacks: List[Callable] = []
        
    def add_dashboard_callback(self, callback: Callable) -> None:
        """Add callback for dashboard updates."""
        self.dashboard_callbacks.append(callback)
    
    def _load_models(self) -> bool:
        """
        Load all AI models for the inspection pipeline.
        Returns True if all models loaded successfully.
        """
        try:
            self.logger.info("Loading AI models...")
            start_time = time.time()
            
            # TODO: Load actual models - placeholder for now
            # self.gatekeeper = load_gatekeeper_model(f"{self.config.model_dir}/gatekeeper.onnx")
            # self.sci_preprocessor = create_sci_preprocessor(model_variant="medium", device="cuda")
            # self.yolo_detector = load_yolo_model(f"{self.config.model_dir}/yolo_wagon.onnx")
            # self.deblur_gan = load_deblur_model(f"{self.config.model_dir}/deblur_gan.onnx")
            # self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            
            # Placeholder - simulate model loading
            self.gatekeeper = MockGatekeeper()
            self.sci_preprocessor = create_sci_preprocessor(model_variant="medium", device="cuda")
            self.yolo_detector = MockYOLODetector()
            self.deblur_gan = MockDeblurGAN()
            self.paddle_ocr = MockPaddleOCR()
            
            load_time = time.time() - start_time
            self.logger.info(f"All models loaded in {load_time:.2f}s")
            self.models_loaded = True
            
            # Start forensic agent
            if not self.forensic_agent.start():
                self.logger.warning("Failed to start forensic agent - OCR fallback disabled")
            
            # Start semantic search engine
            if not self.search_engine.start():
                self.logger.warning("Failed to start search engine - semantic search disabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the Iron-Sight inspection pipeline.
        Returns True if started successfully.
        """
        if self.is_running:
            self.logger.warning("Pipeline already running")
            return True
        
        # Load models if not already loaded
        if not self.models_loaded:
            if not self._load_models():
                return False
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(self.config.video_source)
        if not self.video_capture.isOpened():
            self.logger.error(f"Failed to open video source: {self.config.video_source}")
            return False
        
        # Configure capture settings for high-speed capture
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        self.video_capture.set(cv2.CAP_PROP_FPS, 60)        # Target 60 FPS
        
        # Start processing threads
        self.is_running = True
        
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="IronSight-Capture",
            daemon=True
        )
        
        self.process_thread = threading.Thread(
            target=self._process_loop,
            name="IronSight-Process",
            daemon=True
        )
        
        self.capture_thread.start()
        self.process_thread.start()
        
        self.logger.info("Iron-Sight pipeline started")
        return True
    
    def stop(self) -> None:
        """Stop the Iron-Sight inspection pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop video capture
        if self.video_capture:
            self.video_capture.release()
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        
        # Stop forensic agent
        self.forensic_agent.stop()
        
        # Stop search engine
        self.search_engine.stop()
        
        self.logger.info("Iron-Sight pipeline stopped")
    
    def _capture_loop(self) -> None:
        """
        Stream A: Fast capture and visualization loop.
        Captures frames at 60 FPS and feeds both visualization and processing.
        """
        self.logger.info("Capture loop started")
        
        while self.is_running and self.video_capture:
            try:
                # Capture frame
                ret, frame = self.video_capture.read()
                if not ret:
                    self.logger.warning("Failed to read frame from video source")
                    continue
                
                capture_time = time.time()
                
                # Stream A: Fast visualization (resize and send to dashboard)
                viz_frame = cv2.resize(frame, (1280, 720))  # 720p for dashboard
                self._notify_dashboard("frame", viz_frame, capture_time)
                
                # Stream B: Queue for AI processing (non-blocking)
                try:
                    # If queue is full, drop oldest frame and add newest
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Drop oldest
                            self.state.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put_nowait((frame, capture_time))
                    
                except queue.Full:
                    self.state.frames_dropped += 1
                
                # Update FPS tracking
                self._update_fps_stats(capture_time)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                
        self.logger.info("Capture loop ended")
    
    def _process_loop(self) -> None:
        """
        Stream B: AI inspection processing loop.
        Processes frames through the full AI pipeline with SmolVLM fallback.
        """
        self.logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                
                frame, capture_time = frame_data
                process_start = time.time()
                
                # Process frame through AI pipeline
                result = self._process_frame(frame, capture_time)
                
                if result:
                    # Update processing stats
                    self.state.frames_processed += 1
                    self.state.current_latency_ms = result.processing_time_ms
                    
                    # Send result to dashboard
                    self._notify_dashboard("result", result, process_start)
                    
                    # Queue result for logging
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        self.logger.warning("Result queue full, dropping result")
                
                # Mark frame as processed
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                
        self.logger.info("Processing loop ended")
    
    def _process_frame(self, frame: np.ndarray, capture_time: float) -> Optional[InspectionResult]:
        """
        Process a single frame through the AI inspection pipeline.
        
        Args:
            frame: Input frame (BGR)
            capture_time: Timestamp when frame was captured
            
        Returns:
            InspectionResult if processing successful, None otherwise
        """
        frame_id = self.state.frames_processed + 1
        latency = LatencyMetrics(0, 0, 0, 0, 0, 0, 0)
        
        try:
            # Stage 1: Gatekeeper pre-filtering
            gate_start = time.time()
            is_wagon_present, is_blurry = self._run_gatekeeper(frame)
            latency.gatekeeper_ms = (time.time() - gate_start) * 1000
            
            if not is_wagon_present:
                self.state.gatekeeper_skips += 1
                return None  # Skip frame
            
            # Stage 2: Spectral decomposition
            channels = self._extract_spectral_channels(frame)
            
            # Stage 3: Low-light enhancement using SCI
            enhance_start = time.time()
            enhanced_frame, enhancement_applied = self._enhance_frame_sci(frame)
            latency.enhancement_ms = (time.time() - enhance_start) * 1000
            
            # Stage 4: Wagon detection
            detect_start = time.time()
            detections = self._detect_wagons(enhanced_frame)
            latency.detection_ms = (time.time() - detect_start) * 1000
            
            if not detections:
                return None  # No wagons detected
            
            # Process the best detection
            best_detection = max(detections, key=lambda d: d.confidence)
            
            # Stage 5: Crop extraction
            crop_start = time.time()
            crop = self._extract_crop(enhanced_frame, best_detection)
            latency.crop_ms = (time.time() - crop_start) * 1000
            
            # Stage 6: Conditional deblurring
            deblur_start = time.time()
            deblurred_crop, deblur_applied = self._conditional_deblur(
                crop, is_blurry, best_detection.confidence
            )
            latency.deblur_ms = (time.time() - deblur_start) * 1000
            
            # Stage 7: OCR with SmolVLM fallback
            ocr_start = time.time()
            ocr_result = self._run_ocr_with_fallback(deblurred_crop, best_detection)
            latency.ocr_ms = (time.time() - ocr_start) * 1000
            
            # Calculate total latency
            latency.total_ms = sum([
                latency.gatekeeper_ms, latency.enhancement_ms, latency.detection_ms,
                latency.crop_ms, latency.deblur_ms, latency.ocr_ms
            ])
            
            # Check for damage assessment
            if best_detection.class_name == "damage_door":
                self._queue_damage_assessment(deblurred_crop, best_detection)
            
            # Save crop and add to semantic search (background thread)
            self._save_and_index_crop(deblurred_crop, best_detection, result)
            
            # Create inspection result
            result = InspectionResult(
                frame_id=frame_id,
                timestamp=datetime.fromtimestamp(capture_time),
                wagon_id=ocr_result.text if ocr_result.is_valid_wagon_id else None,
                detection_confidence=best_detection.confidence,
                ocr_confidence=ocr_result.confidence,
                blur_score=0.5 if is_blurry else 0.8,  # Placeholder
                enhancement_applied=enhancement_applied,
                deblur_applied=deblur_applied,
                processing_time_ms=int(latency.total_ms),
                spectral_channel="red",
                bounding_box=asdict(best_detection),
                wagon_angle=best_detection.angle,
                fallback_ocr_used=getattr(ocr_result, 'fallback_used', False),
                damage_assessment=None  # Will be updated by damage assessment callback
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            return None
    
    def _run_gatekeeper(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """Run gatekeeper model to check for wagon presence and blur."""
        # Convert to grayscale and resize to 64x64
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(gray, (64, 64))
        
        # Run gatekeeper model
        is_wagon, is_blurry = self.gatekeeper.predict(thumbnail)
        return is_wagon, is_blurry
    
    def _extract_spectral_channels(self, frame: np.ndarray) -> SpectralChannels:
        """Extract red and saturation channels for specialized processing."""
        # Red channel for OCR (maximum contrast for white text)
        red_channel = frame[:, :, 2]
        
        # Saturation channel for damage detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv[:, :, 1]
        
        return SpectralChannels(
            red=red_channel,
            saturation=saturation_channel,
            original=frame
        )
    
    def _enhance_frame_sci(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Enhance frame using SCI (Self-Calibrated Illumination) for low-light conditions.
        
        SCI is ~6x faster than Zero-DCE (~0.5ms vs ~3ms) and provides better
        low-light enhancement with automatic daytime skip optimization.
        
        Args:
            frame: Input frame [H, W, C] in BGR format
            
        Returns:
            Tuple of (enhanced_frame, enhancement_applied)
        """
        try:
            # SCI automatically checks brightness and skips enhancement for daytime
            enhanced_frame, processing_info = self.sci_preprocessor.enhance_image(frame)
            
            # Log performance if enhancement was applied
            if processing_info['enhanced']:
                self.logger.debug(f"SCI enhancement: {processing_info['processing_time_ms']:.2f}ms")
            
            return enhanced_frame, processing_info['enhanced']
            
        except Exception as e:
            self.logger.warning(f"SCI enhancement failed: {e}")
            return frame, False
    
    def _detect_wagons(self, image: np.ndarray) -> List[Detection]:
        """Detect wagon components using YOLO."""
        return self.yolo_detector.detect(image)
    
    def _extract_crop(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        """Extract crop with 10% padding around detection."""
        h, w = image.shape[:2]
        
        # Calculate crop bounds with padding
        padding = 0.1
        x1 = max(0, int(detection.x - detection.width * (0.5 + padding)))
        y1 = max(0, int(detection.y - detection.height * (0.5 + padding)))
        x2 = min(w, int(detection.x + detection.width * (0.5 + padding)))
        y2 = min(h, int(detection.y + detection.height * (0.5 + padding)))
        
        return image[y1:y2, x1:x2]
    
    def _conditional_deblur(
        self, 
        crop: np.ndarray, 
        is_blurry: bool, 
        detection_confidence: float
    ) -> Tuple[np.ndarray, bool]:
        """Apply deblurring only when necessary."""
        # THE FALLBACK TRIGGER: Apply deblur if blurry AND low confidence
        if is_blurry and detection_confidence < 0.6:
            try:
                deblurred = self.deblur_gan.deblur(crop)
                return deblurred, True
            except Exception as e:
                self.logger.warning(f"Deblurring failed: {e}")
                return crop, False
        
        return crop, False
    
    def _run_ocr_with_fallback(self, crop: np.ndarray, detection: Detection) -> OCRResult:
        """
        Run OCR with SmolVLM fallback for difficult cases.
        This implements the core fallback logic requested.
        """
        # Primary OCR: PaddleOCR
        try:
            text_result, confidence = self.paddle_ocr.predict(crop)
            
            # THE FALLBACK TRIGGER: If confidence < 0.50, use SmolVLM
            if confidence < 0.50:
                self.logger.info(f"OCR confidence {confidence:.2f} < 0.50, triggering SmolVLM fallback")
                self.state.ocr_fallbacks += 1
                
                # Queue for SmolVLM analysis (async)
                bbox = {
                    "x": detection.x,
                    "y": detection.y,
                    "width": detection.width,
                    "height": detection.height
                }
                
                # Try synchronous fallback first (with timeout)
                forensic_result = self.forensic_agent.analyze_crop(
                    image=crop,
                    prompt="Read all the text visible in this image. Focus on any alphanumeric codes, serial numbers, or identification text.",
                    task_type="ocr_fallback",
                    bbox=bbox,
                    blocking=True  # Wait for result
                )
                
                if forensic_result and forensic_result.success and forensic_result.confidence > confidence:
                    self.logger.info(f"SmolVLM improved OCR: '{forensic_result.text}' (conf: {forensic_result.confidence:.2f})")
                    return OCRResult(
                        text=forensic_result.text,
                        confidence=forensic_result.confidence,
                        is_valid_wagon_id=self._validate_wagon_id(forensic_result.text),
                        raw_results=[(forensic_result.text, forensic_result.confidence)]
                    )
                else:
                    # Fallback to PaddleOCR result with placeholder display
                    return OCRResult(
                        text="Analyzing...",  # Placeholder as requested
                        confidence=confidence,
                        is_valid_wagon_id=False,
                        raw_results=[(text_result, confidence)]
                    )
            else:
                # PaddleOCR confidence is good enough
                return OCRResult(
                    text=text_result,
                    confidence=confidence,
                    is_valid_wagon_id=self._validate_wagon_id(text_result),
                    raw_results=[(text_result, confidence)]
                )
                
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                is_valid_wagon_id=False,
                raw_results=[]
            )
    
    def _queue_damage_assessment(self, crop: np.ndarray, detection: Detection) -> None:
        """
        Queue damage assessment for detected damage.
        This implements the "Bonus" feature requested.
        """
        if detection.class_name == "damage_door":
            self.state.damage_assessments += 1
            
            bbox = {
                "x": detection.x,
                "y": detection.y,
                "width": detection.width,
                "height": detection.height
            }
            
            # Queue damage assessment (async)
            self.forensic_agent.analyze_crop(
                image=crop,
                prompt="Describe the damage severity: is it a dent, a hole, or rust?",
                task_type="damage_assessment",
                bbox=bbox,
                callback=self._handle_damage_result
            )
    
    def _handle_damage_result(self, result: ForensicResult) -> None:
        """Handle damage assessment result."""
        if result.success:
            self.logger.info(f"Damage assessment: {result.text}")
            # Send to dashboard log
            self._notify_dashboard("damage_report", {
                "description": result.text,
                "confidence": result.confidence,
                "timestamp": datetime.now()
            }, time.time())
    
    def _save_and_index_crop(
        self, 
        crop: np.ndarray, 
        detection: Detection, 
        result: InspectionResult
    ) -> None:
        """
        Save wagon crop and add to semantic search index (background processing).
        This runs in lowest priority to not affect main pipeline performance.
        """
        try:
            # Generate unique filename
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            wagon_id_str = result.wagon_id or "unknown"
            filename = f"{timestamp_str}_{wagon_id_str}_{result.frame_id}.jpg"
            image_path = self.crop_storage_dir / filename
            
            # Save crop image
            cv2.imwrite(str(image_path), crop)
            
            # Prepare data for semantic search
            inspection_data = {
                "timestamp": result.timestamp,
                "frame_id": result.frame_id,
                "wagon_id": result.wagon_id,
                "detection_confidence": result.detection_confidence,
                "ocr_confidence": result.ocr_confidence,
                "blur_score": result.blur_score,
                "enhancement_applied": result.enhancement_applied,
                "deblur_applied": result.deblur_applied,
                "fallback_ocr_used": result.fallback_ocr_used,
                "damage_assessment": result.damage_assessment,
                "bounding_box": result.bounding_box,
                "wagon_angle": result.wagon_angle
            }
            
            # Queue for embedding generation (lowest priority background task)
            success = self.search_engine.add_wagon_embedding(
                image_array=crop,
                image_path=str(image_path),
                inspection_result=inspection_data,
                priority=False  # Lowest priority
            )
            
            if not success:
                self.logger.debug(f"Failed to queue crop for semantic indexing: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Error saving/indexing crop: {e}")
    
    def search_wagon_history(
        self,
        query: str,
        limit: int = 10,
        time_filter: Optional[Tuple[datetime, datetime]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search wagon inspection history using natural language.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            time_filter: Optional (start_time, end_time) filter
            min_confidence: Minimum detection confidence filter
            
        Returns:
            List of search results with similarity scores
        """
        try:
            results = self.search_engine.search(
                query=query,
                limit=limit,
                time_filter=time_filter,
                min_confidence=min_confidence
            )
            
            # Convert to dictionary format for easy serialization
            search_results = []
            for result in results:
                search_result = {
                    "wagon_id": result.wagon_id,
                    "timestamp": result.timestamp.isoformat(),
                    "image_path": result.image_path,
                    "similarity_score": result.similarity_score,
                    "inspection_data": result.inspection_data
                }
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error searching wagon history: {e}")
            return []
        """Validate wagon ID against pattern [A-Z]{4}\\d{6}."""
        import re
        pattern = re.compile(r'^[A-Z]{4}\d{6}$')
        return bool(pattern.match(text.strip()))
    
    def _update_fps_stats(self, current_time: float) -> None:
    
    def _validate_wagon_id(self, text: str) -> bool:
        """Update FPS and performance statistics."""
        self.frame_times.append(current_time)
        
        # Keep only last 60 frame times (1 second at 60 FPS)
        if len(self.frame_times) > 60:
            self.frame_times = self.frame_times[-60:]
        
        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            if len(self.frame_times) > 1:
                time_span = self.frame_times[-1] - self.frame_times[0]
                if time_span > 0:
                    self.state.avg_fps = (len(self.frame_times) - 1) / time_span
            
            self.state.queue_depth = self.frame_queue.qsize()
            self.last_fps_update = current_time
    
    def _notify_dashboard(self, event_type: str, data: Any, timestamp: float) -> None:
        """Notify dashboard callbacks of events."""
        for callback in self.dashboard_callbacks:
            try:
                callback(event_type, data, timestamp)
            except Exception as e:
                self.logger.error(f"Dashboard callback error: {e}")
    
    def get_state(self) -> ProcessingState:
        """Get current processing state for monitoring."""
        return self.state
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get semantic search engine statistics."""
        return self.search_engine.get_stats()


# Mock classes for development (replace with actual implementations)
class MockGatekeeper:
    def predict(self, thumbnail: np.ndarray) -> Tuple[bool, bool]:
        # Simulate gatekeeper logic
        mean_intensity = np.mean(thumbnail)
        is_wagon = mean_intensity > 50  # Simple threshold
        is_blurry = np.std(thumbnail) < 20  # Low variance = blurry
        return is_wagon, is_blurry


class MockYOLODetector:
    def detect(self, image: np.ndarray) -> List[Detection]:
        # Mock detection - return a single wagon detection
        h, w = image.shape[:2]
        return [Detection(
            x=w//2, y=h//2, width=w//3, height=h//4,
            angle=0.0, confidence=0.75, class_id=0, class_name="wagon_body"
        )]


class MockDeblurGAN:
    def deblur(self, crop: np.ndarray) -> np.ndarray:
        # Simple sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(crop, -1, kernel)


class MockPaddleOCR:
    def predict(self, crop: np.ndarray) -> Tuple[str, float]:
        # Mock OCR - simulate varying confidence
        import random
        confidence = random.uniform(0.3, 0.9)
        if confidence > 0.5:
            text = "ABCD123456"  # Valid wagon ID
        else:
            text = "unclear_text"
        return text, confidence
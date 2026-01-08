"""
SmolVLM 2 Forensic Agent for OCR Fallback and Damage Analysis.

This module provides a fallback OCR system using SmolVLM2-256M-Video-Instruct
for cases where PaddleOCR fails due to severe rust, damage, or poor image quality.
Also provides damage assessment capabilities for detected defects.
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2


@dataclass
class ForensicTask:
    """Task for the forensic agent queue."""
    image: np.ndarray
    task_type: str  # "ocr_fallback" or "damage_assessment"
    bbox: Dict[str, Any]
    prompt: Optional[str] = None
    callback: Optional[callable] = None


@dataclass
class ForensicResult:
    """Result from forensic analysis."""
    task_type: str
    text: str
    confidence: float
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


class SmolVLMForensicAgent:
    """
    SmolVLM 2 based forensic agent for OCR fallback and damage analysis.
    
    Uses HuggingFaceTB/SmolVLM2-256M-Video-Instruct with quantization
    to fit within Jetson memory constraints while providing superior
    text recognition on damaged/rusted surfaces.
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        device: str = "auto",
        quantization_bits: int = 8,
        max_queue_size: int = 10,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize the forensic agent.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("auto", "cuda", "cpu")
            quantization_bits: 4 or 8 bit quantization (8 recommended for Jetson)
            max_queue_size: Maximum number of queued tasks
            timeout_seconds: Maximum time per inference
        """
        self.model_name = model_name
        self.device = device
        self.quantization_bits = quantization_bits
        self.timeout_seconds = timeout_seconds
        
        # Task queue and processing thread
        self.task_queue: queue.Queue[ForensicTask] = queue.Queue(maxsize=max_queue_size)
        self.result_cache: Dict[str, ForensicResult] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Model components (loaded lazily)
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForVision2Seq] = None
        self.model_loaded = False
        
        # Performance tracking
        self.total_inferences = 0
        self.total_processing_time_ms = 0
        self.failed_inferences = 0
        
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self) -> bool:
        """
        Load the SmolVLM model with quantization.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger.info(f"Loading SmolVLM model: {self.model_name}")
            start_time = time.time()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.quantization_bits == 4:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization_bits == 8:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            # Load model with quantization
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "auto" else None,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"SmolVLM model loaded in {load_time:.2f}s")
            self.model_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load SmolVLM model: {e}")
            self.model_loaded = False
            return False
    
    def start(self) -> bool:
        """
        Start the forensic agent processing thread.
        Returns True if started successfully.
        """
        if self.is_running:
            self.logger.warning("Forensic agent already running")
            return True
            
        # Load model if not already loaded
        if not self.model_loaded:
            if not self._load_model():
                return False
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="SmolVLM-ForensicAgent",
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("SmolVLM Forensic Agent started")
        return True
    
    def stop(self) -> None:
        """Stop the forensic agent processing thread."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Add sentinel to wake up processing thread
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            
        self.logger.info("SmolVLM Forensic Agent stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        self.logger.info("SmolVLM processing loop started")
        
        while self.is_running:
            try:
                # Get task from queue (blocking with timeout)
                task = self.task_queue.get(timeout=1.0)
                
                # Sentinel value to stop processing
                if task is None:
                    break
                
                # Process the task
                result = self._process_task(task)
                
                # Store result in cache for retrieval
                task_id = self._generate_task_id(task)
                self.result_cache[task_id] = result
                
                # Call callback if provided
                if task.callback:
                    try:
                        task.callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                
        self.logger.info("SmolVLM processing loop ended")
    
    def _process_task(self, task: ForensicTask) -> ForensicResult:
        """
        Process a single forensic task.
        
        Args:
            task: The forensic task to process
            
        Returns:
            ForensicResult with analysis results
        """
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image
            if task.image.dtype != np.uint8:
                image_uint8 = (task.image * 255).astype(np.uint8)
            else:
                image_uint8 = task.image
                
            # Convert BGR to RGB if needed
            if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_uint8
                
            pil_image = Image.fromarray(image_rgb)
            
            # Generate appropriate prompt based on task type
            if task.task_type == "ocr_fallback":
                prompt = task.prompt or "Read all the text visible in this image. Focus on any alphanumeric codes, serial numbers, or identification text."
            elif task.task_type == "damage_assessment":
                prompt = task.prompt or "Describe the damage severity visible in this image: is it a dent, a hole, rust, or other type of damage? Provide a brief assessment."
            else:
                prompt = task.prompt or "Analyze this image and describe what you see."
            
            # Process with SmolVLM
            result_text = self._analyze_image(pil_image, prompt)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update statistics
            self.total_inferences += 1
            self.total_processing_time_ms += processing_time_ms
            
            # Estimate confidence based on response quality
            confidence = self._estimate_confidence(result_text, task.task_type)
            
            return ForensicResult(
                task_type=task.task_type,
                text=result_text,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                success=True
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.failed_inferences += 1
            
            self.logger.error(f"Error processing forensic task: {e}")
            
            return ForensicResult(
                task_type=task.task_type,
                text="",
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
    
    def _analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze image with SmolVLM model.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the analysis
            
        Returns:
            Generated text response
        """
        if not self.model_loaded or not self.processor or not self.model:
            raise RuntimeError("SmolVLM model not loaded")
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response with timeout
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated text (remove prompt)
        if prompt in response:
            generated_text = response.split(prompt)[-1].strip()
        else:
            generated_text = response.strip()
            
        return generated_text
    
    def _estimate_confidence(self, text: str, task_type: str) -> float:
        """
        Estimate confidence based on response quality.
        
        Args:
            text: Generated text response
            task_type: Type of task performed
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        if task_type == "ocr_fallback":
            # Higher confidence for responses with alphanumeric patterns
            import re
            if re.search(r'[A-Z]{2,}', text):  # Contains uppercase letters
                confidence += 0.2
            if re.search(r'\d{3,}', text):     # Contains number sequences
                confidence += 0.2
            if re.search(r'[A-Z]{4}\d{6}', text):  # Wagon ID pattern
                confidence += 0.3
        
        elif task_type == "damage_assessment":
            # Higher confidence for damage-related keywords
            damage_keywords = ['dent', 'hole', 'rust', 'scratch', 'damage', 'corrosion', 'wear']
            for keyword in damage_keywords:
                if keyword.lower() in text.lower():
                    confidence += 0.1
                    break
        
        return min(confidence, 1.0)
    
    def _generate_task_id(self, task: ForensicTask) -> str:
        """Generate unique ID for task result caching."""
        bbox_str = f"{task.bbox.get('x', 0):.1f}_{task.bbox.get('y', 0):.1f}"
        return f"{task.task_type}_{bbox_str}_{int(time.time() * 1000)}"
    
    def analyze_crop(
        self,
        image: np.ndarray,
        prompt: str = "Read the text inside this image",
        task_type: str = "ocr_fallback",
        bbox: Optional[Dict[str, Any]] = None,
        callback: Optional[callable] = None,
        blocking: bool = False
    ) -> Optional[ForensicResult]:
        """
        Analyze a cropped image region.
        
        Args:
            image: Numpy array image (BGR or grayscale)
            prompt: Analysis prompt
            task_type: Type of analysis ("ocr_fallback" or "damage_assessment")
            bbox: Bounding box information for tracking
            callback: Optional callback function for async results
            blocking: If True, wait for result synchronously
            
        Returns:
            ForensicResult if blocking=True, None otherwise
        """
        if not self.is_running:
            self.logger.warning("Forensic agent not running")
            return None
        
        if bbox is None:
            bbox = {"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]}
        
        task = ForensicTask(
            image=image,
            task_type=task_type,
            bbox=bbox,
            prompt=prompt,
            callback=callback
        )
        
        try:
            if blocking:
                # Process synchronously
                result = self._process_task(task)
                return result
            else:
                # Queue for async processing
                self.task_queue.put_nowait(task)
                return None
                
        except queue.Full:
            self.logger.warning("Forensic agent queue full, dropping task")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = 0.0
        if self.total_inferences > 0:
            avg_time = self.total_processing_time_ms / self.total_inferences
            
        return {
            "model_loaded": self.model_loaded,
            "is_running": self.is_running,
            "queue_size": self.task_queue.qsize(),
            "total_inferences": self.total_inferences,
            "failed_inferences": self.failed_inferences,
            "success_rate": (self.total_inferences - self.failed_inferences) / max(self.total_inferences, 1),
            "avg_processing_time_ms": avg_time,
            "cached_results": len(self.result_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.result_cache.clear()
        self.logger.info("Forensic agent cache cleared")


# Global instance for easy access
_forensic_agent: Optional[SmolVLMForensicAgent] = None


def get_forensic_agent() -> SmolVLMForensicAgent:
    """Get or create the global forensic agent instance."""
    global _forensic_agent
    if _forensic_agent is None:
        _forensic_agent = SmolVLMForensicAgent()
    return _forensic_agent


def initialize_forensic_agent(**kwargs) -> bool:
    """Initialize the global forensic agent with custom parameters."""
    global _forensic_agent
    _forensic_agent = SmolVLMForensicAgent(**kwargs)
    return _forensic_agent.start()


def shutdown_forensic_agent() -> None:
    """Shutdown the global forensic agent."""
    global _forensic_agent
    if _forensic_agent:
        _forensic_agent.stop()
        _forensic_agent = None
"""
SmolVLM Integration Module for IronSight Command Center.

This module provides a wrapper around the existing SmolVLM forensic agent
(src/agent_forensic.py) with specific prompts for OCR and damage assessment,
8-bit quantization for Jetson compatibility, and timeout handling.

Requirements: 7.1, 7.5, 14.2
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.agent_forensic import (
        ForensicResult,
        ForensicTask,
        SmolVLMForensicAgent,
        get_forensic_agent,
        initialize_forensic_agent,
        shutdown_forensic_agent,
    )
    FORENSIC_AGENT_AVAILABLE = True
except Exception:  # Catch all exceptions including OSError for DLL issues
    FORENSIC_AGENT_AVAILABLE = False
    ForensicResult = None
    ForensicTask = None
    SmolVLMForensicAgent = None
    
    # Create mock functions
    def get_forensic_agent():
        return None
    
    def initialize_forensic_agent(**kwargs):
        return False
    
    def shutdown_forensic_agent():
        pass


@dataclass
class OCRResult:
    """Result from OCR analysis."""
    text: str
    confidence: float
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass
class DamageAssessmentResult:
    """Result from damage assessment analysis."""
    assessment: str
    severity: str  # "minor", "moderate", "severe", "unknown"
    damage_type: str  # "dent", "hole", "rust", "scratch", "unknown"
    confidence: float
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


class SmolVLMIntegration:
    """
    Wrapper for existing SmolVLM forensic agent with specific prompts
    for OCR and damage assessment in railway wagon inspection.
    
    Features:
    - Specific prompts for OCR and damage assessment
    - 8-bit quantization for Jetson compatibility
    - Timeout handling and async processing
    - Performance tracking and statistics
    """
    
    # Default prompts for different analysis types
    OCR_PROMPT = (
        "Read the serial number painted on this metal surface. "
        "Return only the alphanumeric string. If no text is visible, return 'UNREADABLE'."
    )
    
    DAMAGE_PROMPT = (
        "Describe the damage severity visible in this image. "
        "Classify as: dent, hole, rust, scratch, or other. "
        "Rate severity as: minor, moderate, or severe. "
        "Format: TYPE: [type], SEVERITY: [severity], DESCRIPTION: [brief description]"
    )
    
    GENERAL_ANALYSIS_PROMPT = (
        "Analyze this railway wagon component image. "
        "Describe any visible defects, damage, or notable features."
    )
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        quantization_bits: int = 8,
        timeout_seconds: float = 30.0,
        max_queue_size: int = 10,
        device: str = "auto",
        lazy_load: bool = True
    ):
        """
        Initialize SmolVLM integration.
        
        Args:
            model_name: HuggingFace model identifier
            quantization_bits: 4 or 8 bit quantization (8 recommended for Jetson)
            timeout_seconds: Maximum time per inference
            max_queue_size: Maximum number of queued tasks
            device: Device to run on ("auto", "cuda", "cpu")
            lazy_load: If True, defer model loading until first use
        """
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.timeout_seconds = timeout_seconds
        self.max_queue_size = max_queue_size
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Forensic agent instance
        self._forensic_agent: Optional[SmolVLMForensicAgent] = None
        self._is_initialized = False
        self._is_available = FORENSIC_AGENT_AVAILABLE
        
        # Thread pool for async processing with timeout
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SmolVLM")
        
        # Performance tracking
        self._total_ocr_requests = 0
        self._total_damage_requests = 0
        self._successful_ocr = 0
        self._successful_damage = 0
        self._total_ocr_time_ms = 0
        self._total_damage_time_ms = 0
        self._timeout_count = 0
        
        # Initialize immediately if not lazy loading
        if not lazy_load:
            self._initialize_agent()
    
    def _initialize_agent(self) -> bool:
        """
        Initialize the forensic agent with configured parameters.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            return True
        
        if not self._is_available:
            self.logger.warning(
                "SmolVLM forensic agent not available. "
                "Install required dependencies or check import path."
            )
            return False
        
        try:
            self.logger.info(
                f"Initializing SmolVLM agent with {self.quantization_bits}-bit quantization"
            )
            
            # Initialize the global forensic agent with our configuration
            success = initialize_forensic_agent(
                model_name=self.model_name,
                device=self.device,
                quantization_bits=self.quantization_bits,
                max_queue_size=self.max_queue_size,
                timeout_seconds=self.timeout_seconds
            )
            
            if success:
                self._forensic_agent = get_forensic_agent()
                self._is_initialized = True
                self.logger.info("SmolVLM agent initialized successfully")
            else:
                self.logger.error("Failed to initialize SmolVLM agent")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing SmolVLM agent: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if SmolVLM integration is available."""
        return self._is_available
    
    def is_initialized(self) -> bool:
        """Check if SmolVLM agent is initialized and ready."""
        return self._is_initialized and self._forensic_agent is not None
    
    def ensure_initialized(self) -> bool:
        """Ensure agent is initialized, initializing if necessary."""
        if not self._is_initialized:
            return self._initialize_agent()
        return True
    
    def analyze_identification_plate(
        self,
        crop: np.ndarray,
        custom_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> OCRResult:
        """
        Analyze identification plate with specific OCR prompt.
        
        Args:
            crop: Numpy array of the plate crop (BGR or grayscale)
            custom_prompt: Optional custom prompt to override default
            timeout: Optional timeout override
            
        Returns:
            OCRResult with extracted text and metadata
        """
        self._total_ocr_requests += 1
        start_time = time.time()
        timeout = timeout or self.timeout_seconds
        
        # Check if agent is available
        if not self.ensure_initialized():
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time_ms=0,
                success=False,
                error_message="SmolVLM agent not available",
                fallback_used=True
            )
        
        prompt = custom_prompt or self.OCR_PROMPT
        
        try:
            # Use thread pool with timeout for blocking call
            future = self._executor.submit(
                self._forensic_agent.analyze_crop,
                image=crop,
                prompt=prompt,
                task_type="ocr_fallback",
                bbox={"x": 0, "y": 0, "width": crop.shape[1], "height": crop.shape[0]},
                blocking=True
            )
            
            result = future.result(timeout=timeout)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            if result and result.success:
                self._successful_ocr += 1
                self._total_ocr_time_ms += processing_time_ms
                
                # Clean up the extracted text
                cleaned_text = self._clean_ocr_text(result.text)
                
                return OCRResult(
                    text=cleaned_text,
                    confidence=result.confidence,
                    processing_time_ms=processing_time_ms,
                    success=True,
                    fallback_used=True
                )
            else:
                error_msg = result.error_message if result else "No result returned"
                return OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time_ms=processing_time_ms,
                    success=False,
                    error_message=error_msg,
                    fallback_used=True
                )
                
        except FuturesTimeoutError:
            self._timeout_count += 1
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.warning(f"OCR analysis timed out after {timeout}s")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=f"Timeout after {timeout}s",
                fallback_used=True
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error in OCR analysis: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e),
                fallback_used=True
            )
    
    def assess_damage(
        self,
        crop: np.ndarray,
        custom_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        blocking: bool = True
    ) -> DamageAssessmentResult:
        """
        Assess damage with specific damage prompt.
        
        Args:
            crop: Numpy array of the damage crop (BGR or grayscale)
            custom_prompt: Optional custom prompt to override default
            timeout: Optional timeout override
            blocking: If True, wait for result; if False, return immediately
            
        Returns:
            DamageAssessmentResult with assessment and metadata
        """
        self._total_damage_requests += 1
        start_time = time.time()
        timeout = timeout or self.timeout_seconds
        
        # Check if agent is available
        if not self.ensure_initialized():
            return DamageAssessmentResult(
                assessment="Assessment unavailable",
                severity="unknown",
                damage_type="unknown",
                confidence=0.0,
                processing_time_ms=0,
                success=False,
                error_message="SmolVLM agent not available"
            )
        
        prompt = custom_prompt or self.DAMAGE_PROMPT
        
        try:
            if blocking:
                # Use thread pool with timeout for blocking call
                future = self._executor.submit(
                    self._forensic_agent.analyze_crop,
                    image=crop,
                    prompt=prompt,
                    task_type="damage_assessment",
                    bbox={"x": 0, "y": 0, "width": crop.shape[1], "height": crop.shape[0]},
                    blocking=True
                )
                
                result = future.result(timeout=timeout)
            else:
                # Non-blocking: queue for async processing
                self._forensic_agent.analyze_crop(
                    image=crop,
                    prompt=prompt,
                    task_type="damage_assessment",
                    bbox={"x": 0, "y": 0, "width": crop.shape[1], "height": crop.shape[0]},
                    blocking=False
                )
                
                return DamageAssessmentResult(
                    assessment="Assessment pending",
                    severity="unknown",
                    damage_type="unknown",
                    confidence=0.0,
                    processing_time_ms=0,
                    success=True,
                    error_message=None
                )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            if result and result.success:
                self._successful_damage += 1
                self._total_damage_time_ms += processing_time_ms
                
                # Parse the damage assessment response
                severity, damage_type, description = self._parse_damage_response(result.text)
                
                return DamageAssessmentResult(
                    assessment=description,
                    severity=severity,
                    damage_type=damage_type,
                    confidence=result.confidence,
                    processing_time_ms=processing_time_ms,
                    success=True
                )
            else:
                error_msg = result.error_message if result else "No result returned"
                return DamageAssessmentResult(
                    assessment="Assessment failed",
                    severity="unknown",
                    damage_type="unknown",
                    confidence=0.0,
                    processing_time_ms=processing_time_ms,
                    success=False,
                    error_message=error_msg
                )
                
        except FuturesTimeoutError:
            self._timeout_count += 1
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.warning(f"Damage assessment timed out after {timeout}s")
            
            return DamageAssessmentResult(
                assessment="Assessment timed out",
                severity="unknown",
                damage_type="unknown",
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=f"Timeout after {timeout}s"
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error in damage assessment: {e}")
            
            return DamageAssessmentResult(
                assessment="Assessment error",
                severity="unknown",
                damage_type="unknown",
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
    
    def analyze_general(
        self,
        crop: np.ndarray,
        custom_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ForensicResult:
        """
        Perform general analysis with custom or default prompt.
        
        Args:
            crop: Numpy array of the image crop
            custom_prompt: Optional custom prompt
            timeout: Optional timeout override
            
        Returns:
            ForensicResult from the underlying agent
        """
        if not self.ensure_initialized():
            return ForensicResult(
                task_type="general",
                text="",
                confidence=0.0,
                processing_time_ms=0,
                success=False,
                error_message="SmolVLM agent not available"
            ) if ForensicResult else None
        
        prompt = custom_prompt or self.GENERAL_ANALYSIS_PROMPT
        timeout = timeout or self.timeout_seconds
        
        try:
            future = self._executor.submit(
                self._forensic_agent.analyze_crop,
                image=crop,
                prompt=prompt,
                task_type="general",
                bbox={"x": 0, "y": 0, "width": crop.shape[1], "height": crop.shape[0]},
                blocking=True
            )
            
            return future.result(timeout=timeout)
            
        except FuturesTimeoutError:
            self._timeout_count += 1
            self.logger.warning(f"General analysis timed out after {timeout}s")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in general analysis: {e}")
            return None
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up OCR text by removing common artifacts.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove common prefixes/suffixes from model responses
        text = text.strip()
        
        # Remove quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove common response prefixes
        prefixes_to_remove = [
            "The serial number is:",
            "The text reads:",
            "I can see:",
            "The alphanumeric string is:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        return text.strip()
    
    def _parse_damage_response(self, text: str) -> Tuple[str, str, str]:
        """
        Parse damage assessment response into structured fields.
        
        Args:
            text: Raw damage assessment text
            
        Returns:
            Tuple of (severity, damage_type, description)
        """
        severity = "unknown"
        damage_type = "unknown"
        description = text
        
        if not text:
            return severity, damage_type, description
        
        text_lower = text.lower()
        
        # Extract severity
        severity_keywords = {
            "minor": ["minor", "slight", "small", "minimal"],
            "moderate": ["moderate", "medium", "noticeable"],
            "severe": ["severe", "major", "significant", "extensive", "serious"]
        }
        
        for sev, keywords in severity_keywords.items():
            if any(kw in text_lower for kw in keywords):
                severity = sev
                break
        
        # Extract damage type
        damage_keywords = {
            "dent": ["dent", "dented", "indentation"],
            "hole": ["hole", "puncture", "perforation"],
            "rust": ["rust", "corrosion", "oxidation", "rusted"],
            "scratch": ["scratch", "scratched", "scrape", "abrasion"]
        }
        
        for dtype, keywords in damage_keywords.items():
            if any(kw in text_lower for kw in keywords):
                damage_type = dtype
                break
        
        # Try to extract structured response if formatted
        if "TYPE:" in text.upper() and "SEVERITY:" in text.upper():
            try:
                parts = text.split(",")
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith("TYPE:"):
                        damage_type = part.split(":", 1)[1].strip().lower()
                    elif part.upper().startswith("SEVERITY:"):
                        severity = part.split(":", 1)[1].strip().lower()
                    elif part.upper().startswith("DESCRIPTION:"):
                        description = part.split(":", 1)[1].strip()
            except Exception:
                pass  # Fall back to keyword extraction
        
        return severity, damage_type, description
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        agent_stats = {}
        if self._forensic_agent:
            agent_stats = self._forensic_agent.get_stats()
        
        avg_ocr_time = 0.0
        if self._successful_ocr > 0:
            avg_ocr_time = self._total_ocr_time_ms / self._successful_ocr
        
        avg_damage_time = 0.0
        if self._successful_damage > 0:
            avg_damage_time = self._total_damage_time_ms / self._successful_damage
        
        return {
            "is_available": self._is_available,
            "is_initialized": self._is_initialized,
            "model_name": self.model_name,
            "quantization_bits": self.quantization_bits,
            "timeout_seconds": self.timeout_seconds,
            "total_ocr_requests": self._total_ocr_requests,
            "successful_ocr": self._successful_ocr,
            "ocr_success_rate": self._successful_ocr / max(self._total_ocr_requests, 1),
            "avg_ocr_time_ms": avg_ocr_time,
            "total_damage_requests": self._total_damage_requests,
            "successful_damage": self._successful_damage,
            "damage_success_rate": self._successful_damage / max(self._total_damage_requests, 1),
            "avg_damage_time_ms": avg_damage_time,
            "timeout_count": self._timeout_count,
            "agent_stats": agent_stats
        }
    
    def shutdown(self) -> None:
        """Shutdown the SmolVLM integration and release resources."""
        self.logger.info("Shutting down SmolVLM integration")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=False)
        
        # Shutdown forensic agent
        if self._is_initialized:
            shutdown_forensic_agent()
            self._forensic_agent = None
            self._is_initialized = False
        
        self.logger.info("SmolVLM integration shutdown complete")


# Module-level instance for easy access
_smolvlm_integration: Optional[SmolVLMIntegration] = None


def get_smolvlm_integration() -> SmolVLMIntegration:
    """Get or create the global SmolVLM integration instance."""
    global _smolvlm_integration
    if _smolvlm_integration is None:
        _smolvlm_integration = SmolVLMIntegration(lazy_load=True)
    return _smolvlm_integration


def initialize_smolvlm_integration(**kwargs) -> bool:
    """Initialize the global SmolVLM integration with custom parameters."""
    global _smolvlm_integration
    _smolvlm_integration = SmolVLMIntegration(**kwargs)
    return _smolvlm_integration.ensure_initialized()


def shutdown_smolvlm_integration() -> None:
    """Shutdown the global SmolVLM integration."""
    global _smolvlm_integration
    if _smolvlm_integration:
        _smolvlm_integration.shutdown()
        _smolvlm_integration = None

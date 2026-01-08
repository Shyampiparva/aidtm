"""
SigLIP Semantic Search Integration Module for IronSight Command Center.

This module provides a wrapper around the existing SigLIP semantic search
(src/semantic_search.py) with background embedding generation for inspection
crops, natural language query processing, and LanceDB vector storage integration.

Requirements: 9.1, 9.2, 14.3
"""

import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue, Empty, Full

import numpy as np

# Add parent directory to path for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.semantic_search import (
        SearchResult,
        SemanticSearchEngine,
        WagonEmbedding,
        get_search_engine,
        initialize_search_engine,
        shutdown_search_engine,
    )
    SEMANTIC_SEARCH_AVAILABLE = True
except Exception:  # Catch all exceptions including OSError for DLL issues
    SEMANTIC_SEARCH_AVAILABLE = False
    SearchResult = None
    SemanticSearchEngine = None
    WagonEmbedding = None
    
    # Create mock functions
    def get_search_engine():
        return None
    
    def initialize_search_engine(**kwargs):
        return False
    
    def shutdown_search_engine():
        pass


@dataclass
class InspectionCrop:
    """Data class for inspection crop to be embedded."""
    image_array: np.ndarray
    image_path: str
    wagon_id: Optional[str]
    timestamp: datetime
    frame_id: int
    detection_confidence: float
    ocr_confidence: float
    blur_score: float
    enhancement_applied: bool
    deblur_applied: bool
    fallback_ocr_used: bool
    damage_assessment: Optional[str]
    bounding_box: Dict[str, Any]
    wagon_angle: float
    priority: bool = False


@dataclass
class QueryResult:
    """Result from a semantic search query."""
    wagon_id: Optional[str]
    timestamp: datetime
    image_path: str
    similarity_score: float
    inspection_data: Dict[str, Any]
    thumbnail: Optional[np.ndarray] = None


@dataclass
class SearchFilter:
    """Filters for semantic search queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: float = 0.0
    wagon_id_pattern: Optional[str] = None
    damage_types: Optional[List[str]] = None


class SigLIPIntegration:
    """
    Wrapper for existing SigLIP semantic search with UI integration.
    
    Features:
    - Background embedding generation for inspection crops
    - Natural language query processing interface
    - LanceDB vector storage integration
    - Performance tracking and statistics
    """
    
    def __init__(
        self,
        db_path: str = "data/wagon_embeddings.lancedb",
        model_name: str = "ViT-B-16-SigLIP-256",
        pretrained: str = "webli",
        device: str = "auto",
        batch_size: int = 32,
        max_queue_size: int = 1000,
        lazy_load: bool = True
    ):
        """
        Initialize SigLIP integration.
        
        Args:
            db_path: Path to LanceDB database
            model_name: SigLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to run on ("auto", "cuda", "cpu")
            batch_size: Batch size for embedding generation
            max_queue_size: Maximum number of queued embedding tasks
            lazy_load: If True, defer model loading until first use
        """
        self.db_path = db_path
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        
        self.logger = logging.getLogger(__name__)
        
        # Search engine instance
        self._search_engine: Optional[SemanticSearchEngine] = None
        self._is_initialized = False
        self._is_available = SEMANTIC_SEARCH_AVAILABLE
        
        # Background embedding queue (thread-safe)
        self._embedding_queue: Queue = Queue(maxsize=max_queue_size)
        self._background_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Performance tracking
        self._total_embeddings_queued = 0
        self._total_embeddings_processed = 0
        self._total_searches = 0
        self._total_search_time_ms = 0
        self._queue_drops = 0
        
        # Initialize immediately if not lazy loading
        if not lazy_load:
            self._initialize_engine()
    
    def _initialize_engine(self) -> bool:
        """
        Initialize the search engine with configured parameters.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            return True
        
        if not self._is_available:
            self.logger.warning(
                "SigLIP semantic search not available. "
                "Install required dependencies or check import path."
            )
            return False
        
        try:
            self.logger.info(f"Initializing SigLIP search engine with model: {self.model_name}")
            
            # Initialize the global search engine with our configuration
            success = initialize_search_engine(
                db_path=self.db_path,
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device,
                batch_size=self.batch_size,
                max_queue_size=self.max_queue_size
            )
            
            if success:
                self._search_engine = get_search_engine()
                self._is_initialized = True
                self._start_background_processing()
                self.logger.info("SigLIP search engine initialized successfully")
            else:
                self.logger.error("Failed to initialize SigLIP search engine")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing SigLIP search engine: {e}")
            return False
    
    def _start_background_processing(self) -> None:
        """Start background thread for processing embedding queue."""
        if self._is_running:
            return
        
        self._is_running = True
        self._background_thread = threading.Thread(
            target=self._background_processing_loop,
            name="SigLIP-BackgroundEmbedder",
            daemon=True
        )
        self._background_thread.start()
        self.logger.info("Background embedding processing started")
    
    def _stop_background_processing(self) -> None:
        """Stop background processing thread."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Add sentinel to wake up thread
        try:
            self._embedding_queue.put_nowait(None)
        except Full:
            pass
        
        # Wait for thread to finish
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)
        
        self.logger.info("Background embedding processing stopped")
    
    def _background_processing_loop(self) -> None:
        """Background loop for processing embedding queue."""
        self.logger.info("Background embedding loop started")
        
        while self._is_running:
            try:
                # Get item from queue with timeout
                try:
                    crop = self._embedding_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Sentinel value to stop
                if crop is None:
                    break
                
                # Process the crop
                self._process_embedding(crop)
                self._embedding_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in background processing loop: {e}")
        
        self.logger.info("Background embedding loop ended")
    
    def _process_embedding(self, crop: InspectionCrop) -> bool:
        """
        Process a single inspection crop for embedding.
        
        Args:
            crop: InspectionCrop to process
            
        Returns:
            True if successful, False otherwise
        """
        if not self._search_engine:
            return False
        
        try:
            # Convert to inspection result dict format expected by search engine
            inspection_result = {
                "timestamp": crop.timestamp,
                "frame_id": crop.frame_id,
                "wagon_id": crop.wagon_id,
                "detection_confidence": crop.detection_confidence,
                "ocr_confidence": crop.ocr_confidence,
                "blur_score": crop.blur_score,
                "enhancement_applied": crop.enhancement_applied,
                "deblur_applied": crop.deblur_applied,
                "fallback_ocr_used": crop.fallback_ocr_used,
                "damage_assessment": crop.damage_assessment,
                "bounding_box": crop.bounding_box,
                "wagon_angle": crop.wagon_angle
            }
            
            success = self._search_engine.add_wagon_embedding(
                image_array=crop.image_array,
                image_path=crop.image_path,
                inspection_result=inspection_result,
                priority=crop.priority
            )
            
            if success:
                self._total_embeddings_processed += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing embedding: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if SigLIP integration is available."""
        return self._is_available
    
    def is_initialized(self) -> bool:
        """Check if search engine is initialized and ready."""
        return self._is_initialized and self._search_engine is not None
    
    def ensure_initialized(self) -> bool:
        """Ensure engine is initialized, initializing if necessary."""
        if not self._is_initialized:
            return self._initialize_engine()
        return True
    
    def add_inspection_crop(
        self,
        image_array: np.ndarray,
        image_path: str,
        wagon_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        frame_id: int = 0,
        detection_confidence: float = 0.0,
        ocr_confidence: float = 0.0,
        blur_score: float = 0.0,
        enhancement_applied: bool = False,
        deblur_applied: bool = False,
        fallback_ocr_used: bool = False,
        damage_assessment: Optional[str] = None,
        bounding_box: Optional[Dict[str, Any]] = None,
        wagon_angle: float = 0.0,
        priority: bool = False
    ) -> bool:
        """
        Add an inspection crop for background embedding generation.
        
        Args:
            image_array: Numpy array of the crop (BGR or RGB)
            image_path: Path where the image is/will be saved
            wagon_id: Optional wagon identifier
            timestamp: Inspection timestamp (defaults to now)
            frame_id: Frame identifier
            detection_confidence: YOLO detection confidence
            ocr_confidence: OCR confidence score
            blur_score: Blur detection score
            enhancement_applied: Whether SCI enhancement was applied
            deblur_applied: Whether NAFNet deblurring was applied
            fallback_ocr_used: Whether SmolVLM fallback OCR was used
            damage_assessment: Optional damage assessment text
            bounding_box: Detection bounding box
            wagon_angle: Wagon angle in degrees
            priority: Whether to prioritize this embedding
            
        Returns:
            True if successfully queued, False otherwise
        """
        if not self.ensure_initialized():
            return False
        
        # Create inspection crop object
        crop = InspectionCrop(
            image_array=image_array,
            image_path=image_path,
            wagon_id=wagon_id,
            timestamp=timestamp or datetime.now(),
            frame_id=frame_id,
            detection_confidence=detection_confidence,
            ocr_confidence=ocr_confidence,
            blur_score=blur_score,
            enhancement_applied=enhancement_applied,
            deblur_applied=deblur_applied,
            fallback_ocr_used=fallback_ocr_used,
            damage_assessment=damage_assessment,
            bounding_box=bounding_box or {},
            wagon_angle=wagon_angle,
            priority=priority
        )
        
        try:
            # Try to add to queue (non-blocking)
            self._embedding_queue.put_nowait(crop)
            self._total_embeddings_queued += 1
            return True
            
        except Full:
            self._queue_drops += 1
            self.logger.warning("Embedding queue full, dropping crop")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[QueryResult]:
        """
        Search inspection history using natural language query.
        
        Args:
            query: Natural language search query (e.g., "wagons with rust damage")
            limit: Maximum number of results
            filters: Optional search filters
            
        Returns:
            List of QueryResult objects sorted by similarity
        """
        if not self.ensure_initialized():
            return []
        
        start_time = time.time()
        
        try:
            # Build time filter if provided
            time_filter = None
            if filters and (filters.start_date or filters.end_date):
                start = filters.start_date or datetime.min
                end = filters.end_date or datetime.max
                time_filter = (start, end)
            
            # Get minimum confidence
            min_confidence = filters.min_confidence if filters else 0.0
            
            # Perform search
            results = self._search_engine.search(
                query=query,
                limit=limit,
                time_filter=time_filter,
                min_confidence=min_confidence
            )
            
            # Convert to QueryResult objects
            query_results = []
            for result in results:
                query_result = QueryResult(
                    wagon_id=result.wagon_id,
                    timestamp=result.timestamp,
                    image_path=result.image_path,
                    similarity_score=result.similarity_score,
                    inspection_data=result.inspection_data
                )
                
                # Apply additional filters
                if filters:
                    # Filter by wagon ID pattern
                    if filters.wagon_id_pattern and query_result.wagon_id:
                        if filters.wagon_id_pattern.lower() not in query_result.wagon_id.lower():
                            continue
                    
                    # Filter by damage types
                    if filters.damage_types:
                        damage = query_result.inspection_data.get("damage_assessment", "")
                        if damage:
                            damage_lower = damage.lower()
                            if not any(dt.lower() in damage_lower for dt in filters.damage_types):
                                continue
                
                query_results.append(query_result)
            
            # Update statistics
            search_time_ms = (time.time() - start_time) * 1000
            self._total_searches += 1
            self._total_search_time_ms += search_time_ms
            
            self.logger.info(
                f"Search '{query}' returned {len(query_results)} results in {search_time_ms:.1f}ms"
            )
            
            return query_results
            
        except Exception as e:
            self.logger.error(f"Error performing search: {e}")
            return []
    
    def search_similar_images(
        self,
        reference_image: np.ndarray,
        limit: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[QueryResult]:
        """
        Search for similar images using a reference image.
        
        Args:
            reference_image: Numpy array of reference image
            limit: Maximum number of results
            filters: Optional search filters
            
        Returns:
            List of QueryResult objects sorted by similarity
        """
        if not self.ensure_initialized():
            return []
        
        # For image-based search, we need to generate embedding first
        # This is a placeholder - the actual implementation would need
        # to expose image embedding functionality from the search engine
        self.logger.warning("Image-based search not yet implemented")
        return []
    
    def get_recent_inspections(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[QueryResult]:
        """
        Get recent inspection results.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
            
        Returns:
            List of QueryResult objects sorted by timestamp
        """
        if not self.ensure_initialized():
            return []
        
        # Create time filter for recent inspections
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        filters = SearchFilter(
            start_date=start_time,
            end_date=end_time
        )
        
        # Use a generic query to get all recent results
        return self.search("wagon inspection", limit=limit, filters=filters)
    
    def get_damage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Get summary of damage types found in inspections.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Dict mapping damage type to count
        """
        if not self.ensure_initialized():
            return {}
        
        # Search for damage-related results
        damage_types = ["rust", "dent", "hole", "scratch", "corrosion"]
        summary = {}
        
        for damage_type in damage_types:
            filters = SearchFilter(
                start_date=start_date,
                end_date=end_date,
                damage_types=[damage_type]
            )
            
            results = self.search(f"wagon with {damage_type}", limit=1000, filters=filters)
            summary[damage_type] = len(results)
        
        return summary
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the vector index for better search performance.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._search_engine:
            return False
        
        try:
            return self._search_engine.rebuild_index()
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        engine_stats = {}
        if self._search_engine:
            engine_stats = self._search_engine.get_stats()
        
        avg_search_time = 0.0
        if self._total_searches > 0:
            avg_search_time = self._total_search_time_ms / self._total_searches
        
        return {
            "is_available": self._is_available,
            "is_initialized": self._is_initialized,
            "is_running": self._is_running,
            "model_name": self.model_name,
            "db_path": self.db_path,
            "total_embeddings_queued": self._total_embeddings_queued,
            "total_embeddings_processed": self._total_embeddings_processed,
            "queue_size": self._embedding_queue.qsize(),
            "queue_drops": self._queue_drops,
            "total_searches": self._total_searches,
            "avg_search_time_ms": avg_search_time,
            "engine_stats": engine_stats
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self._embedding_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "is_processing": self._is_running,
            "total_queued": self._total_embeddings_queued,
            "total_processed": self._total_embeddings_processed,
            "pending": self._total_embeddings_queued - self._total_embeddings_processed,
            "drops": self._queue_drops
        }
    
    def shutdown(self) -> None:
        """Shutdown the SigLIP integration and release resources."""
        self.logger.info("Shutting down SigLIP integration")
        
        # Stop background processing
        self._stop_background_processing()
        
        # Shutdown search engine
        if self._is_initialized:
            shutdown_search_engine()
            self._search_engine = None
            self._is_initialized = False
        
        self.logger.info("SigLIP integration shutdown complete")


# Module-level instance for easy access
_siglip_integration: Optional[SigLIPIntegration] = None


def get_siglip_integration() -> SigLIPIntegration:
    """Get or create the global SigLIP integration instance."""
    global _siglip_integration
    if _siglip_integration is None:
        _siglip_integration = SigLIPIntegration(lazy_load=True)
    return _siglip_integration


def initialize_siglip_integration(**kwargs) -> bool:
    """Initialize the global SigLIP integration with custom parameters."""
    global _siglip_integration
    _siglip_integration = SigLIPIntegration(**kwargs)
    return _siglip_integration.ensure_initialized()


def shutdown_siglip_integration() -> None:
    """Shutdown the global SigLIP integration."""
    global _siglip_integration
    if _siglip_integration:
        _siglip_integration.shutdown()
        _siglip_integration = None

"""
Semantic Search Engine for Railway Wagon Inspection History.

This module provides semantic search capabilities using SigLIP 2 embeddings
and LanceDB for vector storage. Enables natural language queries like:
- "Show me wagons with rust damage"
- "Find wagons inspected last week"
- "Wagons with identification plate issues"
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
import open_clip
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel


@dataclass
class SearchResult:
    """Result from semantic search."""
    wagon_id: Optional[str]
    timestamp: datetime
    image_path: str
    similarity_score: float
    inspection_data: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class WagonEmbedding(LanceModel):
    """LanceDB schema for wagon embeddings."""
    vector: Vector(512)  # SigLIP embedding dimension
    wagon_id: Optional[str]
    timestamp: datetime
    image_path: str
    frame_id: int
    detection_confidence: float
    ocr_confidence: float
    blur_score: float
    enhancement_applied: bool
    deblur_applied: bool
    fallback_ocr_used: bool
    damage_assessment: Optional[str]
    bounding_box: str  # JSON string
    wagon_angle: float


class SemanticSearchEngine:
    """
    Semantic search engine for wagon inspection history.
    
    Uses SigLIP 2 (google/siglip2-base-patch16-224) for generating embeddings
    and LanceDB for efficient vector similarity search.
    """
    
    def __init__(
        self,
        db_path: str = "data/wagon_embeddings.lancedb",
        model_name: str = "ViT-B-16-SigLIP-256",
        pretrained: str = "webli",
        device: str = "auto",
        batch_size: int = 32,
        max_queue_size: int = 1000
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            db_path: Path to LanceDB database
            model_name: SigLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to run on ("auto", "cuda", "cpu")
            batch_size: Batch size for embedding generation
            max_queue_size: Maximum number of queued embedding tasks
        """
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.batch_size = batch_size
        
        # Model components (loaded lazily)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Database
        self.db = None
        self.table = None
        
        # Background processing
        self.embedding_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Performance tracking
        self.total_embeddings = 0
        self.total_searches = 0
        self.avg_embedding_time_ms = 0.0
        self.avg_search_time_ms = 0.0
        
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self) -> bool:
        """
        Load SigLIP model and initialize database.
        Returns True if successful.
        """
        try:
            self.logger.info(f"Loading SigLIP model: {self.model_name}")
            start_time = time.time()
            
            # Load SigLIP model
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device if self.device != "auto" else None
            )
            
            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            load_time = time.time() - start_time
            self.logger.info(f"SigLIP model loaded in {load_time:.2f}s")
            
            # Initialize database
            self._init_database()
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load SigLIP model: {e}")
            return False
    
    def _init_database(self) -> None:
        """Initialize LanceDB database and table."""
        try:
            # Create database directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to LanceDB
            self.db = lancedb.connect(str(self.db_path))
            
            # Create or open table
            try:
                self.table = self.db.open_table("wagon_embeddings")
                self.logger.info(f"Opened existing embeddings table with {len(self.table)} records")
            except FileNotFoundError:
                # Create new table with schema
                self.table = self.db.create_table("wagon_embeddings", schema=WagonEmbedding)
                self.logger.info("Created new embeddings table")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def start(self) -> bool:
        """
        Start the semantic search engine.
        Returns True if started successfully.
        """
        if self.is_running:
            self.logger.warning("Semantic search engine already running")
            return True
        
        # Load model if not already loaded
        if not self.model_loaded:
            if not self._load_model():
                return False
        
        # Start background processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="SemanticSearch-Embedder",
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("Semantic search engine started")
        return True
    
    def stop(self) -> None:
        """Stop the semantic search engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("Semantic search engine stopped")
    
    def _processing_loop(self) -> None:
        """Background loop for processing embedding queue."""
        self.logger.info("Semantic search processing loop started")
        
        # Run async loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_processing_loop())
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
        finally:
            loop.close()
        
        self.logger.info("Semantic search processing loop ended")
    
    async def _async_processing_loop(self) -> None:
        """Async processing loop for batched embedding generation."""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch of items
                while len(batch) < self.batch_size and self.is_running:
                    try:
                        item = await asyncio.wait_for(
                            self.embedding_queue.get(), 
                            timeout=1.0
                        )
                        if item is None:  # Sentinel to stop
                            break
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have items
                if batch:
                    await self._process_embedding_batch(batch)
                    batch.clear()
                    
            except Exception as e:
                self.logger.error(f"Error in async processing loop: {e}")
    
    async def _process_embedding_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of embedding requests."""
        try:
            start_time = time.time()
            
            # Load and preprocess images
            images = []
            valid_items = []
            
            for item in batch:
                try:
                    image_path = item["image_path"]
                    if isinstance(image_path, str):
                        # Load from file path
                        image = Image.open(image_path).convert("RGB")
                    else:
                        # Convert from numpy array
                        image_array = item["image_array"]
                        if image_array.dtype != np.uint8:
                            image_array = (image_array * 255).astype(np.uint8)
                        
                        # Convert BGR to RGB if needed
                        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        
                        image = Image.fromarray(image_array)
                    
                    # Preprocess image
                    processed_image = self.preprocess(image)
                    images.append(processed_image)
                    valid_items.append(item)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load image for embedding: {e}")
                    continue
            
            if not images:
                return
            
            # Generate embeddings
            with torch.no_grad():
                image_tensor = torch.stack(images)
                if torch.cuda.is_available() and self.device != "cpu":
                    image_tensor = image_tensor.cuda()
                
                # Get image embeddings
                embeddings = self.model.encode_image(image_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
                embeddings = embeddings.cpu().numpy()
            
            # Store embeddings in database
            records = []
            for item, embedding in zip(valid_items, embeddings):
                record = WagonEmbedding(
                    vector=embedding.tolist(),
                    wagon_id=item.get("wagon_id"),
                    timestamp=item["timestamp"],
                    image_path=item["image_path"],
                    frame_id=item["frame_id"],
                    detection_confidence=item["detection_confidence"],
                    ocr_confidence=item["ocr_confidence"],
                    blur_score=item["blur_score"],
                    enhancement_applied=item["enhancement_applied"],
                    deblur_applied=item["deblur_applied"],
                    fallback_ocr_used=item["fallback_ocr_used"],
                    damage_assessment=item.get("damage_assessment"),
                    bounding_box=str(item["bounding_box"]),
                    wagon_angle=item["wagon_angle"]
                )
                records.append(record)
            
            # Add to database
            self.table.add(records)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.total_embeddings += len(records)
            self.avg_embedding_time_ms = (
                (self.avg_embedding_time_ms * (self.total_embeddings - len(records)) + processing_time)
                / self.total_embeddings
            )
            
            self.logger.debug(f"Generated {len(records)} embeddings in {processing_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Error processing embedding batch: {e}")
    
    def add_wagon_embedding(
        self,
        image_array: np.ndarray,
        image_path: str,
        inspection_result: Dict[str, Any],
        priority: bool = False
    ) -> bool:
        """
        Add a wagon crop for embedding generation.
        
        Args:
            image_array: Numpy array of the wagon crop
            image_path: Path where the image will be/is saved
            inspection_result: Inspection result data
            priority: Whether to prioritize this embedding
            
        Returns:
            True if successfully queued
        """
        if not self.is_running:
            return False
        
        try:
            # Create embedding task
            task = {
                "image_array": image_array,
                "image_path": image_path,
                "timestamp": inspection_result["timestamp"],
                "frame_id": inspection_result["frame_id"],
                "wagon_id": inspection_result.get("wagon_id"),
                "detection_confidence": inspection_result["detection_confidence"],
                "ocr_confidence": inspection_result["ocr_confidence"],
                "blur_score": inspection_result["blur_score"],
                "enhancement_applied": inspection_result["enhancement_applied"],
                "deblur_applied": inspection_result["deblur_applied"],
                "fallback_ocr_used": inspection_result.get("fallback_ocr_used", False),
                "damage_assessment": inspection_result.get("damage_assessment"),
                "bounding_box": inspection_result["bounding_box"],
                "wagon_angle": inspection_result["wagon_angle"]
            }
            
            # Queue for processing (non-blocking)
            try:
                # Use asyncio-safe method to add to queue
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(
                            self.embedding_queue.put(task),
                            timeout=0.1
                        )
                    )
                    return True
                finally:
                    loop.close()
                    
            except (asyncio.TimeoutError, asyncio.QueueFull):
                self.logger.warning("Embedding queue full, dropping task")
                return False
                
        except Exception as e:
            self.logger.error(f"Error queuing embedding task: {e}")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 10,
        time_filter: Optional[Tuple[datetime, datetime]] = None,
        min_confidence: float = 0.0
    ) -> List[SearchResult]:
        """
        Search wagon history using natural language query.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            time_filter: Optional (start_time, end_time) filter
            min_confidence: Minimum detection confidence filter
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not self.model_loaded or not self.table:
            self.logger.warning("Semantic search not initialized")
            return []
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self._encode_text(query)
            
            # Build search query
            search_query = self.table.search(query_embedding.tolist())
            
            # Apply filters
            if min_confidence > 0:
                search_query = search_query.where(f"detection_confidence >= {min_confidence}")
            
            if time_filter:
                start_ts, end_ts = time_filter
                search_query = search_query.where(
                    f"timestamp >= '{start_ts.isoformat()}' AND timestamp <= '{end_ts.isoformat()}'"
                )
            
            # Execute search
            results = search_query.limit(limit).to_list()
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    wagon_id=result["wagon_id"],
                    timestamp=result["timestamp"],
                    image_path=result["image_path"],
                    similarity_score=1.0 - result["_distance"],  # Convert distance to similarity
                    inspection_data={
                        "frame_id": result["frame_id"],
                        "detection_confidence": result["detection_confidence"],
                        "ocr_confidence": result["ocr_confidence"],
                        "blur_score": result["blur_score"],
                        "enhancement_applied": result["enhancement_applied"],
                        "deblur_applied": result["deblur_applied"],
                        "fallback_ocr_used": result["fallback_ocr_used"],
                        "damage_assessment": result["damage_assessment"],
                        "bounding_box": result["bounding_box"],
                        "wagon_angle": result["wagon_angle"]
                    }
                )
                search_results.append(search_result)
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000
            self.total_searches += 1
            self.avg_search_time_ms = (
                (self.avg_search_time_ms * (self.total_searches - 1) + search_time)
                / self.total_searches
            )
            
            self.logger.info(f"Search '{query}' returned {len(search_results)} results in {search_time:.1f}ms")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error performing search: {e}")
            return []
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding vector."""
        with torch.no_grad():
            text_tokens = self.tokenizer([text])
            if torch.cuda.is_available() and self.device != "cpu":
                text_tokens = text_tokens.cuda()
            
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            return text_embedding.cpu().numpy()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        queue_size = 0
        try:
            queue_size = self.embedding_queue.qsize()
        except:
            pass
        
        total_records = 0
        if self.table:
            try:
                total_records = len(self.table)
            except:
                pass
        
        return {
            "model_loaded": self.model_loaded,
            "is_running": self.is_running,
            "total_embeddings": self.total_embeddings,
            "total_searches": self.total_searches,
            "avg_embedding_time_ms": self.avg_embedding_time_ms,
            "avg_search_time_ms": self.avg_search_time_ms,
            "queue_size": queue_size,
            "total_records": total_records,
            "db_path": str(self.db_path)
        }
    
    def rebuild_index(self) -> bool:
        """Rebuild the vector index for better performance."""
        if not self.table:
            return False
        
        try:
            self.logger.info("Rebuilding vector index...")
            start_time = time.time()
            
            # Create IVF index for faster similarity search
            self.table.create_index(
                "vector",
                config=lancedb.index.IvfPq(
                    num_partitions=256,
                    num_sub_vectors=16
                )
            )
            
            rebuild_time = time.time() - start_time
            self.logger.info(f"Vector index rebuilt in {rebuild_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            return False


# Global instance for easy access
_search_engine: Optional[SemanticSearchEngine] = None


def get_search_engine() -> SemanticSearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SemanticSearchEngine()
    return _search_engine


def initialize_search_engine(**kwargs) -> bool:
    """Initialize the global search engine with custom parameters."""
    global _search_engine
    _search_engine = SemanticSearchEngine(**kwargs)
    return _search_engine.start()


def shutdown_search_engine() -> None:
    """Shutdown the global search engine."""
    global _search_engine
    if _search_engine:
        _search_engine.stop()
        _search_engine = None
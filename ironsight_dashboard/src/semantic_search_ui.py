#!/usr/bin/env python3
"""
Semantic Search UI - Natural Language Search Interface for IronSight Command Center.

This module implements the Semantic Search tab for the IronSight Command Center.
It provides a natural language search interface for querying historical inspection
data using SigLIP embeddings.

Key Features:
- Text input for natural language queries (e.g., "wagons with rust damage")
- Gallery display for search results with similarity scores
- Metadata display: wagon ID, timestamp, confidence scores
- Filtering options for date range and confidence thresholds

Requirements: 9.1, 9.3, 9.4, 9.5
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class SearchResultDisplay:
    """Display-ready search result with all metadata."""
    wagon_id: Optional[str]
    timestamp: datetime
    image_path: str
    similarity_score: float
    confidence_score: float
    damage_assessment: Optional[str]
    ocr_result: Optional[str]
    blur_score: float
    enhancement_applied: bool
    deblur_applied: bool
    thumbnail: Optional[np.ndarray] = None
    
    def get_formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_similarity_percentage(self) -> str:
        """Get similarity as percentage string."""
        return f"{self.similarity_score * 100:.1f}%"
    
    def get_confidence_percentage(self) -> str:
        """Get confidence as percentage string."""
        return f"{self.confidence_score * 100:.1f}%"


@dataclass
class SearchFilters:
    """Filters for semantic search queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: float = 0.0
    min_similarity: float = 0.0
    wagon_id_filter: Optional[str] = None
    damage_types: Optional[List[str]] = None
    include_enhanced_only: bool = False
    include_deblurred_only: bool = False


@dataclass
class SearchStats:
    """Statistics for search operations."""
    total_searches: int = 0
    total_results_returned: int = 0
    avg_search_time_ms: float = 0.0
    last_search_time_ms: float = 0.0
    last_query: str = ""
    last_result_count: int = 0


class SemanticSearchUI:
    """
    Natural language search interface for inspection history.
    
    This class provides:
    - Natural language query processing
    - Gallery display of search results
    - Metadata display with filtering
    - Integration with SigLIP semantic search
    
    Requirements:
    - 9.1: Accept text queries like "Show me rusted doors" or "wagons with damage"
    - 9.3: Display gallery of matching defects with similarity scores
    - 9.4: Include metadata: wagon ID, timestamp, confidence scores, damage descriptions
    - 9.5: Support filtering by date range and confidence thresholds
    """
    
    # Common damage type keywords for filtering
    DAMAGE_TYPES = ["rust", "dent", "hole", "scratch", "corrosion", "crack", "wear"]
    
    # Example queries for user guidance
    EXAMPLE_QUERIES = [
        "wagons with rust damage",
        "dented doors",
        "identification plates",
        "corroded surfaces",
        "damaged wheels",
        "recent inspections with defects"
    ]
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialize Semantic Search UI.
        
        Args:
            lazy_load: If True, defer SigLIP initialization until first use
        """
        self._siglip = None
        self._is_initialized = False
        self._lazy_load = lazy_load
        self._stats = SearchStats()
        
        # Initialize immediately if not lazy loading
        if not lazy_load:
            self._initialize_siglip()
        
        logger.info(f"SemanticSearchUI initialized (lazy_load={lazy_load})")
    
    def _initialize_siglip(self) -> bool:
        """Initialize SigLIP integration."""
        if self._is_initialized:
            return True
        
        try:
            from siglip_integration import get_siglip_integration
            self._siglip = get_siglip_integration()
            self._is_initialized = self._siglip.ensure_initialized()
            
            if self._is_initialized:
                logger.info("SigLIP integration initialized successfully")
            else:
                logger.warning("SigLIP integration not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize SigLIP: {e}")
            self._is_initialized = False
        
        return self._is_initialized
    
    def is_available(self) -> bool:
        """Check if semantic search is available."""
        if not self._is_initialized:
            self._initialize_siglip()
        return self._is_initialized and self._siglip is not None
    
    def search(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResultDisplay]:
        """
        Search inspection history using natural language query.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            filters: Optional search filters
            
        Returns:
            List of SearchResultDisplay objects sorted by similarity
        """
        if not self.is_available():
            logger.warning("Semantic search not available")
            return []
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from siglip_integration import SearchFilter
            
            # Convert our filters to SigLIP filters
            siglip_filters = None
            if filters:
                siglip_filters = SearchFilter(
                    start_date=filters.start_date,
                    end_date=filters.end_date,
                    min_confidence=filters.min_confidence,
                    wagon_id_pattern=filters.wagon_id_filter,
                    damage_types=filters.damage_types
                )
            
            # Perform search
            results = self._siglip.search(
                query=query,
                limit=limit * 2,  # Get extra for post-filtering
                filters=siglip_filters
            )
            
            # Convert to display format and apply additional filters
            display_results = []
            for result in results:
                display_result = self._convert_to_display_result(result)
                
                # Apply additional filters
                if filters:
                    if not self._passes_filters(display_result, filters):
                        continue
                
                display_results.append(display_result)
                
                # Stop if we have enough results
                if len(display_results) >= limit:
                    break
            
            # Update statistics
            search_time_ms = (time.time() - start_time) * 1000
            self._update_stats(query, len(display_results), search_time_ms)
            
            logger.info(
                f"Search '{query}' returned {len(display_results)} results "
                f"in {search_time_ms:.1f}ms"
            )
            
            return display_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _convert_to_display_result(self, result) -> SearchResultDisplay:
        """Convert SigLIP QueryResult to SearchResultDisplay."""
        inspection_data = result.inspection_data or {}
        
        return SearchResultDisplay(
            wagon_id=result.wagon_id,
            timestamp=result.timestamp,
            image_path=result.image_path,
            similarity_score=result.similarity_score,
            confidence_score=inspection_data.get("detection_confidence", 0.0),
            damage_assessment=inspection_data.get("damage_assessment"),
            ocr_result=inspection_data.get("wagon_id"),
            blur_score=inspection_data.get("blur_score", 0.0),
            enhancement_applied=inspection_data.get("enhancement_applied", False),
            deblur_applied=inspection_data.get("deblur_applied", False),
            thumbnail=result.thumbnail
        )
    
    def _passes_filters(
        self, 
        result: SearchResultDisplay, 
        filters: SearchFilters
    ) -> bool:
        """Check if result passes all filters."""
        # Similarity threshold
        if result.similarity_score < filters.min_similarity:
            return False
        
        # Enhancement filter
        if filters.include_enhanced_only and not result.enhancement_applied:
            return False
        
        # Deblur filter
        if filters.include_deblurred_only and not result.deblur_applied:
            return False
        
        return True
    
    def _update_stats(
        self, 
        query: str, 
        result_count: int, 
        search_time_ms: float
    ) -> None:
        """Update search statistics."""
        self._stats.total_searches += 1
        self._stats.total_results_returned += result_count
        self._stats.last_search_time_ms = search_time_ms
        self._stats.last_query = query
        self._stats.last_result_count = result_count
        
        # Update average
        if self._stats.total_searches > 0:
            # Running average
            prev_avg = self._stats.avg_search_time_ms
            n = self._stats.total_searches
            self._stats.avg_search_time_ms = prev_avg + (search_time_ms - prev_avg) / n
    
    def get_recent_inspections(
        self, 
        hours: int = 24, 
        limit: int = 50
    ) -> List[SearchResultDisplay]:
        """
        Get recent inspection results.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
            
        Returns:
            List of recent inspection results
        """
        filters = SearchFilters(
            start_date=datetime.now() - timedelta(hours=hours),
            end_date=datetime.now()
        )
        
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
        if not self.is_available():
            return {}
        
        try:
            return self._siglip.get_damage_summary(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get damage summary: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        siglip_stats = {}
        if self._siglip:
            siglip_stats = self._siglip.get_stats()
        
        return {
            "is_available": self.is_available(),
            "total_searches": self._stats.total_searches,
            "total_results_returned": self._stats.total_results_returned,
            "avg_search_time_ms": self._stats.avg_search_time_ms,
            "last_search_time_ms": self._stats.last_search_time_ms,
            "last_query": self._stats.last_query,
            "last_result_count": self._stats.last_result_count,
            "siglip_stats": siglip_stats
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get embedding queue status."""
        if not self._siglip:
            return {"queue_size": 0, "is_processing": False}
        
        return self._siglip.get_queue_status()
    
    def load_thumbnail(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load thumbnail image from path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Thumbnail as numpy array or None if loading fails
        """
        try:
            path = Path(image_path)
            if not path.exists():
                return None
            
            # Load image
            image = cv2.imread(str(path))
            if image is None:
                return None
            
            # Resize to thumbnail
            max_size = 200
            h, w = image.shape[:2]
            if h > w:
                new_h = max_size
                new_w = int(w * max_size / h)
            else:
                new_w = max_size
                new_h = int(h * max_size / w)
            
            thumbnail = cv2.resize(image, (new_w, new_h))
            return thumbnail
            
        except Exception as e:
            logger.error(f"Failed to load thumbnail: {e}")
            return None


def create_semantic_search_ui(lazy_load: bool = True) -> SemanticSearchUI:
    """
    Factory function to create SemanticSearchUI instance.
    
    Args:
        lazy_load: If True, defer SigLIP initialization until first use
        
    Returns:
        Configured SemanticSearchUI instance
    """
    return SemanticSearchUI(lazy_load=lazy_load)


def render_semantic_search_ui(st_module):
    """
    Render the Semantic Search UI in Streamlit.
    
    This function provides the complete UI for the Semantic Search tab,
    including text input, gallery display, and filtering options.
    
    Args:
        st_module: Streamlit module (passed to avoid import issues)
    """
    st = st_module
    
    # Initialize search UI in session state
    if 'semantic_search_ui' not in st.session_state:
        st.session_state.semantic_search_ui = create_semantic_search_ui()
    
    search_ui = st.session_state.semantic_search_ui
    
    # Check availability
    is_available = search_ui.is_available()
    
    # Status indicator
    if is_available:
        st.success("🟢 Semantic Search: Online")
    else:
        st.warning("🟡 Semantic Search: Offline (SigLIP not loaded)")
        st.info(
            "Semantic search requires SigLIP model and indexed inspection data. "
            "The interface is shown for demonstration purposes."
        )
    
    # Search section
    st.subheader("🔍 Natural Language Search")
    
    # Search input with examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., 'wagons with rust damage' or 'dented doors'",
            help="Enter a natural language query to search inspection history"
        )
    
    with col2:
        search_btn = st.button("🔎 Search", type="primary", width="stretch")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        example_cols = st.columns(3)
        for i, example in enumerate(SemanticSearchUI.EXAMPLE_QUERIES):
            with example_cols[i % 3]:
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    st.session_state.search_query = example
                    st.rerun()
    
    # Check for query from example button
    if 'search_query' in st.session_state:
        query = st.session_state.search_query
        del st.session_state.search_query
    
    # Filters section
    st.subheader("🎛️ Filters")
    
    filter_cols = st.columns(4)
    
    with filter_cols[0]:
        # Date range filter
        date_range = st.selectbox(
            "Date Range",
            options=["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
            index=0
        )
        
        start_date = None
        end_date = None
        
        if date_range == "Last 24 Hours":
            start_date = datetime.now() - timedelta(hours=24)
        elif date_range == "Last 7 Days":
            start_date = datetime.now() - timedelta(days=7)
        elif date_range == "Last 30 Days":
            start_date = datetime.now() - timedelta(days=30)
        elif date_range == "Custom":
            start_date = st.date_input("Start Date", value=None)
            end_date = st.date_input("End Date", value=None)
            if start_date:
                start_date = datetime.combine(start_date, datetime.min.time())
            if end_date:
                end_date = datetime.combine(end_date, datetime.max.time())
    
    with filter_cols[1]:
        # Confidence threshold
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Minimum detection confidence score"
        )
    
    with filter_cols[2]:
        # Similarity threshold
        min_similarity = st.slider(
            "Min Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Minimum similarity score to query"
        )
    
    with filter_cols[3]:
        # Result limit
        result_limit = st.selectbox(
            "Max Results",
            options=[10, 20, 50, 100],
            index=1
        )
    
    # Additional filters
    adv_filter_cols = st.columns(3)
    
    with adv_filter_cols[0]:
        # Wagon ID filter
        wagon_id_filter = st.text_input(
            "Wagon ID Contains",
            placeholder="e.g., 'ABC'",
            help="Filter by wagon ID pattern"
        )
    
    with adv_filter_cols[1]:
        # Damage type filter
        damage_types = st.multiselect(
            "Damage Types",
            options=SemanticSearchUI.DAMAGE_TYPES,
            help="Filter by specific damage types"
        )
    
    with adv_filter_cols[2]:
        # Processing filters
        enhanced_only = st.checkbox("Enhanced Only", value=False)
        deblurred_only = st.checkbox("Deblurred Only", value=False)
    
    # Build filters
    filters = SearchFilters(
        start_date=start_date,
        end_date=end_date,
        min_confidence=min_confidence,
        min_similarity=min_similarity,
        wagon_id_filter=wagon_id_filter if wagon_id_filter else None,
        damage_types=damage_types if damage_types else None,
        include_enhanced_only=enhanced_only,
        include_deblurred_only=deblurred_only
    )
    
    # Store search results in session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Perform search
    if search_btn and query:
        with st.spinner(f"Searching for '{query}'..."):
            results = search_ui.search(query, limit=result_limit, filters=filters)
            st.session_state.search_results = results
    
    # Display results
    results = st.session_state.search_results
    
    if results:
        st.subheader(f"📊 Results ({len(results)} found)")
        
        # Results summary
        stats = search_ui.get_stats()
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Results Found", len(results))
        with metric_cols[1]:
            st.metric("Search Time", f"{stats['last_search_time_ms']:.1f}ms")
        with metric_cols[2]:
            avg_similarity = sum(r.similarity_score for r in results) / len(results) if results else 0
            st.metric("Avg Similarity", f"{avg_similarity * 100:.1f}%")
        with metric_cols[3]:
            st.metric("Total Searches", stats['total_searches'])
        
        # Gallery display
        st.subheader("🖼️ Gallery")
        
        # Display results in a grid
        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            row_results = results[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for col_idx, result in enumerate(row_results):
                with cols[col_idx]:
                    _render_result_card(st, result, search_ui)
    
    elif query and search_btn:
        st.info("No results found. Try a different query or adjust filters.")
    
    # Queue status
    with st.expander("📈 System Status"):
        queue_status = search_ui.get_queue_status()
        stats = search_ui.get_stats()
        
        status_cols = st.columns(3)
        
        with status_cols[0]:
            st.metric("Embedding Queue", queue_status.get("queue_size", 0))
        with status_cols[1]:
            st.metric("Total Indexed", queue_status.get("total_processed", 0))
        with status_cols[2]:
            st.metric("Avg Search Time", f"{stats['avg_search_time_ms']:.1f}ms")


def _render_result_card(st_module, result: SearchResultDisplay, search_ui: SemanticSearchUI):
    """
    Render a single result card in the gallery.
    
    Args:
        st_module: Streamlit module
        result: SearchResultDisplay to render
        search_ui: SemanticSearchUI instance for loading thumbnails
    """
    st = st_module
    
    # Card container
    with st.container():
        # Try to load and display thumbnail
        thumbnail = result.thumbnail
        if thumbnail is None and result.image_path:
            thumbnail = search_ui.load_thumbnail(result.image_path)
        
        if thumbnail is not None:
            # Convert BGR to RGB for display
            if len(thumbnail.shape) == 3 and thumbnail.shape[2] == 3:
                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            else:
                thumbnail_rgb = thumbnail
            st.image(thumbnail_rgb, width="stretch")
        else:
            # Placeholder for missing image
            st.markdown(
                """
                <div style="
                    background-color: #333;
                    height: 150px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 5px;
                ">
                    <span style="color: #888;">No Image</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Similarity score badge
        similarity_pct = result.similarity_score * 100
        if similarity_pct >= 80:
            badge_color = "green"
        elif similarity_pct >= 60:
            badge_color = "orange"
        else:
            badge_color = "red"
        
        st.markdown(
            f"**Similarity:** :{badge_color}[{similarity_pct:.1f}%]"
        )
        
        # Wagon ID
        wagon_id = result.wagon_id or "Unknown"
        st.markdown(f"**Wagon:** {wagon_id}")
        
        # Timestamp
        st.caption(result.get_formatted_timestamp())
        
        # Damage assessment
        if result.damage_assessment:
            st.markdown(f"**Damage:** {result.damage_assessment}")
        
        # Confidence score
        st.caption(f"Confidence: {result.get_confidence_percentage()}")
        
        # Processing indicators
        indicators = []
        if result.enhancement_applied:
            indicators.append("🌙 Enhanced")
        if result.deblur_applied:
            indicators.append("🔍 Deblurred")
        
        if indicators:
            st.caption(" | ".join(indicators))


# Module-level instance for easy access
_semantic_search_ui: Optional[SemanticSearchUI] = None


def get_semantic_search_ui() -> SemanticSearchUI:
    """Get or create the global SemanticSearchUI instance."""
    global _semantic_search_ui
    if _semantic_search_ui is None:
        _semantic_search_ui = SemanticSearchUI(lazy_load=True)
    return _semantic_search_ui


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 Semantic Search UI Module")
    print("="*60)
    
    # Create search UI
    search_ui = create_semantic_search_ui()
    
    # Print status
    print(f"Available: {search_ui.is_available()}")
    
    # Get stats
    stats = search_ui.get_stats()
    print(f"Stats: {stats}")
    
    # Test search (will return empty if SigLIP not available)
    print("\nTesting search...")
    results = search_ui.search("wagons with rust damage", limit=5)
    print(f"Results: {len(results)}")
    
    print("\n🚂 Ready for IronSight Command Center Integration!")
    print("="*60)

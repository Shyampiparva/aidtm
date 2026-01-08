#!/usr/bin/env python3
"""
Property-Based Tests for Semantic Search UI.

Tests Property 13: Natural Language Query Processing using Hypothesis.
Minimum 100 iterations per property test.

Requirements: 9.1, 9.2
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck, assume

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.semantic_search_ui import (
    SemanticSearchUI,
    SearchResultDisplay,
    SearchFilters,
    SearchStats,
    create_semantic_search_ui,
    get_semantic_search_ui
)


# Custom strategies for generating test data
@st.composite
def search_query_strategy(draw):
    """Generate valid search queries."""
    # Generate queries that are realistic natural language
    query_templates = [
        "wagons with {} damage",
        "show me {} defects",
        "{} on doors",
        "damaged {}",
        "inspection of {}",
        "{}"
    ]
    
    damage_types = ["rust", "dent", "hole", "scratch", "corrosion", "crack", "wear"]
    components = ["door", "wheel", "plate", "surface", "frame", "body"]
    
    template = draw(st.sampled_from(query_templates))
    
    if "{}" in template:
        fill = draw(st.sampled_from(damage_types + components))
        query = template.format(fill)
    else:
        query = template
    
    return query


@st.composite
def search_filters_strategy(draw):
    """Generate valid SearchFilters."""
    # Date range
    use_date_filter = draw(st.booleans())
    start_date = None
    end_date = None
    
    if use_date_filter:
        days_back = draw(st.integers(min_value=1, max_value=365))
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now()
    
    # Confidence thresholds
    min_confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    min_similarity = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Wagon ID filter
    use_wagon_filter = draw(st.booleans())
    wagon_id_filter = None
    if use_wagon_filter:
        wagon_id_filter = draw(st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    
    # Damage types
    use_damage_filter = draw(st.booleans())
    damage_types = None
    if use_damage_filter:
        damage_types = draw(st.lists(
            st.sampled_from(SemanticSearchUI.DAMAGE_TYPES),
            min_size=1,
            max_size=3,
            unique=True
        ))
    
    # Processing filters
    enhanced_only = draw(st.booleans())
    deblurred_only = draw(st.booleans())
    
    return SearchFilters(
        start_date=start_date,
        end_date=end_date,
        min_confidence=min_confidence,
        min_similarity=min_similarity,
        wagon_id_filter=wagon_id_filter,
        damage_types=damage_types,
        include_enhanced_only=enhanced_only,
        include_deblurred_only=deblurred_only
    )


@st.composite
def search_result_display_strategy(draw):
    """Generate valid SearchResultDisplay objects."""
    # Generate timestamp within last year
    days_back = draw(st.integers(min_value=0, max_value=365))
    timestamp = datetime.now() - timedelta(days=days_back)
    
    # Generate wagon ID
    wagon_id = draw(st.one_of(
        st.none(),
        st.text(min_size=5, max_size=15, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
    ))
    
    # Generate damage assessment
    damage_assessment = draw(st.one_of(
        st.none(),
        st.sampled_from(["rust damage", "minor dent", "hole detected", "scratch marks", "corrosion"])
    ))
    
    return SearchResultDisplay(
        wagon_id=wagon_id,
        timestamp=timestamp,
        image_path=f"/path/to/image_{draw(st.integers(min_value=1, max_value=10000))}.jpg",
        similarity_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        damage_assessment=damage_assessment,
        ocr_result=wagon_id,
        blur_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        enhancement_applied=draw(st.booleans()),
        deblur_applied=draw(st.booleans()),
        thumbnail=None
    )


class TestProperty13NaturalLanguageQueryProcessing:
    """
    Property-based tests for Property 13: Natural Language Query Processing.
    
    For any text query input to Semantic Search, the system SHALL generate
    embeddings and compare against stored crop embeddings using SigLIP.
    
    **Validates: Requirements 9.1, 9.2**
    """
    
    @pytest.fixture(scope="class")
    def search_ui(self):
        """Create SemanticSearchUI for testing."""
        return create_semantic_search_ui(lazy_load=True)
    
    @given(query=st.text(min_size=1, max_size=200))
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_13_query_accepts_any_text(self, search_ui, query):
        """
        Feature: ironsight-command-center, Property 13: Natural Language Query Processing
        
        For any text query input to Semantic Search, the system SHALL accept
        the query without raising exceptions.
        
        **Validates: Requirements 9.1, 9.2**
        """
        # The search method should accept any text query without raising
        # It may return empty results if SigLIP is not available, but should not crash
        try:
            results = search_ui.search(query, limit=10)
            
            # Results should always be a list
            assert isinstance(results, list), "Search results must be a list"
            
            # Each result should be a SearchResultDisplay
            for result in results:
                assert isinstance(result, SearchResultDisplay), \
                    "Each result must be a SearchResultDisplay"
                    
        except Exception as e:
            # Only acceptable exception is if the query is completely empty after strip
            if query.strip():
                pytest.fail(f"Search should not raise for valid query: {e}")
    
    @given(query=search_query_strategy())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_13_realistic_queries_accepted(self, search_ui, query):
        """
        Feature: ironsight-command-center, Property 13: Natural Language Query Processing
        
        For any realistic natural language query, the system SHALL process
        the query and return valid results.
        
        **Validates: Requirements 9.1, 9.2**
        """
        results = search_ui.search(query, limit=20)
        
        # Results should be a list
        assert isinstance(results, list)
        
        # Results should not exceed limit
        assert len(results) <= 20
    
    @given(
        query=search_query_strategy(),
        limit=st.integers(min_value=1, max_value=100)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_13_result_limit_respected(self, search_ui, query, limit):
        """
        Feature: ironsight-command-center, Property 13: Natural Language Query Processing
        
        For any query with a specified limit, the system SHALL return
        at most that many results.
        
        **Validates: Requirements 9.1, 9.2**
        """
        results = search_ui.search(query, limit=limit)
        
        assert len(results) <= limit, \
            f"Results ({len(results)}) should not exceed limit ({limit})"
    
    @given(
        query=search_query_strategy(),
        filters=search_filters_strategy()
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_13_filters_accepted(self, search_ui, query, filters):
        """
        Feature: ironsight-command-center, Property 13: Natural Language Query Processing
        
        For any query with filters, the system SHALL accept the filters
        and apply them to the search.
        
        **Validates: Requirements 9.1, 9.2**
        """
        results = search_ui.search(query, limit=20, filters=filters)
        
        # Results should be a list
        assert isinstance(results, list)
        
        # If we have results and filters, verify filter application
        for result in results:
            # Check similarity threshold
            if filters.min_similarity > 0:
                assert result.similarity_score >= filters.min_similarity, \
                    f"Result similarity {result.similarity_score} below threshold {filters.min_similarity}"
            
            # Check enhanced only filter
            if filters.include_enhanced_only:
                assert result.enhancement_applied, \
                    "Result should have enhancement applied when filter is set"
            
            # Check deblurred only filter
            if filters.include_deblurred_only:
                assert result.deblur_applied, \
                    "Result should have deblur applied when filter is set"


class TestSearchResultDisplayProperties:
    """Property tests for SearchResultDisplay data class."""
    
    @given(result=search_result_display_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_formatted_timestamp_is_string(self, result):
        """Test that formatted timestamp returns a string."""
        formatted = result.get_formatted_timestamp()
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    @given(result=search_result_display_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_similarity_percentage_format(self, result):
        """Test that similarity percentage is correctly formatted."""
        percentage = result.get_similarity_percentage()
        assert isinstance(percentage, str)
        assert "%" in percentage
        
        # Extract numeric value
        numeric_str = percentage.replace("%", "").strip()
        numeric_value = float(numeric_str)
        
        # Should be between 0 and 100
        assert 0.0 <= numeric_value <= 100.0
    
    @given(result=search_result_display_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_confidence_percentage_format(self, result):
        """Test that confidence percentage is correctly formatted."""
        percentage = result.get_confidence_percentage()
        assert isinstance(percentage, str)
        assert "%" in percentage
        
        # Extract numeric value
        numeric_str = percentage.replace("%", "").strip()
        numeric_value = float(numeric_str)
        
        # Should be between 0 and 100
        assert 0.0 <= numeric_value <= 100.0


class TestSearchFiltersProperties:
    """Property tests for SearchFilters data class."""
    
    @given(filters=search_filters_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_filters_have_valid_confidence_range(self, filters):
        """Test that confidence thresholds are in valid range."""
        assert 0.0 <= filters.min_confidence <= 1.0
        assert 0.0 <= filters.min_similarity <= 1.0
    
    @given(filters=search_filters_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_filters_date_range_valid(self, filters):
        """Test that date range is valid when specified."""
        if filters.start_date and filters.end_date:
            assert filters.start_date <= filters.end_date, \
                "Start date should be before or equal to end date"


class TestSearchStatsProperties:
    """Property tests for SearchStats tracking."""
    
    @given(
        total_searches=st.integers(min_value=0, max_value=10000),
        total_results=st.integers(min_value=0, max_value=100000),
        avg_time=st.floats(min_value=0.0, max_value=10000.0)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_stats_store_values_correctly(self, total_searches, total_results, avg_time):
        """Test that SearchStats stores values correctly."""
        stats = SearchStats(
            total_searches=total_searches,
            total_results_returned=total_results,
            avg_search_time_ms=avg_time
        )
        
        assert stats.total_searches == total_searches
        assert stats.total_results_returned == total_results
        assert stats.avg_search_time_ms == avg_time


class TestSemanticSearchUIStateMethods:
    """Tests for SemanticSearchUI state methods."""
    
    def test_is_available_returns_bool(self):
        """Test that is_available returns a boolean."""
        search_ui = create_semantic_search_ui(lazy_load=True)
        result = search_ui.is_available()
        assert isinstance(result, bool)
    
    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a dictionary."""
        search_ui = create_semantic_search_ui(lazy_load=True)
        stats = search_ui.get_stats()
        assert isinstance(stats, dict)
        assert "is_available" in stats
        assert "total_searches" in stats
    
    def test_get_queue_status_returns_dict(self):
        """Test that get_queue_status returns a dictionary."""
        search_ui = create_semantic_search_ui(lazy_load=True)
        status = search_ui.get_queue_status()
        assert isinstance(status, dict)
        assert "queue_size" in status


class TestExampleQueries:
    """Tests for example queries constant."""
    
    def test_example_queries_not_empty(self):
        """Test that example queries list is not empty."""
        assert len(SemanticSearchUI.EXAMPLE_QUERIES) > 0
    
    def test_example_queries_are_strings(self):
        """Test that all example queries are strings."""
        for query in SemanticSearchUI.EXAMPLE_QUERIES:
            assert isinstance(query, str)
            assert len(query) > 0


class TestDamageTypes:
    """Tests for damage types constant."""
    
    def test_damage_types_not_empty(self):
        """Test that damage types list is not empty."""
        assert len(SemanticSearchUI.DAMAGE_TYPES) > 0
    
    def test_damage_types_are_strings(self):
        """Test that all damage types are strings."""
        for damage_type in SemanticSearchUI.DAMAGE_TYPES:
            assert isinstance(damage_type, str)
            assert len(damage_type) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Demo Data Package for IronSight Command Center.

This package provides sample data and scenarios for demonstrating
the IronSight Command Center dashboard functionality.

Contents:
- Sample wagon images for Restoration Lab testing
- Mock inspection history for Semantic Search demo
- Video samples for Mission Control demonstration
- Performance benchmark scenarios

Requirements: Demo preparation (Task 16.2)
"""

from .demo_data_generator import (
    DemoDataGenerator,
    create_demo_data_generator,
    generate_all_demo_data,
)

from .mock_inspection_history import (
    MockInspectionHistory,
    InspectionRecord,
    create_mock_inspection_history,
)

from .benchmark_scenarios import (
    BenchmarkScenario,
    BenchmarkRunner,
    create_benchmark_runner,
)

__all__ = [
    "DemoDataGenerator",
    "create_demo_data_generator",
    "generate_all_demo_data",
    "MockInspectionHistory",
    "InspectionRecord",
    "create_mock_inspection_history",
    "BenchmarkScenario",
    "BenchmarkRunner",
    "create_benchmark_runner",
]

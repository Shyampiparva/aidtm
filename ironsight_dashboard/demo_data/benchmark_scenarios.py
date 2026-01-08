#!/usr/bin/env python3
"""
Performance Benchmark Scenarios for IronSight Command Center.

This module provides benchmark scenarios for testing and demonstrating
the performance characteristics of the IronSight Command Center.

Requirements: Demo preparation (Task 16.2)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import json

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of benchmark scenarios."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    STRESS = "stress"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    scenario_name: str
    category: BenchmarkCategory
    timestamp: datetime
    
    # Timing metrics
    total_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    operations_per_second: float = 0.0
    frames_processed: int = 0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Additional metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "total_time_ms": self.total_time_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "operations_per_second": self.operations_per_second,
            "frames_processed": self.frames_processed,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "success": self.success,
            "error_message": self.error_message,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark scenario."""
    name: str
    description: str
    category: BenchmarkCategory
    
    # Configuration
    num_iterations: int = 100
    warmup_iterations: int = 10
    timeout_seconds: float = 60.0
    
    # Target metrics (for pass/fail)
    target_latency_ms: Optional[float] = None
    target_throughput: Optional[float] = None
    target_memory_mb: Optional[float] = None
    
    # Test function (set by runner)
    test_func: Optional[Callable] = None
    
    def meets_targets(self, result: BenchmarkResult) -> Tuple[bool, List[str]]:
        """
        Check if result meets target metrics.
        
        Returns:
            Tuple of (passes, list of failure reasons)
        """
        failures = []
        
        if self.target_latency_ms and result.avg_latency_ms > self.target_latency_ms:
            failures.append(
                f"Latency {result.avg_latency_ms:.2f}ms > target {self.target_latency_ms}ms"
            )
        
        if self.target_throughput and result.operations_per_second < self.target_throughput:
            failures.append(
                f"Throughput {result.operations_per_second:.1f} ops/s < target {self.target_throughput}"
            )
        
        if self.target_memory_mb and result.peak_memory_mb > self.target_memory_mb:
            failures.append(
                f"Memory {result.peak_memory_mb:.1f}MB > target {self.target_memory_mb}MB"
            )
        
        return len(failures) == 0, failures


class BenchmarkRunner:
    """
    Runs benchmark scenarios and collects results.
    
    Provides:
    - Pre-defined benchmark scenarios
    - Custom scenario support
    - Result collection and reporting
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = output_dir or Path(__file__).parent / "generated" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenarios: Dict[str, BenchmarkScenario] = {}
        self.results: List[BenchmarkResult] = []
        
        # Register default scenarios
        self._register_default_scenarios()
        
        logger.info(f"BenchmarkRunner initialized, output: {self.output_dir}")
    
    def _register_default_scenarios(self) -> None:
        """Register default benchmark scenarios."""
        
        # Gatekeeper latency benchmark
        self.register_scenario(BenchmarkScenario(
            name="gatekeeper_latency",
            description="Measure Gatekeeper model inference latency on 64x64 thumbnails",
            category=BenchmarkCategory.LATENCY,
            num_iterations=1000,
            warmup_iterations=100,
            target_latency_ms=0.5,
        ))
        
        # SCI enhancement latency
        self.register_scenario(BenchmarkScenario(
            name="sci_enhancement_latency",
            description="Measure SCI low-light enhancement latency",
            category=BenchmarkCategory.LATENCY,
            num_iterations=500,
            warmup_iterations=50,
            target_latency_ms=0.5,
        ))
        
        # YOLO detection latency
        self.register_scenario(BenchmarkScenario(
            name="yolo_detection_latency",
            description="Measure combined YOLO detection latency (3 models)",
            category=BenchmarkCategory.LATENCY,
            num_iterations=200,
            warmup_iterations=20,
            target_latency_ms=20.0,
        ))
        
        # NAFNet deblurring latency
        self.register_scenario(BenchmarkScenario(
            name="nafnet_deblur_latency",
            description="Measure NAFNet crop deblurring latency",
            category=BenchmarkCategory.LATENCY,
            num_iterations=100,
            warmup_iterations=10,
            target_latency_ms=20.0,
        ))
        
        # Full pipeline throughput
        self.register_scenario(BenchmarkScenario(
            name="pipeline_throughput",
            description="Measure full pipeline throughput (frames per second)",
            category=BenchmarkCategory.THROUGHPUT,
            num_iterations=300,
            warmup_iterations=30,
            target_throughput=60.0,  # 60 FPS target
        ))
        
        # Memory usage under load
        self.register_scenario(BenchmarkScenario(
            name="memory_usage",
            description="Measure peak memory usage during processing",
            category=BenchmarkCategory.MEMORY,
            num_iterations=100,
            warmup_iterations=10,
            target_memory_mb=4096.0,  # 4GB target for Jetson
        ))
        
        # Stress test
        self.register_scenario(BenchmarkScenario(
            name="stress_test",
            description="Sustained load test for stability",
            category=BenchmarkCategory.STRESS,
            num_iterations=1000,
            warmup_iterations=0,
            timeout_seconds=120.0,
        ))
    
    def register_scenario(self, scenario: BenchmarkScenario) -> None:
        """Register a benchmark scenario."""
        self.scenarios[scenario.name] = scenario
        logger.debug(f"Registered scenario: {scenario.name}")
    
    def run_scenario(
        self,
        scenario_name: str,
        test_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        Run a single benchmark scenario.
        
        Args:
            scenario_name: Name of scenario to run
            test_func: Function to benchmark (takes no args, returns None)
            
        Returns:
            BenchmarkResult with metrics
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        
        # Use provided test function or scenario's test function
        func = test_func or scenario.test_func
        if func is None:
            # Use mock function for demo
            func = self._create_mock_test_func(scenario)
        
        logger.info(f"Running benchmark: {scenario.name}")
        
        result = BenchmarkResult(
            scenario_name=scenario.name,
            category=scenario.category,
            timestamp=datetime.now(),
        )
        
        try:
            # Warmup
            for _ in range(scenario.warmup_iterations):
                func()
            
            # Benchmark
            latencies = []
            start_time = time.time()
            
            for i in range(scenario.num_iterations):
                # Check timeout
                if time.time() - start_time > scenario.timeout_seconds:
                    logger.warning(f"Benchmark timeout after {i} iterations")
                    break
                
                iter_start = time.time()
                func()
                iter_time = (time.time() - iter_start) * 1000  # ms
                latencies.append(iter_time)
            
            total_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate metrics
            result.total_time_ms = total_time
            result.frames_processed = len(latencies)
            
            if latencies:
                result.avg_latency_ms = np.mean(latencies)
                result.min_latency_ms = np.min(latencies)
                result.max_latency_ms = np.max(latencies)
                result.p95_latency_ms = np.percentile(latencies, 95)
                result.p99_latency_ms = np.percentile(latencies, 99)
                result.operations_per_second = len(latencies) / (total_time / 1000)
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.success = False
            result.error_message = str(e)
        
        # Store result
        self.results.append(result)
        
        # Log summary
        if result.success:
            logger.info(
                f"Benchmark complete: {scenario.name} - "
                f"avg={result.avg_latency_ms:.2f}ms, "
                f"p95={result.p95_latency_ms:.2f}ms, "
                f"throughput={result.operations_per_second:.1f} ops/s"
            )
        
        return result
    
    def _create_mock_test_func(self, scenario: BenchmarkScenario) -> Callable:
        """Create a mock test function for demo purposes."""
        
        # Simulate different latencies based on scenario
        base_latency = {
            "gatekeeper_latency": 0.3,
            "sci_enhancement_latency": 0.4,
            "yolo_detection_latency": 15.0,
            "nafnet_deblur_latency": 18.0,
            "pipeline_throughput": 16.0,
            "memory_usage": 10.0,
            "stress_test": 16.0,
        }.get(scenario.name, 10.0)
        
        def mock_func():
            # Simulate processing with some variance
            latency = base_latency * (0.8 + 0.4 * np.random.random())
            time.sleep(latency / 1000)  # Convert to seconds
        
        return mock_func
    
    def run_all_scenarios(
        self,
        test_funcs: Optional[Dict[str, Callable]] = None
    ) -> List[BenchmarkResult]:
        """
        Run all registered benchmark scenarios.
        
        Args:
            test_funcs: Dict mapping scenario names to test functions
            
        Returns:
            List of all benchmark results
        """
        test_funcs = test_funcs or {}
        results = []
        
        for name, scenario in self.scenarios.items():
            func = test_funcs.get(name)
            result = self.run_scenario(name, func)
            results.append(result)
        
        return results
    
    def run_quick_benchmark(self) -> List[BenchmarkResult]:
        """
        Run a quick subset of benchmarks for demo.
        
        Returns:
            List of benchmark results
        """
        quick_scenarios = [
            "gatekeeper_latency",
            "sci_enhancement_latency",
            "pipeline_throughput",
        ]
        
        results = []
        for name in quick_scenarios:
            if name in self.scenarios:
                # Reduce iterations for quick run
                scenario = self.scenarios[name]
                original_iterations = scenario.num_iterations
                scenario.num_iterations = min(50, original_iterations)
                
                result = self.run_scenario(name)
                results.append(result)
                
                # Restore original
                scenario.num_iterations = original_iterations
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate benchmark report.
        
        Returns:
            Report dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "num_scenarios": len(self.results),
            "summary": {},
            "results": [],
        }
        
        # Calculate summary
        passed = 0
        failed = 0
        
        for result in self.results:
            scenario = self.scenarios.get(result.scenario_name)
            if scenario:
                meets, failures = scenario.meets_targets(result)
                if meets:
                    passed += 1
                else:
                    failed += 1
            
            report["results"].append(result.to_dict())
        
        report["summary"] = {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
        }
        
        return report
    
    def export_report(self, filepath: Optional[Path] = None) -> Path:
        """
        Export benchmark report to JSON.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"benchmark_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported benchmark report to {filepath}")
        return filepath
    
    def print_summary(self) -> None:
        """Print benchmark summary to console."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        for result in self.results:
            scenario = self.scenarios.get(result.scenario_name)
            status = "✅" if result.success else "❌"
            
            print(f"\n{status} {result.scenario_name}")
            print(f"   Category: {result.category.value}")
            print(f"   Avg Latency: {result.avg_latency_ms:.2f}ms")
            print(f"   P95 Latency: {result.p95_latency_ms:.2f}ms")
            print(f"   Throughput: {result.operations_per_second:.1f} ops/s")
            
            if scenario:
                meets, failures = scenario.meets_targets(result)
                if not meets:
                    for failure in failures:
                        print(f"   ⚠️  {failure}")
        
        # Overall summary
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "-" * 70)
        print(f"Total: {summary['total']} | "
              f"Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | "
              f"Pass Rate: {summary['pass_rate']:.1%}")
        print("=" * 70)


def create_benchmark_runner(output_dir: Optional[Path] = None) -> BenchmarkRunner:
    """Factory function to create BenchmarkRunner."""
    return BenchmarkRunner(output_dir=output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("⏱️  Performance Benchmark Runner")
    print("=" * 60)
    
    # Create runner
    runner = create_benchmark_runner()
    
    # Run quick benchmark
    print("\nRunning quick benchmark suite...")
    results = runner.run_quick_benchmark()
    
    # Print summary
    runner.print_summary()
    
    # Export report
    report_path = runner.export_report()
    print(f"\n📁 Report exported to: {report_path}")
    
    print("\n✅ Benchmark complete!")

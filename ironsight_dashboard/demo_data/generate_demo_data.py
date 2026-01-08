#!/usr/bin/env python3
"""
Main script to generate all demo data for IronSight Command Center.

This script generates:
1. Sample wagon images for Restoration Lab testing
2. Mock inspection history for Semantic Search demo
3. Demo video for Mission Control demonstration
4. Performance benchmark scenarios

Usage:
    python generate_demo_data.py [--output-dir PATH] [--num-samples N]

Requirements: Demo preparation (Task 16.2)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from demo_data_generator import (
    DemoDataGenerator,
    generate_all_demo_data,
)
from mock_inspection_history import (
    MockInspectionHistory,
    create_mock_inspection_history,
)
from benchmark_scenarios import (
    BenchmarkRunner,
    create_benchmark_runner,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_restoration_lab_data(
    output_dir: Path,
    num_samples: int = 10
) -> dict:
    """Generate sample images for Restoration Lab."""
    print("\n📸 Generating Restoration Lab samples...")
    
    generator = DemoDataGenerator(output_dir=output_dir)
    samples = generator.generate_restoration_lab_samples(num_samples)
    
    print(f"   Generated {len(samples)} sample images")
    return {"samples": samples, "count": len(samples)}


def generate_semantic_search_data(
    output_dir: Path,
    num_records: int = 100
) -> dict:
    """Generate mock inspection history for Semantic Search."""
    print("\n📋 Generating Semantic Search history...")
    
    history = create_mock_inspection_history(
        output_dir=output_dir / "semantic_search",
        num_records=num_records
    )
    
    # Export to JSON
    export_path = history.export_to_json()
    
    stats = history.get_stats()
    print(f"   Generated {stats['num_records']} inspection records")
    print(f"   Damage ratio: {stats['damage_ratio']:.1%}")
    print(f"   Exported to: {export_path}")
    
    return {
        "stats": stats,
        "export_path": str(export_path),
        "count": stats['num_records']
    }


def generate_mission_control_data(
    output_dir: Path,
    num_frames: int = 60,
    fps: int = 15
) -> dict:
    """Generate demo video for Mission Control."""
    print("\n🎬 Generating Mission Control video...")
    
    generator = DemoDataGenerator(output_dir=output_dir)
    frames, metadata = generator.generate_mission_control_frames(num_frames, fps)
    
    video_path = generator.save_demo_video(frames, "demo_wagon_inspection.mp4", fps)
    
    if video_path:
        print(f"   Generated {num_frames} frames at {fps} FPS")
        print(f"   Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"   Saved to: {video_path}")
    else:
        print("   ⚠️  Video generation skipped (OpenCV not available)")
    
    return {
        "video_path": str(video_path) if video_path else None,
        "metadata": metadata,
        "frame_count": num_frames
    }


def generate_benchmark_scenarios(output_dir: Path) -> dict:
    """Generate benchmark scenario definitions."""
    print("\n⏱️  Setting up benchmark scenarios...")
    
    runner = create_benchmark_runner(output_dir=output_dir / "benchmarks")
    
    # Run quick benchmark to generate sample results
    print("   Running quick benchmark suite...")
    results = runner.run_quick_benchmark()
    
    # Export report
    report_path = runner.export_report()
    
    print(f"   Configured {len(runner.scenarios)} benchmark scenarios")
    print(f"   Ran {len(results)} quick benchmarks")
    print(f"   Report saved to: {report_path}")
    
    return {
        "scenarios": list(runner.scenarios.keys()),
        "results_count": len(results),
        "report_path": str(report_path)
    }


def create_demo_readme(output_dir: Path, results: dict) -> Path:
    """Create README file for demo data."""
    readme_path = output_dir / "README.md"
    
    content = f"""# IronSight Command Center Demo Data

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

This directory contains demo data for testing and demonstrating the
IronSight Command Center dashboard.

### 1. Restoration Lab Samples (`restoration_lab/`)

Sample wagon images with various blur levels for testing the NAFNet
image restoration functionality.

- **Count**: {results.get('restoration_lab', {}).get('count', 'N/A')} images
- **Formats**: PNG
- **Variations**: Different blur levels, noise, brightness

### 2. Semantic Search History (`semantic_search/`)

Mock inspection history data for demonstrating the natural language
search functionality.

- **Records**: {results.get('semantic_search', {}).get('count', 'N/A')} inspection records
- **Format**: JSON
- **Includes**: Wagon IDs, timestamps, damage assessments, OCR results

### 3. Mission Control Video (`mission_control/`)

Demo video for testing the live processing interface.

- **Duration**: {results.get('mission_control', {}).get('metadata', {}).get('duration_seconds', 'N/A')} seconds
- **Format**: MP4
- **Content**: Synthetic wagon footage with serial numbers

### 4. Benchmark Results (`benchmarks/`)

Performance benchmark scenarios and sample results.

- **Scenarios**: {len(results.get('benchmarks', {}).get('scenarios', []))}
- **Format**: JSON reports

## Usage

### Restoration Lab Testing

1. Open the IronSight Command Center dashboard
2. Navigate to "Restoration Lab" tab
3. Upload any image from `restoration_lab/` folder
4. Click "Restore Image" to test NAFNet deblurring

### Semantic Search Demo

1. Navigate to "Semantic Search" tab
2. Try queries like:
   - "wagons with rust damage"
   - "dented doors"
   - "recent inspections"
3. Use filters to narrow results

### Mission Control Demo

1. Navigate to "Mission Control" tab
2. Select "File Upload" as video source
3. Upload `mission_control/demo_wagon_inspection.mp4`
4. Observe real-time processing overlays

### Running Benchmarks

```python
from demo_data.benchmark_scenarios import create_benchmark_runner

runner = create_benchmark_runner()
results = runner.run_all_scenarios()
runner.print_summary()
```

## Regenerating Demo Data

To regenerate all demo data:

```bash
cd ironsight_dashboard/demo_data
python generate_demo_data.py --num-samples 20 --num-records 200
```

## Notes

- Demo data is synthetic and for demonstration purposes only
- Actual inspection data should be used for production testing
- Benchmark results may vary based on hardware configuration
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    return readme_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate demo data for IronSight Command Center"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "generated",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of restoration lab samples to generate"
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=100,
        help="Number of inspection history records to generate"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=60,
        help="Number of video frames to generate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmark generation"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("=" * 70)
    print("🚂 IronSight Command Center - Demo Data Generator")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate all demo data
    try:
        results['restoration_lab'] = generate_restoration_lab_data(
            args.output_dir,
            args.num_samples
        )
        
        results['semantic_search'] = generate_semantic_search_data(
            args.output_dir,
            args.num_records
        )
        
        results['mission_control'] = generate_mission_control_data(
            args.output_dir,
            args.num_frames
        )
        
        if not args.skip_benchmarks:
            results['benchmarks'] = generate_benchmark_scenarios(args.output_dir)
        
        # Create README
        readme_path = create_demo_readme(args.output_dir, results)
        print(f"\n📄 Created README: {readme_path}")
        
    except Exception as e:
        logging.error(f"Error generating demo data: {e}")
        raise
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Demo Data Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated:")
    print(f"  - {results.get('restoration_lab', {}).get('count', 0)} restoration lab samples")
    print(f"  - {results.get('semantic_search', {}).get('count', 0)} inspection history records")
    print(f"  - {results.get('mission_control', {}).get('frame_count', 0)} video frames")
    if 'benchmarks' in results:
        print(f"  - {len(results['benchmarks'].get('scenarios', []))} benchmark scenarios")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

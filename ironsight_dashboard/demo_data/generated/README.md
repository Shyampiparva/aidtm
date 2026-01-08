# IronSight Command Center Demo Data

Generated: 2026-01-08 21:46:36

## Contents

This directory contains demo data for testing and demonstrating the
IronSight Command Center dashboard.

### 1. Restoration Lab Samples (`restoration_lab/`)

Sample wagon images with various blur levels for testing the NAFNet
image restoration functionality.

- **Count**: 10 images
- **Formats**: PNG
- **Variations**: Different blur levels, noise, brightness

### 2. Semantic Search History (`semantic_search/`)

Mock inspection history data for demonstrating the natural language
search functionality.

- **Records**: 100 inspection records
- **Format**: JSON
- **Includes**: Wagon IDs, timestamps, damage assessments, OCR results

### 3. Mission Control Video (`mission_control/`)

Demo video for testing the live processing interface.

- **Duration**: 4.0 seconds
- **Format**: MP4
- **Content**: Synthetic wagon footage with serial numbers

### 4. Benchmark Results (`benchmarks/`)

Performance benchmark scenarios and sample results.

- **Scenarios**: 7
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

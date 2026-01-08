#!/usr/bin/env python3
"""
DeblurGAN Performance Benchmark

Tests inference speed and memory usage of the trained model.
"""

import time
import logging
from pathlib import Path
import argparse

import torch
import numpy as np
import onnxruntime as ort
import cv2

logger = logging.getLogger(__name__)


def benchmark_onnx_model(model_path: Path, num_runs: int = 100) -> dict:
    """Benchmark ONNX model performance."""
    
    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test different input sizes
    test_sizes = [
        (128, 128, "Small Crop"),
        (256, 256, "Standard Crop"), 
        (320, 240, "Railway Frame"),
        (512, 512, "Large Crop")
    ]
    
    results = {}
    
    for h, w, size_name in test_sizes:
        logger.info(f"Benchmarking {size_name} ({h}x{w})...")
        
        # Create test input
        test_input = np.random.randn(1, 3, h, w).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run([output_name], {input_name: test_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            output = session.run([output_name], {input_name: test_input})
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        throughput_fps = 1000 / avg_time_ms
        
        results[size_name] = {
            'size': f"{h}x{w}",
            'avg_time_ms': avg_time_ms,
            'throughput_fps': throughput_fps,
            'pixels': h * w
        }
        
        logger.info(f"  Average time: {avg_time_ms:.2f}ms")
        logger.info(f"  Throughput: {throughput_fps:.1f} FPS")
    
    return results


def benchmark_real_image(model_path: Path, image_path: Path) -> dict:
    """Benchmark on a real image."""
    
    # Load model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to 256x256 for processing
    image_resized = cv2.resize(image, (256, 256))
    
    # Convert to model input format
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5  # [-1, 1]
    image_tensor = np.transpose(image_normalized, (2, 0, 1))  # CHW
    image_batch = np.expand_dims(image_tensor, axis=0)  # NCHW
    
    # Benchmark inference
    num_runs = 50
    
    # Warmup
    for _ in range(5):
        _ = session.run([output_name], {input_name: image_batch})
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        output = session.run([output_name], {input_name: image_batch})
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    
    return {
        'image_path': str(image_path),
        'original_size': f"{image.shape[1]}x{image.shape[0]}",
        'processed_size': "256x256",
        'avg_time_ms': avg_time_ms,
        'throughput_fps': 1000 / avg_time_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeblurGAN performance")
    parser.add_argument("--model-path", type=str, default="models/deblur_gan.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of benchmark runs")
    parser.add_argument("--test-image", type=str, 
                       help="Path to test image for real-world benchmark")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("DEBLURGAN PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    try:
        # Synthetic benchmark
        logger.info("Running synthetic input benchmark...")
        synthetic_results = benchmark_onnx_model(model_path, args.num_runs)
        
        print(f"\n📊 SYNTHETIC INPUT BENCHMARK ({args.num_runs} runs):")
        print(f"{'Size':<15} {'Dimensions':<12} {'Time (ms)':<12} {'FPS':<8} {'Pixels':<10}")
        print("-" * 60)
        
        for size_name, result in synthetic_results.items():
            print(f"{size_name:<15} {result['size']:<12} {result['avg_time_ms']:<12.2f} "
                  f"{result['throughput_fps']:<8.1f} {result['pixels']:<10}")
        
        # Real image benchmark (if provided)
        if args.test_image:
            test_image_path = Path(args.test_image)
            if test_image_path.exists():
                logger.info(f"Running real image benchmark on: {test_image_path}")
                real_result = benchmark_real_image(model_path, test_image_path)
                
                print(f"\n🖼️  REAL IMAGE BENCHMARK:")
                print(f"   Image: {real_result['image_path']}")
                print(f"   Original Size: {real_result['original_size']}")
                print(f"   Processed Size: {real_result['processed_size']}")
                print(f"   Average Time: {real_result['avg_time_ms']:.2f}ms")
                print(f"   Throughput: {real_result['throughput_fps']:.1f} FPS")
            else:
                logger.warning(f"Test image not found: {test_image_path}")
        else:
            # Try to find a test image automatically
            test_dirs = [
                Path("data/blurred_sharp/blurred"),
                Path("data/wagon_detection/train/images")
            ]
            
            for test_dir in test_dirs:
                if test_dir.exists():
                    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                    if test_images:
                        test_image = test_images[0]
                        logger.info(f"Running real image benchmark on: {test_image}")
                        real_result = benchmark_real_image(model_path, test_image)
                        
                        print(f"\n🖼️  REAL IMAGE BENCHMARK:")
                        print(f"   Image: {real_result['image_path']}")
                        print(f"   Original Size: {real_result['original_size']}")
                        print(f"   Processed Size: {real_result['processed_size']}")
                        print(f"   Average Time: {real_result['avg_time_ms']:.2f}ms")
                        print(f"   Throughput: {real_result['throughput_fps']:.1f} FPS")
                        break
        
        # Performance summary
        standard_result = synthetic_results.get("Standard Crop", {})
        standard_time = standard_result.get('avg_time_ms', 0)
        
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   Model: DeblurGAN-v2 MobileNet-DSC")
        print(f"   Standard Crop (256x256): {standard_time:.2f}ms")
        print(f"   Target: <40ms ({'✅ PASSED' if standard_time < 40 else '❌ FAILED'})")
        print(f"   GPU Acceleration: {'✅ CUDA' if torch.cuda.is_available() else '❌ CPU Only'}")
        print(f"   Production Ready: ✅ ONNX Exported")
        
        # Railway inspection context
        print(f"\n🚂 RAILWAY INSPECTION CONTEXT:")
        print(f"   Train Speed: 50-80 km/h")
        print(f"   Frame Rate: ~30 FPS")
        print(f"   Processing Budget: ~33ms per frame")
        print(f"   Crop Processing: {standard_time:.2f}ms ({'✅ Real-time capable' if standard_time < 33 else '⚠️ May need optimization'})")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
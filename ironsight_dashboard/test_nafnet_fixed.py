#!/usr/bin/env python3
"""
Test script for the fixed NAFNet implementation.

This script tests:
1. GPU detection and device selection
2. Checkpoint analysis and architecture detection
3. Model loading with correct architecture
4. Image processing functionality
5. Performance benchmarking
"""

import sys
import logging
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test GPU detection and device selection."""
    print("🔍 Testing Device Detection")
    print("-" * 40)
    
    try:
        from nafnet_fixed import detect_device
        import torch
        
        device = detect_device()
        print(f"✅ Detected device: {device}")
        
        # Print detailed GPU info
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("   No CUDA devices available")
            print("   Reasons could be:")
            print("   - No NVIDIA GPU installed")
            print("   - CUDA drivers not installed")
            print("   - PyTorch CPU-only version installed")
        
        return True
        
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_checkpoint_analysis():
    """Test checkpoint analysis functionality."""
    print("\n🔍 Testing Checkpoint Analysis")
    print("-" * 40)
    
    try:
        from nafnet_fixed import analyze_checkpoint
        
        # Find the NAFNet model
        model_paths = [
            Path("../NAFNet-REDS-width64.pth"),
            Path("../../NAFNet-REDS-width64.pth"),
            Path("NAFNet-REDS-width64.pth")
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if not model_path:
            print("❌ NAFNet model file not found")
            return False
        
        print(f"✅ Found model: {model_path.absolute()}")
        
        # Analyze the checkpoint
        arch_info = analyze_checkpoint(str(model_path))
        print(f"✅ Architecture analysis completed:")
        print(f"   Encoder blocks: {arch_info['enc_blk_nums']}")
        print(f"   Decoder blocks: {arch_info['dec_blk_nums']}")
        print(f"   Middle blocks: {arch_info['middle_blk_num']}")
        print(f"   Width: {arch_info['width']}")
        
        return True, str(model_path)
        
    except Exception as e:
        print(f"❌ Checkpoint analysis failed: {e}")
        return False, None

def test_model_loading(model_path: str):
    """Test model loading with the fixed implementation."""
    print("\n🔍 Testing Model Loading")
    print("-" * 40)
    
    try:
        from nafnet_fixed import FixedNAFNetLoader
        
        # Test with auto device detection
        loader = FixedNAFNetLoader(model_path)
        success = loader.load_model()
        
        if success:
            print("✅ Model loaded successfully!")
            
            # Print model info
            info = loader.get_info()
            print(f"   Device: {info['device']}")
            print(f"   Architecture: {info['architecture_info']}")
            print(f"   CUDA available: {info['cuda_available']}")
            
            return True, loader
        else:
            print("❌ Model loading failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Model loading crashed: {e}")
        return False, None

def test_image_processing(loader):
    """Test image processing functionality."""
    print("\n🔍 Testing Image Processing")
    print("-" * 40)
    
    try:
        # Create test images of different sizes
        test_cases = [
            (64, 64, "Small image"),
            (256, 256, "Medium image"),
            (512, 512, "Large image"),
            (100, 150, "Non-square image")
        ]
        
        for height, width, description in test_cases:
            # Create random test image (BGR format)
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Process the image
            start_time = time.time()
            processed_image = loader.process_image(test_image)
            processing_time = (time.time() - start_time) * 1000
            
            # Verify output
            if processed_image.shape == test_image.shape:
                print(f"✅ {description} ({height}x{width}): {processing_time:.1f}ms")
            else:
                print(f"❌ {description}: Shape mismatch {test_image.shape} -> {processed_image.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def test_performance_benchmark(loader):
    """Run performance benchmarks."""
    print("\n🔍 Performance Benchmark")
    print("-" * 40)
    
    try:
        # Test different image sizes
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        
        for height, width in sizes:
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(3):
                loader.process_image(test_image)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                loader.process_image(test_image)
                times.append((time.time() - start_time) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"   {height}x{width}: {avg_time:.1f}±{std_time:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios."""
    print("\n🔍 Testing Error Handling")
    print("-" * 40)
    
    try:
        from nafnet_fixed import FixedNAFNetLoader
        
        # Test with non-existent model
        loader = FixedNAFNetLoader("non_existent_model.pth")
        success = loader.load_model()
        
        if not success:
            print("✅ Correctly handled non-existent model")
        else:
            print("❌ Should have failed with non-existent model")
            return False
        
        # Test processing without loaded model
        test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = loader.process_image(test_image)
        
        if np.array_equal(result, test_image):
            print("✅ Correctly returned original image when model not loaded")
        else:
            print("❌ Should have returned original image when model not loaded")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Fixed NAFNet Implementation")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Device Detection
    tests_total += 1
    if test_device_detection():
        tests_passed += 1
    
    # Test 2: Checkpoint Analysis
    tests_total += 1
    analysis_success, model_path = test_checkpoint_analysis()
    if analysis_success:
        tests_passed += 1
    
    if not analysis_success:
        print("\n❌ Cannot continue without model file")
        return 1
    
    # Test 3: Model Loading
    tests_total += 1
    loading_success, loader = test_model_loading(model_path)
    if loading_success:
        tests_passed += 1
    
    if loading_success and loader:
        # Test 4: Image Processing
        tests_total += 1
        if test_image_processing(loader):
            tests_passed += 1
        
        # Test 5: Performance Benchmark
        tests_total += 1
        if test_performance_benchmark(loader):
            tests_passed += 1
    
    # Test 6: Error Handling
    tests_total += 1
    if test_error_handling():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! Fixed NAFNet is working correctly.")
        
        if loader and loader.is_loaded:
            info = loader.get_info()
            print(f"\n🚀 Ready for production:")
            print(f"   Device: {info['device']}")
            print(f"   Model: {info['model_path']}")
            print(f"   GPU Available: {info['cuda_available']}")
            
            if not info['cuda_available']:
                print("\n⚠️  GPU Acceleration Tips:")
                print("   1. Install NVIDIA GPU drivers")
                print("   2. Install CUDA toolkit")
                print("   3. Install PyTorch with CUDA support:")
                print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("💥 Some tests failed. Check the errors above.")
    
    return 0 if tests_passed == tests_total else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script to verify NAFNet mock fallback works.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nafnet_mock_fallback():
    """Test NAFNet with mock fallback."""
    
    print("🔬 Testing NAFNet Mock Fallback")
    print("="*60)
    
    try:
        from crop_first_nafnet import CropFirstNAFNet
        print("✅ NAFNet module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NAFNet module: {e}")
        return False
    
    # Test creating NAFNet instance
    try:
        nafnet = CropFirstNAFNet(
            model_path="../NAFNet-REDS-width64.pth",
            device="cpu"
        )
        print("✅ NAFNet instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create NAFNet instance: {e}")
        return False
    
    # Test model loading
    try:
        success = nafnet.load_model()
        if success:
            print("✅ NAFNet model loaded successfully!")
            print(f"   Model loaded: {nafnet.model_loaded}")
        else:
            print("❌ NAFNet model loading failed")
            return False
    except Exception as e:
        print(f"❌ NAFNet model loading failed: {e}")
        return False
    
    # Test image processing
    try:
        # Create test image
        test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        # Process image
        result = nafnet.nafnet_deblur(test_image)
        
        if result is not None and result.shape == test_image.shape:
            print("✅ Image deblurring successful!")
            print(f"   Input shape: {test_image.shape}")
            print(f"   Output shape: {result.shape}")
        else:
            print("❌ Image deblurring failed")
            return False
            
    except Exception as e:
        print(f"❌ Image deblurring failed: {e}")
        return False
    
    print("\n🎯 NAFNet Status:")
    print(f"   Model loaded: {nafnet.model_loaded}")
    print(f"   Using mock: {hasattr(nafnet, 'mock_loader')}")
    print(f"   Using standalone: {hasattr(nafnet, 'standalone_loader')}")
    
    return True

if __name__ == "__main__":
    success = test_nafnet_mock_fallback()
    print("\n" + "="*60)
    if success:
        print("🎉 NAFNet mock fallback test completed successfully!")
    else:
        print("❌ NAFNet mock fallback test failed!")
    sys.exit(0 if success else 1)
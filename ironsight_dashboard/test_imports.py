#!/usr/bin/env python3
"""
Test script to verify that the application can start without BasicSR.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test critical imports for the application."""
    
    print("Testing critical imports...")
    
    # Test basic imports
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV import successful")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch import successful (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ Torchvision import successful (version: {torchvision.__version__})")
    except ImportError as e:
        print(f"❌ Torchvision import failed: {e}")
        return False
    
    # Test BasicSR (should handle gracefully if it fails)
    try:
        import basicsr
        print("✅ BasicSR import successful")
    except ImportError as e:
        print(f"⚠️ BasicSR import failed (expected): {e}")
        print("   This is OK - the app will use mock restoration")
    
    # Test our modules
    try:
        from src.restoration_lab import RestorationLab
        print("✅ RestorationLab import successful")
    except ImportError as e:
        print(f"❌ RestorationLab import failed: {e}")
        return False
    
    try:
        from src.crop_first_nafnet import CropFirstNAFNet
        print("✅ CropFirstNAFNet import successful")
    except ImportError as e:
        print(f"⚠️ CropFirstNAFNet import failed (expected): {e}")
        print("   This is OK - the app will use mock restoration")
    
    # Test creating restoration lab without NAFNet
    try:
        lab = RestorationLab(nafnet_model_path=None, device="cpu")
        status = lab.get_model_status()
        print(f"✅ RestorationLab created successfully")
        print(f"   Model loaded: {status['model_loaded']}")
        print(f"   Model type: {status['model_type']}")
    except Exception as e:
        print(f"❌ RestorationLab creation failed: {e}")
        return False
    
    print("\n🎉 All critical imports successful! Application should start properly.")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
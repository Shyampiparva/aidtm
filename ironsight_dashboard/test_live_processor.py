#!/usr/bin/env python3
"""
Test script for live_processor.py dependencies and basic functionality.

This script verifies that all required dependencies are available and
tests basic functionality without requiring an actual IP webcam.
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestLiveProcessor')

def test_imports():
    """Test that all required imports work."""
    logger.info("Testing imports...")
    
    try:
        import cv2
        logger.info(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import niquests
        logger.info(f"✅ niquests version: {niquests.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import niquests: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import NumPy: {e}")
        return False
    
    return True

def test_webcam_config():
    """Test WebcamConfig class."""
    logger.info("Testing WebcamConfig...")
    
    try:
        from live_processor import WebcamConfig
        
        # Test basic config
        config = WebcamConfig(base_url="192.168.1.100:8080")
        assert config.base_url == "http://192.168.1.100:8080"
        logger.info("✅ Basic config creation works")
        
        # Test quality settings
        config_high = WebcamConfig(base_url="http://test.com", quality="high")
        assert config_high.max_width == 1920
        assert config_high.max_height == 1080
        logger.info("✅ Quality settings work")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebcamConfig test failed: {e}")
        return False

def test_api_client():
    """Test WebcamAPIClient class (without actual connection)."""
    logger.info("Testing WebcamAPIClient...")
    
    try:
        from live_processor import WebcamConfig, WebcamAPIClient
        
        config = WebcamConfig(base_url="http://test.example.com:8080")
        client = WebcamAPIClient(config)
        
        # Test URL generation
        video_url = client.get_video_url()
        assert video_url == "http://test.example.com:8080/video"
        logger.info("✅ Video URL generation works")
        
        # Test session creation
        assert client.session is not None
        logger.info("✅ HTTP session creation works")
        
        client.close()
        logger.info("✅ Client cleanup works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebcamAPIClient test failed: {e}")
        return False

def test_processing_stats():
    """Test ProcessingStats class."""
    logger.info("Testing ProcessingStats...")
    
    try:
        from live_processor import ProcessingStats
        
        stats = ProcessingStats()
        
        # Test initial values
        assert stats.frames_processed == 0
        assert stats.fps == 0.0
        logger.info("✅ Initial stats values correct")
        
        # Test property calculations
        stats.frames_processed = 100
        stats.total_processing_time = 10.0
        assert abs(stats.fps - 10.0) < 0.1
        logger.info("✅ FPS calculation works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ProcessingStats test failed: {e}")
        return False

def test_niquests_functionality():
    """Test niquests HTTP functionality."""
    logger.info("Testing niquests functionality...")
    
    try:
        import niquests
        
        # Test session creation
        session = niquests.Session()
        session.headers.update({'User-Agent': 'Test-Agent/1.0'})
        logger.info("✅ niquests session creation works")
        
        # Test basic request (to a reliable endpoint)
        try:
            response = session.get('https://httpbin.org/get', timeout=5)
            if response.status_code == 200:
                logger.info("✅ niquests HTTP request works")
            else:
                logger.warning(f"⚠️ HTTP request returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ HTTP request failed (network issue?): {e}")
        
        session.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ niquests functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting live_processor.py tests...")
    logger.info("=" * 50)
    
    tests = [
        test_imports,
        test_webcam_config,
        test_api_client,
        test_processing_stats,
        test_niquests_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        logger.info("-" * 30)
    
    logger.info("=" * 50)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! live_processor.py is ready to use.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
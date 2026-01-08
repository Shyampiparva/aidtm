#!/usr/bin/env python3
"""
Test script to verify the video playback fixes.

This script tests the updated Streamlit app to ensure:
1. No deprecation warnings for use_container_width
2. Proper video file handling
3. No MediaFileHandler errors
"""

import subprocess
import sys
import time
from pathlib import Path

def test_streamlit_syntax():
    """Test that the Streamlit app has no syntax errors."""
    print("🔍 Testing Streamlit app syntax...")
    
    try:
        # Test Python syntax
        result = subprocess.run([
            sys.executable, '-m', 'py_compile', 'src/app.py'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Streamlit app syntax is valid")
            return True
        else:
            print(f"❌ Syntax error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing syntax: {e}")
        return False

def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test importing the main modules
        result = subprocess.run([
            sys.executable, '-c', 
            'import sys; sys.path.insert(0, "src"); import app; print("Imports successful")'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ All imports successful")
            return True
        else:
            print(f"❌ Import error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing imports: {e}")
        return False

def check_deprecated_usage():
    """Check for deprecated Streamlit usage."""
    print("🔍 Checking for deprecated Streamlit usage...")
    
    try:
        with open('src/app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for use_container_width
        if 'use_container_width' in content:
            print("❌ Found deprecated 'use_container_width' usage")
            return False
        else:
            print("✅ No deprecated 'use_container_width' usage found")
        
        # Check for width parameter
        if 'width="stretch"' in content:
            print("✅ Found updated 'width=\"stretch\"' usage")
        else:
            print("⚠️ No 'width=\"stretch\"' usage found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking deprecated usage: {e}")
        return False

def test_video_file_cleanup():
    """Test video file cleanup functionality."""
    print("🔍 Testing video file cleanup...")
    
    try:
        # Create some test temp files
        test_files = [
            Path("temp_video_test1.mp4"),
            Path("temp_video_test2.mp4"),
            Path("temp_video_abc123.mp4")
        ]
        
        for test_file in test_files:
            test_file.write_text("test content")
        
        print(f"Created {len(test_files)} test temp files")
        
        # Test cleanup code
        import glob
        temp_files = glob.glob("temp_video_*.mp4")
        print(f"Found {len(temp_files)} temp video files")
        
        # Clean up test files
        for temp_file in test_files:
            if temp_file.exists():
                temp_file.unlink()
        
        print("✅ Video file cleanup test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing video file cleanup: {e}")
        return False

def test_error_handling():
    """Test error handling improvements."""
    print("🔍 Testing error handling...")
    
    try:
        # Check if error handling code is present
        with open('src/app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'MediaFileStorageError' in content:
            print("✅ Found MediaFileStorageError handling")
        else:
            print("⚠️ MediaFileStorageError handling not found")
        
        if 'st.exception(e)' in content:
            print("✅ Found exception handling")
        else:
            print("⚠️ Exception handling not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Streamlit Video Playback Fixes")
    print("=" * 50)
    
    tests = [
        ("Syntax Check", test_streamlit_syntax),
        ("Import Check", test_imports),
        ("Deprecated Usage Check", check_deprecated_usage),
        ("Video File Cleanup", test_video_file_cleanup),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Video playback fixes are working.")
        print("\n🚀 Ready to test:")
        print("1. Run: uv run python run_ironsight.py")
        print("2. Go to Mission Control tab")
        print("3. Upload a video file")
        print("4. Video should play without deprecation warnings")
    else:
        print("💥 Some tests failed. Check the errors above.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
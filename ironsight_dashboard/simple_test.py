#!/usr/bin/env python3
"""Simple test to verify dependencies are working."""

print("Testing dependencies...")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__} - OK")
except ImportError as e:
    print(f"❌ OpenCV failed: {e}")

try:
    import niquests
    print(f"✅ niquests {niquests.__version__} - OK")
except ImportError as e:
    print(f"❌ niquests failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} - OK")
except ImportError as e:
    print(f"❌ NumPy failed: {e}")

print("\nTesting niquests functionality...")
try:
    session = niquests.Session()
    session.headers.update({'User-Agent': 'Test/1.0'})
    print("✅ niquests session created successfully")
    session.close()
except Exception as e:
    print(f"❌ niquests session failed: {e}")

print("\nAll basic tests completed!")
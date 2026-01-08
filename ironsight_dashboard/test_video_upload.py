#!/usr/bin/env python3
"""
Test script for video upload functionality in Streamlit.

This script creates a simple test video and verifies that the video upload
and playback functionality works correctly.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VideoUploadTest')

def create_test_video(output_path: str = "test_video.mp4", duration_seconds: int = 5):
    """Create a simple test video for upload testing."""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    total_frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Creating test video: {output_path}")
    logger.info(f"Properties: {width}x{height} @ {fps} FPS, {duration_seconds}s duration")
    
    try:
        for frame_num in range(total_frames):
            # Create a frame with changing colors and text
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Background color that changes over time
            hue = int((frame_num / total_frames) * 180)
            frame[:, :] = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
            
            # Add frame number text
            text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add timestamp
            time_text = f"Time: {frame_num / fps:.2f}s"
            cv2.putText(frame, time_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add moving circle
            center_x = int(width * 0.5 + 100 * np.sin(2 * np.pi * frame_num / fps))
            center_y = int(height * 0.7)
            cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
            
            # Write frame
            out.write(frame)
            
            if frame_num % 30 == 0:  # Log progress every second
                logger.info(f"Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
        
        logger.info("✅ Test video created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating test video: {e}")
        return False
    finally:
        out.release()

def verify_video_file(video_path: str):
    """Verify that the created video file is valid."""
    
    logger.info(f"Verifying video file: {video_path}")
    
    if not Path(video_path).exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error("❌ Could not open video file")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"✅ Video properties:")
        logger.info(f"   Resolution: {width}x{height}")
        logger.info(f"   FPS: {fps}")
        logger.info(f"   Frame count: {frame_count}")
        logger.info(f"   Duration: {duration:.2f} seconds")
        
        # Try to read a few frames
        frames_read = 0
        for i in range(min(10, frame_count)):
            ret, frame = cap.read()
            if ret:
                frames_read += 1
            else:
                break
        
        cap.release()
        
        if frames_read > 0:
            logger.info(f"✅ Successfully read {frames_read} frames")
            return True
        else:
            logger.error("❌ Could not read any frames")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verifying video: {e}")
        return False

def test_video_formats():
    """Test different video formats for compatibility."""
    
    formats = [
        ("test_video_mp4.mp4", "mp4v"),
        ("test_video_avi.avi", "XVID"),
    ]
    
    results = {}
    
    for filename, codec in formats:
        logger.info(f"Testing format: {filename} with codec {codec}")
        
        try:
            # Create a short test video
            width, height = 320, 240
            fps = 15
            duration = 2
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            for i in range(duration * fps):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
            
            out.release()
            
            # Verify the file
            if verify_video_file(filename):
                results[filename] = "✅ Success"
                logger.info(f"✅ {filename} created and verified successfully")
            else:
                results[filename] = "❌ Verification failed"
                
        except Exception as e:
            results[filename] = f"❌ Error: {e}"
            logger.error(f"❌ Failed to create {filename}: {e}")
    
    return results

def main():
    """Run video upload tests."""
    
    logger.info("🎬 Starting video upload functionality tests")
    logger.info("=" * 50)
    
    # Test 1: Create a standard test video
    logger.info("Test 1: Creating standard test video")
    success = create_test_video("test_video.mp4", duration_seconds=3)
    
    if success:
        verify_video_file("test_video.mp4")
    
    logger.info("-" * 30)
    
    # Test 2: Test different formats
    logger.info("Test 2: Testing different video formats")
    format_results = test_video_formats()
    
    logger.info("Format test results:")
    for filename, result in format_results.items():
        logger.info(f"  {filename}: {result}")
    
    logger.info("-" * 30)
    
    # Test 3: Check OpenCV video capabilities
    logger.info("Test 3: Checking OpenCV video capabilities")
    
    try:
        # List available video codecs
        logger.info("Available video backends:")
        backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
        backend_names = ["ANY", "V4L2", "FFMPEG", "GSTREAMER"]
        
        for backend, name in zip(backends, backend_names):
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    logger.info(f"  ✅ {name} backend available")
                    cap.release()
                else:
                    logger.info(f"  ❌ {name} backend not available")
            except:
                logger.info(f"  ❌ {name} backend not available")
                
    except Exception as e:
        logger.error(f"Error checking OpenCV capabilities: {e}")
    
    logger.info("=" * 50)
    logger.info("🎉 Video upload tests completed!")
    logger.info("")
    logger.info("To test in Streamlit:")
    logger.info("1. Run: uv run python run_ironsight.py")
    logger.info("2. Go to Mission Control tab")
    logger.info("3. Select 'Video File' as input source")
    logger.info("4. Upload test_video.mp4")
    logger.info("5. Click 'Start Processing'")

if __name__ == "__main__":
    main()
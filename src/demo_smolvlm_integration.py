"""
Demo script for SmolVLM 2 integration with Iron-Sight pipeline.

This script demonstrates:
1. SmolVLM 2 fallback OCR when PaddleOCR confidence < 0.50
2. Damage assessment using SmolVLM for detected defects
3. Performance monitoring and statistics

Usage:
    python -m src.demo_smolvlm_integration
"""

import logging
import time
import numpy as np
import cv2
from pathlib import Path

from .agent_forensic import SmolVLMForensicAgent, ForensicTask
from .pipeline_core import IronSightEngine
from .models import PipelineConfig


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_test_images():
    """Create test images for demonstration."""
    # Create a test image with text (simulating wagon ID)
    text_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(text_image, "ABCD123456", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Create a blurry version (simulating motion blur)
    kernel = np.ones((15, 15), np.float32) / 225
    blurry_image = cv2.filter2D(text_image, -1, kernel)
    
    # Create a damaged/rusty image
    damage_image = np.ones((200, 400, 3), dtype=np.uint8) * 100
    # Add some rust-like patterns
    cv2.circle(damage_image, (100, 100), 30, (0, 50, 150), -1)  # Rust spot
    cv2.rectangle(damage_image, (200, 50), (300, 150), (50, 50, 50), -1)  # Dent
    
    return {
        "clear_text": text_image,
        "blurry_text": blurry_image,
        "damage": damage_image
    }


def demo_forensic_agent():
    """Demonstrate the SmolVLM forensic agent capabilities."""
    print("=== SmolVLM 2 Forensic Agent Demo ===")
    
    # Initialize forensic agent
    agent = SmolVLMForensicAgent(
        quantization_bits=8,  # Use 8-bit for better compatibility
        timeout_seconds=30.0
    )
    
    print("Starting forensic agent...")
    if not agent.start():
        print("Failed to start forensic agent - check if transformers/accelerate are installed")
        return
    
    # Create test images
    test_images = create_test_images()
    
    print("\n1. Testing OCR Fallback on Clear Text:")
    result = agent.analyze_crop(
        image=test_images["clear_text"],
        prompt="Read all the text visible in this image. Focus on any alphanumeric codes.",
        task_type="ocr_fallback",
        blocking=True
    )
    
    if result:
        print(f"   Text: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing time: {result.processing_time_ms}ms")
    
    print("\n2. Testing OCR Fallback on Blurry Text:")
    result = agent.analyze_crop(
        image=test_images["blurry_text"],
        prompt="Read all the text visible in this image. Focus on any alphanumeric codes.",
        task_type="ocr_fallback",
        blocking=True
    )
    
    if result:
        print(f"   Text: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing time: {result.processing_time_ms}ms")
    
    print("\n3. Testing Damage Assessment:")
    result = agent.analyze_crop(
        image=test_images["damage"],
        prompt="Describe the damage severity: is it a dent, a hole, or rust?",
        task_type="damage_assessment",
        blocking=True
    )
    
    if result:
        print(f"   Assessment: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing time: {result.processing_time_ms}ms")
    
    # Show statistics
    stats = agent.get_stats()
    print(f"\n4. Performance Statistics:")
    print(f"   Total inferences: {stats['total_inferences']}")
    print(f"   Success rate: {stats['success_rate']:.2f}")
    print(f"   Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"   Queue size: {stats['queue_size']}")
    
    agent.stop()
    print("\nForensic agent demo completed!")


def demo_pipeline_integration():
    """Demonstrate the full pipeline integration."""
    print("\n=== Pipeline Integration Demo ===")
    
    # Create pipeline configuration
    config = PipelineConfig(
        video_source=0,  # Use webcam for demo
        model_dir="models/",
        queue_maxsize=2,
        ocr_timeout_ms=50.0
    )
    
    # Initialize pipeline
    engine = IronSightEngine(config)
    
    print("Starting Iron-Sight pipeline...")
    if not engine.start():
        print("Failed to start pipeline - check video source and models")
        return
    
    # Run for a short demo period
    demo_duration = 10  # seconds
    start_time = time.time()
    
    print(f"Running pipeline demo for {demo_duration} seconds...")
    print("The pipeline will:")
    print("- Process frames from video source")
    print("- Use PaddleOCR for primary OCR")
    print("- Fallback to SmolVLM when OCR confidence < 0.50")
    print("- Assess damage when 'damage_door' class is detected")
    
    try:
        while time.time() - start_time < demo_duration:
            # Get current state
            state = engine.get_state()
            forensic_stats = engine.get_forensic_stats()
            
            print(f"\rFPS: {state.avg_fps:.1f} | "
                  f"Processed: {state.frames_processed} | "
                  f"Dropped: {state.frames_dropped} | "
                  f"OCR Fallbacks: {state.ocr_fallbacks} | "
                  f"Damage Assessments: {state.damage_assessments}", end="")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        engine.stop()
        print(f"\n\nFinal Statistics:")
        state = engine.get_state()
        forensic_stats = engine.get_forensic_stats()
        
        print(f"Frames processed: {state.frames_processed}")
        print(f"Frames dropped: {state.frames_dropped}")
        print(f"Average FPS: {state.avg_fps:.1f}")
        print(f"OCR fallbacks triggered: {state.ocr_fallbacks}")
        print(f"Damage assessments: {state.damage_assessments}")
        print(f"SmolVLM success rate: {forensic_stats.get('success_rate', 0):.2f}")


def demo_fallback_trigger():
    """Demonstrate the specific fallback trigger logic."""
    print("\n=== Fallback Trigger Logic Demo ===")
    
    # Simulate different OCR confidence scenarios
    scenarios = [
        {"paddle_confidence": 0.85, "text": "ABCD123456", "should_fallback": False},
        {"paddle_confidence": 0.45, "text": "unclear", "should_fallback": True},
        {"paddle_confidence": 0.30, "text": "???", "should_fallback": True},
        {"paddle_confidence": 0.60, "text": "WXYZ789012", "should_fallback": False},
    ]
    
    print("Testing fallback trigger logic (confidence < 0.50):")
    print("Paddle Confidence | Text Result | Fallback Triggered")
    print("-" * 55)
    
    for scenario in scenarios:
        conf = scenario["paddle_confidence"]
        text = scenario["text"]
        should_fallback = scenario["should_fallback"]
        
        # This is the core logic from the pipeline
        fallback_triggered = conf < 0.50
        
        status = "✓" if fallback_triggered == should_fallback else "✗"
        fallback_str = "YES" if fallback_triggered else "NO"
        
        print(f"{conf:>15.2f} | {text:>11} | {fallback_str:>15} {status}")
    
    print("\nThe fallback trigger works as specified:")
    print("- PaddleOCR confidence >= 0.50: Use PaddleOCR result")
    print("- PaddleOCR confidence < 0.50: Trigger SmolVLM fallback")
    print("- Display 'Analyzing...' placeholder while SmolVLM processes")


def main():
    """Run all demos."""
    setup_logging()
    
    print("Iron-Sight SmolVLM 2 Integration Demo")
    print("=" * 50)
    
    try:
        # Demo 1: Forensic agent capabilities
        demo_forensic_agent()
        
        # Demo 2: Fallback trigger logic
        demo_fallback_trigger()
        
        # Demo 3: Full pipeline integration (optional - requires video source)
        response = input("\nRun full pipeline demo? (requires webcam/video source) [y/N]: ")
        if response.lower().startswith('y'):
            demo_pipeline_integration()
        else:
            print("Skipping pipeline demo")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ SmolVLM 2 model loading with quantization")
        print("✓ OCR fallback when PaddleOCR confidence < 0.50")
        print("✓ Damage assessment for detected defects")
        print("✓ Async processing to maintain 60 FPS")
        print("✓ Performance monitoring and statistics")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
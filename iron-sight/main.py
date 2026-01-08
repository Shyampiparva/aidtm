"""
Main entry point for the Iron-Sight railway inspection system.
"""

import logging
import sys
from pathlib import Path

from .pipeline_core import IronSightEngine
from .models import PipelineConfig
from .agent_forensic import initialize_forensic_agent


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for Iron-Sight."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Iron-Sight Railway Inspection System")
    
    # Configuration
    config = PipelineConfig(
        video_source=0,  # Default to webcam, can be changed to video file
        model_dir="models/",
        queue_maxsize=2,
        total_timeout_ms=100.0
    )
    
    # Initialize forensic agent
    logger.info("Initializing SmolVLM 2 forensic agent...")
    if not initialize_forensic_agent(quantization_bits=8):
        logger.warning("Failed to initialize forensic agent - OCR fallback disabled")
    
    # Create and start pipeline
    engine = IronSightEngine(config)
    
    try:
        if engine.start():
            logger.info("Iron-Sight pipeline started successfully")
            logger.info("Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while True:
                import time
                time.sleep(1)
                
                # Print stats periodically
                state = engine.get_state()
                if state.frames_processed % 60 == 0 and state.frames_processed > 0:
                    forensic_stats = engine.get_forensic_stats()
                    logger.info(
                        f"Stats - FPS: {state.avg_fps:.1f}, "
                        f"Processed: {state.frames_processed}, "
                        f"OCR Fallbacks: {state.ocr_fallbacks}, "
                        f"SmolVLM Success Rate: {forensic_stats.get('success_rate', 0):.2f}"
                    )
        else:
            logger.error("Failed to start Iron-Sight pipeline")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Stopping Iron-Sight pipeline...")
        engine.stop()
        logger.info("Iron-Sight stopped")


if __name__ == "__main__":
    main()

# Implementation Plan: IronSight Command Center

## Overview

This implementation plan creates the IronSight Command Center dashboard by integrating existing models and AI agents into a unified Streamlit interface. The system leverages existing codebase components including NAFNet-Width64, SmolVLM forensic agent, SigLIP semantic search, and SCI preprocessor, while adding the missing YOLO models and Gatekeeper for complete functionality.

## Tasks

- [x] 1. Project Setup and Environment Configuration




  - [x] 1.1 Initialize IronSight Command Center project structure


    - Create `ironsight_dashboard/` directory with proper structure
    - Initialize with `uv init` and configure `pyproject.toml`
    - Add dependencies: `streamlit`, `streamlit-image-comparison`, `torch`, `torchvision`, `ultralytics`, `opencv-python-headless`, `pillow`, `numpy`, `pandas`, `einops`, `basicsr`
    - _Requirements: 1.1, 14.1_

  - [x] 1.2 Asset discovery and configuration management


    - Scan `aidtm/` folder for existing config files, fonts, and utility scripts
    - Import existing configuration from `config/vehicle_detection.yaml`
    - Set up logging and monitoring configurations from existing codebase
    - _Requirements: 15.1, 15.2, 15.3_

  - [x] 1.3 Write property test for asset scanning



    - **Property 15: Asset Discovery Completeness**
    - **Validates: Requirements 15.1**

- [x] 2. Core Engine Integration




  - [x] 2.1 Create IronSightEngine main orchestrator


    - Implement `src/ironsight_engine.py` integrating existing `src/pipeline_core.py` as foundational logic
    - Create model loading manager with graceful error handling
    - Implement GPU memory optimization with FP16 quantization
    - Add model status tracking (loaded/offline/error)
    - _Requirements: 1.1, 1.2, 1.4, 14.1_

  - [x] 2.2 Write property test for multi-model loading



    - **Property 1: Multi-Model Loading Success**
    - **Validates: Requirements 1.1**

  - [x] 2.3 Write property test for GPU memory optimization


    - **Property 2: GPU Memory Optimization**
    - **Validates: Requirements 1.2**

- [x] 3. Gatekeeper Model Implementation






  - [x] 3.1 Create Gatekeeper binary classifier






    - Implement `src/gatekeeper_model.py` using MobileNetV3-Small architecture
    - Create training script for joint prediction `[is_wagon_present, is_blurry]`
    - Train on existing car dataset and mock wagon data for dual classification
    - Export to ONNX with FP16 precision, target <0.5ms inference
    - _Requirements: 3.1, 3.2, 13.1_

  - [x] 3.2 Write property test for gatekeeper output format


    - **Property 3: Gatekeeper Dual Output Format**
    - **Validates: Requirements 3.2**

  - [x] 3.3 Write property test for gatekeeper performance


    - **Property 4: Gatekeeper Performance Constraint**
    - **Validates: Requirements 3.1**

- [x] 4. SCI Enhancement Integration




  - [x] 4.1 Integrate existing SCI preprocessor


    - Create wrapper `src/sci_enhancer.py` around existing `src/preprocessor_sci.py`
    - Add performance monitoring and statistics tracking
    - Implement brightness-based skip logic for daytime optimization
    - Configure for ~0.5ms target inference time
    - _Requirements: 4.1, 4.2, 14.3_

  - [x] 4.2 Write property test for SCI performance


    - **Property 5: SCI Enhancement Performance**
    - **Validates: Requirements 4.1**

  - [x] 4.3 Write property test for brightness skip logic


    - **Property 6: SCI Brightness Skip Logic**
    - **Validates: Requirements 4.2**

- [x] 5. Multi-YOLO Detection System




  - [x] 5.1 Create placeholder YOLO models for demonstration


    - Implement `src/multi_yolo_detector.py` with 3 model slots
    - Create mock implementations for: sideview_damage_obb, structure_obb, wagon_number_obb
    - Implement detection merging logic into single JSON result
    - Add oriented bounding box support and visualization
    - _Requirements: 5.1, 5.2, 5.5_

  - [x] 5.2 Write property test for detection merging




    - **Property 7: YOLO Detection Merging**
    - **Validates: Requirements 5.2**

- [-] 6. Crop-First NAFNet Integration


  - [x] 6.1 Integrate NAFNet-Width64 model


    - Create `src/crop_first_nafnet.py` loading from `NAFNet-GoPro-width64.pth`
    - Implement crop-first strategy processing only identification_plate detections
    - Add 10% padding extraction for ROI processing
    - Optimize for 85% computation reduction vs full-frame processing
    - Add dependencies: `basicsr`, `einops` for NAFNet support
    - _Requirements: 6.1, 6.2, 6.5, 14.4_

  - [x] 6.2 Write property test for crop-first processing











    - **Property 8: Crop-First Processing Logic**
    - **Validates: Requirements 6.1**

  - [x] 6.3 Write property test for crop padding





    - **Property 9: Crop Padding Correctness**
    - **Validates: Requirements 6.2**

- [x] 7. Spectral Processing Module





  - [x] 7.1 Implement spectral channel extraction


    - Create `src/spectral_processor.py` for red and saturation channel extraction
    - Implement red channel extraction for OCR optimization
    - Implement saturation channel extraction for damage detection
    - Add 40% efficiency gain validation vs full RGB processing
    - _Requirements: 10.1, 10.2, 10.3_

  - [x] 7.2 Write property test for spectral channel extraction


    - **Property 10: Spectral Channel Extraction**
    - **Validates: Requirements 10.1, 10.2**

- [x] 8. SmolVLM and SigLIP Integration





  - [x] 8.1 Integrate existing SmolVLM forensic agent


    - Create wrapper `src/smolvlm_integration.py` around existing `src/agent_forensic.py`
    - Configure specific prompts for OCR and damage assessment
    - Implement 8-bit quantization for Jetson compatibility
    - Add timeout handling and async processing
    - _Requirements: 7.1, 7.5, 14.2_

  - [x] 8.2 Integrate existing SigLIP semantic search


    - Create wrapper `src/siglip_integration.py` around existing `src/semantic_search.py`
    - Implement background embedding generation for inspection crops
    - Add natural language query processing interface
    - Configure LanceDB vector storage integration
    - _Requirements: 9.1, 9.2, 14.3_

- [x] 9. Checkpoint - Core Engine Complete





  - Ensure all core components load and integrate properly
  - Verify model status reporting works correctly
  - Test basic processing pipeline end-to-end

- [x] 10. Mission Control Tab Implementation





  - [x] 10.1 Create live processing interface


    - Implement `src/mission_control.py` with video input selection (webcam/RTSP/upload)
    - Add real-time video display with oriented bounding box overlays
    - Create metric cards for latest serial number, FPS, and processing stats
    - Implement different colored overlays for each YOLO model type
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 10.2 Write property test for video input acceptance


    - **Property 11: Video Input Acceptance**
    - **Validates: Requirements 2.1**

  - [x] 10.3 Write property test for real-time overlay display


    - **Property 12: Real-time Overlay Display**
    - **Validates: Requirements 2.2**

- [x] 11. Restoration Lab Tab Implementation






  - [x] 11.1 Create interactive restoration interface

    - Implement `src/restoration_lab.py` with file upload for JPG/PNG images
    - Add streamlit-image-comparison for before/after visualization
    - Create slider for comparison mix adjustment
    - Display processing time and quality metrics
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [x] 12. Semantic Search Tab Implementation





  - [x] 12.1 Create natural language search interface


    - Implement `src/semantic_search_ui.py` with text input for queries
    - Add gallery display for search results with similarity scores
    - Include metadata display: wagon ID, timestamp, confidence scores
    - Add filtering options for date range and confidence thresholds
    - _Requirements: 9.1, 9.3, 9.4, 9.5_

  - [x] 12.2 Write property test for natural language query processing


    - **Property 13: Natural Language Query Processing**
    - **Validates: Requirements 9.1, 9.2**

- [x] 13. Dashboard Integration and Styling





  - [x] 13.1 Create main Streamlit application


    - Implement `app.py` with dark industrial theme
    - Create 3-tab layout: Mission Control, Restoration Lab, Semantic Search
    - Add performance monitoring dashboard with FPS, latency, queue depth
    - Implement model status indicators and "Model Offline" badges
    - _Requirements: 12.1, 12.2, 12.4, 11.1, 11.2_

  - [x] 13.2 Write property test for model status reporting


    - **Property 14: Model Status Reporting**
    - **Validates: Requirements 1.4**

- [x] 14. Performance Monitoring and Error Handling




  - [x] 14.1 Implement performance tracking system

    - Create `src/performance_monitor.py` with latency budgets and violation logging
    - Add GPU utilization and temperature monitoring
    - Implement graceful degradation when models exceed latency budgets
    - Create incident logging for post-mortem analysis
    - _Requirements: 11.1, 11.3, 11.4_


  - [ ] 14.2 Implement error handling and fallbacks















    - Add model loading failure handling with mock model fallbacks
    - Implement timeout handling for all AI processing stages
    - Create user-friendly error messages and recovery suggestions
    - Add automatic retry logic for transient failures
    - _Requirements: 1.4, 11.5_

- [-] 15. Model Export and Optimization




  - [x] 15.1 Enable FP16 acceleration for models


    - Load .pt models directly using YOLO('model.pt') instead of ONNX export
    - Enable FP16 acceleration via model(source, device='cuda', half=True)
    - Support dynamic input shapes for varying crop sizes
    - Test all models for performance and accuracy with FP16
    - _Requirements: 13.1, 13.2, 13.4_

  - [x] 15.2 Write property test for FP16 model loading











    - **Property 15: FP16 Model Loading**
    - **Validates: Requirements 13.1**

- [x] 16. Integration Testing and Demo Preparation







  - [x] 16.1 Create comprehensive integration tests


    - Test all 3 dashboard tabs with real and mock data
    - Verify end-to-end processing pipeline performance
    - Test graceful degradation when models are offline
    - Validate memory usage stays within Jetson constraints
    - _Requirements: All requirements integration_

  - [x] 16.2 Prepare demo data and scenarios


    - Create sample wagon images for restoration lab testing
    - Generate mock inspection history for semantic search demo
    - Prepare video samples for mission control demonstration
    - Create performance benchmark scenarios
    - _Requirements: Demo preparation_

- [x] 17. Documentation and Deployment










  - [x] 17.1 Create user documentation


    - Write README with installation and usage instructions
    - Create user guide for each dashboard tab
    - Document model requirements and optional components
    - Add troubleshooting guide for common issues
    - _Requirements: Documentation_

  - [x] 17.2 Prepare deployment configuration


    - Create Docker configuration for containerized deployment
    - Add environment variable configuration for model paths
    - Create startup scripts and health checks
    - Document hardware requirements and performance expectations
    - _Requirements: Deployment readiness_

- [x] 18. Final Checkpoint and Validation





  - Ensure all tests pass and system runs end-to-end
  - Verify performance meets targets on available hardware
  - Test all dashboard functionality with real data
  - Validate graceful handling of missing models
  - Confirm system ready for demonstration

## Notes

- **Existing Asset Leverage**: Maximum reuse of existing codebase including `pipeline_core.py`, `agent_forensic.py`, `semantic_search.py`, `preprocessor_sci.py`, and NAFNet-GoPro-width64.pth model
- **SCI over Zero-DCE**: Use existing SCI preprocessor for 6x performance improvement (~0.5ms vs ~3ms)
- **Graceful Degradation**: System works with missing models by showing "Model Offline" badges and using mock implementations
- **Performance Focus**: Target 60 FPS real-time processing with comprehensive latency monitoring
- **Property-Based Testing**: Use Hypothesis library with minimum 100 iterations per property test
- **Mock Model Strategy**: Create functional mock implementations for missing YOLO models to enable full demo
- **Memory Optimization**: 8-bit quantization for SmolVLM, FP16 for ONNX models, GPU memory fraction limiting
- **Dark Industrial Theme**: Professional styling appropriate for railway inspection context
- **Three-Tier Architecture**: Real-time ONNX models, async SmolVLM processing, background SigLIP indexing
- **NAFNet Integration**: Use pre-trained NAFNet-GoPro-width64.pth model with basicsr and einops dependencies
- **Natural Language Search**: Enable queries like "wagons with rust damage" using SigLIP embeddings
- **Interactive Testing**: Restoration Lab allows upload and before/after comparison of any image
- **Real-time Monitoring**: Mission Control shows live processing with performance metrics and model status
- **Integration Priority**: Focus on connecting existing proven components rather than training new models
- **Demo Readiness**: System designed to work immediately with existing assets and gracefully handle missing components

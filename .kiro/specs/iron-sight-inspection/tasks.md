# Implementation Plan: Iron-Sight Railway Inspection System

## Overview

This implementation plan breaks down the Iron-Sight system into incremental coding tasks. Each task builds on previous work, ensuring no orphaned code. The system will be implemented in Python using ONNX Runtime for model inference, with TimescaleDB for logging and Streamlit for the dashboard.

## Tasks

- [x] 1. Project Setup and Core Infrastructure
  - [x] 1.1 Initialize project with uv and create directory structure
    - Create `src/`, `models/`, `config/`, `tests/`, `data/`, `scripts/` directories
    - Initialize `pyproject.toml` with dependencies: opencv-python-headless, torch, torchvision, ultralytics, paddleocr, streamlit, psycopg2-binary, onnxruntime-gpu, hypothesis, roboflow
    - _Requirements: 12.2_

  - [x] 1.2 Create configuration data models
    - Implement `PipelineConfig`, `Detection`, `OCRResult`, `InspectionResult`, `SpectralChannels`, `LatencyMetrics` dataclasses in `src/models.py`
    - _Requirements: 8.2, 9.3_

  - [x] 1.3 Write property test for InspectionResult field completeness
    - **Property 17: Inspection Logging Completeness**
    - **Validates: Requirements 8.1, 8.2**

- [x] 2. Data Pipeline and Dataset Preparation
  - [x] 2.1 Implement Roboflow dataset download
    - Create `src/data_ingest.py` with `download_wagon_dataset()` function
    - Download from `vishakha-singh/wagon-detection-eh2ov` in YOLOv8-OBB format
    - Store in `data/wagon_detection/`
    - _Requirements: 5.2 (training data for wagon_body, wheel_assembly, coupling_mechanism, identification_plate)_

  - [x] 2.2 Implement physics-based augmentation and combined dataset preparation
    - Create `src/data_physics.py` with `simulate_railway_conditions()` function
    - Apply gamma correction (γ = 0.3-0.5) for darkness simulation
    - Apply Poisson noise for sensor grain
    - Apply horizontal motion blur (15px kernel)
    - Generate 2000+ augmented images from Roboflow wagon dataset in `data/dataset_train/`
    - Create `src/data_local.py` to prepare local blurred/sharp car dataset
    - Copy and organize `blurred_sharp/blurred_sharp/blurred/` and `blurred_sharp/blurred_sharp/sharp/` for training
    - Create `src/data_combine.py` to merge Roboflow wagon data with local car data
    - Map wagon classes to vehicle classes: `wagon_body`→`vehicle_body`, `identification_plate`→`license_plate`
    - _Requirements: 4.1, 6.1 (training data matching real-world conditions)_

  - [x] 2.3 Create combined dataset configuration for vehicle detection training
    - Create `config/vehicle_detection.yaml` with unified class definitions and data paths
    - Define unified classes: `vehicle_body` (cars + wagons), `license_plate` (cars + wagon IDs), `wheel`, `coupling_mechanism`
    - Point to combined dataset paths: Roboflow wagon data + local car data + augmented images
    - Create train/val/test splits ensuring both car and wagon representation
    - _Requirements: 5.2_

- [x] 3. Model Training
  - [x] 3.1 Train Gatekeeper binary classifier on combined dataset
    - Create `scripts/train_gatekeeper.py`
    - Use MobileNetV3-Small architecture
    - Train for joint prediction of `[is_vehicle_present, is_blurry]` using both car and wagon images
    - Input: 64x64 grayscale thumbnails from both datasets
    - Positive class: Vehicle-present frames (cars from local dataset + wagons from Roboflow)
    - Negative class: Empty backgrounds, partial vehicles, extreme blur
    - Target: 95% accuracy @ <0.5ms inference on combined vehicle types
    - Export to ONNX with FP16
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 10.1, 10.2_

  - [x] 3.2 Fine-tune YOLOv8-OBB on combined vehicle dataset
    - Create `scripts/train_yolo.py`
    - Fine-tune `yolov8n-obb.pt` on combined dataset (Roboflow wagons + local cars)
    - Use unified classes: `vehicle_body`, `license_plate`, `wheel`, `coupling_mechanism`
    - Apply physics-based augmentation to both datasets for consistency
    - Train for 100 epochs at 640x640 resolution
    - Target: mAP@50 >= 0.85 on combined vehicle detection (cars + wagons)
    - Export to ONNX with FP16 and dynamic shapes
    - _Requirements: 5.1, 5.2, 5.3, 5.7, 10.1, 10.2_

  - [x] 3.3 Prepare Zero-DCE++ model
    - Create `scripts/prepare_zero_dce.py`
    - Download or convert pre-trained Zero-DCE++ weights
    - Export to ONNX with FP16
    - Verify inference time <15ms
    - _Requirements: 4.2, 10.1, 10.2_

  - [x] 3.4 Train DeblurGAN-v2 model on combined blurred dataset
    - Create `scripts/train_deblur.py`
    - Use local paired dataset: `blurred_sharp/blurred_sharp/blurred/` (input) and `blurred_sharp/blurred_sharp/sharp/` (target)
    - Augment with artificially blurred Roboflow wagon images (apply motion blur to sharp wagon images)
    - Train DeblurGAN-v2 with MobileNet-DSC backbone on combined dataset (1151 car pairs + augmented wagon pairs)
    - Focus on vehicle images for license plate and identification number deblurring
    - Export trained model to ONNX with FP16 and dynamic input shapes
    - Verify inference time <40ms on crops
    - _Requirements: 6.3, 10.1, 10.2, 10.3_

- [x] 4. Checkpoint - Data and Models Ready
  - Ensure dataset downloaded and augmented
  - Ensure all models exported to ONNX in `models/` directory
  - Verify model files: gatekeeper.onnx, yolov8n_obb.onnx, zero_dce.onnx, deblur_gan.onnx

- [x] 5. SmolVLM 2 Forensic Agent Integration
  - [x] 5.1 Add SmolVLM 2 dependencies
    - Add `transformers` and `accelerate` packages via `uv add transformers accelerate`
    - Configure for Jetson compatibility (skip flash_attn)
    - _Requirements: Enhanced OCR capabilities for damaged/rusted surfaces_

  - [x] 5.2 Implement SmolVLM forensic agent
    - Create `src/agent_forensic.py` with `SmolVLMForensicAgent` class
    - Use HuggingFaceTB/SmolVLM2-256M-Video-Instruct (256M fits Jetson RAM)
    - Implement 8-bit quantization for memory efficiency
    - Support both OCR fallback and damage assessment tasks
    - Run in separate thread to avoid blocking main pipeline
    - _Requirements: OCR fallback when PaddleOCR confidence < 0.50_

  - [x] 5.3 Integrate fallback OCR logic in pipeline
    - Update `src/pipeline_core.py` with fallback trigger logic
    - When PaddleOCR confidence < 0.50, queue SmolVLM analysis
    - Display "Analyzing..." placeholder during SmolVLM processing
    - Maintain 60 FPS by async processing in forensic queue
    - _Requirements: Real-time performance with intelligent fallback_

  - [x] 5.4 Implement damage assessment feature
    - When YOLO detects `class="damage_door"`, trigger SmolVLM analysis
    - Use prompt: "Describe the damage severity: is it a dent, a hole, or rust?"
    - Log damage descriptions to dashboard
    - _Requirements: AI-powered damage assessment for detected defects_

  - [x] 5.5 Update data models for forensic integration
    - Add `fallback_used` and `forensic_processing_time_ms` to `OCRResult`
    - Add `fallback_ocr_used` and `damage_assessment` to `InspectionResult`
    - Update database schema to track forensic agent usage
    - _Requirements: Comprehensive logging of AI processing pipeline_

- [x] 6. Semantic Search Engine Integration
  - [x] 6.1 Add semantic search dependencies
    - Add `open_clip_torch` and `lancedb` packages via `uv add open_clip_torch lancedb`
    - Configure SigLIP 2 model for embedding generation
    - _Requirements: Natural language search of inspection history_

  - [x] 6.2 Implement semantic search engine
    - Create `src/semantic_search.py` with `SemanticSearchEngine` class
    - Use google/siglip2-base-patch16-224 (86M parameters, fast inference)
    - Implement LanceDB vector storage with schema `WagonEmbedding`
    - Support background embedding generation in lowest priority thread
    - _Requirements: Efficient vector similarity search with metadata_

  - [x] 6.3 Integrate crop storage and indexing
    - Update `src/pipeline_core.py` to save wagon crops automatically
    - Generate embeddings for every saved wagon crop in background
    - Store [vector, timestamp, image_path, inspection_data] in LanceDB
    - Maintain performance by async processing (no blocking main loop)
    - _Requirements: Automatic indexing of all inspected wagons_

  - [x] 6.4 Implement natural language search API
    - Add `search_wagon_history()` method to pipeline core
    - Support queries like "wagons with rust damage", "inspected last week"
    - Include time filters and confidence thresholds
    - Return results with similarity scores and inspection metadata
    - _Requirements: User-friendly search interface for historical data_

- [ ] 7. Streamlit Dashboard with Search UI
  - [ ] 7.1 Add search interface to dashboard
    - Create search box in Streamlit UI for natural language queries
    - Display search results with images, similarity scores, and metadata
    - Add filters for time range, confidence thresholds, wagon ID
    - Show search statistics and performance metrics
    - _Requirements: 11.1-11.8 + semantic search capabilities_

  - [ ] 7.2 Integrate forensic agent monitoring
    - Display SmolVLM fallback statistics (triggers, success rate, avg time)
    - Show damage assessment results in real-time log
    - Monitor forensic agent queue depth and processing status
    - _Requirements: Real-time monitoring of AI agent performance_

- [ ] 8. Spectral Decomposition Module
  - [ ] 8.1 Implement SpectralDecomposer class
    - Create `src/spectral.py` with `extract_red_channel()`, `extract_saturation()`, and `decompose()` methods
    - _Requirements: 3.1, 3.2_

  - [ ] 8.2 Write property test for spectral decomposition correctness
    - **Property 7: Spectral Decomposition Correctness**
    - **Validates: Requirements 3.1, 3.2**

- [ ] 9. Wagon ID Validation
  - [ ] 9.1 Implement wagon ID validation function
    - Create regex-based validator in `src/validation.py` for pattern `[A-Z]{4}\d{6}`
    - _Requirements: 7.3_

  - [ ] 9.2 Write property test for wagon ID validation pattern
    - **Property 14: Wagon ID Validation Pattern**
    - **Validates: Requirements 7.3**

- [ ] 10. Checkpoint - Core Utilities
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Frame Queue Implementation
  - [ ] 11.1 Implement thread-safe frame queue with drop-oldest behavior
    - Create `src/frame_queue.py` with `FrameQueue` class
    - Implement `put_nowait()` that drops oldest frame when full
    - Set maxsize=2
    - _Requirements: 1.3, 1.4_

  - [ ] 11.2 Write property test for queue overflow behavior
    - **Property 3: Queue Overflow Behavior**
    - **Validates: Requirements 1.3**

- [ ] 12. Gatekeeper Model Wrapper
  - [ ] 12.1 Implement Gatekeeper class
    - Create `src/gatekeeper.py` with ONNX model loading
    - Implement `preprocess()` for 64x64 grayscale conversion
    - Implement `predict()` returning `(is_wagon_present, is_blurry)` tuple
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 12.2 Write property test for gatekeeper output format
    - **Property 5: Gatekeeper Output Format**
    - **Validates: Requirements 2.3**

- [ ] 13. Zero-DCE Enhancer
  - [ ] 13.1 Implement ZeroDCEEnhancer class
    - Create `src/enhancer.py` with ONNX model loading
    - Implement `enhance()` with timeout handling and fallback to raw frame
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 13.2 Write property test for enhancement latency and fallback
    - **Property 8: Enhancement Latency and Fallback**
    - **Validates: Requirements 4.2, 4.3, 13.1**

- [ ] 14. YOLO Detector
  - [ ] 14.1 Implement YOLODetector class
    - Create `src/detector.py` with ONNX model loading
    - Implement `detect()` returning list of `Detection` objects with OBB
    - Implement `postprocess()` with NMS (threshold 0.45) and confidence filter (0.35)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ] 14.2 Write property test for detection output format
    - **Property 9: Detection Output Format**
    - **Validates: Requirements 5.3, 5.5**

- [ ] 15. Crop Extraction Utility
  - [ ] 15.1 Implement ROI extraction with padding
    - Create `extract_rotated_crop()` function in `src/crop.py`
    - Apply 10% padding on all sides
    - Handle OBB rotation
    - _Requirements: 6.2_

  - [ ] 15.2 Write property test for crop padding
    - **Property 12: Crop Padding**
    - **Validates: Requirements 6.2**

- [ ] 16. Checkpoint - Model Wrappers
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. DeblurGAN Module
  - [ ] 17.1 Implement DeblurGAN class
    - Create `src/deblur.py` with ONNX model loading
    - Implement `deblur()` with timeout handling and fallback
    - Support dynamic input shapes
    - _Requirements: 6.3, 6.4, 6.5, 10.3_

  - [ ] 17.2 Write property test for deblur latency and fallback
    - **Property 13: Deblur Latency and Fallback**
    - **Validates: Requirements 6.3, 6.4, 13.2**

  - [ ] 17.3 Write property test for dynamic input shapes
    - **Property 21: Dynamic Input Shapes for DeblurGAN**
    - **Validates: Requirements 10.3**

- [ ] 18. OCR Engine
  - [ ] 18.1 Implement OCREngine class
    - Create `src/ocr.py` with PaddleOCR initialization
    - Implement `read()` returning `OCRResult`
    - Implement confidence filtering (reject < 0.80)
    - Integrate wagon ID validation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

  - [ ] 18.2 Write property test for OCR confidence rejection
    - **Property 15: OCR Confidence Rejection**
    - **Validates: Requirements 7.4**

- [ ] 19. Database Layer
  - [ ] 19.1 Implement TimescaleDBClient
    - Create `src/database.py` with connection management
    - Implement `log_inspection()` method
    - Implement `health_check()` method
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ] 19.2 Implement SQLite fallback client
    - Create `SQLiteClient` class in `src/database.py`
    - Mirror TimescaleDB interface
    - _Requirements: 13.4_

  - [ ] 19.3 Implement DatabaseManager with automatic fallback
    - Create `DatabaseManager` class that switches to SQLite when TimescaleDB unavailable
    - _Requirements: 13.4_

  - [ ] 19.4 Write property test for database fallback
    - **Property 22: Database Fallback to SQLite**
    - **Validates: Requirements 13.4**

- [ ] 20. Checkpoint - All Components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 21. Latency Tracking
  - [ ] 21.1 Implement latency tracking utilities
    - Create `src/latency.py` with `LatencyTracker` class
    - Implement context manager for timing each stage
    - Implement `LatencyMetrics` aggregation
    - _Requirements: 9.3_

  - [ ] 21.2 Write property test for latency tracking per stage
    - **Property 19: Latency Tracking Per Stage**
    - **Validates: Requirements 9.3**

- [ ] 22. Error Handling and Incident Logging
  - [ ] 22.1 Implement exception classes and handlers
    - Create `src/exceptions.py` with `PipelineError`, `ModelTimeoutError`, `DatabaseConnectionError`
    - Implement `handle_model_timeout()` with incident logging
    - _Requirements: 13.5_

  - [ ] 22.2 Write property test for incident logging on latency violations
    - **Property 23: Incident Logging for Latency Violations**
    - **Validates: Requirements 13.5**

- [ ] 23. Inspection Pipeline Core
  - [ ] 23.1 Implement InspectionPipeline class
    - Create `src/pipeline.py` with model loading and initialization
    - Implement `capture_loop()` for Stream A (visualization)
    - Implement `process_loop()` for Stream B (inspection)
    - Wire all components together
    - _Requirements: 1.1, 1.2, 2.4, 9.1, 9.2, 9.4_

  - [ ] 23.2 Write property test for gatekeeper skip behavior
    - **Property 6: Gatekeeper Skip Behavior**
    - **Validates: Requirements 2.4**

  - [ ] 23.3 Write property test for conditional deblur invocation
    - **Property 11: Conditional Deblur Invocation**
    - **Validates: Requirements 6.1, 6.5**

  - [ ] 23.4 Write property test for graceful degradation
    - **Property 20: Graceful Degradation**
    - **Validates: Requirements 9.4**

- [ ] 24. Checkpoint - Pipeline Integration
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 25. Main Entry Point
  - [ ] 25.1 Create main application entry point
    - Create `src/main.py` with CLI argument parsing
    - Initialize pipeline with configuration
    - Start capture and processing threads
    - Launch dashboard
    - _Requirements: 1.1_

- [ ] 26. Docker Configuration
  - [ ] 26.1 Create Dockerfile
    - Use `nvidia/cuda:12.1.0-runtime-ubuntu22.04` base image
    - Install uv package manager
    - Copy application code and models
    - Add health check for ONNX Runtime GPU
    - _Requirements: 12.1, 12.2, 12.4, 12.5_

  - [ ] 26.2 Create docker-compose.yml
    - Configure inspection service with GPU support
    - Configure TimescaleDB service
    - Set memory limit to 16GB
    - _Requirements: 12.3_

- [ ] 27. Database Schema
  - [ ] 27.1 Create TimescaleDB schema migration
    - Create `scripts/init_db.sql` with inspection_log table
    - Create hypertable for time-series optimization
    - Create indexes on wagon_id, detection_confidence, processing_time_ms
    - Add 90-day retention policy
    - _Requirements: 8.3, 8.4_

- [ ] 28. Final Checkpoint
  - Ensure all tests pass, ask the user if questions arise.
  - Verify Docker build succeeds
  - Verify end-to-end pipeline with sample video

## Notes

- All tasks are required for comprehensive testing from the start
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation uses Python with Hypothesis for property-based testing
- Data pipeline tasks (2.1-2.3) require Roboflow API key set as `ROBOFLOW_KEY` environment variable
- Model training tasks (3.1-3.4) require GPU with CUDA support
- Training outputs should be placed in `models/` directory as ONNX files
- **Combined dataset approach**: Use both Roboflow wagon dataset (`vishakha-singh/wagon-detection-eh2ov`) AND local car dataset (`blurred_sharp/blurred_sharp/`) for comprehensive vehicle detection and deblurring
- **Dataset mapping**: Wagon classes mapped to vehicle classes for unified training (wagon_body→vehicle_body, identification_plate→license_plate)
- **Hackathon focus**: Primary goal is demonstrating robust vehicle detection, deblurring, and OCR across both cars and railway wagons
- **Data augmentation**: Apply consistent physics-based augmentation (blur, darkness, noise) to both datasets for training robustness
- **SmolVLM 2 Integration**: Added HuggingFaceTB/SmolVLM2-256M-Video-Instruct as fallback OCR agent for damaged/rusted surfaces when PaddleOCR confidence < 0.50
- **Semantic Search**: Integrated SigLIP 2 (google/siglip2-base-patch16-224) with LanceDB for natural language search of inspection history
- **AI Agent Architecture**: Three-tier AI system: (1) Fast ONNX models for real-time processing, (2) SmolVLM for intelligent fallback, (3) SigLIP for semantic indexing
- **Performance Optimization**: All AI agents run in separate threads/processes to maintain 60 FPS main pipeline performance
- **Memory Management**: 8-bit quantization for SmolVLM on Jetson, background embedding generation for semantic search
- **Forensic Capabilities**: Damage assessment using SmolVLM when YOLO detects damage classes, with natural language damage descriptions

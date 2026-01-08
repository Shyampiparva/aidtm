# Requirements Document

## Introduction

Iron-Sight is a high-velocity railway inspection system designed to detect and identify wagons moving at 50-80 km/h in low-light environments. The system uses a conditional restoration pipeline with spectral decomposition to efficiently process video frames, detect wagons, read identification plates via OCR, and log inspection data. The architecture employs a two-stream approach: a fast visualization lane and an AI-powered inspection lane with gatekeeper filtering.

## Glossary

- **Inspection_Pipeline**: The main processing system that orchestrates frame capture, AI inference, and result logging
- **Gatekeeper**: A lightweight MobileNetV3-Small binary classifier that determines if a frame contains a wagon and if it's blurry
- **Zero_DCE_Enhancer**: A Zero-Reference Deep Curve Estimation model that enhances low-light images
- **YOLO_Detector**: YOLOv8-OBB nano model for detecting wagon components with oriented bounding boxes
- **DeblurGAN**: DeblurGAN-v2 with MobileNet backbone for correcting horizontal motion blur
- **OCR_Engine**: PaddleOCR Mobile v2.6 for reading wagon identification text
- **Frame_Queue**: A thread-safe queue managing frames between capture and processing threads
- **Spectral_Decomposition**: The process of extracting red channel (for OCR) and saturation channel (for damage detection) from frames
- **TimescaleDB**: PostgreSQL extension for time-series inspection data storage
- **Wagon_ID**: Alphanumeric identifier in format `[A-Z]{4}\d{6}` (4 letters followed by 6 digits)

## Requirements

### Requirement 1: Two-Stream Frame Processing

**User Story:** As a system operator, I want the system to process video in two parallel streams, so that I can view real-time footage while AI inspection runs without blocking.

#### Acceptance Criteria

1. THE Inspection_Pipeline SHALL maintain two concurrent processing streams: a visualization stream and an inspection stream
2. WHEN a frame is captured, THE Inspection_Pipeline SHALL render it to the dashboard within 5ms for the visualization stream
3. WHEN the Frame_Queue is full, THE Inspection_Pipeline SHALL drop the oldest frame and accept the newest frame
4. THE Frame_Queue SHALL have a maximum capacity of 2 frames to ensure latest-frame-first processing

### Requirement 2: Gatekeeper Pre-filtering

**User Story:** As a system architect, I want frames to be pre-filtered before expensive processing, so that computation is not wasted on empty or unusable frames.

#### Acceptance Criteria

1. WHEN a frame enters the inspection stream, THE Gatekeeper SHALL classify it within 0.5ms
2. THE Gatekeeper SHALL accept a 64x64 grayscale thumbnail as input
3. THE Gatekeeper SHALL output two boolean predictions: `is_wagon_present` and `is_blurry`
4. WHEN the Gatekeeper predicts `is_wagon_present` as false, THE Inspection_Pipeline SHALL skip all subsequent processing for that frame
5. THE Gatekeeper SHALL achieve at least 95% accuracy on the validation dataset

### Requirement 3: Spectral Decomposition

**User Story:** As a computer vision engineer, I want frames decomposed into spectral channels, so that OCR and damage detection can use optimized inputs.

#### Acceptance Criteria

1. WHEN a frame passes the Gatekeeper, THE Inspection_Pipeline SHALL extract the red channel from the BGR image
2. WHEN a frame passes the Gatekeeper, THE Inspection_Pipeline SHALL convert the frame to HSV and extract the saturation channel
3. THE Inspection_Pipeline SHALL use the red channel for OCR processing
4. THE Inspection_Pipeline SHALL use the saturation channel for damage detection processing

### Requirement 4: Low-Light Enhancement

**User Story:** As a system operator, I want dark frames enhanced before detection, so that wagons can be detected in low-light conditions.

#### Acceptance Criteria

1. WHEN a frame passes the Gatekeeper, THE Zero_DCE_Enhancer SHALL process the red channel
2. THE Zero_DCE_Enhancer SHALL complete processing within 8ms (target) and no more than 15ms (maximum)
3. IF the Zero_DCE_Enhancer exceeds 15ms, THEN THE Inspection_Pipeline SHALL use the raw frame instead
4. THE Zero_DCE_Enhancer SHALL use FP16 precision for inference

### Requirement 5: Wagon Detection

**User Story:** As an inspection operator, I want wagons and their components detected in each frame, so that I can track and inspect them.

#### Acceptance Criteria

1. WHEN an enhanced frame is available, THE YOLO_Detector SHALL detect wagon components
2. THE YOLO_Detector SHALL detect the following classes: `wagon_body`, `wheel_assembly`, `coupling_mechanism`, `identification_plate`
3. THE YOLO_Detector SHALL use oriented bounding boxes (OBB) to handle angled wagons
4. THE YOLO_Detector SHALL complete detection within 15ms (target) and no more than 25ms (maximum)
5. THE YOLO_Detector SHALL use a confidence threshold of 0.35
6. THE YOLO_Detector SHALL use an NMS threshold of 0.45
7. THE YOLO_Detector SHALL achieve at least 95% recall on wagons moving at 50-80 km/h

### Requirement 6: Conditional Motion Deblurring

**User Story:** As a system architect, I want motion blur corrected only when necessary, so that computation is saved on clear frames.

#### Acceptance Criteria

1. WHEN the Gatekeeper indicates `is_blurry` is true AND detection confidence is below 0.6, THE DeblurGAN SHALL process the cropped region
2. THE Inspection_Pipeline SHALL extract the detected region with 10% padding before deblurring
3. THE DeblurGAN SHALL complete processing within 20ms (target) and no more than 40ms (maximum)
4. IF the DeblurGAN exceeds 50ms, THEN THE Inspection_Pipeline SHALL skip deblurring and log degraded quality
5. THE DeblurGAN SHALL only process cropped regions, not full frames

### Requirement 7: OCR for Wagon Identification

**User Story:** As an inspection operator, I want wagon identification numbers read automatically, so that I can track individual wagons.

#### Acceptance Criteria

1. WHEN a wagon region is detected, THE OCR_Engine SHALL process the red channel crop
2. THE OCR_Engine SHALL complete processing within 25ms (target) and no more than 50ms (maximum)
3. THE OCR_Engine SHALL validate wagon IDs against the pattern `[A-Z]{4}\d{6}`
4. WHEN OCR confidence is below 0.80, THE OCR_Engine SHALL reject the reading
5. THE OCR_Engine SHALL achieve at least 92% character-level accuracy
6. THE OCR_Engine SHALL use PaddleOCR, not Tesseract, Docling, or Tika

### Requirement 8: Inspection Data Logging

**User Story:** As a data analyst, I want all inspection results logged to a time-series database, so that I can analyze inspection history.

#### Acceptance Criteria

1. WHEN a wagon is detected, THE Inspection_Pipeline SHALL log the result to TimescaleDB
2. THE Inspection_Pipeline SHALL log the following fields: frame_id, timestamp, wagon_id, detection_confidence, ocr_confidence, blur_score, enhancement_applied, deblur_applied, processing_time_ms, spectral_channel, bounding_box, wagon_angle
3. THE TimescaleDB SHALL retain inspection logs for 90 days
4. THE TimescaleDB SHALL create indexes on wagon_id, detection_confidence, and processing_time_ms

### Requirement 9: Pipeline Latency Management

**User Story:** As a system architect, I want the total pipeline latency managed within budget, so that real-time processing is maintained.

#### Acceptance Criteria

1. THE Inspection_Pipeline SHALL complete full processing within 70ms (target) and no more than 100ms (maximum) per frame
2. WHEN total pipeline latency exceeds 100ms, THE Inspection_Pipeline SHALL drop the frame and log the incident
3. THE Inspection_Pipeline SHALL track and report latency for each processing stage
4. THE Inspection_Pipeline SHALL implement graceful degradation by skipping expensive operations when latency budget is exceeded

### Requirement 10: Model Format and Optimization

**User Story:** As a deployment engineer, I want all models in ONNX format with FP16 precision, so that they can be optimized for edge deployment.

#### Acceptance Criteria

1. THE Inspection_Pipeline SHALL load all AI models in ONNX format
2. THE Inspection_Pipeline SHALL use FP16 precision for all model inference
3. THE Inspection_Pipeline SHALL support dynamic input shapes for the DeblurGAN model
4. THE Inspection_Pipeline SHALL use ONNX Runtime with TensorRT optimization path

### Requirement 11: Dashboard Visualization

**User Story:** As a system operator, I want a real-time dashboard showing inspection metrics, so that I can monitor system health.

#### Acceptance Criteria

1. THE Dashboard SHALL display frames processed per second (FPS)
2. THE Dashboard SHALL display current latency per pipeline stage
3. THE Dashboard SHALL display queue depth (frame buffer status)
4. THE Dashboard SHALL display model inference times as a histogram
5. THE Dashboard SHALL display detection confidence distribution
6. THE Dashboard SHALL display OCR success rate for the last 100 frames
7. THE Dashboard SHALL display GPU utilization and temperature
8. THE Dashboard SHALL be implemented using Streamlit

### Requirement 12: Docker Containerization

**User Story:** As a deployment engineer, I want the system containerized with GPU support, so that it can be deployed consistently across environments.

#### Acceptance Criteria

1. THE Docker_Container SHALL use `nvidia/cuda:12.1.0-runtime-ubuntu22.04` as the base image
2. THE Docker_Container SHALL use uv as the package manager
3. THE Docker_Container SHALL limit memory to 16GB to simulate Jetson constraints
4. THE Docker_Container SHALL include a health check that verifies ONNX Runtime GPU availability
5. THE Docker_Container SHALL verify all required model files exist at build time

### Requirement 13: Error Handling and Fallbacks

**User Story:** As a system operator, I want the system to handle errors gracefully, so that inspection continues even when components fail.

#### Acceptance Criteria

1. IF the Zero_DCE_Enhancer fails, THEN THE Inspection_Pipeline SHALL continue with the raw frame
2. IF the DeblurGAN fails, THEN THE Inspection_Pipeline SHALL continue with the blurry crop
3. IF the OCR_Engine fails, THEN THE Inspection_Pipeline SHALL log the detection without wagon_id
4. IF TimescaleDB is unavailable, THEN THE Inspection_Pipeline SHALL fall back to SQLite logging
5. WHEN any model exceeds its maximum latency by more than 20%, THE Inspection_Pipeline SHALL log an incident for post-mortem analysis

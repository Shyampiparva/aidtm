# Requirements Document

## Introduction

IronSight Command Center is a production-grade, real-time rail inspection dashboard that integrates existing YOLO models, a restoration engine (NAFNet-Width64), and Vision-Language Models (SmolVLM2, SigLIP 2) into a unified Streamlit interface. The system moves from experimentation to integration, connecting pre-trained models into a cohesive inspection workflow with three main interfaces: Mission Control for live processing, Restoration Lab for image enhancement testing, and Semantic Search for historical analysis.

## Glossary

- **IronSight_Engine**: The backend processing system that manages 5 neural networks simultaneously and coordinates frame processing
- **Mission_Control**: Live processing tab showing real-time video feed with AI detection overlays and metrics
- **Restoration_Lab**: Interactive tab for testing image restoration with before/after comparisons
- **Semantic_Search**: Natural language search interface for querying historical inspection data
- **Gatekeeper**: MobileNetV3-Small binary classifier predicting `[is_wagon_present, is_blurry]`
- **SCI_Enhancer**: Self-Calibrated Illumination model for fast low-light enhancement (~0.5ms vs Zero-DCE ~3ms)
- **NAFNet**: Pre-trained NAFNet-Width64 GoPro deblurring model using crop-first strategy for efficiency
- **YOLO_Detector**: YOLOv8-OBB model detecting wagon components with oriented bounding boxes
- **SmolVLM_Agent**: HuggingFaceTB/SmolVLM2-256M-Video-Instruct for OCR fallback and damage assessment
- **SigLIP_Engine**: Google SigLIP 2 model for generating embeddings and semantic search capabilities
- **Spectral_Processing**: Red channel extraction for OCR and saturation channel for damage detection
- **Crop_First_Strategy**: Processing only detected regions instead of full frames for 85% computation reduction

## Requirements

### Requirement 1: Multi-Model Engine Integration

**User Story:** As a system architect, I want all 5 neural networks loaded and managed efficiently, so that the system can perform comprehensive inspection without memory issues.

#### Acceptance Criteria

1. THE IronSight_Engine SHALL load and manage 5 neural networks simultaneously: Gatekeeper, SCI_Enhancer, YOLO_Detector, NAFNet, and SmolVLM_Agent
2. THE IronSight_Engine SHALL optimize GPU VRAM usage through FP16 quantization and model sharing
3. THE IronSight_Engine SHALL use 8-bit quantization for SmolVLM to fit within Jetson memory constraints
4. WHEN any model fails to load, THE IronSight_Engine SHALL display a "Model Offline" badge in the dashboard
5. THE IronSight_Engine SHALL verify CUDA availability and attempt CPU fallback with latency warnings

### Requirement 2: Mission Control Live Processing

**User Story:** As an operator, I want a live mission control interface, so that I can monitor real-time inspection with visual feedback and performance metrics.

#### Acceptance Criteria

1. THE Mission_Control SHALL accept input from webcam, RTSP stream, or uploaded video file
2. THE Mission_Control SHALL display live video feed with oriented bounding box overlays from all 3 YOLO models
3. THE Mission_Control SHALL show "Latest Serial Number" from NAFNet + SmolVLM processing in a large metric card
4. THE Mission_Control SHALL display real-time metadata including FPS, processing latency, and queue depth
5. THE Mission_Control SHALL maintain visual updates even when AI processing is slower than video framerate

### Requirement 3: Gatekeeper Pre-filtering with Dual Prediction

**User Story:** As a performance engineer, I want frames pre-filtered efficiently, so that expensive processing is only applied to relevant frames.

#### Acceptance Criteria

1. THE Gatekeeper SHALL process 64x64 grayscale thumbnails within 0.5ms
2. THE Gatekeeper SHALL output joint predictions: `[is_wagon_present, is_blurry]` as boolean values
3. WHEN `is_wagon_present` is false, THE IronSight_Engine SHALL skip all subsequent AI processing
4. THE Gatekeeper SHALL achieve at least 95% accuracy on the validation dataset
5. THE Mission_Control SHALL display gatekeeper skip statistics in real-time

### Requirement 4: SCI Low-Light Enhancement

**User Story:** As a computer vision engineer, I want fast low-light enhancement, so that dark frames are processed efficiently without blocking the pipeline.

#### Acceptance Criteria

1. THE SCI_Enhancer SHALL process single-channel images within 0.5ms target time
2. THE SCI_Enhancer SHALL automatically skip enhancement for bright images (mean brightness > 50) for daytime optimization
3. THE SCI_Enhancer SHALL use Self-Calibrated Illumination with physics-based curve learning
4. THE SCI_Enhancer SHALL export to ONNX with FP16 precision for deployment
5. IF SCI_Enhancer processing exceeds 1ms, THE system SHALL log performance degradation

### Requirement 5: Multi-Model YOLO Detection

**User Story:** As an inspection operator, I want comprehensive wagon detection, so that all components are identified and tracked.

#### Acceptance Criteria

1. THE YOLO_Detector SHALL run 3 specialized models: sideview damage OBB, structure OBB, and wagon number OBB
2. THE YOLO_Detector SHALL merge detections from all 3 models into a single JSON result
3. THE YOLO_Detector SHALL use oriented bounding boxes to handle angled wagons in curved sections
4. THE YOLO_Detector SHALL complete all 3 detections within 20ms combined latency budget
5. THE Mission_Control SHALL display different colored overlays for each detection type

### Requirement 6: Crop-First NAFNet Strategy

**User Story:** As a performance architect, I want motion blur corrected efficiently, so that only relevant regions are processed.

#### Acceptance Criteria

1. THE NAFNet SHALL only process crops of identification_plate detections from YOLO
2. THE NAFNet SHALL extract ROI with 10% padding before deblurring
3. THE NAFNet SHALL achieve 85% computation reduction compared to full-frame processing
4. THE NAFNet SHALL use the pre-trained model from `NAFNet-GoPro-width64.pth`
5. THE NAFNet SHALL complete crop processing within 20ms per region

### Requirement 7: SmolVLM OCR and Damage Assessment

**User Story:** As an AI engineer, I want intelligent OCR fallback and damage analysis, so that difficult cases are handled with superior text recognition.

#### Acceptance Criteria

1. THE SmolVLM_Agent SHALL process deblurred plate crops with prompt "Read the serial number painted on this metal surface. Return only the alphanumeric string."
2. THE SmolVLM_Agent SHALL generate embeddings for damage crops using SigLIP 2 and store in local vector list
3. THE SmolVLM_Agent SHALL run in separate thread to avoid blocking main processing pipeline
4. THE SmolVLM_Agent SHALL use HuggingFaceTB/SmolVLM2-256M-Video-Instruct with 8-bit quantization
5. THE SmolVLM_Agent SHALL timeout after 30 seconds and return partial results

### Requirement 8: Restoration Lab Interactive Testing

**User Story:** As a quality engineer, I want to test image restoration interactively, so that I can validate enhancement quality on specific images.

#### Acceptance Criteria

1. THE Restoration_Lab SHALL provide file upload interface for blurry images (JPG, PNG formats)
2. THE Restoration_Lab SHALL display split-screen comparison using streamlit-image-comparison
3. THE Restoration_Lab SHALL show [Raw Blurry Crop] vs [NAFNet Restored Crop]
4. THE Restoration_Lab SHALL provide a slider to adjust the comparison mix ratio
5. THE Restoration_Lab SHALL display processing time and quality metrics for each restoration

### Requirement 9: Semantic Search with Natural Language

**User Story:** As a data analyst, I want to search inspection history using natural language, so that I can find relevant cases quickly.

#### Acceptance Criteria

1. THE Semantic_Search SHALL accept text queries like "Show me rusted doors" or "wagons with damage"
2. THE Semantic_Search SHALL compare query embeddings using SigLIP against stored crop embeddings
3. THE Semantic_Search SHALL display gallery of matching defects with similarity scores
4. THE Semantic_Search SHALL include metadata: wagon ID, timestamp, confidence scores, and damage descriptions
5. THE Semantic_Search SHALL support filtering by date range and confidence thresholds

### Requirement 10: Spectral Processing Optimization

**User Story:** As a computer vision engineer, I want optimized channel processing, so that OCR and damage detection use the most effective inputs.

#### Acceptance Criteria

1. THE IronSight_Engine SHALL extract red channel from BGR images for OCR processing
2. THE IronSight_Engine SHALL extract saturation channel from HSV conversion for damage detection
3. THE Spectral_Processing SHALL provide 40% efficiency gain over full RGB processing
4. THE OCR processing SHALL use red channel exclusively for maximum text contrast
5. THE damage detection SHALL use saturation channel to highlight rust and oxidation

### Requirement 11: Performance Monitoring and Error Handling

**User Story:** As a system operator, I want comprehensive performance monitoring, so that I can identify bottlenecks and system health issues.

#### Acceptance Criteria

1. THE Dashboard SHALL display FPS, latency per stage, queue depth, and model inference times
2. THE Dashboard SHALL show GPU utilization, temperature, and memory usage
3. THE Dashboard SHALL track SmolVLM fallback triggers and success rates
4. WHEN any model exceeds latency budget, THE system SHALL log incidents for analysis
5. THE Dashboard SHALL provide "Model Offline" indicators when components fail

### Requirement 12: Streamlit Dashboard Architecture

**User Story:** As a user interface designer, I want a professional dark industrial theme, so that the dashboard matches the railway inspection context.

#### Acceptance Criteria

1. THE Dashboard SHALL implement "Dark Industrial" theme with appropriate color scheme
2. THE Dashboard SHALL organize functionality into 3 tabs: Mission Control, Restoration Lab, Semantic Search
3. THE Dashboard SHALL use streamlit-image-comparison for before/after visualizations
4. THE Dashboard SHALL display real-time metrics using Streamlit metric cards and charts
5. THE Dashboard SHALL maintain responsive layout across different screen sizes

### Requirement 13: Model Export and Deployment Format

**User Story:** As a deployment engineer, I want all models in optimized formats, so that they can be deployed efficiently on edge hardware.

#### Acceptance Criteria

1. THE IronSight_Engine SHALL load models in ONNX format with FP16 precision where possible
2. THE DeblurGAN SHALL support dynamic input shapes for varying crop sizes
3. THE SmolVLM_Agent SHALL use 8-bit quantization for memory efficiency on Jetson
4. THE SigLIP_Engine SHALL export embeddings in efficient vector format for LanceDB storage
5. THE system SHALL verify model compatibility during initialization

### Requirement 14: Integration with Existing Codebase

**User Story:** As a software engineer, I want to leverage existing implementations, so that development time is minimized and proven components are reused.

#### Acceptance Criteria

1. THE IronSight_Engine SHALL use existing `src/pipeline_core.py` as foundational logic
2. THE system SHALL integrate existing `src/agent_forensic.py` for SmolVLM functionality
3. THE system SHALL use existing `src/semantic_search.py` for SigLIP search capabilities
4. THE system SHALL use existing `NAFNet-GoPro-width64.pth` for motion deblurring
5. THE system SHALL maintain compatibility with existing configuration and data structures

### Requirement 15: Asset and Configuration Management

**User Story:** As a system administrator, I want consistent styling and configuration, so that the system maintains professional appearance and behavior.

#### Acceptance Criteria

1. THE system SHALL scan the `aidtm/` folder for config files, fonts, and utility scripts
2. THE system SHALL maintain style consistency using discovered assets
3. THE system SHALL use existing `config/vehicle_detection.yaml` for model configuration
4. THE system SHALL preserve existing logging and monitoring configurations
5. THE system SHALL support hot-reloading of configuration changes during development
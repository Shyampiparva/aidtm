"""
IronSight Command Center - Main Streamlit Application

Production-grade real-time rail inspection dashboard integrating 5 neural networks.
Features:
- Dark industrial theme appropriate for railway inspection context
- 3-tab layout: Mission Control, Restoration Lab, Semantic Search
- Performance monitoring dashboard with FPS, latency, queue depth
- Model status indicators and "Model Offline" badges

Requirements: 12.1, 12.2, 12.4, 11.1, 11.2
"""

import streamlit as st
import sys
import time
import numpy as np
import glob
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for accessing existing aidtm modules
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import components with error handling
try:
    from mission_control import (
        MissionControl, VideoInputType, VideoInputConfig, 
        OverlayConfig, create_mission_control
    )
    MISSION_CONTROL_AVAILABLE = True
except ImportError as e:
    MISSION_CONTROL_AVAILABLE = False
    print(f"Mission Control import error: {e}")

try:
    from ironsight_engine import IronSightEngine, EngineConfig, create_engine, ModelStatus
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"Engine import error: {e}")
    
    # Define fallback ModelStatus
    class ModelStatus(Enum):
        NOT_LOADED = "not_loaded"
        LOADING = "loading"
        LOADED = "loaded"
        OFFLINE = "offline"
        ERROR = "error"

# Import error handler for UI integration
try:
    from error_handler import (
        ErrorHandler, create_error_handler, ErrorDisplayHelper,
        ErrorCategory, get_global_error_handler
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError as e:
    ERROR_HANDLER_AVAILABLE = False
    print(f"Error handler import error: {e}")


# ============================================================================
# Dark Industrial Theme CSS
# ============================================================================

DARK_INDUSTRIAL_THEME = """
<style>
/* Main background and text colors */
.main {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

.stApp {
    background-color: #1a1a1a;
}

/* Header styling */
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #252525;
    border-right: 1px solid #3a3a3a;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0f0f0 !important;
}

/* Metric cards styling */
[data-testid="stMetric"] {
    background-color: #2d2d2d;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #3a3a3a;
}

[data-testid="stMetricLabel"] {
    color: #a0a0a0 !important;
}

[data-testid="stMetricValue"] {
    color: #f0f0f0 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #252525;
    border-radius: 8px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: #a0a0a0;
    background-color: transparent;
    border-radius: 5px;
}

.stTabs [aria-selected="true"] {
    background-color: #3a3a3a !important;
    color: #f0f0f0 !important;
}

/* Button styling */
.stButton > button {
    background-color: #3a3a3a;
    color: #f0f0f0;
    border: 1px solid #4a4a4a;
    border-radius: 5px;
}

.stButton > button:hover {
    background-color: #4a4a4a;
    border-color: #5a5a5a;
}

.stButton > button[kind="primary"] {
    background-color: #1e5f74;
    border-color: #2a7a94;
}

.stButton > button[kind="primary"]:hover {
    background-color: #2a7a94;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    background-color: #2d2d2d;
    color: #f0f0f0;
    border-color: #3a3a3a;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #2d2d2d;
    color: #f0f0f0;
    border-radius: 5px;
}

/* Info/Warning/Error boxes */
.stAlert {
    background-color: #2d2d2d;
    border-radius: 5px;
}

/* Slider styling */
.stSlider > div > div > div {
    background-color: #3a3a3a;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #2d2d2d;
    border: 1px dashed #4a4a4a;
    border-radius: 8px;
    padding: 10px;
}

/* Model status badges */
.model-status-online {
    background-color: #1e5f2e;
    color: #90ee90;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    display: inline-block;
    margin: 2px 0;
}

.model-status-offline {
    background-color: #5f1e1e;
    color: #ff9090;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    display: inline-block;
    margin: 2px 0;
}

.model-status-loading {
    background-color: #5f5f1e;
    color: #ffff90;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    display: inline-block;
    margin: 2px 0;
}

/* Performance dashboard */
.perf-card {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 15px;
    margin: 5px 0;
}

.perf-value {
    font-size: 1.5em;
    font-weight: bold;
    color: #f0f0f0;
}

.perf-label {
    font-size: 0.85em;
    color: #a0a0a0;
}

/* Detection overlay legend */
.legend-item {
    display: flex;
    align-items: center;
    margin: 5px 0;
}

.legend-color {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    margin-right: 10px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4a4a4a;
}
</style>
"""


# ============================================================================
# Error Display Component
# ============================================================================

def render_error_display():
    """Render error display panel showing recent errors and recovery suggestions."""
    if not ERROR_HANDLER_AVAILABLE:
        return
    
    # Get error handler from session state or global
    error_handler = st.session_state.get('error_handler')
    if not error_handler:
        try:
            error_handler = get_global_error_handler()
            st.session_state.error_handler = error_handler
        except Exception:
            return
    
    # Get error summary
    summary = ErrorDisplayHelper.get_error_summary_for_ui(error_handler)
    
    # Only show if there are errors
    if summary["total_errors"] == 0:
        return
    
    # Create expandable error panel
    with st.expander(f"⚠️ System Alerts ({summary['total_errors']} issues)", expanded=summary["has_critical_errors"]):
        # Show critical warning if needed
        if summary["has_critical_errors"]:
            st.error("🚨 Critical system issues detected. Some features may be unavailable.")
        
        # Show fallback warning
        if summary["fallback_models_active"]:
            st.warning("⚡ Some models are using fallback mode. Results may be limited.")
        
        # Show recent errors
        recent_errors = ErrorDisplayHelper.format_errors_for_streamlit(error_handler, max_errors=5)
        
        for error in recent_errors:
            severity_icon = {
                "error": "🔴",
                "warning": "🟡",
                "info": "🔵"
            }.get(error["severity"], "⚪")
            
            st.markdown(f"""
            **{severity_icon} {error['title']}** - {error['technical_details']['stage']}
            
            {error['description']}
            
            *{error['technical_details']['timestamp']}*
            """)
            
            # Show recovery status
            if error['recovery']['successful']:
                st.success(f"✅ Recovery: {error['recovery']['action_taken']}")
            else:
                st.error(f"❌ Recovery failed: {error['recovery']['action_taken']}")
            
            # Show suggestions in a collapsed section
            with st.expander("💡 Suggestions"):
                for suggestion in error['suggestions']:
                    st.markdown(f"• {suggestion}")
            
            st.markdown("---")
        
        # Clear errors button
        if st.button("🗑️ Clear Error History", key="clear_errors"):
            error_handler.clear_history()
            st.rerun()


# ============================================================================
# Model Status Display Component
# ============================================================================

@dataclass
class ModelInfo:
    """Information about a model for display."""
    name: str
    display_name: str
    status: str
    description: str


def get_model_display_info(engine: Optional[Any] = None) -> Dict[str, ModelInfo]:
    """Get display information for all models."""
    models = {
        "gatekeeper": ModelInfo(
            name="gatekeeper",
            display_name="Gatekeeper",
            status="offline",
            description="Pre-filter (wagon/blur detection)"
        ),
        "sci_enhancer": ModelInfo(
            name="sci_enhancer",
            display_name="SCI Enhancer",
            status="offline",
            description="Low-light enhancement"
        ),
        "yolo_sideview": ModelInfo(
            name="yolo_sideview",
            display_name="YOLO Sideview",
            status="offline",
            description="Damage detection"
        ),
        "yolo_structure": ModelInfo(
            name="yolo_structure",
            display_name="YOLO Structure",
            status="offline",
            description="Component detection"
        ),
        "yolo_wagon_number": ModelInfo(
            name="yolo_wagon_number",
            display_name="YOLO Wagon Number",
            status="offline",
            description="ID plate detection"
        ),
        "nafnet": ModelInfo(
            name="nafnet",
            display_name="NAFNet",
            status="offline",
            description="Motion deblurring"
        ),
        "smolvlm_agent": ModelInfo(
            name="smolvlm_agent",
            display_name="SmolVLM Agent",
            status="offline",
            description="OCR & damage assessment"
        ),
        "siglip_search": ModelInfo(
            name="siglip_search",
            display_name="SigLIP Search",
            status="offline",
            description="Semantic search"
        ),
    }
    
    # Update status from engine if available
    if engine:
        try:
            status_dict = engine.get_model_status()
            for name, status in status_dict.items():
                if name in models:
                    models[name].status = status
        except Exception:
            pass
    
    return models


def render_model_status_badge(status: str) -> str:
    """Render HTML for model status badge."""
    if status == "loaded":
        return '<span class="model-status-online">🟢 Online</span>'
    elif status == "loading":
        return '<span class="model-status-loading">🟡 Loading</span>'
    else:
        return '<span class="model-status-offline">🔴 Offline</span>'


def render_model_status_sidebar():
    """Render model status in sidebar with badges."""
    st.header("🔧 Model Status")
    
    # Get engine from session state
    engine = st.session_state.get('engine')
    models = get_model_display_info(engine)
    
    # Count online/offline
    online_count = sum(1 for m in models.values() if m.status == "loaded")
    total_count = len(models)
    
    # Summary
    st.markdown(f"**{online_count}/{total_count}** models online")
    st.markdown("---")
    
    # Individual model status
    for model in models.values():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{model.display_name}**")
            st.caption(model.description)
        with col2:
            badge_html = render_model_status_badge(model.status)
            st.markdown(badge_html, unsafe_allow_html=True)


# ============================================================================
# Performance Monitoring Dashboard
# ============================================================================

def render_performance_dashboard():
    """Render performance monitoring dashboard."""
    st.subheader("📊 Performance Monitor")
    
    # Get stats from mission control or engine
    mission_control = st.session_state.get('mission_control')
    engine = st.session_state.get('engine')
    
    # Default values
    fps = 0.0
    latency_ms = 0.0
    queue_depth = 0
    frames_processed = 0
    gatekeeper_skips = 0
    gpu_memory_mb = 0.0
    gpu_temp_c = 0.0
    
    # Get actual values if available
    if mission_control and st.session_state.get('is_processing'):
        stats = mission_control.get_stats()
        fps = stats.fps
        latency_ms = stats.processing_latency_ms
        queue_depth = stats.queue_depth
        frames_processed = stats.frames_processed
        gatekeeper_skips = stats.gatekeeper_skips
    
    if engine:
        try:
            metrics = engine.get_performance_metrics()
            gpu_memory_mb = metrics.gpu_memory_usage_mb
            gpu_temp_c = metrics.gpu_temperature_c
        except Exception:
            pass
    
    # Render metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FPS", f"{fps:.1f}", delta=None)
    with col2:
        st.metric("Latency", f"{latency_ms:.1f}ms", delta=None)
    with col3:
        st.metric("Queue Depth", queue_depth, delta=None)
    with col4:
        st.metric("Frames", frames_processed, delta=None)
    
    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Gatekeeper Skips", gatekeeper_skips, delta=None)
    with col6:
        st.metric("GPU Memory", f"{gpu_memory_mb:.0f}MB", delta=None)
    with col7:
        st.metric("GPU Temp", f"{gpu_temp_c:.0f}°C" if gpu_temp_c > 0 else "N/A", delta=None)
    with col8:
        # Calculate efficiency
        efficiency = (gatekeeper_skips / max(frames_processed, 1)) * 100 if frames_processed > 0 else 0
        st.metric("Skip Rate", f"{efficiency:.1f}%", delta=None)


# ============================================================================
# Mission Control Tab
# ============================================================================

def render_mission_control_tab():
    """Render the Mission Control tab with live processing interface."""
    st.header("🎯 Mission Control")
    st.caption("Real-time wagon inspection with AI detection overlays")
    
    # Initialize session state
    if 'mission_control' not in st.session_state:
        st.session_state.mission_control = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'temp_video_files' not in st.session_state:
        st.session_state.temp_video_files = []
    
    # Clean up any orphaned media files on page refresh
    if 'page_refresh_count' not in st.session_state:
        st.session_state.page_refresh_count = 0
        # Clear any temporary files from previous sessions
        import glob
        for temp_file in glob.glob("temp_video_*.mp4"):
            try:
                Path(temp_file).unlink()
            except:
                pass
    st.session_state.page_refresh_count += 1

    # Video input selection
    st.subheader("📹 Video Input")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_type = st.selectbox(
            "Input Source",
            options=["None", "Webcam", "RTSP Stream", "Video File"],
            key="video_input_type",
            help="Select video input source for live processing"
        )
    
    with col2:
        if input_type == "Webcam":
            source = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
            video_type = VideoInputType.WEBCAM if MISSION_CONTROL_AVAILABLE else None
        elif input_type == "RTSP Stream":
            source = st.text_input("RTSP URL", placeholder="rtsp://...")
            video_type = VideoInputType.RTSP if MISSION_CONTROL_AVAILABLE else None
        elif input_type == "Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv', 'webm'])
            if uploaded_file:
                # Create a unique temporary file name based on file content and timestamp
                import hashlib
                import time as time_module
                
                file_content = uploaded_file.read()
                file_hash = hashlib.md5(file_content).hexdigest()[:8]
                timestamp = str(int(time_module.time()))[-6:]  # Last 6 digits of timestamp
                uploaded_file.seek(0)  # Reset file pointer
                
                temp_path = Path(f"temp_video_{file_hash}_{timestamp}.mp4")
                
                # Only create new file if it doesn't exist
                if not temp_path.exists():
                    with open(temp_path, "wb") as f:
                        f.write(file_content)
                
                source = str(temp_path)
                
                # Store temp file path for cleanup
                if temp_path not in st.session_state.temp_video_files:
                    st.session_state.temp_video_files.append(temp_path)
            else:
                source = ""
            video_type = VideoInputType.FILE if MISSION_CONTROL_AVAILABLE else None
        else:
            source = ""
            video_type = VideoInputType.NONE if MISSION_CONTROL_AVAILABLE else None
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_btn = st.button("▶️ Start Processing", type="primary", width="stretch")
    with col2:
        stop_btn = st.button("⏹️ Stop Processing", width="stretch")
    with col3:
        refresh_btn = st.button("🔄 Refresh", width="stretch")
    
    # Handle start/stop
    if start_btn and video_type and video_type != VideoInputType.NONE and MISSION_CONTROL_AVAILABLE:
        if st.session_state.mission_control is None:
            st.session_state.mission_control = create_mission_control(
                input_type=video_type,
                source=source
            )
        else:
            st.session_state.mission_control.set_video_input(video_type, source)
        
        if st.session_state.mission_control.start_processing():
            st.session_state.is_processing = True
            st.success("✅ Processing started!")
        else:
            if video_type == VideoInputType.WEBCAM:
                st.error("❌ Failed to start camera. Please check if a camera is connected and not in use by another application.")
            elif video_type == VideoInputType.FILE:
                st.error("❌ Failed to open video file. Please check if the file exists and is a valid video format.")
            elif video_type == VideoInputType.RTSP:
                st.error("❌ Failed to connect to RTSP stream. Please check the URL and network connection.")
            else:
                st.error("❌ Failed to start processing. Check video source.")
    
    if stop_btn and st.session_state.mission_control:
        st.session_state.mission_control.stop_processing()
        st.session_state.is_processing = False
        
        # Clean up temporary video files
        for temp_file in st.session_state.get('temp_video_files', []):
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")
        st.session_state.temp_video_files = []
        
        st.info("⏹️ Processing stopped.")

    # Performance dashboard
    render_performance_dashboard()
    
    # Latest serial number display (prominent)
    st.subheader("🔢 Latest Serial Number")
    serial_container = st.container()
    
    with serial_container:
        if st.session_state.mission_control and st.session_state.is_processing:
            stats = st.session_state.mission_control.get_stats()
            serial = stats.last_serial_number
        else:
            serial = "N/A"
        
        st.markdown(
            f"""
            <div style="
                background-color: #2d2d2d;
                border: 2px solid #1e5f74;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            ">
                <span style="font-size: 2.5em; font-weight: bold; color: #90caf9;">
                    {serial}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Video display area
    st.subheader("📺 Live Video Feed")
    
    # Add playback controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="auto_refresh_video")
    with col2:
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    with col3:
        if st.session_state.is_processing:
            st.success("🟢 Processing Active")
        else:
            st.info("⏸️ Processing Stopped")
    
    video_placeholder = st.empty()
    
    if st.session_state.is_processing and st.session_state.mission_control:
        # Live processing mode
        try:
            import cv2
            frame, stats = st.session_state.mission_control.process_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    video_placeholder.image(
                        frame_rgb, 
                        channels="RGB", 
                        width="stretch",
                        caption=f"Live Feed - FPS: {stats.fps:.1f} | Frames: {stats.frames_processed}"
                    )
                except Exception as img_error:
                    # Handle image display errors gracefully
                    if "MediaFileStorageError" in str(img_error):
                        video_placeholder.warning("⚠️ Image display issue - refreshing...")
                    else:
                        video_placeholder.error(f"Image display error: {img_error}")
                
                # Auto-refresh for continuous playback
                if auto_refresh:
                    time.sleep(0.033)  # ~30 FPS
                    st.rerun()
            else:
                video_placeholder.info("⏳ Waiting for video frames...")
                if auto_refresh:
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            video_placeholder.error(f"Error processing frame: {e}")
            # Don't show full stack trace for media file errors
            if "MediaFileStorageError" not in str(e):
                st.exception(e)
    else:
        # Preview mode for uploaded videos
        if input_type == "Video File" and uploaded_file is not None and source:
            try:
                # Show video preview using Streamlit's native video player
                st.info("📹 Video uploaded successfully! Use the controls above to start processing.")
                video_placeholder.video(source)
                
                # Show video info
                try:
                    import cv2
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        st.caption(f"📊 Video Info: {width}x{height} | {fps:.1f} FPS | {duration:.1f}s | {frame_count} frames")
                        cap.release()
                except Exception as e:
                    st.caption(f"Could not read video info: {e}")
                    
            except Exception as e:
                video_placeholder.error(f"Error displaying video preview: {e}")
                st.error(f"Video format may not be supported. Try MP4 format.")
        else:
            video_placeholder.info("👆 Select a video source and click 'Start Processing' to begin.")
    
    # Detection overlay legend
    st.subheader("🎨 Detection Overlay Legend")
    legend_cols = st.columns(3)
    
    with legend_cols[0]:
        st.markdown("""
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff4444;"></div>
            <span><b>Sideview Damage</b><br/>Dents, holes, rust, scratches</span>
        </div>
        """, unsafe_allow_html=True)
    with legend_cols[1]:
        st.markdown("""
        <div class="legend-item">
            <div class="legend-color" style="background-color: #44ff44;"></div>
            <span><b>Structure</b><br/>Doors, wheels, couplers, brakes</span>
        </div>
        """, unsafe_allow_html=True)
    with legend_cols[2]:
        st.markdown("""
        <div class="legend-item">
            <div class="legend-color" style="background-color: #44ffff;"></div>
            <span><b>Wagon Numbers</b><br/>ID plates, serial numbers</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Restoration Lab Tab
# ============================================================================

def render_restoration_lab_tab():
    """Render the Restoration Lab tab with interactive image restoration."""
    st.header("🔬 Restoration Lab")
    st.caption("Interactive image restoration testing with NAFNet")
    
    try:
        from restoration_lab import render_restoration_lab_ui
        render_restoration_lab_ui(st)
    except ImportError as e:
        st.error(f"❌ Failed to load Restoration Lab module: {e}")
        st.info("Please ensure restoration_lab.py is in the src directory.")
        
        # Show placeholder UI
        st.subheader("📤 Upload Image")
        st.file_uploader(
            "Upload a blurry image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            disabled=True
        )
        st.warning("Restoration Lab is currently unavailable.")


# ============================================================================
# Semantic Search Tab
# ============================================================================

def render_semantic_search_tab():
    """Render the Semantic Search tab."""
    st.header("🔍 Semantic Search")
    st.caption("Natural language search for inspection history")
    
    try:
        from semantic_search_ui import render_semantic_search_ui
        render_semantic_search_ui(st)
    except ImportError as e:
        st.error(f"❌ Failed to load Semantic Search module: {e}")
        st.info("Please ensure semantic_search_ui.py is in the src directory.")
        
        # Show placeholder UI
        st.subheader("🔍 Natural Language Search")
        query = st.text_input(
            "Search Query",
            placeholder="e.g., 'wagons with rust damage'",
            disabled=True
        )
        st.warning("Semantic Search is currently unavailable.")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render the sidebar with model status and system info."""
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <span style="font-size: 2em;">🚂</span>
            <h2 style="margin: 0;">IronSight</h2>
            <p style="color: #888; margin: 0;">Command Center</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model status
        render_model_status_sidebar()
        
        st.markdown("---")
        
        # System info
        st.header("💻 System Info")
        
        # Check CUDA availability
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                st.success(f"🟢 GPU: {gpu_name}")
            else:
                st.warning("🟡 GPU: Not available (CPU mode)")
        except ImportError:
            st.warning("🟡 PyTorch not installed")
        
        # Engine status
        engine = st.session_state.get('engine')
        if engine:
            st.success("🟢 Engine: Initialized")
        else:
            st.info("⚪ Engine: Not initialized")
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🔄 Initialize Engine", width="stretch"):
            with st.spinner("Initializing engine..."):
                try:
                    if ENGINE_AVAILABLE:
                        st.session_state.engine = create_engine()
                        st.session_state.engine.load_models()
                        st.success("✅ Engine initialized!")
                    else:
                        st.error("Engine module not available")
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")
        
        if st.button("📊 Refresh Status", width="stretch"):
            st.rerun()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main entry point for IronSight Command Center dashboard."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="IronSight Command Center",
        page_icon="🚂",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply dark industrial theme
    st.markdown(DARK_INDUSTRIAL_THEME, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="margin: 0;">🚂 IronSight Command Center</h1>
        <p style="color: #888; margin: 5px 0 0 0;">
            Production-grade real-time rail inspection dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Render error display (shows alerts if there are errors)
    render_error_display()
    
    # Create tabs for the three main interfaces
    tab1, tab2, tab3 = st.tabs([
        "🎯 Mission Control", 
        "🔬 Restoration Lab", 
        "🔍 Semantic Search"
    ])
    
    with tab1:
        render_mission_control_tab()
    
    with tab2:
        render_restoration_lab_tab()
    
    with tab3:
        render_semantic_search_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85em;">
        IronSight Command Center v1.0 | 
        Real-time AI-powered rail inspection
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

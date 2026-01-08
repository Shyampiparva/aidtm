# 🎥 Video Playback Troubleshooting Guide

## 🔍 Common Issues and Solutions

### **Issue 1: Video Not Playing After Upload**

**Symptoms:**
- Video file uploads successfully but doesn't play
- Only shows a static image or placeholder
- No error messages displayed

**Solutions:**

#### **Solution A: Use Streamlit's Native Video Player**
The updated implementation now uses `st.video()` for preview mode:

```python
# Preview mode - shows actual video player
if input_type == "Video File" and uploaded_file is not None:
    st.video(source)  # Native Streamlit video player
```

#### **Solution B: Check Video Format Compatibility**
Supported formats:
- ✅ **MP4** (recommended) - Best compatibility
- ✅ **WebM** - Good web compatibility  
- ✅ **AVI** - Basic support
- ✅ **MOV** - Apple format
- ✅ **MKV** - Container format

**Test your video format:**
```bash
uv run python test_video_upload.py
```

#### **Solution C: Verify File Upload Process**
The system now creates unique temporary files:

```python
# Creates unique temp files to avoid conflicts
file_hash = hashlib.md5(uploaded_file.read()).hexdigest()[:8]
temp_path = Path(f"temp_video_{file_hash}.mp4")
```

### **Issue 2: Live Processing Shows Static Images**

**Symptoms:**
- Video processing starts but shows only static frames
- No continuous playback during processing
- FPS counter shows 0 or very low values

**Solutions:**

#### **Solution A: Enable Auto-Refresh**
The new implementation includes auto-refresh controls:

```python
# Auto-refresh checkbox for continuous playback
auto_refresh = st.checkbox("Auto Refresh", value=True)

if auto_refresh:
    time.sleep(0.033)  # ~30 FPS
    st.rerun()
```

#### **Solution B: Manual Refresh Option**
Use the manual refresh button if auto-refresh causes issues:

```python
if st.button("🔄 Manual Refresh"):
    st.rerun()
```

#### **Solution C: Check Processing Status**
Monitor the processing status indicator:
- 🟢 **Processing Active** - System is running
- ⏸️ **Processing Stopped** - System is idle

### **Issue 3: Video Upload Fails**

**Symptoms:**
- File upload widget shows error
- "Video format may not be supported" message
- Upload progress bar fails

**Solutions:**

#### **Solution A: Convert Video Format**
Convert your video to MP4 using FFmpeg:

```bash
# Install FFmpeg first, then convert
ffmpeg -i input_video.avi -c:v libx264 -c:a aac output_video.mp4
```

#### **Solution B: Check File Size**
Streamlit has file size limits:
- Default limit: 200MB
- For larger files, use video compression

#### **Solution C: Verify Video Properties**
Use the test script to check video compatibility:

```bash
uv run python test_video_upload.py
```

### **Issue 4: Poor Video Performance**

**Symptoms:**
- Choppy playback
- Low FPS during processing
- High CPU usage

**Solutions:**

#### **Solution A: Adjust Refresh Rate**
Modify the refresh rate in the code:

```python
# Slower refresh for better performance
time.sleep(0.1)  # 10 FPS instead of 30 FPS
```

#### **Solution B: Reduce Video Resolution**
The system automatically resizes large videos:

```python
# Automatic resizing in mission_control.py
if width > self.config.max_width or height > self.config.max_height:
    # Calculate scaling factor and resize
```

#### **Solution C: Use CPU-Only Mode**
Run with CPU-only processing:

```bash
uv run python run_ironsight.py --cpu
```

### **Issue 5: Memory Issues with Large Videos**

**Symptoms:**
- System crashes with large video files
- "Out of memory" errors
- Slow performance with high-resolution videos

**Solutions:**

#### **Solution A: Video Compression**
Compress videos before upload:

```bash
# Compress video to reduce size
ffmpeg -i input.mp4 -vf scale=1280:720 -crf 23 compressed.mp4
```

#### **Solution B: Frame Buffering**
The system uses limited frame queues:

```python
# Limited queue size to prevent memory issues
self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
```

#### **Solution C: Temporary File Cleanup**
The system automatically cleans up temporary files:

```python
# Automatic cleanup on stop
for temp_file in st.session_state.get('temp_video_files', []):
    if temp_file.exists():
        temp_file.unlink()
```

## 🛠️ Debugging Steps

### **Step 1: Check System Requirements**
```bash
# Verify dependencies
uv run python simple_test.py

# Check OpenCV video capabilities
uv run python test_video_upload.py
```

### **Step 2: Test with Sample Video**
```bash
# Create test video
uv run python test_video_upload.py

# Upload test_video.mp4 in the dashboard
```

### **Step 3: Monitor Logs**
Check the application logs for errors:
- Console output in terminal
- `live_processor.log` file
- Browser developer console (F12)

### **Step 4: Verify Video Properties**
Use OpenCV to check video properties:

```python
import cv2
cap = cv2.VideoCapture("your_video.mp4")
print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
cap.release()
```

## 🎯 Best Practices

### **Video Upload Best Practices**
1. **Use MP4 format** with H.264 codec
2. **Keep file size under 100MB** for best performance
3. **Use standard resolutions** (720p, 1080p)
4. **Frame rate 15-30 FPS** for optimal processing

### **Processing Best Practices**
1. **Enable auto-refresh** for smooth playback
2. **Monitor performance metrics** (FPS, latency)
3. **Use appropriate quality settings** based on hardware
4. **Stop processing** when switching video sources

### **Performance Optimization**
1. **Close unused browser tabs** to free memory
2. **Use CPU-only mode** if GPU issues occur
3. **Reduce video resolution** for faster processing
4. **Enable hardware acceleration** in browser settings

## 🚨 Known Limitations

### **Streamlit Limitations**
- No native real-time video streaming
- File upload size limits
- Browser-dependent video codec support

### **OpenCV Limitations**
- Platform-specific codec availability
- Limited video format support on some systems
- Memory usage with large videos

### **Browser Limitations**
- Video format compatibility varies by browser
- Memory limits for large files
- Performance varies by browser engine

## 📞 Getting Help

If you're still experiencing issues:

1. **Check the logs** for specific error messages
2. **Try the test video** created by `test_video_upload.py`
3. **Use different video formats** (MP4, WebM, AVI)
4. **Reduce video size and resolution**
5. **Test with different browsers** (Chrome, Firefox, Edge)

## 🔧 Advanced Troubleshooting

### **Custom Video Processing**
For advanced users, you can modify the video processing pipeline:

```python
# In mission_control.py - customize frame processing
def process_frame(self) -> Tuple[Optional[np.ndarray], ProcessingStats]:
    # Add custom processing logic here
    pass
```

### **Alternative Video Sources**
Test with different video sources:
- Webcam: `input_type = "Webcam"`
- RTSP stream: `input_type = "RTSP Stream"`
- Local file: `input_type = "Video File"`

### **Performance Profiling**
Monitor system performance:

```python
# Check processing statistics
stats = mission_control.get_stats()
print(f"FPS: {stats.fps}")
print(f"Latency: {stats.processing_latency_ms}ms")
print(f"Frames processed: {stats.frames_processed}")
```

---

**Remember: The system now supports both video preview (using Streamlit's native player) and live processing (using OpenCV frame-by-frame processing). Choose the appropriate mode based on your needs!** 🎬✨
# 📱 Live IP Webcam Video Processor

A Python script that processes live video feeds from IP webcam applications (like phone cameras) using OpenCV for video capture and niquests for HTTP-based metadata and API requests.

## 🚀 Features

- **Real-time Video Processing**: Live stream capture and processing from IP webcam URLs
- **HTTP API Integration**: Uses niquests for webcam settings and metadata fetching
- **Performance Monitoring**: Real-time FPS, latency, and connection statistics
- **Error Recovery**: Automatic reconnection and graceful error handling
- **Interactive Controls**: Save frames, view statistics, and quit with keyboard shortcuts
- **Configurable Quality**: Low, medium, and high quality settings
- **Headless Mode**: Run without display for server deployments

## 📦 Dependencies

Installed via UV (as per project standards):
- `opencv-python` - Video capture and processing
- `niquests` - Modern HTTP client for API requests

## 🎯 Usage

### Basic Usage

```bash
# Using UV (recommended)
uv run python live_processor.py --url http://192.168.1.100:8080

# Direct Python
python live_processor.py --url http://192.168.1.100:8080
```

### Advanced Usage

```bash
# High quality with custom FPS
uv run python live_processor.py --url http://phone.local:8080 --quality high --fps 15

# Debug mode with custom resolution
uv run python live_processor.py --url http://192.168.1.50:8080 --debug --max-width 1280 --max-height 720

# Headless mode for server deployment
uv run python live_processor.py --url http://192.168.1.100:8080 --no-display
```

## 📱 IP Webcam Setup

### Android - IP Webcam App

1. Install "IP Webcam" app from Google Play Store
2. Configure video settings (resolution, quality, FPS)
3. Start the server
4. Note the IP address and port (usually 8080)
5. Use URL format: `http://<phone_ip>:8080`

### iOS - Similar Apps

1. Install apps like "EpocCam" or "iVCam"
2. Follow app-specific setup instructions
3. Note the streaming URL provided by the app

## 🎮 Interactive Controls

While the video window is active:

- **`q`** - Quit the application
- **`s`** - Save current frame as JPG
- **`i`** - Show detailed statistics

## 📊 Statistics and Monitoring

The processor provides comprehensive statistics:

- **Performance**: FPS, frame processing time, dropped frames
- **Connection**: Success rate, reconnection attempts
- **API**: HTTP request statistics and errors
- **Uptime**: Total running time and last frame timestamp

## 🔧 Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | Required | Base URL of IP webcam |
| `--quality` | medium | Video quality (low/medium/high) |
| `--fps` | 30 | Target frames per second |
| `--timeout` | 10 | Connection timeout (seconds) |
| `--max-width` | 1920 | Maximum frame width |
| `--max-height` | 1080 | Maximum frame height |
| `--debug` | False | Enable debug logging |
| `--no-display` | False | Run without video display |

## 🏗️ Architecture

### Core Components

1. **WebcamConfig**: Configuration management with quality presets
2. **WebcamAPIClient**: HTTP client using niquests for API interactions
3. **ProcessingStats**: Performance monitoring and statistics
4. **LiveVideoProcessor**: Main processing pipeline with threading

### Processing Pipeline

```
IP Webcam → HTTP Stream → OpenCV Capture → Frame Queue → Processing → Display
     ↓
API Client → niquests → Status/Settings → Metadata
```

### Threading Model

- **Main Thread**: Display loop and user interaction
- **Capture Thread**: Frame capture from video stream
- **Queue**: Thread-safe frame buffer (max 10 frames)

## 🔍 API Integration

The processor uses niquests to interact with IP webcam APIs:

### Status Endpoint
```python
GET http://<webcam_ip>:8080/status.json
```
Returns webcam status, settings, and metadata.

### Settings Endpoint
```python
POST http://<webcam_ip>:8080/settings
Content-Type: application/json

{"quality": "high", "fps": 30}
```
Updates webcam configuration.

### Video Stream
```python
GET http://<webcam_ip>:8080/video
```
MJPEG video stream for OpenCV capture.

## 🚨 Error Handling

The processor includes robust error handling:

- **Connection Failures**: Automatic reconnection with exponential backoff
- **Frame Drops**: Queue overflow protection and statistics
- **API Errors**: Graceful degradation when API unavailable
- **Signal Handling**: Clean shutdown on SIGINT/SIGTERM

## 📝 Logging

Logs are written to both console and `live_processor.log`:

```
2026-01-08 22:45:30,123 - LiveProcessor - INFO - Starting live video processor...
2026-01-08 22:45:30,456 - LiveProcessor - INFO - Successfully connected to video stream. Frame size: (720, 1280, 3)
2026-01-08 22:45:30,789 - LiveProcessor - INFO - Starting display loop (press 'q' to quit)
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if webcam is accessible
   curl http://192.168.1.100:8080/status.json
   
   # Try different endpoint
   python live_processor.py --url http://192.168.1.100:8080
   ```

2. **Poor Performance**
   ```bash
   # Reduce quality and FPS
   python live_processor.py --url http://192.168.1.100:8080 --quality low --fps 15
   ```

3. **Network Issues**
   ```bash
   # Increase timeout
   python live_processor.py --url http://192.168.1.100:8080 --timeout 30
   ```

### Debug Mode

Enable debug logging for detailed information:

```bash
uv run python live_processor.py --url http://192.168.1.100:8080 --debug
```

## 🚀 Integration with IronSight

The live processor can be integrated with the IronSight Command Center:

1. **Mission Control**: Use as video input source
2. **Real-time Analysis**: Process frames through AI pipeline
3. **Performance Monitoring**: Integrate statistics with dashboard
4. **API Integration**: Use niquests for all HTTP operations

## 📈 Performance Tips

1. **Quality Settings**: Use appropriate quality for your network
2. **Frame Rate**: Lower FPS reduces CPU usage
3. **Resolution**: Limit max width/height for better performance
4. **Network**: Use wired connection for stability
5. **Headless Mode**: Disable display for server deployments

## 🔒 Security Considerations

- IP webcams may not have authentication
- Use on trusted networks only
- Consider VPN for remote access
- Monitor for unauthorized access attempts

---

**Ready to process some live video feeds! 📱✨**
#!/usr/bin/env python3
"""
Example usage of the Live IP Webcam Video Processor.

This script demonstrates how to use the live_processor.py script
with different configurations and scenarios.
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Run a command and show the description."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    print("Press Ctrl+C to stop this example and continue to the next one...")
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Example timed out (this is expected for demo purposes)")
        return True
    except KeyboardInterrupt:
        print("⏹️ Example stopped by user")
        return True
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False

def main():
    """Run example usage scenarios."""
    print("🎥 Live IP Webcam Video Processor - Usage Examples")
    print("=" * 60)
    print()
    print("This script demonstrates various ways to use live_processor.py")
    print("Note: These examples will fail to connect (no actual webcam)")
    print("but they show the correct command syntax and options.")
    print()
    
    examples = [
        {
            "cmd": ["uv", "run", "python", "live_processor.py", "--help"],
            "desc": "Show help and available options"
        },
        {
            "cmd": ["uv", "run", "python", "live_processor.py", 
                   "--url", "http://192.168.1.100:8080", "--debug"],
            "desc": "Connect to IP webcam with debug logging"
        },
        {
            "cmd": ["uv", "run", "python", "live_processor.py", 
                   "--url", "http://phone.local:8080", "--quality", "high", "--fps", "15"],
            "desc": "High quality video at 15 FPS"
        },
        {
            "cmd": ["uv", "run", "python", "live_processor.py", 
                   "--url", "http://192.168.1.50:8080", "--no-display", "--timeout", "30"],
            "desc": "Headless mode with extended timeout"
        },
        {
            "cmd": ["uv", "run", "python", "live_processor.py", 
                   "--url", "http://test.example.com:8080", "--quality", "low", 
                   "--max-width", "640", "--max-height", "480"],
            "desc": "Low quality with custom resolution limits"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Example {i}/{len(examples)}")
        success = run_command(example["cmd"], example["desc"])
        
        if not success:
            print("❌ Example failed")
        else:
            print("✅ Example completed")
        
        if i < len(examples):
            print("\nPress Enter to continue to the next example...")
            try:
                input()
            except KeyboardInterrupt:
                print("\n👋 Examples interrupted by user")
                break
    
    print("\n" + "="*60)
    print("🎉 All examples completed!")
    print()
    print("To use with a real IP webcam:")
    print("1. Install 'IP Webcam' app on your phone")
    print("2. Start the server and note the IP address")
    print("3. Run: uv run python live_processor.py --url http://<phone_ip>:8080")
    print()
    print("Interactive controls while running:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'i' to show statistics")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
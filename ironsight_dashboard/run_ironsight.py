#!/usr/bin/env python3
"""
IronSight Command Center - Unified Startup Script

This script provides a single entry point to run the complete IronSight system,
including backend services and frontend dashboard.

Usage:
    python run_ironsight.py [OPTIONS]

Options:
    --dev           Run in development mode with auto-reload
    --cpu           Force CPU-only mode (disable GPU)
    --port PORT     Set Streamlit port (default: 8501)
    --host HOST     Set host address (default: 0.0.0.0)
    --no-browser    Don't open browser automatically
    --debug         Enable debug logging
    --help          Show this help message

Examples:
    python run_ironsight.py                    # Standard production mode
    python run_ironsight.py --dev              # Development mode
    python run_ironsight.py --cpu --port 8080  # CPU mode on port 8080
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import webbrowser
from dataclasses import dataclass

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def colorize(text: str, color: str) -> str:
        """Add color to text if terminal supports it."""
        if os.name == 'nt':  # Windows
            return text  # Windows terminal may not support colors
        return f"{color}{text}{Colors.END}"


@dataclass
class SystemCheck:
    """Result of a system check."""
    name: str
    passed: bool
    message: str
    critical: bool = False


class IronSightLauncher:
    """Main launcher for IronSight Command Center."""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.src_dir = self.project_dir / "src"
        self.config_dir = self.project_dir / "config"
        self.logs_dir = self.project_dir / "logs"
        
        # Runtime state
        self.processes: List[subprocess.Popen] = []
        self.shutdown_requested = False
        
        # Configuration
        self.dev_mode = False
        self.cpu_only = False
        self.port = 8501
        self.host = "0.0.0.0"
        self.open_browser = True
        self.debug = False
        self.use_uv = False  # Will be set during system checks
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logs_dir.mkdir(exist_ok=True)
        
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'ironsight_launcher.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('IronSightLauncher')
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        self.shutdown()
    
    def print_banner(self):
        """Print startup banner."""
        banner = f"""
{Colors.colorize('=' * 60, Colors.CYAN)}
{Colors.colorize('🚂 IronSight Command Center', Colors.BOLD + Colors.WHITE)}
{Colors.colorize('   Production-grade rail inspection dashboard', Colors.WHITE)}
{Colors.colorize('=' * 60, Colors.CYAN)}

{Colors.colorize('Configuration:', Colors.YELLOW)}
  • Mode: {Colors.colorize('Development' if self.dev_mode else 'Production', Colors.GREEN)}
  • Compute: {Colors.colorize('CPU-only' if self.cpu_only else 'GPU + CPU', Colors.GREEN)}
  • Runtime: {Colors.colorize('uv run' if getattr(self, 'use_uv', False) else 'python', Colors.GREEN)}
  • Port: {Colors.colorize(str(self.port), Colors.GREEN)}
  • Host: {Colors.colorize(self.host, Colors.GREEN)}
  • Debug: {Colors.colorize('Enabled' if self.debug else 'Disabled', Colors.GREEN)}

"""
        print(banner)
    
    def run_system_checks(self) -> List[SystemCheck]:
        """Run comprehensive system checks."""
        checks = []
        
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 10) and python_version < (3, 13):
            checks.append(SystemCheck(
                "Python Version", True, 
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            ))
        else:
            checks.append(SystemCheck(
                "Python Version", False,
                f"Python {python_version.major}.{python_version.minor} (requires 3.10-3.12)",
                critical=True
            ))
        
        # UV availability check
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                uv_version = result.stdout.strip()
                checks.append(SystemCheck("UV Package Manager", True, f"Available: {uv_version}"))
                self.use_uv = True
            else:
                checks.append(SystemCheck("UV Package Manager", False, "Not available"))
                self.use_uv = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            checks.append(SystemCheck("UV Package Manager", False, "Not installed"))
            self.use_uv = False
        
        # Required packages check (using uv if available)
        required_packages = [
            'streamlit', 'torch', 'torchvision', 'opencv-python', 
            'numpy', 'pillow', 'transformers', 'accelerate'
        ]
        
        if self.use_uv:
            # Check if pyproject.toml exists for uv
            pyproject_path = self.project_dir / "pyproject.toml"
            if pyproject_path.exists():
                checks.append(SystemCheck("pyproject.toml", True, "Found for uv dependency management"))
            else:
                checks.append(SystemCheck("pyproject.toml", False, "Missing (recommended for uv)"))
        
        for package in required_packages:
            try:
                if self.use_uv:
                    # For uv, we'll assume packages are managed via pyproject.toml
                    # We can still do a basic import check
                    __import__(package.replace('-', '_'))
                    checks.append(SystemCheck(package, True, "Available"))
                else:
                    __import__(package.replace('-', '_'))
                    checks.append(SystemCheck(package, True, "Installed"))
            except ImportError:
                checks.append(SystemCheck(
                    package, False, "Not available", 
                    critical=package in ['streamlit', 'torch', 'numpy']
                ))
        
        # CUDA availability check
        try:
            import torch
            if torch.cuda.is_available() and not self.cpu_only:
                gpu_name = torch.cuda.get_device_name(0)
                checks.append(SystemCheck(
                    "CUDA GPU", True, f"Available: {gpu_name}"
                ))
            else:
                checks.append(SystemCheck(
                    "CUDA GPU", False, 
                    "Not available (CPU mode)" if self.cpu_only else "Not available"
                ))
        except ImportError:
            checks.append(SystemCheck(
                "CUDA GPU", False, "PyTorch not available", critical=True
            ))
        
        # Model files check
        model_paths = [
            ("NAFNet Model", self.project_dir.parent / "NAFNet-GoPro-width64.pth"),
            ("YOLO Sideview", self.project_dir.parent / "yolo_sideview_damage_obb.pt"),
            ("YOLO Structure", self.project_dir.parent / "yolo_structure_obb.pt"),
        ]
        
        for name, path in model_paths:
            if path.exists():
                checks.append(SystemCheck(name, True, f"Found at {path}"))
            else:
                checks.append(SystemCheck(name, False, f"Not found at {path}"))
        
        # Directory structure check
        required_dirs = [self.src_dir, self.config_dir]
        for dir_path in required_dirs:
            if dir_path.exists():
                checks.append(SystemCheck(f"Directory {dir_path.name}", True, "Exists"))
            else:
                checks.append(SystemCheck(
                    f"Directory {dir_path.name}", False, "Missing", critical=True
                ))
        
        # Port availability check
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, self.port))
            checks.append(SystemCheck(
                f"Port {self.port}", True, "Available"
            ))
        except OSError:
            checks.append(SystemCheck(
                f"Port {self.port}", False, "Already in use", critical=True
            ))
        
        return checks
    
    def display_system_checks(self, checks: List[SystemCheck]):
        """Display system check results."""
        print(f"{Colors.colorize('System Checks:', Colors.YELLOW)}")
        print()
        
        passed_count = 0
        critical_failures = 0
        
        for check in checks:
            if check.passed:
                icon = "✅"
                color = Colors.GREEN
                passed_count += 1
            else:
                icon = "❌" if check.critical else "⚠️"
                color = Colors.RED if check.critical else Colors.YELLOW
                if check.critical:
                    critical_failures += 1
            
            status = Colors.colorize(f"{icon} {check.name}", color)
            print(f"  {status:<40} {check.message}")
        
        print()
        print(f"Results: {Colors.colorize(f'{passed_count}/{len(checks)} passed', Colors.GREEN)}")
        
        if critical_failures > 0:
            print(f"{Colors.colorize(f'❌ {critical_failures} critical failures detected!', Colors.RED)}")
            return False
        
        return True
    
    def setup_environment(self):
        """Setup environment variables and configuration."""
        # Set CUDA visibility
        if self.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.logger.info("CUDA disabled - running in CPU-only mode")
        
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = str(self.port)
        os.environ['STREAMLIT_SERVER_ADDRESS'] = self.host
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true' if not self.dev_mode else 'false'
        os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'true' if self.dev_mode else 'false'
        
        # Set project paths
        os.environ['IRONSIGHT_PROJECT_DIR'] = str(self.project_dir)
        os.environ['IRONSIGHT_SRC_DIR'] = str(self.src_dir)
        os.environ['IRONSIGHT_CONFIG_DIR'] = str(self.config_dir)
        
        # Load .env file if exists
        env_file = self.project_dir / '.env'
        if env_file.exists():
            self.logger.info(f"Loading environment from {env_file}")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def start_backend_services(self):
        """Start any required backend services."""
        # For now, IronSight is primarily a Streamlit app
        # Future: Could start separate API servers, model servers, etc.
        self.logger.info("Backend services: Using integrated Streamlit backend")
    
    def start_frontend(self):
        """Start the Streamlit frontend."""
        self.logger.info(f"Starting Streamlit frontend on {self.host}:{self.port}")
        
        # Build command based on whether we're using uv or not
        if self.use_uv:
            cmd = [
                'uv', 'run', 'streamlit', 'run',
                str(self.src_dir / 'app.py'),
                '--server.port', str(self.port),
                '--server.address', self.host,
            ]
            self.logger.info("Using uv run for better dependency management")
        else:
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                str(self.src_dir / 'app.py'),
                '--server.port', str(self.port),
                '--server.address', self.host,
            ]
            self.logger.info("Using standard python -m streamlit")
        
        if not self.dev_mode:
            cmd.extend(['--server.headless', 'true'])
        
        if self.dev_mode:
            cmd.extend(['--server.runOnSave', 'true'])
        
        # Start Streamlit process
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(process)
            
            # Monitor output in separate thread
            def monitor_output():
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        self.logger.info(f"Streamlit: {line.strip()}")
                    if self.shutdown_requested:
                        break
            
            output_thread = threading.Thread(target=monitor_output, daemon=True)
            output_thread.start()
            
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to start Streamlit: {e}")
            return None
    
    def wait_for_startup(self, timeout: int = 30):
        """Wait for services to be ready."""
        import requests
        
        url = f"http://{self.host}:{self.port}"
        self.logger.info(f"Waiting for services to be ready at {url}")
        
        for i in range(timeout):
            if self.shutdown_requested:
                return False
            
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    self.logger.info("Services are ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            if i % 5 == 0:
                print(f"  Waiting... ({i}/{timeout}s)")
        
        self.logger.warning("Services may not be fully ready")
        return False
    
    def open_browser_tab(self):
        """Open browser tab to the dashboard."""
        if not self.open_browser:
            return
        
        url = f"http://localhost:{self.port}"
        self.logger.info(f"Opening browser to {url}")
        
        try:
            webbrowser.open(url)
        except Exception as e:
            self.logger.warning(f"Could not open browser: {e}")
    
    def monitor_processes(self):
        """Monitor running processes and handle failures."""
        while not self.shutdown_requested:
            for process in self.processes[:]:  # Copy list to avoid modification during iteration
                if process.poll() is not None:  # Process has terminated
                    self.logger.error(f"Process {process.pid} has terminated unexpectedly")
                    self.processes.remove(process)
                    
                    if not self.shutdown_requested:
                        self.logger.error("Critical process failure - initiating shutdown")
                        self.shutdown_requested = True
                        break
            
            time.sleep(2)
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        if self.shutdown_requested:
            return
        
        self.shutdown_requested = True
        self.logger.info("Initiating graceful shutdown...")
        
        # Terminate all processes
        for process in self.processes:
            try:
                self.logger.info(f"Terminating process {process.pid}")
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing process {process.pid}")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                self.logger.error(f"Error terminating process {process.pid}: {e}")
        
        self.processes.clear()
        self.logger.info("Shutdown complete")
    
    def run(self):
        """Main run method."""
        try:
            # Print banner
            self.print_banner()
            
            # Run system checks
            checks = self.run_system_checks()
            if not self.display_system_checks(checks):
                print(f"\n{Colors.colorize('❌ Critical system checks failed. Please fix the issues above.', Colors.RED)}")
                return 1
            
            print(f"{Colors.colorize('✅ All system checks passed!', Colors.GREEN)}")
            print()
            
            # Setup environment
            self.setup_environment()
            
            # Start services
            print(f"{Colors.colorize('🚀 Starting IronSight services...', Colors.CYAN)}")
            
            self.start_backend_services()
            frontend_process = self.start_frontend()
            
            if not frontend_process:
                self.logger.error("Failed to start frontend")
                return 1
            
            # Wait for startup
            if self.wait_for_startup():
                self.open_browser_tab()
            
            # Display ready message
            print()
            print(f"{Colors.colorize('🎉 IronSight Command Center is ready!', Colors.GREEN + Colors.BOLD)}")
            print(f"   Dashboard URL: {Colors.colorize(f'http://localhost:{self.port}', Colors.CYAN + Colors.UNDERLINE)}")
            print(f"   Press {Colors.colorize('Ctrl+C', Colors.YELLOW)} to stop")
            print()
            
            # Monitor processes
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            # Keep main thread alive
            while not self.shutdown_requested:
                time.sleep(1)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            self.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IronSight Command Center - Unified Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ironsight.py                    # Standard production mode
  python run_ironsight.py --dev              # Development mode
  python run_ironsight.py --cpu --port 8080  # CPU mode on port 8080
        """
    )
    
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with auto-reload')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU-only mode (disable GPU)')
    parser.add_argument('--port', type=int, default=8501,
                       help='Set Streamlit port (default: 8501)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Set host address (default: 0.0.0.0)')
    parser.add_argument('--no-browser', action='store_true',
                       help="Don't open browser automatically")
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create launcher
    launcher = IronSightLauncher()
    
    # Configure from arguments
    launcher.dev_mode = args.dev
    launcher.cpu_only = args.cpu
    launcher.port = args.port
    launcher.host = args.host
    launcher.open_browser = not args.no_browser
    launcher.debug = args.debug
    
    # Update logging level if debug enabled
    if launcher.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the launcher
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
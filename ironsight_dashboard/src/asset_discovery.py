"""
Asset Discovery and Configuration Management

Scans the aidtm/ folder for existing config files, fonts, and utility scripts.
Imports existing configuration and sets up logging and monitoring.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path for accessing existing aidtm modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


@dataclass
class AssetInventory:
    """Inventory of discovered assets from aidtm folder."""
    config_files: List[Path] = field(default_factory=list)
    font_files: List[Path] = field(default_factory=list)
    utility_scripts: List[Path] = field(default_factory=list)
    model_files: List[Path] = field(default_factory=list)
    data_directories: List[Path] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert inventory to dictionary for serialization."""
        return {
            'config_files': [str(p) for p in self.config_files],
            'font_files': [str(p) for p in self.font_files],
            'utility_scripts': [str(p) for p in self.utility_scripts],
            'model_files': [str(p) for p in self.model_files],
            'data_directories': [str(p) for p in self.data_directories]
        }


class AssetDiscoveryManager:
    """Manages discovery and configuration of existing aidtm assets."""
    
    def __init__(self, aidtm_root: Optional[Path] = None):
        """
        Initialize asset discovery manager.
        
        Args:
            aidtm_root: Root path to aidtm folder. If None, uses parent directory.
        """
        self.aidtm_root = aidtm_root or parent_dir
        self.logger = self._setup_logging()
        self.inventory = AssetInventory()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration based on existing codebase patterns."""
        logger = logging.getLogger(__name__)
        
        # Configure logging format similar to existing codebase
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)  # Set to DEBUG for testing
        
        return logger
    
    def scan_assets(self) -> AssetInventory:
        """
        Scan aidtm folder for existing assets.
        
        Returns:
            AssetInventory containing discovered assets
        """
        self.logger.info(f"Scanning assets in: {self.aidtm_root}")
        
        # Reset inventory
        self.inventory = AssetInventory()
        
        # Scan for different asset types
        self._scan_config_files()
        self._scan_font_files()
        self._scan_utility_scripts()
        self._scan_model_files()
        self._scan_data_directories()
        
        self.logger.info(f"Asset scan complete:")
        self.logger.info(f"  - Config files: {len(self.inventory.config_files)}")
        self.logger.info(f"  - Font files: {len(self.inventory.font_files)}")
        self.logger.info(f"  - Utility scripts: {len(self.inventory.utility_scripts)}")
        self.logger.info(f"  - Model files: {len(self.inventory.model_files)}")
        self.logger.info(f"  - Data directories: {len(self.inventory.data_directories)}")
        
        return self.inventory
    
    def _scan_config_files(self):
        """Scan for configuration files (YAML, JSON, TOML)."""
        config_patterns = ['*.yaml', '*.yml', '*.json', '*.toml']
        
        for pattern in config_patterns:
            for config_file in self.aidtm_root.rglob(pattern):
                # Skip hidden directories and cache files
                if any(part.startswith('.') for part in config_file.parts):
                    continue
                if '__pycache__' in str(config_file):
                    continue
                    
                self.inventory.config_files.append(config_file)
                self.logger.debug(f"Found config: {config_file}")
    
    def _scan_font_files(self):
        """Scan for font files (TTF, OTF, WOFF)."""
        font_patterns = ['*.ttf', '*.otf', '*.woff', '*.woff2']
        
        for pattern in font_patterns:
            for font_file in self.aidtm_root.rglob(pattern):
                # Skip hidden directories
                if any(part.startswith('.') for part in font_file.parts):
                    continue
                    
                self.inventory.font_files.append(font_file)
                self.logger.debug(f"Found font: {font_file}")
    
    def _scan_utility_scripts(self):
        """Scan for utility scripts in scripts/ and src/ directories."""
        script_dirs = ['scripts', 'src']
        
        for script_dir in script_dirs:
            script_path = self.aidtm_root / script_dir
            self.logger.debug(f"Checking script directory: {script_path}")
            if script_path.exists():
                self.logger.debug(f"Script directory exists: {script_path}")
                for script_file in script_path.glob('*.py'):
                    self.logger.debug(f"Found Python file: {script_file}")
                    # Skip __pycache__ and test files
                    if '__pycache__' in str(script_file):
                        self.logger.debug(f"Skipping __pycache__ file: {script_file}")
                        continue
                    if script_file.name.startswith('test_'):
                        self.logger.debug(f"Skipping test file: {script_file}")
                        continue
                        
                    self.inventory.utility_scripts.append(script_file)
                    self.logger.debug(f"Added script: {script_file}")
            else:
                self.logger.debug(f"Script directory does not exist: {script_path}")
    
    def _scan_model_files(self):
        """Scan for model files (PTH, PT, ONNX, BIN)."""
        model_patterns = ['*.pth', '*.pt', '*.onnx', '*.bin']
        
        for pattern in model_patterns:
            for model_file in self.aidtm_root.rglob(pattern):
                # Skip hidden directories and cache files
                if any(part.startswith('.') for part in model_file.parts):
                    continue
                if '__pycache__' in str(model_file):
                    continue
                    
                self.inventory.model_files.append(model_file)
                self.logger.debug(f"Found model: {model_file}")
    
    def _scan_data_directories(self):
        """Scan for data directories."""
        data_dir_names = ['data', 'datasets', 'images', 'models']
        
        for dir_name in data_dir_names:
            for data_dir in self.aidtm_root.rglob(dir_name):
                if data_dir.is_dir():
                    # Skip hidden directories
                    if any(part.startswith('.') for part in data_dir.parts):
                        continue
                        
                    self.inventory.data_directories.append(data_dir)
                    self.logger.debug(f"Found data dir: {data_dir}")
    
    def load_vehicle_detection_config(self) -> Dict[str, Any]:
        """
        Load existing vehicle detection configuration.
        
        Returns:
            Parsed configuration dictionary
        """
        config_path = self.aidtm_root / 'config' / 'vehicle_detection.yaml'
        
        if not config_path.exists():
            self.logger.warning(f"Vehicle detection config not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded vehicle detection config: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load vehicle detection config: {e}")
            return {}
    
    def create_ironsight_config(self) -> Dict[str, Any]:
        """
        Create IronSight configuration based on discovered assets.
        
        Returns:
            IronSight configuration dictionary
        """
        vehicle_config = self.load_vehicle_detection_config()
        
        # Create IronSight configuration
        ironsight_config = {
            'engine': {
                'target_fps': 60,
                'gpu_memory_fraction': 0.8,
                'use_fp16': True,
                'smolvlm_quantization_bits': 8
            },
            'models': {
                'gatekeeper': {
                    'path': 'models/gatekeeper.onnx',
                    'timeout_ms': 0.5,
                    'input_size': [64, 64]
                },
                'sci_enhancer': {
                    'path': 'models/sci_enhancer.onnx',
                    'timeout_ms': 0.5,
                    'brightness_threshold': 50
                },
                'yolo_sideview': {
                    'path': 'models/yolo_sideview_damage_obb_extended.pt',
                    'timeout_ms': 7.0
                },
                'yolo_structure': {
                    'path': 'models/yolo_structure_obb.pt',
                    'timeout_ms': 7.0
                },
                'yolo_wagon_number': {
                    'path': 'models/yolo_wagon_number_obb.pt',
                    'timeout_ms': 6.0
                },
                'nafnet': {
                    'path': str(self._find_nafnet_model()),
                    'timeout_ms': 20.0,
                    'crop_padding_percent': 0.1
                }
            },
            'dashboard': {
                'theme': 'dark_industrial',
                'enable_performance_monitoring': True,
                'tabs': ['mission_control', 'restoration_lab', 'semantic_search']
            },
            'assets': self.inventory.to_dict()
        }
        
        # Merge vehicle detection config if available
        if vehicle_config:
            ironsight_config['vehicle_detection'] = vehicle_config
        
        return ironsight_config
    
    def _find_nafnet_model(self) -> Path:
        """Find NAFNet model file in discovered assets."""
        nafnet_candidates = [
            'NAFNet-REDS-width64.pth',
            'NAFNet-GoPro-width64.pth',  # Keep as fallback
            'nafnet.pth',
            'nafnet_gopro.pth'
        ]
        
        for model_file in self.inventory.model_files:
            if model_file.name in nafnet_candidates:
                return model_file
        
        # Default path if not found
        return self.aidtm_root / 'NAFNet-REDS-width64.pth'
    
    def save_config(self, config: Dict[str, Any], output_path: Path):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path to save configuration file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def setup_logging_config(self) -> Dict[str, Any]:
        """
        Set up logging configuration based on existing codebase patterns.
        
        Returns:
            Logging configuration dictionary
        """
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': 'logs/ironsight.log',
                    'mode': 'a'
                }
            },
            'loggers': {
                'ironsight': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            }
        }
        
        return logging_config


def main():
    """Main function for testing asset discovery."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create asset discovery manager
    manager = AssetDiscoveryManager()
    
    # Scan assets
    inventory = manager.scan_assets()
    
    # Create configuration
    config = manager.create_ironsight_config()
    
    # Save configuration
    config_path = Path(__file__).parent.parent / 'config' / 'ironsight_config.yaml'
    manager.save_config(config, config_path)
    
    print(f"Asset discovery complete. Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
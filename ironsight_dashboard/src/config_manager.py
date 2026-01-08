"""
Configuration Manager for IronSight Command Center

Manages loading and validation of configuration files, including integration
with existing aidtm configurations and logging setup.
"""

import os
import sys
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for accessing existing aidtm modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    path: str
    timeout_ms: float
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(
            path=data['path'],
            timeout_ms=data['timeout_ms'],
            enabled=data.get('enabled', True)
        )


@dataclass
class EngineConfig:
    """Configuration for IronSight Engine."""
    target_fps: int
    gpu_memory_fraction: float
    use_fp16: bool
    smolvlm_quantization_bits: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineConfig':
        """Create EngineConfig from dictionary."""
        return cls(
            target_fps=data['target_fps'],
            gpu_memory_fraction=data['gpu_memory_fraction'],
            use_fp16=data['use_fp16'],
            smolvlm_quantization_bits=data['smolvlm_quantization_bits']
        )


class ConfigManager:
    """Manages IronSight configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.logger = self._setup_basic_logging()
        
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        return Path(__file__).parent.parent / 'config' / 'ironsight_config.yaml'
    
    def _setup_basic_logging(self) -> logging.Logger:
        """Set up basic logging before full configuration is loaded."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Loaded configuration dictionary
        """
        if not self.config_path.exists():
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded from: {self.config_path}")
            
            # Set up full logging configuration
            self._setup_logging()
            
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_logging(self):
        """Set up full logging configuration from config file."""
        if 'logging' in self.config:
            try:
                # Create logs directory if it doesn't exist
                logs_dir = Path(__file__).parent.parent / 'logs'
                logs_dir.mkdir(exist_ok=True)
                
                # Configure logging
                logging.config.dictConfig(self.config['logging'])
                
                # Update logger reference
                self.logger = logging.getLogger('ironsight.config')
                self.logger.info("Full logging configuration applied")
                
            except Exception as e:
                self.logger.warning(f"Failed to apply logging configuration: {e}")
    
    def get_engine_config(self) -> EngineConfig:
        """Get engine configuration."""
        if 'engine' not in self.config:
            raise ValueError("Engine configuration not found")
        
        return EngineConfig.from_dict(self.config['engine'])
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for specific model.
        
        Args:
            model_name: Name of the model (e.g., 'gatekeeper', 'nafnet')
            
        Returns:
            ModelConfig for the specified model
        """
        if 'models' not in self.config:
            raise ValueError("Models configuration not found")
        
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        return ModelConfig.from_dict(self.config['models'][model_name])
    
    def get_model_paths(self) -> Dict[str, str]:
        """
        Get all model paths from configuration.
        
        Returns:
            Dictionary mapping model names to their file paths
        """
        if 'models' not in self.config:
            return {}
        
        model_paths = {}
        for model_name, model_config in self.config['models'].items():
            model_paths[model_name] = model_config['path']
        
        return model_paths
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get('dashboard', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration."""
        return self.config.get('performance', {})
    
    def get_vehicle_detection_config(self) -> Dict[str, Any]:
        """Get vehicle detection configuration (imported from existing config)."""
        return self.config.get('vehicle_detection', {})
    
    def get_asset_paths(self) -> Dict[str, Any]:
        """Get discovered asset paths."""
        return self.config.get('assets', {})
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration for existing codebase."""
        return self.config.get('integration', {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration completeness and correctness.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['engine', 'models', 'dashboard']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Required configuration section missing: {section}")
                return False
        
        # Validate engine config
        try:
            engine_config = self.get_engine_config()
            if engine_config.target_fps <= 0:
                self.logger.error("Invalid target_fps: must be positive")
                return False
            
            if not (0.0 < engine_config.gpu_memory_fraction <= 1.0):
                self.logger.error("Invalid gpu_memory_fraction: must be between 0 and 1")
                return False
                
        except Exception as e:
            self.logger.error(f"Invalid engine configuration: {e}")
            return False
        
        # Validate model configs
        required_models = ['gatekeeper', 'sci_enhancer', 'nafnet']
        for model_name in required_models:
            try:
                model_config = self.get_model_config(model_name)
                if model_config.timeout_ms <= 0:
                    self.logger.error(f"Invalid timeout for {model_name}: must be positive")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Invalid configuration for {model_name}: {e}")
                return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def resolve_model_path(self, model_name: str) -> Path:
        """
        Resolve model path relative to configuration file or project root.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Resolved absolute path to model file
        """
        model_config = self.get_model_config(model_name)
        model_path = Path(model_config.path)
        
        # If path is relative, resolve relative to config file directory
        if not model_path.is_absolute():
            config_dir = self.config_path.parent
            model_path = config_dir / model_path
        
        return model_path.resolve()
    
    def get_existing_module_path(self, module_name: str) -> Optional[Path]:
        """
        Get path to existing aidtm module for integration.
        
        Args:
            module_name: Name of the module (e.g., 'pipeline_core', 'agent_forensic')
            
        Returns:
            Path to existing module or None if not found
        """
        integration_config = self.get_integration_config()
        
        if module_name + '_path' in integration_config:
            module_path = Path(integration_config[module_name + '_path'])
            
            # Resolve relative to config file
            if not module_path.is_absolute():
                config_dir = self.config_path.parent
                module_path = config_dir / module_path
            
            if module_path.exists():
                return module_path.resolve()
        
        return None


def load_config(config_path: Optional[Path] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ConfigManager instance
    """
    manager = ConfigManager(config_path)
    manager.load_config()
    
    if not manager.validate_config():
        raise ValueError("Configuration validation failed")
    
    return manager


def main():
    """Main function for testing configuration loading."""
    try:
        # Load configuration
        config_manager = load_config()
        
        print("Configuration loaded successfully!")
        print(f"Target FPS: {config_manager.get_engine_config().target_fps}")
        print(f"Model paths: {config_manager.get_model_paths()}")
        
        # Test model path resolution
        nafnet_path = config_manager.resolve_model_path('nafnet')
        print(f"NAFNet path: {nafnet_path}")
        print(f"NAFNet exists: {nafnet_path.exists()}")
        
    except Exception as e:
        print(f"Configuration loading failed: {e}")


if __name__ == "__main__":
    main()
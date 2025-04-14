"""
Configuration handling for the logging system.

This module provides functions for loading, validating, and applying
logging configurations.
"""
import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    Load logging configuration from a file.
    
    Args:
        config_file (str): Path to configuration file (JSON or YAML)
        
    Returns:
        dict: Loaded configuration
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Logging config file not found: {config_file}")
    
    # Load based on file extension
    file_ext = Path(config_file).suffix.lower()
    
    if file_ext == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    elif file_ext in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ValueError("PyYAML is required for YAML configuration files")
    else:
        raise ValueError(f"Unsupported config file format: {file_ext}")
    
    # Validate and return
    validate_config(config)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate logging configuration.
    
    Args:
        config (dict): Logging configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Basic validation
    if not isinstance(config, dict):
        raise ValueError("Logging configuration must be a dictionary")
    
    # Check required keys
    if 'version' not in config:
        raise ValueError("Logging configuration missing 'version' key")
    
    # Check formatters if present
    if 'formatters' in config and not isinstance(config['formatters'], dict):
        raise ValueError("'formatters' must be a dictionary")
    
    # Check handlers if present
    if 'handlers' in config and not isinstance(config['handlers'], dict):
        raise ValueError("'handlers' must be a dictionary")
    
    # Check loggers if present
    if 'loggers' in config and not isinstance(config['loggers'], dict):
        raise ValueError("'loggers' must be a dictionary")

def apply_config(config: Dict[str, Any]) -> None:
    """
    Apply logging configuration.
    
    Args:
        config (dict): Logging configuration dictionary
    """
    # Apply configuration
    try:
        logging.config.dictConfig(config)
    except Exception as e:
        # Fall back to basic configuration
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to apply logging configuration: {str(e)}")
        
def get_default_config() -> Dict[str, Any]:
    """
    Get default logging configuration.
    
    Returns:
        dict: Default configuration
    """
    return {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'brief': {
                'format': '%(asctime)s [%(levelname)s] %(message)s',
                'datefmt': '%H:%M:%S'
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
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/trading.log',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': True
        },
        'loggers': {
            'trading': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }

def create_logger_config(
    name: str, 
    level: str = 'INFO',
    handlers: Optional[list] = None,
    propagate: bool = False
) -> Dict[str, Any]:
    """
    Create a logger configuration.
    
    Args:
        name (str): Logger name
        level (str, optional): Log level
        handlers (list, optional): List of handler names
        propagate (bool, optional): Whether to propagate to parent
        
    Returns:
        dict: Logger configuration
    """
    return {
        'level': level,
        'handlers': handlers or ['console'],
        'propagate': propagate
    }

def create_handler_config(
    handler_type: str,
    level: str = 'INFO',
    formatter: str = 'standard',
    **kwargs
) -> Dict[str, Any]:
    """
    Create a handler configuration.
    
    Args:
        handler_type (str): Handler type ('console', 'file', etc.)
        level (str, optional): Log level
        formatter (str, optional): Formatter name
        **kwargs: Additional handler-specific parameters
        
    Returns:
        dict: Handler configuration
    """
    config = {
        'level': level,
        'formatter': formatter
    }
    
    if handler_type == 'console':
        config['class'] = 'logging.StreamHandler'
        config['stream'] = kwargs.get('stream', 'ext://sys.stdout')
    elif handler_type == 'file':
        config['class'] = 'logging.handlers.RotatingFileHandler'
        config['filename'] = kwargs.get('filename', 'logs/app.log')
        config['maxBytes'] = kwargs.get('max_bytes', 10485760)  # 10 MB
        config['backupCount'] = kwargs.get('backup_count', 5)
    elif handler_type == 'syslog':
        config['class'] = 'logging.handlers.SysLogHandler'
        config['address'] = kwargs.get('address', ('localhost', 514))
        config['facility'] = kwargs.get('facility', 'local1')
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")
    
    return config

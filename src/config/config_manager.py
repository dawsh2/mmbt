# config/config_manager.py

"""
Main configuration manager for the trading system.
"""

class ConfigManager:
    """
    Central configuration management for the trading system.
    """
    
    def __init__(self, config_dict=None, config_file=None):
        """
        Initialize with configuration from dictionary or file.
        
        Args:
            config_dict: Optional configuration dictionary
            config_file: Optional path to JSON/YAML config file
        """
        self.config = {}
        
        # Load defaults
        self._load_defaults()
        
        # Override with provided config
        if config_file:
            self._load_from_file(config_file)
            
        if config_dict:
            self._update_config(config_dict)
            
        # Validate configuration
        self.validate()
            
    def _load_defaults(self):
        """Load default configuration values."""
        from .defaults import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()
    
    def _load_from_file(self, config_file):
        """Load configuration from file."""
        import os
        
        _, ext = os.path.splitext(config_file)
        
        if ext.lower() == '.json':
            self._load_from_json(config_file)
        elif ext.lower() in ('.yaml', '.yml'):
            self._load_from_yaml(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    
    def _load_from_json(self, json_file):
        """Load configuration from JSON file."""
        import json
        
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
                
            self._update_config(config)
        except Exception as e:
            raise ValueError(f"Error loading JSON config: {str(e)}")
    
    def _load_from_yaml(self, yaml_file):
        """Load configuration from YAML file."""
        try:
            import yaml
            
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self._update_config(config)
        except ImportError:
            raise ImportError("PyYAML not installed. Install with 'pip install pyyaml'")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {str(e)}")
    
    def _update_config(self, config_dict):
        """Update configuration with new values."""
        self._recursive_update(self.config, config_dict)
    
    def _recursive_update(self, target, source):
        """Recursively update nested dictionaries."""
        for key, value in source.items():
            # If both are dicts, recurse
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._recursive_update(target[key], value)
            else:
                # Otherwise simply overwrite
                target[key] = value
    
    def validate(self):
        """Validate the configuration."""
        from .validators import validate_config
        
        try:
            validate_config(self.config)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def get(self, path, default=None):
        """
        Get a configuration value using dot notation path.
        
        Args:
            path: Dot notation path (e.g., "backtester.market_simulation.slippage_model")
            default: Default value if path not found
            
        Returns:
            The configuration value or default
        """
        parts = path.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, path, value):
        """
        Set a configuration value using dot notation path.
        
        Args:
            path: Dot notation path
            value: Value to set
        """
        parts = path.split('.')
        current = self.config
        
        # Navigate to the parent of the target
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
        
        # Validate after changing
        self.validate()
    
    def save(self, file_path):
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save the config
        """
        import os
        
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.json':
            self._save_to_json(file_path)
        elif ext.lower() in ('.yaml', '.yml'):
            self._save_to_yaml(file_path)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    
    def _save_to_json(self, file_path):
        """Save configuration to JSON file."""
        import json
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving JSON config: {str(e)}")
    
    def _save_to_yaml(self, file_path):
        """Save configuration to YAML file."""
        try:
            import yaml
            
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML not installed. Install with 'pip install pyyaml'")
        except Exception as e:
            raise ValueError(f"Error saving YAML config: {str(e)}")
    
    def to_dict(self):
        """
        Get the full configuration as a dictionary.
        
        Returns:
            dict: Complete configuration
        """
        return self.config.copy()

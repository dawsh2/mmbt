# config/validators.py

"""
Validation utilities for configuration.
"""

def validate_config(config):
    """
    Validate the configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If validation fails
    """
    from .schema import CONFIG_SCHEMA
    
    try:
        validate_section(config, CONFIG_SCHEMA)
    except Exception as e:
        raise ValueError(f"Config validation error: {str(e)}")

def validate_section(config_section, schema_section, path=""):
    """
    Recursively validate a configuration section.
    
    Args:
        config_section: Configuration section to validate
        schema_section: Schema section to validate against
        path: Current path for error messages
    """
    # Check type
    if 'type' in schema_section:
        expected_type = schema_section['type']
        
        # Check for correct type
        if expected_type == 'dict':
            if not isinstance(config_section, dict):
                raise ValueError(f"{path}: Expected dictionary, got {type(config_section).__name__}")
        elif expected_type == 'list':
            if not isinstance(config_section, list):
                raise ValueError(f"{path}: Expected list, got {type(config_section).__name__}")
        elif expected_type == 'int':
            if not isinstance(config_section, int):
                raise ValueError(f"{path}: Expected integer, got {type(config_section).__name__}")
        elif expected_type == 'float':
            if not isinstance(config_section, (int, float)):
                raise ValueError(f"{path}: Expected number, got {type(config_section).__name__}")
        elif expected_type == 'str':
            if not isinstance(config_section, str):
                raise ValueError(f"{path}: Expected string, got {type(config_section).__name__}")
        elif expected_type == 'bool':
            if not isinstance(config_section, bool):
                raise ValueError(f"{path}: Expected boolean, got {type(config_section).__name__}")
    
    # Check required fields
    if 'required' in schema_section and isinstance(config_section, dict):
        for field in schema_section['required']:
            if field not in config_section:
                raise ValueError(f"{path}: Missing required field '{field}'")
    
    # Check constraints
    if 'min' in schema_section:
        if isinstance(config_section, (int, float)) and config_section < schema_section['min']:
            raise ValueError(f"{path}: Value {config_section} is less than minimum {schema_section['min']}")
        elif isinstance(config_section, (list, str)) and len(config_section) < schema_section['min']:
            raise ValueError(f"{path}: Length {len(config_section)} is less than minimum {schema_section['min']}")
            
    if 'max' in schema_section:
        if isinstance(config_section, (int, float)) and config_section > schema_section['max']:
            raise ValueError(f"{path}: Value {config_section} is greater than maximum {schema_section['max']}")
        elif isinstance(config_section, (list, str)) and len(config_section) > schema_section['max']:
            raise ValueError(f"{path}: Length {len(config_section)} is greater than maximum {schema_section['max']}")
            
    if 'enum' in schema_section and config_section not in schema_section['enum']:
        raise ValueError(f"{path}: Value {config_section} not in allowed values {schema_section['enum']}")
        
    # Recursively validate children
    if 'properties' in schema_section and isinstance(config_section, dict):
        for key, schema in schema_section['properties'].items():
            if key in config_section:
                new_path = f"{path}.{key}" if path else key
                validate_section(config_section[key], schema, new_path)
                
    # Validate array items
    if 'items' in schema_section and isinstance(config_section, list):
        for i, item in enumerate(config_section):
            new_path = f"{path}[{i}]"
            validate_section(item, schema_section['items'], new_path)

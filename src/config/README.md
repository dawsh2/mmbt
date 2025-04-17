# Config Module

Configuration management for the trading system.

This module provides a centralized way to manage configuration for all components
of the trading system, with validation and defaults.

## Contents

- [config_manager](#config_manager)
- [defaults](#defaults)
- [example_usage](#example_usage)
- [schema](#schema)
- [validators](#validators)

## config_manager

Main configuration manager for the trading system.

### Classes

#### `ConfigManager`

Central configuration management for the trading system.

##### Methods

###### `__init__(config_dict=None, config_file=None)`

Initialize with configuration from dictionary or file.

Args:
    config_dict: Optional configuration dictionary
    config_file: Optional path to JSON/YAML config file

###### `_load_defaults()`

Load default configuration values.

###### `_load_from_file(config_file)`

Load configuration from file.

###### `_load_from_json(json_file)`

Load configuration from JSON file.

###### `_load_from_yaml(yaml_file)`

Load configuration from YAML file.

###### `_update_config(config_dict)`

Update configuration with new values.

###### `_recursive_update(target, source)`

Recursively update nested dictionaries.

###### `validate()`

Validate the configuration.

###### `get(path, default=None)`

Get a configuration value using dot notation path.

Args:
    path: Dot notation path (e.g., "backtester.market_simulation.slippage_model")
    default: Default value if path not found
    
Returns:
    The configuration value or default

*Returns:* The configuration value or default

###### `set(path, value)`

Set a configuration value using dot notation path.

Args:
    path: Dot notation path
    value: Value to set

###### `save(file_path)`

Save configuration to file.

Args:
    file_path: Path to save the config

###### `_save_to_json(file_path)`

Save configuration to JSON file.

###### `_save_to_yaml(file_path)`

Save configuration to YAML file.

###### `to_dict()`

Get the full configuration as a dictionary.

Returns:
    dict: Complete configuration

*Returns:* dict: Complete configuration

## defaults

Default configuration values.

## example_usage

Example usage of the ConfigManager.

### Functions

#### `main()`

No docstring provided.

## schema

Configuration schema for validation.

## validators

Validation utilities for configuration.

### Functions

#### `validate_config(config)`

Validate the configuration against schema.

Args:
    config: Configuration dictionary to validate
    
Raises:
    ValueError: If validation fails

#### `validate_section(config_section, schema_section, path='')`

Recursively validate a configuration section.

Args:
    config_section: Configuration section to validate
    schema_section: Schema section to validate against
    path: Current path for error messages

# Trading System Configuration Module

## Overview

This module provides a unified configuration management system for all components of the trading system. It includes:

- Centralized configuration management via the `ConfigManager` class
- Configuration validation against a defined schema
- Default values for all settings
- Support for loading from and saving to JSON/YAML files
- Simple API for getting and setting configuration values

## Directory Structure

```
config/
├── __init__.py             # Package exports
├── config_manager.py       # Main configuration manager
├── validators.py           # Configuration validation
├── defaults.py             # Default configurations
└── schema.py               # Configuration schema definitions
```

## Usage

### Basic Usage

```python
from config import ConfigManager

# Create config manager with defaults
config = ConfigManager()

# Get values using dot notation
slippage_model = config.get('backtester.market_simulation.slippage_model')
initial_capital = config.get('backtester.initial_capital')

# Update configuration values
config.set('backtester.initial_capital', 200000)
config.set('position_management.position_sizing.method', 'volatility')
```

### Loading from Files

```python
# Load from JSON file
config = ConfigManager(config_file="trading_config.json")

# Load from YAML file
config = ConfigManager(config_file="trading_config.yaml")
```

### Overriding with Custom Dictionary

```python
custom_config = {
    'backtester': {
        'initial_capital': 500000,
        'market_simulation': {
            'slippage_model': 'volume'
        }
    }
}

# Override defaults with custom values
config = ConfigManager(config_dict=custom_config)
```

### Saving Configuration

```python
# Save to JSON
config.save("trading_config.json")

# Save to YAML
config.save("trading_config.yaml")
```

### Access Full Configuration

```python
# Get the complete configuration as a dictionary
full_config = config.to_dict()
```

## Configuration Schema

The configuration is validated against a schema defined in `schema.py`. The schema specifies:

- The type of each configuration value
- Allowed value ranges (min/max)
- Enumerated options for string values
- Default values

### Main Configuration Sections

1. **Backtester**
   - Market simulation parameters (slippage, fees)
   - Initial capital
   - EOD position handling

2. **Position Management**
   - Position sizing methods
   - Risk management parameters
   - Asset allocation

3. **Signals**
   - Signal processing options
   - Filtering and transformation
   - Confidence scoring

4. **Regime Detection**
   - Detector types and parameters
   - Trend and volatility thresholds
   - Composite regime detection

## Validation

All configuration values are validated when:
- The ConfigManager is initialized
- A new value is set via the `set()` method
- Configuration is loaded from a file

Validation ensures:
- Types are correct
- Values are within specified ranges
- Required fields are present
- Enumerated values are valid

If validation fails, a `ValueError` is raised with details about the validation error.

## Error Handling

The module provides clear error messages for:
- Configuration validation failures
- File loading/saving errors
- Invalid configuration paths

## Examples

See [example_usage.py](example_usage.py) for practical examples of how to use the configuration module.
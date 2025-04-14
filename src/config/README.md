# Trading System Configuration Module

## Overview

The Configuration module provides a unified system for managing configuration across all components of the trading system. It enables loading, validating, modifying, and saving configuration settings with a clean API that ensures consistency and type safety.

Key features:
- Centralized configuration management via the `ConfigManager` class
- Schema-based validation with type checking and constraints
- Default values for all settings
- Support for loading from and saving to JSON/YAML files
- Simple dot notation API for accessing and modifying settings

## Directory Structure

```
config/
├── __init__.py             # Package exports
├── config_manager.py       # Main configuration manager
├── validators.py           # Configuration validation
├── defaults.py             # Default configurations
└── schema.py               # Configuration schema definitions
```

## Basic Usage

### Creating a Configuration Manager

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

## Complete Configuration Reference

The configuration is structured in sections that correspond to different components of the trading system. Below is a complete reference of all available configuration options, their types, constraints, and default values.

### Backtester Configuration

This section configures the backtesting engine behavior.

```python
{
    'backtester': {
        'market_simulation': {
            'slippage_model': 'fixed',     # Options: 'none', 'fixed', 'volume'
            'slippage_bps': 5,             # Range: 0-100, basis points of slippage
            'price_impact': 0.1,           # Range: 0-1, impact factor
            'fee_model': 'fixed',          # Options: 'none', 'fixed', 'tiered'
            'fee_bps': 10                  # Range: 0-100, basis points of fees
        },
        'initial_capital': 100000,         # Range: ≥1000, starting capital
        'close_positions_eod': True        # Boolean, whether to close positions at end of day
    }
}
```

#### Slippage Models

- `none`: No slippage applied
- `fixed`: Fixed basis points slippage applied to every trade
- `volume`: Volume-dependent slippage that scales with position size

#### Fee Models

- `none`: No trading fees
- `fixed`: Fixed basis points fee applied to every trade
- `tiered`: Tiered fee structure based on volume

### Position Management Configuration

This section configures how positions are sized, managed, and allocated.

```python
{
    'position_management': {
        'position_sizing': {
            'method': 'percent_equity',    # Options: 'fixed', 'percent_equity', 'volatility', 'kelly'
            'percent': 0.02,               # Range: 0.001-1.0, percent of equity per position
            'fixed_size': 100,             # Range: ≥1, fixed position size in units
            'risk_pct': 0.01,              # Range: 0.001-0.1, risk per trade
            'kelly_fraction': 0.5          # Range: 0.1-1.0, fraction of Kelly criterion
        },
        'risk_management': {
            'max_position_pct': 0.25,      # Range: 0.01-1.0, maximum position size as % of equity
            'max_drawdown_pct': 0.10,      # Range: 0.01-0.5, maximum allowed drawdown
            'use_stop_loss': False,        # Boolean, whether to use stop losses
            'stop_loss_pct': 0.05,         # Range: 0.01-0.5, stop loss percentage
            'use_take_profit': False,      # Boolean, whether to use take profit
            'take_profit_pct': 0.10        # Range: 0.01-0.5, take profit percentage
        },
        'allocation': {
            'method': 'equal',             # Options: 'equal', 'volatility_parity', 'optimize'
            'max_instruments': 10          # Range: ≥1, maximum number of instruments to trade
        }
    }
}
```

#### Position Sizing Methods

- `fixed`: Use a fixed number of units for each position
- `percent_equity`: Size each position as a percentage of account equity
- `volatility`: Size positions based on volatility to maintain consistent risk
- `kelly`: Use the Kelly criterion to determine optimal position size

#### Allocation Methods

- `equal`: Equal weighting across all instruments
- `volatility_parity`: Allocate to maintain equal volatility contribution
- `optimize`: Use optimization to determine allocation weights

### Signal Processing Configuration

This section configures how trading signals are processed and evaluated.

```python
{
    'signals': {
        'processing': {
            'use_filtering': False,        # Boolean, whether to filter signals
            'filter_type': 'moving_average', # Options: 'moving_average', 'kalman', 'exponential'
            'window_size': 5,              # Range: 2-100, window size for filtering
            'use_transformations': False,  # Boolean, whether to transform signals
            'wavelet_type': 'db1',         # Options: 'db1', 'db4', 'sym4', 'haar'
            'wavelet_level': 3             # Range: 1-6, wavelet decomposition level
        },
        'confidence': {
            'use_confidence_score': False, # Boolean, whether to apply confidence scoring
            'min_confidence': 0.5,         # Range: 0-1, minimum confidence threshold
            'prior_accuracy': 0.5          # Range: 0.1-0.9, prior belief of signal accuracy
        }
    }
}
```

#### Signal Filtering Methods

- `moving_average`: Simple moving average filter to smooth signals
- `kalman`: Kalman filter for adaptive signal smoothing
- `exponential`: Exponential smoothing filter

### Regime Detection Configuration

This section configures market regime detection methods and parameters.

```python
{
    'regime_detection': {
        'detector_type': 'trend',          # Options: 'trend', 'volatility', 'composite'
        'trend': {
            'adx_period': 14,              # Range: 5-50, period for ADX calculation
            'adx_threshold': 25            # Range: 10-50, threshold for trend strength
        },
        'volatility': {
            'lookback_period': 20,         # Range: 5-50, period for volatility calculation
            'volatility_threshold': 0.015  # Range: 0.001-0.1, threshold for high volatility
        },
        'composite': {
            'combination_method': 'majority', # Options: 'majority', 'consensus', 'weighted'
            'weights': {}                  # Dictionary of weights for weighted method
        }
    }
}
```

#### Detector Types

- `trend`: Detect trends using ADX or similar indicators
- `volatility`: Detect volatility regimes
- `composite`: Combine multiple detectors using voting or weighted methods

#### Combination Methods

- `majority`: Use the most common regime detected
- `consensus`: Require all detectors to agree
- `weighted`: Use a weighted vote based on specified weights

## Complete Configuration Example

Below is a complete configuration example showing all available settings:

```python
{
    'backtester': {
        'market_simulation': {
            'slippage_model': 'fixed',
            'slippage_bps': 5,
            'price_impact': 0.1,
            'fee_model': 'fixed',
            'fee_bps': 10
        },
        'initial_capital': 100000,
        'close_positions_eod': True
    },
    'position_management': {
        'position_sizing': {
            'method': 'percent_equity',
            'percent': 0.02,
            'fixed_size': 100,
            'risk_pct': 0.01,
            'kelly_fraction': 0.5
        },
        'risk_management': {
            'max_position_pct': 0.25,
            'max_drawdown_pct': 0.10,
            'use_stop_loss': False,
            'stop_loss_pct': 0.05,
            'use_take_profit': False,
            'take_profit_pct': 0.10
        },
        'allocation': {
            'method': 'equal',
            'max_instruments': 10
        }
    },
    'signals': {
        'processing': {
            'use_filtering': False,
            'filter_type': 'moving_average',
            'window_size': 5,
            'use_transformations': False,
            'wavelet_type': 'db1',
            'wavelet_level': 3
        },
        'confidence': {
            'use_confidence_score': False,
            'min_confidence': 0.5,
            'prior_accuracy': 0.5
        }
    },
    'regime_detection': {
        'detector_type': 'trend',
        'trend': {
            'adx_period': 14,
            'adx_threshold': 25
        },
        'volatility': {
            'lookback_period': 20,
            'volatility_threshold': 0.015
        },
        'composite': {
            'combination_method': 'majority',
            'weights': {}
        }
    }
}
```

## Configuration Scenarios

### Example 1: Conservative Backtesting Setup

```python
config = ConfigManager()

# Higher slippage and fees for conservative estimates
config.set('backtester.market_simulation.slippage_bps', 10)
config.set('backtester.market_simulation.fee_bps', 15)

# Smaller position sizes
config.set('position_management.position_sizing.percent', 0.01)

# Enable stop losses
config.set('position_management.risk_management.use_stop_loss', True)
config.set('position_management.risk_management.stop_loss_pct', 0.03)

# Lower maximum drawdown tolerance
config.set('position_management.risk_management.max_drawdown_pct', 0.05)
```

### Example 2: High-Frequency Trading Setup

```python
config = ConfigManager()

# Volume-based slippage model
config.set('backtester.market_simulation.slippage_model', 'volume')
config.set('backtester.market_simulation.price_impact', 0.2)

# Tiered fee structure for high volume
config.set('backtester.market_simulation.fee_model', 'tiered')

# Don't close positions at end of day
config.set('backtester.close_positions_eod', False)

# Enable signal filtering for noise reduction
config.set('signals.processing.use_filtering', True)
config.set('signals.processing.filter_type', 'kalman')

# Use confidence scoring
config.set('signals.confidence.use_confidence_score', True)
config.set('signals.confidence.min_confidence', 0.7)
```

### Example 3: Regime-Based Strategy

```python
config = ConfigManager()

# Set up composite regime detection
config.set('regime_detection.detector_type', 'composite')
config.set('regime_detection.composite.combination_method', 'weighted')
config.set('regime_detection.composite.weights', {
    'trend': 0.5,
    'volatility': 0.3,
    'seasonality': 0.2
})

# Custom trend parameters
config.set('regime_detection.trend.adx_period', 10)
config.set('regime_detection.trend.adx_threshold', 20)

# Custom volatility parameters
config.set('regime_detection.volatility.lookback_period', 15)
config.set('regime_detection.volatility.volatility_threshold', 0.02)
```

## Configuration Validation

All configuration values are validated when:
- The ConfigManager is initialized
- A new value is set via the `set()` method
- Configuration is loaded from a file

Validation ensures:
- Types are correct (int, float, bool, string, etc.)
- Values are within specified ranges
- Required fields are present
- Enumerated values are valid

### Common Validation Errors

Below are common validation errors and how to resolve them:

| Error | Description | Resolution |
|-------|-------------|------------|
| `Value X is less than minimum Y` | Value is below the allowed minimum | Increase the value to be within the allowed range |
| `Value X is greater than maximum Y` | Value exceeds the allowed maximum | Decrease the value to be within the allowed range |
| `Value X not in allowed values [A, B, C]` | Value not in enumerated options | Use one of the allowed values |
| `Missing required field 'X'` | Required configuration field is missing | Add the missing field with a valid value |
| `Expected type X, got Y` | Type mismatch in configuration | Correct the type of the provided value |

## Error Handling

The ConfigManager provides clear error messages when:

1. **Validation Fails**: A `ValueError` is raised with details about the validation failure
2. **File Loading Fails**: A `ValueError` is raised with details about the file loading error
3. **Invalid Path**: When accessing a non-existent path with `get()`, the default value is returned

## Integration with Other Components

The ConfigManager is designed to be used throughout the trading system. Here's how it integrates with other components:

### Backtester Integration

```python
from config import ConfigManager
from backtester import Backtester

# Create configuration
config = ConfigManager()
config.set('backtester.initial_capital', 200000)

# Pass configuration to backtester
backtester = Backtester(data_handler, strategy, config=config)
results = backtester.run()
```

### Signal Processor Integration

```python
from config import ConfigManager
from signals import SignalProcessor

# Create configuration
config = ConfigManager()
config.set('signals.processing.use_filtering', True)
config.set('signals.processing.filter_type', 'exponential')

# Create signal processor with configuration
signal_processor = SignalProcessor(config)
```

### Regime Detection Integration

```python
from config import ConfigManager
from regime_detection import DetectorFactory

# Create configuration
config = ConfigManager()
config.set('regime_detection.detector_type', 'trend')
config.set('regime_detection.trend.adx_period', 14)

# Create detector with configuration
detector_factory = DetectorFactory()
detector = detector_factory.create_from_config(config.get('regime_detection'))
```

## Best Practices

1. **Environment-specific configs**: Create separate config files for development, testing, and production
2. **Validation first**: Always validate configurations before using them in critical components
3. **Sensible defaults**: The default configuration provides safe, middle-ground values for most use cases
4. **Configuration inheritance**: Create a base configuration and extend it for specific use cases
5. **Documentation**: Comment your configuration files to explain non-obvious settings

## Troubleshooting

### Configuration Not Applied

If your configuration changes don't seem to be applied:

1. Verify you're using the right configuration path in `set()` and `get()`
2. Ensure you're passing the configuration object to the components
3. Check if the component has a `reset()` method that needs to be called after config changes

### Schema Validation Errors

If you're getting validation errors:

1. Check the error message for specifics on which field failed
2. Refer to the configuration reference to see valid values
3. Use `to_dict()` to inspect the current configuration state
4. For complex nested structures, build the structure step by step

### Loading/Saving Issues

For issues with loading or saving configurations:

1. Check file permissions and paths
2. Verify JSON/YAML syntax is correct
3. For YAML, ensure proper indentation
4. Try loading the file with a different parser to verify its validity

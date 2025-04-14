# System Module Documentation

The System module serves as the central orchestration layer for the trading system, responsible for initializing components, managing lifecycles, and orchestrating the overall system flow.

## Core Concepts

**System Initialization**: Coordinates component creation and initialization in the correct order.  
**Configuration Management**: Centralizes system configuration and ensures consistency.  
**Component Lifecycle**: Manages starting, stopping, and resetting system components.  
**Event Coordination**: Ensures proper registration of event handlers and emitters.  
**Error Handling**: Provides system-wide error handling and recovery.

## Basic Usage

```python
from system import initialize_system, run_backtest

# Initialize the system with configuration
components = initialize_system("config/trading_config.json")

# Access individual components
config = components['config']
event_bus = components['event_bus']
data_handler = components['data_handler']
strategy = components['strategy']
backtester = components['backtester']

# Run a backtest
results = run_backtest(components)

# Process results
print(f"Total Return: {results['total_percent_return']:.2f}%")
print(f"Number of Trades: {results['num_trades']}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
```

## System Initialization

The system initialization process follows a specific order to ensure proper component dependencies:

1. **Configuration** - Load and validate system configuration
2. **Event Bus** - Create the central event communication channel
3. **Data Handler** - Initialize data sources and transformers
4. **Rules & Strategies** - Create trading rules and strategies
5. **Signal Processing** - Set up signal filters and transformers
6. **Position Management** - Initialize position sizing and risk management
7. **Execution Engine** - Set up order execution and portfolio tracking
8. **Event Handlers** - Register event handlers for system components

```python
from system.initialization import initialize_system

# Initialize with default configuration
components = initialize_system()

# Initialize with custom configuration file
components = initialize_system("config/my_custom_config.json")

# Initialize with specific component overrides
components = initialize_system(
    config_file="config/base_config.json",
    strategy_override=my_custom_strategy,
    data_handler_override=my_custom_data_handler
)
```

## System Components

The System module coordinates the following components:

### Event Bus

Central communication hub that enables component decoupling:

```python
# Access event bus
event_bus = components['event_bus']

# Register custom event handler
event_bus.register(EventType.SIGNAL, my_custom_handler)

# Emit custom event
event_bus.emit(Event(EventType.CUSTOM, {"message": "System initialized"}))
```

### Data Handler

Manages market data loading and iteration:

```python
# Access data handler
data_handler = components['data_handler']

# Load specific data
data_handler.load_data(
    symbols=["AAPL", "MSFT"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2022, 12, 31)
)

# Iterate through data
for bar in data_handler.iter_bars():
    # Process bar data
    pass
```

### Strategy

Encapsulates trading logic:

```python
# Access strategy
strategy = components['strategy']

# Get signal for a specific bar
signal = strategy.on_bar(bar_event)

# Reset strategy state
strategy.reset()
```

### Backtester

Runs backtests with the configured components:

```python
# Access backtester
backtester = components['backtester']

# Run backtest
results = backtester.run()

# Calculate performance metrics
sharpe = backtester.calculate_sharpe()
drawdown = backtester.calculate_max_drawdown()
```

## Module Structure

```
system/
├── __init__.py             # Main exports
├── initialization.py       # System initialization
├── lifecycle.py            # Component lifecycle management
├── error_handling.py       # System error handling
└── utils.py                # Utility functions
```

## Advanced Usage

### Custom Component Initialization

```python
from system.initialization import initialize_system
from strategies import RegimeStrategy
from regime_detection import TrendStrengthRegimeDetector

# Create custom components
regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
custom_strategy = RegimeStrategy(regime_detector=regime_detector)

# Initialize system with custom strategy
components = initialize_system(
    config_file="config/base_config.json",
    strategy_override=custom_strategy
)
```

### System Lifecycle Management

```python
from system.lifecycle import SystemLifecycle

# Create system lifecycle manager
lifecycle = SystemLifecycle(components)

# Start all components
lifecycle.start()

# Pause specific components
lifecycle.pause(['data_handler', 'strategy'])

# Resume components
lifecycle.resume(['data_handler', 'strategy'])

# Stop all components
lifecycle.stop()

# Reset system state
lifecycle.reset()
```

### Error Handling and Recovery

```python
from system.error_handling import setup_error_handlers, recover_from_error

# Set up system-wide error handlers
setup_error_handlers(components)

# Attempt recovery from error
try:
    # Operation that might fail
    result = risky_operation()
except Exception as e:
    # Attempt recovery
    success = recover_from_error(e, components)
    if not success:
        # Handle unrecoverable error
        lifecycle.stop()
        raise
```

## Best Practices

1. **Centralized Configuration**: Use the ConfigManager for all component settings
2. **Proper Initialization Order**: Always initialize components in the correct dependency order
3. **Event-Driven Communication**: Use the event system for inter-component communication
4. **Error Handling**: Implement proper error handling at the system level
5. **Component Lifecycle**: Respect component lifecycle operations (start, stop, reset)
6. **Configuration Validation**: Validate configurations during system initialization
7. **Modular Design**: Keep components loosely coupled through the event system
8. **Consistent Logging**: Use the logging system consistently across components
9. **Resource Management**: Ensure proper cleanup of resources when stopping components
10. **Testing**: Test system initialization with different configurations and scenarios
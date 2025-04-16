# Algorithmic Trading System

An event-driven trading system for backtesting and implementing algorithmic trading strategies.

## Overview

This trading system is a modular, event-driven framework designed for developing, testing, and deploying algorithmic trading strategies. The system supports backtesting with historical data, optimization of strategy parameters, and real-time trading through a plugin architecture.

The architecture employs an event-driven design pattern, where components communicate through a central event bus, promoting loose coupling, flexibility, and extensibility. This design makes it easy to add new strategies, data sources, or execution methods without modifying existing code.

## Event System Architecture

The event system is the core of the application, providing a robust communication framework between components:

- **EventBus**: Central message broker that routes events from emitters to handlers
- **Event**: Base class for all events with type, timestamp, ID, and data payload
- **EventType**: Enumeration of all system event types (BAR, SIGNAL, ORDER, etc.)
- **Event Handlers**: Components that process events of specific types
- **Event Emitters**: Components that generate and emit events

## Key Components

### Event System

The Events module provides a robust event-driven architecture that enables decoupled communication between components through a central event bus. It defines standard event types and provides mechanisms for routing events from producers to consumers.

**Key Classes:**
- `EventBus`: Central message broker for routing events
- `Event`: Base class for events with type, payload, and timestamp
- `EventHandler`: Base class for components that process events
- `EventEmitter`: Mixin for components that emit events
- `EventType`: Enumeration of all event types in the system

### Data Module

The Data module handles all aspects of market data, including fetching, transforming, storing, and providing data to other system components. It supports various data sources (CSV, databases, APIs) and provides a unified interface for accessing market data.

**Key Classes:**
- `DataSource`: Interface for retrieving data from various origins
- `DataHandler`: Core component that manages data flow
- `DataTransformer`: Components for preprocessing and transforming raw data
- `CSVDataSource`, `SQLiteDataSource`, etc.: Specific implementations

### Rules Module

The Rules module provides a framework for creating trading rules that generate signals based on technical indicators. Each rule maintains internal state, processes bar data, and returns standardized Signal objects.

**Key Classes:**
- `Rule`: Base class for all trading rules
- `CompositeRule`: Combines multiple rules with aggregation methods
- `SignalType`: Enumeration of signal types (BUY, SELL, NEUTRAL)
- Various rule implementations (SMA, RSI, etc.)

### Strategies Module

The Strategies module provides a modular framework for creating and combining trading strategies. It supports weighted combinations, ensemble techniques, and regime-based adaptation.

**Key Classes:**
- `Strategy`: Base class for all strategies
- `WeightedStrategy`: Combines multiple components using weights
- `EnsembleStrategy`: Combines multiple strategies
- `RegimeStrategy`: Adapts to different market regimes

### Position Management

The Position Management module handles position creation, tracking, sizing, and risk management. It models individual positions and manages the entire portfolio.

**Key Classes:**
- `Position`: Represents a single trading position
- `Portfolio`: Manages multiple positions
- `PositionManager`: Coordinates position sizing and allocation
- Various position sizers and allocation strategies

### Execution Module

The Engine module is the core execution component, handling backtesting, market simulation, order execution, and portfolio tracking. It transforms trading signals into orders and manages their execution.

**Key Classes:**
- `Backtester`: Orchestrates the backtesting process
- `ExecutionEngine`: Handles order execution and position tracking
- `MarketSimulator`: Simulates market conditions

## Event Types

The system uses a comprehensive set of event types organized into categories:

```
Market Data Events
├── BAR
├── TICK
├── MARKET_OPEN
└── MARKET_CLOSE

Signal Events
└── SIGNAL

Order Events
├── ORDER
├── CANCEL
└── MODIFY

Execution Events
├── FILL
├── PARTIAL_FILL
└── REJECT

Portfolio Events
├── POSITION_OPENED
├── POSITION_CLOSED
└── POSITION_MODIFIED
└── POSITION_STOPPED

System Events
├── START
├── STOP
├── PAUSE
├── RESUME
└── ERROR
```

## Event Handler Types

The system provides various event handler implementations:

- **BasicHandler**: Standard event handler for simple event processing
- **LoggingHandler**: Logs events at configurable levels
- **DebounceHandler**: Prevents processing the same event type too frequently
- **FilterHandler**: Only processes events that meet specific criteria
- **AsyncEventHandler**: Processes events in separate threads
- **CompositeHandler**: Delegates to multiple handlers for the same event

## Event Emitters

Event emitters are specialized for different domain events:

- **MarketDataEmitter**: Emits BAR, TICK, MARKET_OPEN, and MARKET_CLOSE events
- **SignalEmitter**: Emits SIGNAL events
- **OrderEmitter**: Emits ORDER, CANCEL, and MODIFY events
- **FillEmitter**: Emits FILL and PARTIAL_FILL events
- **PortfolioEmitter**: Emits POSITION_OPENED, POSITION_CLOSED, and POSITION_MODIFIED events
- **SystemEmitter**: Emits START, STOP, ERROR, and other system events

## Event Flow

The system follows this general data flow:

1. Market data flows from data sources through the DataHandler
2. Strategy components process market data to generate signals
3. Position Manager determines position size based on signals
4. Execution Engine converts position actions into orders
5. Market Simulator (in backtesting) or broker API (in live trading) executes orders
6. Fill events update the portfolio and risk metrics

## Validation and Error Handling

The system includes validation for events:

- **EventSchema**: Defines expected fields and types for each event type
- **EventValidator**: Validates events against their schemas
- Error events are emitted when validation or processing fails
- Centralized error handling through the ErrorHandler class

## Recent Improvements

### Standardized Event Objects

The system now uses standardized event objects throughout:

- **Event Base Class**: Foundational class for all events
- **BarEvent**: Standard representation of market data bars
- **SignalEvent**: Standard representation of trading signals
- **OrderEvent**: Standard representation of order instructions
- **FillEvent**: Standard representation of execution fills

All emitters now strictly enforce type checking to ensure only proper event objects are used throughout the system.

### Fixed Circular Imports

Circular import issues have been resolved by:

- Moving the base Event class to its own module (event_base.py)
- Ensuring proper import patterns throughout the system
- Using appropriate type hints and imports

### Improved Error Handling

Error handling improvements include:

- Centralized ErrorHandler for system-wide error management
- Standardized error event creation
- Threshold-based system control (auto-stop on repeated errors)

### Metrics Collection

The system now includes metrics collection:

- EventBus tracks event counts, processing times, and rates
- MetricsCollector aggregates metrics from all components
- Performance monitoring for system optimization

## TODOs and Improvements

- **Move trade history to Portfolio**: Currently `get_trade_history()` is located in the ExecutionEngine class, but conceptually this data belongs in the Portfolio class. Move this functionality to Portfolio for better encapsulation and more intuitive API.
- **Fix Portfolio pricing**: The current implementation shows portfolio equity dropping to $0 during backtesting. This needs investigation to ensure proper price updates and position valuation.
- **Improve cross-component documentation**: Make module interdependencies clearer in documentation, especially which components store what data.

## Best Practices

1. **Use the Event System**: All communication between components should go through the event bus
2. **Validate Configurations**: Use the schema validation system to ensure correct configuration
3. **Reset Component State**: Reset internal state when switching symbols or timeframes
4. **Handle Edge Cases**: Implement proper handling for missing data or extreme market conditions
5. **Use Factory Methods**: Create components using factory methods for consistency
6. **Maintain Loose Coupling**: Keep components independent and focused on single responsibilities
7. **Test Extensively**: Use unit tests for individual components and integration tests for the system
8. **Document Components**: Clearly document component interfaces and behavior
9. **Use Proper Event Objects**: Always use properly typed event objects, never dictionaries


# !!!: POSSIBLY OUTDATED -- SYSTEM IS GOING THROUGH A REFACTOR
TO ENSURE CONSISTENT TYPING SYSTEM WIDE WE'RE MOVING TO STRICTLY USING EVENT OBJECTS
AND THEIR SUBCLASSES, NO SIGNAL OBJECTS OR DICTIONARIES. READ THE FOLLOWING WITH CAUTION.

# Algorithmic Trading System

An event-driven trading system for backtesting and implementing algorithmic trading strategies.

## Overview

This trading system is a modular, event-driven framework designed for developing, testing, and deploying algorithmic trading strategies. The system supports backtesting with historical data, optimization of strategy parameters, and real-time trading through a plugin architecture.

The architecture employs an event-driven design pattern, where components communicate through a central event bus, promoting loose coupling, flexibility, and extensibility. This design makes it easy to add new strategies, data sources, or execution methods without modifying existing code.

## TODOs and Improvements

- **Move trade history to Portfolio**: Currently `get_trade_history()` is located in the ExecutionEngine class, but conceptually this data belongs in the Portfolio class. Move this functionality to Portfolio for better encapsulation and more intuitive API.
- **Fix Portfolio pricing**: The current implementation shows portfolio equity dropping to $0 during backtesting. This needs investigation to ensure proper price updates and position valuation.
- **Improve cross-component documentation**: Make module interdependencies clearer in documentation, especially which components store what data.

## System Architecture

![System Architecture](https://mermaid.ink/img/pako:eNqNkl9PwjAUxb-K6ROQ8oAkGpdAwAdjFokm4tN8aOctyNbSdp0J4bvbMRQWYtT7dO85v9O7f_YIUaYQRHCq5YlxOa-W4qIkxw_iRHTKfuTsLOWn4KjVTlwJrbQ6C8lTgsuP2kGrKYlXCy1Mw2jK3pNTKz6D3UCGKuv1BhP-MwmxXdJAGTnVyj5GS9UoUq9UxAo6Ri6ZQ5c-rz0ejmbbpAQz2B55IVvTgHPDKfDLQtKRYmUcQ-0YN1uIoICZcZjGmEAEDa-0Nb4ToAj3i0Zgm2yjYmm76TsBeswdtgITnRt0DUztapV7P9yLzHnlYLQFp1U-b2TDWQM7uGm8l6m5oQo-YAQpRtcFzZTjqnBsAiPGgEeF4XtPFUww0pIZGS2vwPRKgG9aSqIjrhTrvcEIcxdtl2l6B0-rXF86l3XuZIGSyqLnAQoKN8u-BXSDoMR4KPzDYeB7kVo-SsGGwZCO_cg_-kOD8YFPh8PjwaE_iCc4DsLxuH9h-NP-hPaj-Bp_F9o_FfwBCn3iSg?type=png)

### Key Components

1. **Event System**: Manages communication between components
2. **Data Module**: Handles market data acquisition and processing
3. **Rules Module**: Defines trading rules that generate signals
4. **Strategies Module**: Combines rules into coherent trading strategies
5. **Position Management**: Handles position sizing and portfolio management
6. **Risk Management**: Implements risk control measures
7. **Engine**: Orchestrates backtesting and execution
8. **Configuration**: Manages system and component configurations
9. **Analytics**: Provides performance analysis and reporting

## Event-Driven Design

The system operates on an event-driven paradigm, where:

- Components communicate by emitting and handling events
- The event bus routes events from emitters to handlers
- Components are loosely coupled and independently testable
- New functionality can be added by implementing new event handlers

### Core Event Types

```
Market Data Events
├── BAR
├── TICK
├── MARKET_OPEN
└── MARKET_CLOSE

Signal Events
└── SIGNAL

Order Events
├── ORDER
├── CANCEL
└── MODIFY

Execution Events
├── FILL
├── PARTIAL_FILL
└── REJECT

Portfolio Events
├── POSITION_OPENED
├── POSITION_CLOSED
└── POSITION_MODIFIED

System Events
├── START
├── STOP
├── PAUSE
├── RESUME
└── ERROR
```

## Data Flow

The system follows this general data flow:

1. Market data flows from data sources through the DataHandler
2. Strategy components process market data to generate signals
3. Position Manager determines position size based on signals
4. Execution Engine converts position actions into orders
5. Market Simulator (in backtesting) or broker API (in live trading) executes orders
6. Fill events update the portfolio and risk metrics

## Component Relationships

```
                   ┌─────────────────┐
                   │    Event Bus    │
                   └─────────────────┘
                          ▲   │
                          │   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│             │    │                 │    │                 │
│ DataHandler ├───►│    Strategy     ├───►│ Position Manager│
│             │    │                 │    │                 │
└─────────────┘    └─────────────────┘    └─────────────┬───┘
                                                        │
                                                        ▼
                                          ┌─────────────────────┐
                                          │                     │
                                          │  Execution Engine   │
                                          │                     │
                                          └─────────────────────┘
```

## Module Descriptions

### Event System

The Events module provides a robust event-driven architecture that enables decoupled communication between components through a central event bus. It defines standard event types and provides mechanisms for routing events from producers to consumers.

**Key Classes:**
- `EventBus`: Central message broker for routing events
- `Event`: Container for event data with type, payload, and timestamp
- `EventHandler`: Base class for components that process events
- `EventEmitter`: Mixin for components that emit events

### Data Module

The Data module handles all aspects of market data, including fetching, transforming, storing, and providing data to other system components. It supports various data sources (CSV, databases, APIs) and provides a unified interface for accessing market data.

**Key Classes:**
- `DataSource`: Interface for retrieving data from various origins
- `DataHandler`: Core component that manages data flow
- `DataTransformer`: Components for preprocessing and transforming raw data
- `CSVDataSource`, `SQLiteDataSource`, etc.: Specific implementations

### Rules Module

The Rules module provides a framework for creating trading rules that generate signals based on technical indicators. Each rule maintains internal state, processes bar data, and returns standardized Signal objects.

**Key Classes:**
- `Rule`: Base class for all trading rules
- `CompositeRule`: Combines multiple rules with aggregation methods
- `SignalType`: Enumeration of signal types (BUY, SELL, NEUTRAL)
- Various rule implementations (SMA, RSI, etc.)

### Strategies Module

The Strategies module provides a modular framework for creating and combining trading strategies. It supports weighted combinations, ensemble techniques, and regime-based adaptation.

**Key Classes:**
- `Strategy`: Base class for all strategies
- `WeightedStrategy`: Combines multiple components using weights
- `EnsembleStrategy`: Combines multiple strategies
- `RegimeStrategy`: Adapts to different market regimes

### Position Management

The Position Management module handles position creation, tracking, sizing, and risk management. It models individual positions and manages the entire portfolio.

**Key Classes:**
- `Position`: Represents a single trading position
- `Portfolio`: Manages multiple positions
- `PositionManager`: Coordinates position sizing and allocation
- Various position sizers and allocation strategies

### Risk Management

The Risk Management module implements advanced risk management using MAE (Maximum Adverse Excursion), MFE (Maximum Favorable Excursion), and ETD (Entry-To-Exit Duration) analysis to derive data-driven risk parameters.

**Key Classes:**
- `RiskManager`: Applies risk rules to trades
- `RiskMetricsCollector`: Collects trade metrics
- `RiskParameterOptimizer`: Derives optimal risk parameters

### Engine Module

The Engine module is the core execution component, handling backtesting, market simulation, order execution, and portfolio tracking. It transforms trading signals into orders and manages their execution.

**Key Classes:**
- `Backtester`: Orchestrates the backtesting process
- `ExecutionEngine`: Handles order execution and position tracking
- `MarketSimulator`: Simulates market conditions

### Configuration

The Configuration module provides a unified system for managing configuration across all components. It enables loading, validating, modifying, and saving configuration settings.

**Key Classes:**
- `ConfigManager`: Main class for managing configurations
- Configuration schema and validation tools

### Analytics

The Analytics module provides tools for analyzing and visualizing backtesting results and trading performance metrics.

**Key Classes:**
- Various performance metrics calculators
- Visualization tools for equity curves, drawdowns, etc.

## Optimization

The Optimization module provides tools for finding optimal parameters for trading strategies and validating results.

**Key Classes:**
- `OptimizerManager`: Manages different optimization approaches
- `GridOptimizer`, `GeneticOptimizer`: Specific optimization algorithms
- Validation tools like `WalkForwardValidator`

## System Setup and Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# Install dependencies
pip install -r requirements.txt
```

### Startup Sequence

```python
# 1. Initialize the event system
event_bus = EventBus()

# 2. Create and configure the config manager
config = ConfigManager('trading_config.yaml')

# 3. Set up data sources and handler
data_source = CSVDataSource(config.get('data.csv_directory'))
data_handler = DataHandler(data_source)

# 4. Initialize risk management
risk_manager = RiskManager(
    risk_params=config.get('risk_management'),
    metrics_collector=RiskMetricsCollector()
)

# 5. Set up position management
portfolio = Portfolio(initial_capital=config.get('backtester.initial_capital'))
position_manager = PositionManager(
    portfolio=portfolio,
    risk_manager=risk_manager
)

# 6. Create strategy from configuration
strategy = StrategyFactory().create_from_config(config.get('strategy'))

# 7. Set up execution engine
execution_engine = ExecutionEngine(position_manager=position_manager)

# 8. Initialize backtester
backtester = Backtester(config, data_handler, strategy, execution_engine)

# 9. Register components with event system
event_bus.register(EventType.BAR, strategy)
event_bus.register(EventType.SIGNAL, position_manager.on_signal)
event_bus.register(EventType.ORDER, execution_engine.on_order)
event_bus.register(EventType.FILL, portfolio.on_fill)

# 10. Load initial data
data_handler.load_data(
    symbols=config.get('symbols'),
    start_date=config.get('start_date'),
    end_date=config.get('end_date'),
    timeframe=config.get('timeframe')
)

# 11. Run the system
results = backtester.run()
```

### Example Strategy Configuration

```yaml
strategy:
  type: WeightedStrategy
  params:
    components:
      - type: SMAcrossoverRule
        params:
          fast_window: 10
          slow_window: 30
      - type: RSIRule
        params:
          period: 14
          overbought: 70
          oversold: 30
    weights: [0.7, 0.3]
    buy_threshold: 0.4
    sell_threshold: -0.4
```

## Best Practices

1. **Use the Event System**: All communication between components should go through the event bus
2. **Validate Configurations**: Use the schema validation system to ensure correct configuration
3. **Reset Component State**: Reset internal state when switching symbols or timeframes
4. **Handle Edge Cases**: Implement proper handling for missing data or extreme market conditions
5. **Use Factory Methods**: Create components using factory methods for consistency
6. **Maintain Loose Coupling**: Keep components independent and focused on single responsibilities
7. **Test Extensively**: Use unit tests for individual components and integration tests for the system
8. **Document Components**: Clearly document component interfaces and behavior

## Future Enhancements

- **Live Trading**: Integration with broker APIs for live trading
- **Machine Learning**: Support for ML-based prediction models
- **Web Interface**: Dashboard for monitoring and controlling the system
- **Distributed Processing**: Support for parallel processing of strategies
- **Real-Time Data**: Integration with real-time market data providers

## Contributing

Guidelines for contributing to the project would go here.

## License

Information about the project license would go here.


## Recent Improvements

### Signal and BarEvent Standardization

The system has been updated to use standardized objects across all components:

- **Signal Objects**: All components now use proper `Signal` objects with consistent access patterns, improving reliability and type safety throughout the system. Dictionary signals and fallback methods have been removed.

- **BarEvent Objects**: Bar data is now consistently wrapped in `BarEvent` objects, making it easier to process market data across different components.

These improvements reduce the need for defensive programming and make the system more maintainable and less error-prone. See the respective module READMEs for detailed usage information.

## Signal Handling

The engine module has been updated to handle standardized Signal objects:

### ExecutionEngine

The `ExecutionEngine` now expects proper `Signal` objects and processes them directly:

```python
def on_signal(self, signal):
    """Process a Signal object and convert to an order."""
    # Skip neutral signals
    if signal.signal_type == SignalType.NEUTRAL:
        return None
        
    # Extract attributes directly from signal
    direction = signal.signal_type.value
    symbol = getattr(signal, 'symbol', 'default')
    # ...


## 5. Update `src/rules/README.md` for Signal object creation

```markdown
## Signal Generation

All rules now consistently return standardized `Signal` objects:

```python
def generate_signal(self, data):
    """Generate a trading signal from the provided data."""
    # Rule logic to determine signal type
    # ...
    
    # Always return a Signal object
    return Signal(
        timestamp=data.get('timestamp', None),
        signal_type=signal_type_enum,
        price=close,
        rule_id=self.name,
        confidence=confidence,
        metadata={
            'fast_sma': current_fast,
            'slow_sma': current_slow,
            'symbol': symbol
        }
    )

## 6. Update `src/position_management/README.md` for position sizing with Signal objects

```markdown
## Position Sizing with Signal Objects

Position sizing components have been updated to work directly with standardized Signal objects:

```python
def calculate_position_size(self, signal, portfolio, price=None):
    """Calculate position size based on a Signal object."""
    # Get direction directly from signal
    direction = signal.signal_type.value
    
    # Skip neutral signals
    if direction == 0:
        return 0
        
    # Use price from signal if not provided
    if price is None and hasattr(signal, 'price') and signal.price is not None:
        price = signal.price
    
    # Calculate and return sized position
    # ...
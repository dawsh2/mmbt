# Engine Module Documentation

The Engine module is the core execution component of the trading system, handling backtesting, market simulation, order execution, and portfolio tracking. It transforms trading signals into orders, manages their execution with realistic market conditions, and tracks performance metrics.

## Core Concepts

**Backtester**: Orchestrates the backtesting process, connecting data, strategy, and execution components.  
**ExecutionEngine**: Handles order execution, position tracking, and portfolio management.  
**MarketSimulator**: Simulates realistic market conditions including slippage and transaction costs.  
**Event System**: Passes events (bars, signals, orders, fills) between system components.

## Event Flow in Backtesting

The complete event flow in the system follows this sequence:

1. `BAR` events are emitted for each bar of market data
2. Strategy components receive the bar events via their `on_bar(event)` methods
3. If a strategy generates a signal, it is emitted as a `SIGNAL` event
4. The PositionManager receives signal events via its `on_signal(event)` method
5. If the PositionManager decides to act on the signal, it generates an `ORDER` event
6. The ExecutionEngine receives order events via its `on_order(event)` method
7. The ExecutionEngine simulates execution and emits a `FILL` event
8. The Portfolio is updated with the fill information

This event flow ensures that each component is responsible only for its specific role in the trading process.

## Basic Usage

```python
from src.engine import Backtester, MarketSimulator
from src.data.data_handler import CSVDataHandler
from src.strategy import WeightedStrategy
from src.config import ConfigManager
from src.position_management.position_manager import PositionManager

# Create data handler
data_handler = CSVDataHandler('data/AAPL_daily.csv')

# Create a strategy
strategy = WeightedStrategy(rules=rule_objects, weights=[0.4, 0.3, 0.3])

# Create configuration
config = ConfigManager()
config.set('backtester.initial_capital', 100000)
config.set('backtester.market_simulation.slippage_model', 'fixed')
config.set('backtester.market_simulation.slippage_bps', 5)

# Create position manager
position_manager = PositionManager()

# Create and run backtester
backtester = Backtester(config, data_handler, strategy, position_manager)
results = backtester.run()

# Analyze results
print(f"Total Return: {results['total_percent_return']:.2f}%")
print(f"Number of Trades: {results['num_trades']}")
print(f"Sharpe Ratio: {backtester.calculate_sharpe():.4f}")

# Access trade history
for trade in results['trades'][:5]:  # First 5 trades
    print(f"Entry: {trade[0]}, Direction: {trade[1]}, Exit: {trade[3]}, Return: {trade[5]:.4f}")

# Access portfolio history
portfolio_history = results['portfolio_history']
```

## API Reference

### Backtester

The main orchestration class that coordinates the backtesting process.

**Constructor Parameters:**
- `config` (ConfigManager/dict): Configuration for the backtest
- `data_handler` (DataHandler): Data handler providing market data
- `strategy` (Strategy): Trading strategy to test
- `position_manager` (PositionManager, optional): Position manager for risk management

**Methods:**

#### run(use_test_data=False)

Run the backtest.

**Parameters:**
- `use_test_data` (bool, optional): Whether to use test data (True) or training data (False)

**Returns:**
- `dict`: Backtest results containing trades, portfolio history, and performance metrics

**Result Dictionary:**
- `trades` (list): List of executed trades as tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)
- `num_trades` (int): Number of trades executed
- `total_log_return` (float): Total logarithmic return
- `total_percent_return` (float): Total percentage return
- `average_log_return` (float): Average logarithmic return per trade
- `portfolio_history` (list): Snapshot of portfolio at each bar
- `signals` (list): Signal history
- `config` (dict): Configuration used for the backtest

**Example:**
```python
backtester = Backtester(config, data_handler, strategy, position_manager)
results = backtester.run(use_test_data=True)  # Run on test data
```

#### calculate_sharpe()

Calculate the Sharpe ratio for the backtest.

**Returns:**
- `float`: Sharpe ratio value

**Example:**
```python
sharpe_ratio = backtester.calculate_sharpe()
```

#### reset()

Reset the backtester state.

**Example:**
```python
backtester.reset()  # Reset before running another backtest
```

### ExecutionEngine

Handles order execution, position tracking, and portfolio management.

**Constructor Parameters:**
- `position_manager` (PositionManager, optional): Position manager for position sizing

**Methods:**

#### on_order(event)

Handle incoming order events.

**Parameters:**
- `event` (Event): Order event

**Example:**
```python
execution_engine = ExecutionEngine(position_manager)
execution_engine.on_order(order_event)
```

#### execute_pending_orders(bar, market_simulator)

Execute any pending orders based on current bar data.

**Parameters:**
- `bar` (dict): Current bar data
- `market_simulator` (MarketSimulator): Simulator for market effects

**Example:**
```python
execution_engine.execute_pending_orders(current_bar, market_simulator)
```

#### update(bar)

Update portfolio with latest market data and record portfolio state.

**Parameters:**
- `bar` (dict): Current bar data containing at minimum 'symbol' and 'Close' keys

**Example:**
```python
execution_engine.update(current_bar)
```

### Updating Portfolio with Market Data

The ExecutionEngine is responsible for updating the portfolio with current market data:

```python
def update(self, bar_data):
    """
    Update portfolio with latest market data.
    
    Parameters:
    -----------
    bar_data : dict
        Current bar data with at minimum 'symbol' and 'Close' keys
    """
    # Update last known prices
    self.last_known_prices[bar_data['symbol']] = bar_data['Close']
    
    # Record portfolio state for history
    self._record_portfolio_state(bar_data['timestamp'])
```

This method should be called for each new bar to ensure the portfolio state is updated with current market prices.

#### get_trade_history()

Get the history of all executed trades.

**Returns:**
- `list`: List of Fill objects representing executed trades

**Example:**
```python
trades = execution_engine.get_trade_history()
```

#### get_portfolio_history()

Get the history of portfolio states.

**Returns:**
- `list`: List of portfolio state dictionaries

**Example:**
```python
portfolio_history = execution_engine.get_portfolio_history()
```

#### reset()

Reset the execution engine state.

**Example:**
```python
execution_engine.reset()
```

### MarketSimulator

Simulates realistic market conditions including slippage and transaction costs.

**Constructor Parameters:**
- `config` (dict, optional): Configuration dictionary for market simulation

**Methods:**

#### calculate_execution_price(order, bar)

Calculate the execution price including slippage.

**Parameters:**
- `order` (Order): Order object with quantity and direction
- `bar` (dict): Current bar data

**Returns:**
- `float`: Execution price with slippage

**Example:**
```python
market_simulator = MarketSimulator({'slippage_model': 'fixed', 'slippage_bps': 5})
execution_price = market_simulator.calculate_execution_price(order, bar)
```

#### calculate_fees(order, execution_price)

Calculate transaction fees.

**Parameters:**
- `order` (Order): Order object with quantity
- `execution_price` (float): Price after slippage

**Returns:**
- `float`: Fee amount

**Example:**
```python
fees = market_simulator.calculate_fees(order, execution_price)
```

## Integration with Other Modules

### Using Position Manager with Signal Processor

```python
from src.engine import MarketSimulator
from src.position_management.position_manager import PositionManager
from src.signals import SignalProcessor
from src.config import ConfigManager

# Create configuration
config = ConfigManager()
config.set('signals.confidence.use_confidence_score', True)
config.set('signals.confidence.min_confidence', 0.6)

# Create signal processor
signal_processor = SignalProcessor(config)

# Create position manager
position_manager = PositionManager()

def process_signal_with_position_sizing(raw_signal, portfolio):
    """
    Process a raw signal and determine appropriate position size.
    
    Args:
        raw_signal: Raw signal from strategy
        portfolio: Current portfolio state
        
    Returns:
        tuple: (processed_signal, position_size)
    """
    # Process signal for confidence scoring and filtering
    processed_signal = signal_processor.process_signal(raw_signal)
    
    # Check if signal meets confidence threshold
    if processed_signal.confidence >= config.get('signals.confidence.min_confidence'):
        # Calculate position size
        position_size = position_manager.calculate_position_size(processed_signal, portfolio)
        return processed_signal, position_size
    else:
        # Signal doesn't meet confidence threshold
        return processed_signal, 0
```

### Integration with Regime Detection

```python
from src.engine import Backtester, MarketSimulator
from src.position_management.position_manager import PositionManager
from src.regime_detection import RegimeType, VolatilityRegimeDetector

# Create regime detector and position manager
regime_detector = VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015)
position_manager = PositionManager()

# Update position manager with regime data
def update_position_manager_with_regime(signal, position_manager, regime_detector):
    """
    Update position manager with current regime data.
    
    Args:
        signal: Trading signal
        position_manager: Position manager instance
        regime_detector: Regime detector instance
    """
    # Get current regime
    current_regime = regime_detector.current_regime
    
    # Add regime to signal metadata
    if hasattr(signal, 'metadata'):
        signal.metadata['regime'] = current_regime
    
    return signal

# Create backtester with regime-aware position sizing
backtester = Backtester(config, data_handler, strategy, position_manager)
```

## Event System

The engine module uses an event-driven architecture for communication between components. Events are imported from the `src.events.event_types` module.

```python
from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType
from src.engine import Backtester

# Create an event bus
event_bus = EventBus()

# Register an event handler
event_bus.register(EventType.FILL, my_fill_handler)

# Create an event
bar_event = Event(EventType.BAR, bar_data)

# Emit an event
event_bus.emit(bar_event)
```

## Market Simulation Models

The engine module provides several models for simulating market effects:

### Slippage Models

**NoSlippageModel**: Returns the base price unchanged.
**FixedSlippageModel**: Applies a fixed basis point slippage to the price.
**VolumeBasedSlippageModel**: Scales slippage with order size relative to volume.

```python
from src.engine.market_simulator import VolumeBasedSlippageModel

# Create a volume-based slippage model
slippage_model = VolumeBasedSlippageModel(price_impact=0.1)
```

### Fee Models

**NoFeeModel**: Returns zero fees.
**FixedFeeModel**: Applies a fixed basis point fee to the transaction value.
**TieredFeeModel**: Uses different fees based on transaction value.

```python
from src.engine.market_simulator import TieredFeeModel

# Create a tiered fee model
fee_model = TieredFeeModel(
    tier_thresholds=[10000, 100000],  # $10K, $100K thresholds
    tier_fees_bps=[10, 5, 3]          # 0.1%, 0.05%, 0.03% fees
)
```

## Best Practices

1. **Always use realistic slippage and fees**: Configure the MarketSimulator with realistic parameters to avoid overly optimistic results

2. **Test with conservative risk management**: Start with conservative risk parameters and gradually relax them rather than the opposite

3. **Reset state between backtests**: Always call `reset()` on all components when running multiple backtests

4. **Validate position sizing logic**: Verify that position sizing behaves as expected by checking allocation across different market conditions

5. **Monitor execution details**: Keep track of execution prices vs. signal prices to understand the impact of slippage

6. **Use event handlers for custom analytics**: Leverage the event system to monitor specific aspects of the backtest

7. **Separate parameter optimization from backtesting**: Use the backtester to evaluate strategies with pre-optimized parameters rather than doing both at once

8. **Maintain proper event flow**: Ensure all components receive and process events correctly according to the defined event flow

9. **Update portfolio regularly**: Call the `update()` method with each new bar to keep the portfolio state current

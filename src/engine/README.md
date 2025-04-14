# Engine Module Documentation

The Engine module is the core execution component of the trading system, handling backtesting, market simulation, order execution, position management, and portfolio tracking. It transforms trading signals into orders, manages their execution with realistic market conditions, and tracks performance metrics.

## Core Concepts

**Backtester**: Orchestrates the backtesting process, connecting data, strategy, and execution components.  
**ExecutionEngine**: Handles order execution, position tracking, and portfolio management.  
**MarketSimulator**: Simulates realistic market conditions including slippage and transaction costs.  
**PositionManager**: Manages position sizing, risk parameters, and allocation across instruments.  
**Event System**: Passes events (bars, signals, orders, fills) between system components.

## Basic Usage

```python
from engine import Backtester, MarketSimulator
from data_handler import CSVDataHandler
from strategies import WeightedStrategy
from config import ConfigManager

# Create data handler
data_handler = CSVDataHandler('data/AAPL_daily.csv')

# Create a strategy
strategy = WeightedStrategy(rules=rule_objects, weights=[0.4, 0.3, 0.3])

# Create configuration
config = ConfigManager()
config.set('backtester.initial_capital', 100000)
config.set('backtester.market_simulation.slippage_model', 'fixed')
config.set('backtester.market_simulation.slippage_bps', 5)

# Create and run backtester
backtester = Backtester(config, data_handler, strategy)
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
backtester = Backtester(config, data_handler, strategy)
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
execution_engine = ExecutionEngine()
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

Update portfolio with latest market data.

**Parameters:**
- `bar` (dict): Current bar data

**Example:**
```python
execution_engine.update(current_bar)
```

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

### PositionManager

Manages position sizing, risk parameters, and allocation across instruments.

**Constructor Parameters:**
- `sizing_strategy` (SizingStrategy, optional): Strategy for position sizing
- `risk_manager` (RiskManager, optional): Risk management controls
- `allocation_strategy` (AllocationStrategy, optional): Portfolio allocation strategy

**Methods:**

#### calculate_position_size(signal, portfolio)

Calculate the appropriate position size for a signal.

**Parameters:**
- `signal` (Signal): The trading signal
- `portfolio` (Portfolio): Current portfolio state

**Returns:**
- `float`: Position size (positive for buy, negative for sell, 0 for no trade)

**Example:**
```python
from engine import PositionManager

position_manager = PositionManager()
position_size = position_manager.calculate_position_size(signal, portfolio)
```

#### get_risk_metrics(portfolio)

Get current risk metrics for the portfolio.

**Parameters:**
- `portfolio` (Portfolio): Current portfolio state

**Returns:**
- `dict`: Risk metrics

**Example:**
```python
risk_metrics = position_manager.get_risk_metrics(portfolio)
```

#### update_stops(bar_data, portfolio)

Update and check stop losses and take profits.

**Parameters:**
- `bar_data` (dict): Current bar data
- `portfolio` (Portfolio): Current portfolio state

**Returns:**
- `dict`: Positions to close due to stops {symbol: reason}

**Example:**
```python
stops_hit = position_manager.update_stops(bar_data, portfolio)
```

#### reset()

Reset position manager state.

**Example:**
```python
position_manager.reset()
```

### Position Sizing Strategies

The engine module includes various position sizing strategies that can be used with the PositionManager.

#### FixedSizingStrategy

Position sizing using a fixed number of units.

**Constructor Parameters:**
- `fixed_size` (float, optional): Number of units to trade (default: 100)

**Example:**
```python
from engine.position_manager import FixedSizingStrategy, PositionManager

sizing_strategy = FixedSizingStrategy(fixed_size=200)
position_manager = PositionManager(sizing_strategy=sizing_strategy)
```

#### PercentOfEquitySizing

Size position as a percentage of portfolio equity.

**Constructor Parameters:**
- `percent` (float, optional): Percentage of equity to allocate (default: 0.02)

**Example:**
```python
from engine.position_manager import PercentOfEquitySizing, PositionManager

sizing_strategy = PercentOfEquitySizing(percent=0.05)  # 5% of equity per position
position_manager = PositionManager(sizing_strategy=sizing_strategy)
```

#### VolatilityBasedSizing

Size positions based on asset volatility.

**Constructor Parameters:**
- `risk_pct` (float, optional): Percentage of equity to risk per unit of volatility (default: 0.01)
- `lookback_period` (int, optional): Period for calculating volatility (default: 20)

**Example:**
```python
from engine.position_manager import VolatilityBasedSizing, PositionManager

sizing_strategy = VolatilityBasedSizing(risk_pct=0.02, lookback_period=30)
position_manager = PositionManager(sizing_strategy=sizing_strategy)
```

#### KellySizingStrategy

Position sizing based on the Kelly Criterion.

**Constructor Parameters:**
- `win_rate` (float, optional): Historical win rate (default: 0.5)
- `win_loss_ratio` (float, optional): Ratio of average win to average loss (default: 1.0)
- `fraction` (float, optional): Fraction of Kelly to use (default: 0.5)

**Example:**
```python
from engine.position_manager import KellySizingStrategy, PositionManager

sizing_strategy = KellySizingStrategy(win_rate=0.6, win_loss_ratio=1.5, fraction=0.3)
position_manager = PositionManager(sizing_strategy=sizing_strategy)
```

### Risk Management

The engine module provides risk management tools through the RiskManager class.

#### RiskManager

Manages risk controls and limits for trading.

**Constructor Parameters:**
- `max_position_pct` (float, optional): Maximum position size as percentage of portfolio (default: 0.25)
- `max_drawdown_pct` (float, optional): Maximum allowable drawdown from peak equity (default: 0.10)
- `max_concentration_pct` (float, optional): Maximum allocation to a single instrument (default: None)
- `use_stop_loss` (bool, optional): Whether to use stop losses (default: False)
- `stop_loss_pct` (float, optional): Stop loss percentage from entry (default: 0.05)
- `use_take_profit` (bool, optional): Whether to use take profits (default: False)
- `take_profit_pct` (float, optional): Take profit percentage from entry (default: 0.10)

**Example:**
```python
from engine.position_manager import RiskManager, PositionManager

risk_manager = RiskManager(
    max_position_pct=0.1,  # Max 10% of portfolio in one position
    use_stop_loss=True,    # Use stop losses
    stop_loss_pct=0.03     # 3% stop loss
)

position_manager = PositionManager(risk_manager=risk_manager)
```

### Event System

The engine module uses an event-driven architecture for communication between components.

#### EventType

Enumeration of different event types in the system.

```python
class EventType(Enum):
    BAR = auto()      # Market data bar
    SIGNAL = auto()   # Trading signal
    ORDER = auto()    # Order request
    FILL = auto()     # Order fill
```

#### Event

Base class for all events in the system.

**Constructor Parameters:**
- `event_type` (EventType): Type of the event
- `data` (Any, optional): Data payload

**Example:**
```python
from engine import Event, EventType

# Create an event
bar_event = Event(EventType.BAR, bar_data)
```

#### Order

Represents a trading order.

**Constructor Parameters:**
- `symbol` (str): Instrument symbol
- `order_type` (str): Type of order ("MARKET", "LIMIT", etc.)
- `quantity` (float): Order quantity
- `direction` (int): Order direction (1 for buy, -1 for sell)
- `timestamp` (datetime): Order timestamp

**Example:**
```python
from engine import Order

# Create a market buy order
order = Order(
    symbol="AAPL",
    order_type="MARKET",
    quantity=100,
    direction=1,
    timestamp=datetime.now()
)
```

## Advanced Usage

### Custom Position Sizing

Create a custom position sizing strategy:

```python
from engine.position_manager import SizingStrategy, PositionManager

class ATRPercentRiskSizing(SizingStrategy):
    """
    Size positions based on ATR and percent risk.
    
    This sizes positions to risk a fixed percentage of equity
    based on the ATR-based stop loss distance.
    """
    
    def __init__(self, risk_percent=0.01, atr_multiple=2):
        """
        Initialize with risk percentage and ATR multiple.
        
        Args:
            risk_percent: Percentage of equity to risk per trade (0.01 = 1%)
            atr_multiple: Multiple of ATR for stop distance
        """
        self.risk_percent = risk_percent
        self.atr_multiple = atr_multiple
        
    def calculate_size(self, signal, portfolio):
        """Calculate position size based on ATR and risk percentage."""
        if not hasattr(signal, 'price') or signal.price <= 0:
            return 0
            
        # Get ATR if available in signal metadata
        atr = signal.metadata.get('atr', None)
        if atr is None or atr <= 0:
            # Default to 2% of price if ATR not available
            atr = signal.price * 0.02
        
        # Calculate stop distance
        stop_distance = atr * self.atr_multiple
        
        # Risk amount in dollar terms
        risk_amount = portfolio.equity * self.risk_percent
        
        # Calculate position size
        size = risk_amount / stop_distance
        
        # Adjust direction based on signal
        if hasattr(signal, 'signal_type'):
            direction = signal.signal_type.value
        elif hasattr(signal, 'signal'):
            direction = signal.signal
        else:
            direction = 1  # Default to buy
            
        if direction < 0:
            size = -size
            
        return size

# Use the custom sizing strategy
custom_sizing = ATRPercentRiskSizing(risk_percent=0.02, atr_multiple=3)
position_manager = PositionManager(sizing_strategy=custom_sizing)
```

### Advanced Risk Management

Implement more sophisticated risk management:

```python
from engine.position_manager import RiskManager, PositionManager
from datetime import datetime, time

class EnhancedRiskManager(RiskManager):
    """
    Enhanced risk manager with time-based constraints and position scaling.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_open_positions = kwargs.get('max_open_positions', 5)
        self.no_trading_times = kwargs.get('no_trading_times', [])
        self.position_scaling = kwargs.get('position_scaling', False)
        self.scale_factor = kwargs.get('scale_factor', 0.5)
        
    def check_signal(self, signal, portfolio):
        """Check if a signal should be acted upon given risk constraints."""
        # Call parent implementation
        if not super().check_signal(signal, portfolio):
            return False
            
        # Check time restrictions
        if hasattr(signal, 'timestamp'):
            current_time = signal.timestamp
            if isinstance(current_time, datetime):
                # Check no-trading times
                for start_time, end_time in self.no_trading_times:
                    if start_time <= current_time.time() <= end_time:
                        return False  # Don't trade during restricted times
        
        # Check max open positions
        if len(portfolio.positions) >= self.max_open_positions:
            return False  # Too many open positions
            
        return True
        
    def adjust_position_size(self, symbol, size, portfolio):
        """Adjust position size to comply with risk limits."""
        # Call parent implementation
        size = super().adjust_position_size(symbol, size, portfolio)
        
        # Implement position scaling based on drawdown
        if self.position_scaling:
            # Calculate current drawdown
            current_drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity if self.peak_equity > 0 else 0
            
            # Scale position size down as drawdown increases
            if current_drawdown > 0.05:  # Start scaling at 5% drawdown
                # Calculate scaling factor (linearly decrease from 1.0 to scale_factor)
                scale = 1.0 - ((current_drawdown - 0.05) / 0.2) * (1.0 - self.scale_factor)
                scale = max(self.scale_factor, min(1.0, scale))
                
                # Apply scaling
                size *= scale
                
        return size

# Use the enhanced risk manager
enhanced_risk = EnhancedRiskManager(
    max_position_pct=0.1,
    use_stop_loss=True,
    stop_loss_pct=0.03,
    max_open_positions=3,
    no_trading_times=[
        (time(9, 30), time(10, 0)),  # No trading first 30 minutes
        (time(15, 30), time(16, 0))  # No trading last 30 minutes
    ],
    position_scaling=True,
    scale_factor=0.3
)

position_manager = PositionManager(risk_manager=enhanced_risk)
```

### Creating Custom Market Simulator

Implement a custom market simulation model:

```python
from engine import MarketSimulator
from engine.market_simulator import SlippageModel, FeeModel

class JumpSlippageModel(SlippageModel):
    """
    Slippage model that simulates price jumps for large orders.
    
    This model applies minimal slippage to small orders, but
    significant slippage when orders exceed a threshold size.
    """
    
    def __init__(self, threshold_pct=0.01, small_slip_bps=3, large_slip_bps=20):
        """
        Initialize with threshold and slippage parameters.
        
        Args:
            threshold_pct: Threshold for large orders as percentage of volume
            small_slip_bps: Slippage in basis points for small orders
            large_slip_bps: Slippage in basis points for large orders
        """
        self.threshold_pct = threshold_pct
        self.small_slip_bps = small_slip_bps
        self.large_slip_bps = large_slip_bps
    
    def apply_slippage(self, price, quantity, direction, bar):
        """Apply slippage to a base price based on order size."""
        # Default to small slippage
        slippage_bps = self.small_slip_bps
        
        # If volume data is available, check threshold
        if 'Volume' in bar and bar['Volume'] > 0:
            volume_ratio = abs(quantity) / bar['Volume']
            
            # Apply jump slippage for large orders
            if volume_ratio > self.threshold_pct:
                slippage_bps = self.large_slip_bps
                
                # Add random jump component for very large orders
                if volume_ratio > self.threshold_pct * 2:
                    # Add extra slippage with 20% probability
                    if np.random.random() < 0.2:
                        slippage_bps *= 2  # Double slippage to simulate jump
        
        # Convert BPS to factor
        slippage_factor = slippage_bps / 10000
        
        # Apply slippage in the adverse direction
        if direction > 0:  # Buy - price goes up
            return price * (1 + slippage_factor)
        else:  # Sell - price goes down
            return price * (1 - slippage_factor)

# Create custom market simulator
custom_simulator = MarketSimulator({
    'slippage_model': 'custom',  # Not actually used, we'll override
    'fee_model': 'tiered',
    'tier_thresholds': [10000, 100000],
    'tier_fees_bps': [10, 5, 3]
})

# Override the slippage model
custom_simulator.slippage_model = JumpSlippageModel(
    threshold_pct=0.02,
    small_slip_bps=2,
    large_slip_bps=15
)
```

### Backtesting Multiple Strategies

Run backtests on multiple strategies for comparison:

```python
def backtest_multiple_strategies(data_handler, strategies, config=None):
    """
    Backtest multiple strategies and compare results.
    
    Args:
        data_handler: Data handler with market data
        strategies: Dictionary of strategy names and objects
        config: Optional configuration object
        
    Returns:
        dict: Comparison results
    """
    from engine import Backtester
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create default config if none provided
    if config is None:
        from config import ConfigManager
        config = ConfigManager()
    
    # Store results
    all_results = {}
    
    # Run backtest for each strategy
    for name, strategy in strategies.items():
        print(f"Backtesting strategy: {name}")
        
        # Create and run backtester
        backtester = Backtester(config, data_handler, strategy)
        results = backtester.run(use_test_data=True)
        
        # Calculate additional metrics
        sharpe = backtester.calculate_sharpe()
        
        # Store results
        all_results[name] = {
            'trades': results['trades'],
            'num_trades': results['num_trades'],
            'total_return': results['total_percent_return'],
            'sharpe_ratio': sharpe,
            'portfolio_history': results['portfolio_history']
        }
        
    # Create comparison summary
    summary = {
        'num_trades': {name: results['num_trades'] for name, results in all_results.items()},
        'total_return': {name: results['total_return'] for name, results in all_results.items()},
        'sharpe_ratio': {name: results['sharpe_ratio'] for name, results in all_results.items()}
    }
    
    # Create equity curves for visualization
    plt.figure(figsize=(12, 8))
    
    for name, results in all_results.items():
        # Extract equity values
        equity = [p['equity'] for p in results['portfolio_history']]
        
        # Plot equity curve
        plt.plot(equity, label=name)
    
    plt.title("Strategy Comparison: Equity Curves")
    plt.xlabel("Bar Number")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary)
    
    return {
        'detailed_results': all_results,
        'summary': summary_df,
        'plot': plt.gcf()
    }

# Use the function
sma_strategy = WeightedStrategy(rules=sma_rules)
rsi_strategy = WeightedStrategy(rules=rsi_rules)
combined_strategy = WeightedStrategy(rules=combined_rules)

strategies = {
    'SMA': sma_strategy,
    'RSI': rsi_strategy,
    'Combined': combined_strategy
}

comparison = backtest_multiple_strategies(data_handler, strategies, config)
print(comparison['summary'])
```

### Using the Event System

Create a custom event handler for monitoring:

```python
from engine import EventBus, Event, EventType, EventHandler

class PerformanceMonitor(EventHandler):
    """
    Event handler that monitors performance metrics in real-time during backtesting.
    """
    
    def __init__(self):
        super().__init__([EventType.FILL, EventType.BAR])
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_equity = 0
        self.current_equity = 0
        self.trades_by_hour = {}
        
    def _process_event(self, event):
        """Process different event types."""
        if event.event_type == EventType.FILL:
            # Process trade fill
            self.trade_count += 1
            
            # Calculate profit/loss if this closes a position
            fill = event.data
            if hasattr(fill, 'pnl') and fill.pnl is not None:
                if fill.pnl > 0:
                    self.win_count += 1
                elif fill.pnl < 0:
                    self.loss_count += 1
                    
                # Track trades by hour
                hour = fill.timestamp.hour
                self.trades_by_hour[hour] = self.trades_by_hour.get(hour, 0) + 1
                
        elif event.event_type == EventType.BAR:
            # Update equity and drawdown tracking
            if hasattr(event, 'portfolio'):
                portfolio = event.portfolio
                self.current_equity = portfolio.equity
                
                # Update peak equity
                if self.current_equity > self.peak_equity:
                    self.peak_equity = self.current_equity
                    self.current_drawdown = 0
                else:
                    # Calculate current drawdown
                    self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
                    
                    # Update max drawdown
                    if self.current_drawdown > self.max_drawdown:
                        self.max_drawdown = self.current_drawdown
    
    def get_statistics(self):
        """Get current performance statistics."""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        
        return {
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'current_equity': self.current_equity,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'trades_by_hour': dict(sorted(self.trades_by_hour.items()))
        }

# Use the performance monitor
event_bus = EventBus()
monitor = PerformanceMonitor()

# Register the monitor with the event bus
event_bus.register(EventType.FILL, monitor)
event_bus.register(EventType.BAR, monitor)

# After backtesting, get statistics
stats = monitor.get_statistics()
print(f"Win Rate: {stats['win_rate']:.2%}")
print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
```

## Integration with Other Components

### Using Position Manager with Signal Processor

```python
from engine import PositionManager
from signals import SignalProcessor
from config import ConfigManager

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
from engine import Backtester, PositionManager
from regime_detection import RegimeType, VolatilityRegimeDetector

class RegimeAwarePositionManager(PositionManager):
    """
    Position manager that adapts sizing based on detected market regime.
    """
    
    def __init__(self, regime_detector, **kwargs):
        super().__init__(**kwargs)
        self.regime_detector = regime_detector
        self.regime_position_factors = {
            RegimeType.TRENDING_UP: 1.0,      # Normal sizing
            RegimeType.TRENDING_DOWN: 0.5,    # Half sizing
            RegimeType.RANGE_BOUND: 0.75,     # 75% sizing
            RegimeType.VOLATILE: 0.3,         # 30% sizing
            RegimeType.LOW_VOLATILITY: 1.2,   # 120% sizing
            RegimeType.UNKNOWN: 0.5           # Half sizing if unknown
        }
        
    def calculate_position_size(self, signal, portfolio):
        """Calculate position size with regime adjustment."""
        # Get base position size
        base_size = super().calculate_position_size(signal, portfolio)
        
        # Get current regime if available
        current_regime = RegimeType.UNKNOWN
        if hasattr(signal, 'metadata') and 'regime' in signal.metadata:
            current_regime = signal.metadata['regime']
        elif hasattr(self.regime_detector, 'current_regime'):
            current_regime = self.regime_detector.current_regime
            
        # Apply regime-specific sizing factor
        regime_factor = self.regime_position_factors.get(current_regime, 1.0)
        adjusted_size = base_size * regime_factor
        
        return adjusted_size

# Create regime detector and regime-aware position manager
regime_detector = VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015)
position_manager = RegimeAwarePositionManager(regime_detector)

# Create backtester with regime detection
backtester = Backtester(config, data_handler, strategy, position_manager)
```

## Best Practices

1. **Always use realistic slippage and fees**: Configure the MarketSimulator with realistic parameters to avoid overly optimistic results

2. **Test with conservative risk management**: Start with conservative risk parameters and gradually relax them rather than the opposite

3. **Reset state between backtests**: Always call `reset()` on all components when running multiple backtests

4. **Validate position sizing logic**: Verify that position sizing behaves as expected by checking allocation across different market conditions

5. **Monitor execution details**: Keep track of execution prices vs. signal prices to understand the impact of slippage

6. **Use event handlers for custom analytics**: Leverage the event system to monitor specific aspects of the backtest

7. **Separate parameter optimization from backtesting**: Use the backtester to evaluate strategies with pre-optimized parameters rather than doing both at once
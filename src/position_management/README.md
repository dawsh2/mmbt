# Position Management Module

The Position Management module is responsible for handling position creation, tracking, sizing, and risk management within the trading system. It provides a comprehensive framework for managing both individual positions and the entire portfolio.

## Overview

This module implements several key responsibilities:

1. **Position Representation** - Models trading positions and their lifecycle
2. **Portfolio Management** - Tracks multiple positions and their aggregated metrics
3. **Position Sizing** - Calculates appropriate position sizes based on risk parameters
4. **Capital Allocation** - Determines how to distribute capital across multiple trading opportunities
5. **Risk Management Integration** - Works with the Risk Management module to apply risk constraints

## Core Components

### Position

The `Position` class is the fundamental building block that represents a single trading position:

```python
from position_management.position import Position, EntryType, ExitType, PositionStatus
from datetime import datetime

# Create a position
position = Position(
    symbol="AAPL",
    direction=1,  # 1 for long, -1 for short
    quantity=100,
    entry_price=150.0,
    entry_time=datetime.now(),
    stop_loss=145.0,
    take_profit=160.0
)

# Update with current price
exit_info = position.update_price(current_price=152.5, timestamp=datetime.now())
if exit_info:
    print(f"Exit triggered: {exit_info['exit_reason']}")

# Close position
summary = position.close(
    exit_price=153.0,
    exit_time=datetime.now(),
    exit_type=ExitType.STRATEGY
)
```

Key features:
- Tracks entry and exit details
- Maintains current state and position metrics
- Handles stop loss and take profit levels
- Supports trailing stops
- Calculates P&L and other performance metrics
- Manages position lifecycle (open, partially closed, closed)

### PositionFactory

The `PositionFactory` provides convenient methods for creating positions with different configurations:

```python
from position_management.position import PositionFactory
from datetime import datetime

# Create a position with stops based on percentages
position = PositionFactory.create_position_with_stops(
    symbol="MSFT",
    direction=1,
    quantity=100,
    entry_price=250.0,
    entry_time=datetime.now(),
    stop_loss_pct=0.02,  # 2% stop loss
    take_profit_pct=0.05  # 5% take profit
)

# Create a position sized by risk
position = PositionFactory.create_position_with_risk(
    symbol="GOOGL",
    direction=1,
    entry_price=2000.0,
    entry_time=datetime.now(),
    account_size=100000.0,
    risk_pct=0.01,  # Risk 1% of account
    stop_loss_price=1950.0
)

# Create a position from a signal
position = PositionFactory.create_from_signal(
    signal={
        'symbol': 'AAPL',
        'direction': 'buy',
        'price': 150.0,
        'timestamp': datetime.now(),
        'stop_loss': 145.0,
        'take_profit': 160.0,
        'confidence': 0.8
    },
    account_size=100000.0,
    risk_pct=0.01
)
```

### Portfolio

The `Portfolio` class manages multiple positions and tracks overall portfolio metrics:

```python
from position_management.portfolio import Portfolio
from datetime import datetime

# Create a portfolio
portfolio = Portfolio(initial_capital=100000.0)

# Open a position
position = portfolio.open_position(
    symbol="AAPL",
    direction=1,
    quantity=100,
    entry_price=150.0,
    entry_time=datetime.now(),
    stop_loss=145.0,
    take_profit=160.0
)

# Update portfolio with current prices
current_prices = {
    "AAPL": 152.0,
    "MSFT": 250.0
}
exits = portfolio.update_prices(current_prices, datetime.now())

# Close a position
portfolio.close_position(
    position_id=position.position_id,
    exit_price=155.0,
    exit_time=datetime.now()
)

# Get portfolio metrics
metrics = portfolio.get_performance_metrics()
print(f"Portfolio equity: ${metrics['current_equity']:.2f}")
print(f"Return: {metrics['roi_percent']:.2f}%")
```

Key features:
- Tracks multiple positions across different instruments
- Manages portfolio equity, cash, and margin
- Calculates portfolio-level metrics
- Handles position entry and exit
- Provides position lookup by ID or symbol
- Tracks transaction costs

### Position Sizers

The `position_sizers` module provides strategies for determining appropriate position sizes:

```python
from position_management.position_sizers import (
    FixedSizeSizer,
    PercentOfEquitySizer,
    VolatilityPositionSizer,
    KellyCriterionSizer,
    PositionSizerFactory
)

# Create a volatility-based position sizer
volatility_sizer = VolatilityPositionSizer(
    risk_pct=0.01,  # Risk 1% of equity
    atr_multiplier=2.0  # Use 2 ATRs for stop distance
)

# Calculate position size for a signal
size = volatility_sizer.calculate_position_size(
    signal={
        'symbol': 'AAPL',
        'direction': 1,
        'atr': 2.5  # Average True Range
    },
    portfolio=portfolio,
    current_price=150.0
)

# Create a position sizer from configuration
sizer = PositionSizerFactory.create_from_config({
    'type': 'kelly',
    'params': {
        'win_rate': 0.6,
        'reward_risk_ratio': 2.0,
        'fraction': 0.5
    }
})
```

Available position sizers:
- **FixedSizeSizer** - Always uses a fixed number of units
- **PercentOfEquitySizer** - Sizes positions as a percentage of account equity
- **VolatilityPositionSizer** - Sizes based on price volatility (ATR)
- **KellyCriterionSizer** - Uses the Kelly formula for optimal position sizing
- **RiskParityPositionSizer** - Allocates risk equally across positions
- **PSARPositionSizer** - Uses Parabolic SAR for stop placement and sizing
- **AdaptivePositionSizer** - Adjusts sizing based on market conditions

### Allocation Strategies

The `allocation` module provides strategies for distributing capital across multiple trading opportunities:

```python
from position_management.allocation import (
    EqualWeightAllocation,
    SignalStrengthAllocation,
    VolatilityParityAllocation,
    AllocationStrategyFactory
)

# Create an allocation strategy
allocation_strategy = VolatilityParityAllocation()

# Calculate allocation weights
allocation_weights = allocation_strategy.allocate(
    portfolio=portfolio,
    signals={
        'AAPL': {'symbol': 'AAPL', 'direction': 1, 'confidence': 0.8, 'volatility': 0.012},
        'MSFT': {'symbol': 'MSFT', 'direction': 1, 'confidence': 0.6, 'volatility': 0.010}
    },
    prices={'AAPL': 150.0, 'MSFT': 250.0}
)

# Create an allocation strategy from configuration
strategy = AllocationStrategyFactory.create_from_config({
    'type': 'signal_strength',
    'params': {}
})
```

Available allocation strategies:
- **EqualWeightAllocation** - Allocates equal capital to each instrument
- **MarketCapAllocation** - Allocates based on market capitalization
- **SignalStrengthAllocation** - Allocates based on signal confidence
- **VolatilityParityAllocation** - Allocates to achieve equal risk contribution
- **MaximumSharpeAllocation** - Allocates to maximize Sharpe ratio
- **ConstrainedAllocation** - Applies constraints to an underlying allocation

### Position Manager

The `PositionManager` class integrates position sizing, allocation, and risk management:

```python
from position_management.position_manager import PositionManager
from position_management.position_sizers import VolatilityPositionSizer
from position_management.allocation import SignalStrengthAllocation
from position_management.portfolio import Portfolio

# Create portfolio
portfolio = Portfolio(initial_capital=100000.0)

# Create components
position_sizer = VolatilityPositionSizer(risk_pct=0.01)
allocation_strategy = SignalStrengthAllocation()

# Create position manager
position_manager = PositionManager(
    portfolio=portfolio,
    position_sizer=position_sizer,
    allocation_strategy=allocation_strategy,
    max_positions=10
)

# Process a signal event
actions = position_manager.on_signal({
    'symbol': 'AAPL',
    'direction': 1,
    'price': 150.0,
    'confidence': 0.8,
    'volatility': 0.012,
    'atr': 1.8
})

# Execute position actions
for action in actions:
    result = position_manager.execute_position_action(
        action=action,
        current_time=datetime.now()
    )

# Update positions with current prices
exit_actions = position_manager.update_positions(
    current_prices={'AAPL': 152.0},
    current_time=datetime.now()
)

# Get risk exposure
risk_metrics = position_manager.get_portfolio_risk_exposure()
print(f"Capital at risk: {risk_metrics['capital_at_risk_pct']:.2%}")
```

Key features:
- Coordinates position sizing and allocation
- Processes trading signals into position actions
- Executes position entry and exit
- Updates positions with current prices
- Calculates risk exposure metrics
- Provides position queries and reporting

## Integration with Risk Management

The Position Management module integrates with the Risk Management module:

```python
from position_management.position_manager import PositionManager
from risk_management import RiskManager

# Create risk manager
risk_manager = RiskManager(
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    max_position_pct=0.05
)

# Create position manager with risk manager
position_manager = PositionManager(
    portfolio=portfolio,
    position_sizer=position_sizer,
    allocation_strategy=allocation_strategy,
    risk_manager=risk_manager
)
```

The risk manager can:
- Evaluate position sizes against risk limits
- Calculate stop loss and take profit levels
- Apply position-level risk constraints
- Apply portfolio-level risk constraints

## Event Integration

The Position Management module integrates with the Events module:

- Position actions can be emitted as events
- Position manager can respond to signal events
- Position updates can be emitted as events

```python
from events.event_bus import EventBus
from events.event_types import EventType
from position_management.position_manager import PositionManager

# Create event bus
event_bus = EventBus()

# Create position manager with event bus
position_manager = PositionManager(
    portfolio=portfolio,
    position_sizer=position_sizer,
    allocation_strategy=allocation_strategy,
    event_bus=event_bus
)

# Register signal handler
event_bus.register(EventType.SIGNAL, position_manager.on_signal)
```

## Examples

### Creating and Managing an Order with Stop Loss

```python
from position_management.position import PositionFactory, ExitType
from position_management.portfolio import Portfolio
from datetime import datetime

# Create portfolio
portfolio = Portfolio(initial_capital=100000.0)

# Open a position with stop loss
position = portfolio.open_position(
    symbol="AAPL",
    direction=1,  # Long
    quantity=100,
    entry_price=150.0,
    entry_time=datetime.now(),
    stop_loss=145.0,  # $5 stop loss
    take_profit=160.0  # $10 take profit
)

# Update with new prices over time
for price in [151.0, 153.0, 148.0, 144.0]:
    current_time = datetime.now()
    exits = portfolio.update_prices({"AAPL": price}, current_time)
    
    # Check if any exits were triggered
    for exit_info in exits:
        print(f"Exit triggered: {exit_info['exit_reason']} at ${exit_info['exit_price']}")
        
        # Confirm the exit in the portfolio
        portfolio.close_position(
            position_id=exit_info['position_id'],
            exit_price=exit_info['exit_price'],
            exit_time=current_time,
            exit_type=exit_info['exit_type']
        )
```

### Implementing a Risk-Based Position Sizing Strategy

```python
from position_management.position_sizers import VolatilityPositionSizer
from position_management.position import PositionFactory
from datetime import datetime

# Create position sizer
position_sizer = VolatilityPositionSizer(
    risk_pct=0.01,  # Risk 1% of account
    atr_multiplier=2.0  # Use 2 ATRs for stop distance
)

# Account information
account_size = 100000.0

# Signal with ATR information
signal = {
    'symbol': 'TSLA',
    'direction': 1,
    'price': 200.0,
    'atr': 8.5  # $8.50 ATR
}

# Calculate position size
size = position_sizer.calculate_position_size(
    signal=signal,
    portfolio={'equity': account_size},
    current_price=signal['price']
)

# Calculate stop distance and price
stop_distance = signal['atr'] * position_sizer.atr_multiplier
stop_price = signal['price'] - stop_distance

# Create position with calculated size and stop
position = PositionFactory.create_position(
    symbol=signal['symbol'],
    direction=signal['direction'],
    quantity=size,
    entry_price=signal['price'],
    entry_time=datetime.now(),
    stop_loss=stop_price
)

print(f"Position size: {size:.2f} shares")
print(f"Entry price: ${signal['price']:.2f}")
print(f"Stop loss: ${stop_price:.2f}")
print(f"Risk amount: ${account_size * position_sizer.risk_pct:.2f}")
```

### Portfolio Management with Multiple Positions

```python
from position_management.portfolio import Portfolio
from position_management.position import PositionFactory
from datetime import datetime

# Create portfolio
portfolio = Portfolio(initial_capital=100000.0)

# Open positions for multiple symbols
portfolio.open_position(
    symbol="AAPL",
    direction=1,
    quantity=100,
    entry_price=150.0,
    entry_time=datetime.now(),
    stop_loss=145.0
)

portfolio.open_position(
    symbol="MSFT",
    direction=1,
    quantity=50,
    entry_price=250.0,
    entry_time=datetime.now(),
    stop_loss=240.0
)

portfolio.open_position(
    symbol="GOOGL",
    direction=-1,  # Short
    quantity=10,
    entry_price=2000.0,
    entry_time=datetime.now(),
    stop_loss=2050.0
)

# Update portfolio with current prices
current_prices = {
    "AAPL": 155.0,
    "MSFT": 260.0,
    "GOOGL": 1950.0
}

portfolio.update_prices(current_prices, datetime.now())

# Get portfolio metrics
metrics = portfolio.get_performance_metrics()

print(f"Portfolio equity: ${metrics['current_equity']:.2f}")
print(f"Cash: ${metrics['cash']:.2f}")
print(f"Unrealized P&L: ${metrics['unrealized_pnl']:.2f}")
print(f"Return: {metrics['roi_percent']:.2f}%")
print(f"Max drawdown: {metrics['max_drawdown_percent']:.2f}%")
print(f"Open positions: {metrics['open_positions']}")

# Close all positions
portfolio.close_all_positions(current_prices, datetime.now())
```

## Best Practices

1. **Use the PositionFactory** for creating positions with consistent configurations
2. **Implement proper risk management** by always setting stop losses
3. **Choose appropriate position sizing** based on your strategy's characteristics
4. **Monitor portfolio-level metrics** not just individual positions
5. **Update positions regularly** with current market prices
6. **Record transaction costs** to get accurate performance metrics
7. **Implement trailing stops** for trend-following strategies
8. **Use appropriate allocation strategies** for your portfolio's objectives
9. **Handle partial fills/closes** correctly to maintain accurate position tracking
10. **Integrate with the event system** for consistent system-wide communication
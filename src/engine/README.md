# Engine Module

No module overview available.

## Contents

- [backtester](#backtester)
- [execution_engine](#execution_engine)
- [market_simulator](#market_simulator)
- [position_manager](#position_manager)
- [position_sizers](#position_sizers)

## backtester

Backtester for Trading System - Fixed Version

This module provides the main Backtester class which orchestrates the backtesting process,
ensuring proper initialization of components and event flow.

### Classes

#### `Backtester`

Main orchestration class that coordinates the backtest execution.
Acts as the facade for the backtesting subsystem.

##### Methods

###### `__init__(config, data_handler, strategy, position_manager=None)`

Initialize the backtester with configuration and dependencies.

Args:
    config: Configuration dictionary or ConfigManager instance
    data_handler: Data handler providing market data
    strategy: Trading strategy to test
    position_manager: Optional position manager for risk management

###### `_extract_market_sim_config(config)`

Extract market simulation configuration.

###### `_extract_initial_capital(config)`

Extract initial capital from configuration.

###### `_setup_event_handlers()`

Set up event handlers with proper registration.

###### `_create_bar_handler()`

Create a handler for BAR events.

###### `_create_signal_handler()`

Create a handler for SIGNAL events.

###### `_create_order_handler()`

Create a handler for ORDER events.

###### `_create_fill_handler()`

Create a handler for FILL events.

###### `_create_position_action_handler()`

Create a handler for POSITION_ACTION events.

###### `run(use_test_data=False)`

Run the backtest with proper event flow.

Args:
    use_test_data: Whether to use test data (True) or training data (False)

Returns:
    dict: Backtest results

*Returns:* dict: Backtest results

###### `reset()`

Reset the backtester state and all components.

###### `collect_results()`

Collect backtest results for analysis.

Returns:
    dict: Results dictionary with trades, portfolio history, etc.

*Returns:* dict: Results dictionary with trades, portfolio history, etc.

###### `_process_trades(trade_history)`

Process trade history into a standardized format.

###### `_calculate_performance_metrics(processed_trades)`

Calculate performance metrics from processed trades.

###### `calculate_sharpe(risk_free_rate=0.0, annualization_factor=252)`

Calculate Sharpe ratio for the backtest results.

###### `calculate_max_drawdown()`

Calculate maximum drawdown for the backtest.

## execution_engine

Execution Engine with Standardized Event Objects

This is a reference implementation of the ExecutionEngine class
that uses standardized event objects throughout.

### Classes

#### `ExecutionEngine`

Handles order execution, position tracking, and portfolio management.

##### Methods

###### `__init__(position_manager=None, market_simulator=None)`

Initialize the execution engine.

###### `on_signal(event)`

Process a signal and convert to an order if appropriate.

Args:
    event: Event containing a SignalEvent

Returns:
    OrderEvent if order was created, None otherwise

*Returns:* OrderEvent if order was created, None otherwise

###### `on_order(event)`

Handle incoming order events.

Args:
    event: Order event

###### `execute_pending_orders(bar, market_simulator=None)`

Execute any pending orders based on current bar data.

###### `_execute_order(order, price, timestamp, commission=0.0)`

Execute a single order and update portfolio.

Args:
    order: OrderEvent to execute
    price: Execution price
    timestamp: Execution timestamp
    commission: Commission cost

Returns:
    FillEvent if successful

*Returns:* FillEvent if successful

###### `_emit_fill_event(fill)`

Emit fill event to the event bus.

###### `_get_last_known_price(symbol)`

Get the last known price for a symbol.

###### `update(bar)`

Update portfolio with latest market data.

Args:
    bar: Current market data (BarEvent or dict)

###### `get_trade_history()`

Get the history of all executed trades.

###### `get_portfolio_history()`

Get the history of portfolio states.

###### `get_signal_history()`

Get the history of signals received.

###### `on_position_action(event)`

Handle position action events.

Args:
    event: Event containing position action

###### `reset()`

Reset the execution engine state.

## market_simulator

Market Simulator for Trading System

This module simulates realistic market conditions including slippage, transaction
costs, and other market effects that impact trade execution in the backtesting system.

### Classes

#### `SlippageModel`

Base class for slippage models.

##### Methods

###### `apply_slippage(price, quantity, direction, bar)`

*Returns:* `float`

Apply slippage to a base price.

Args:
    price: Base price
    quantity: Order quantity
    direction: Order direction (1 for buy, -1 for sell)
    bar: Current bar data
    
Returns:
    float: Price after slippage

#### `NoSlippageModel`

No slippage model - returns the base price unchanged.

##### Methods

###### `apply_slippage(price, quantity, direction, bar)`

*Returns:* `float`

No docstring provided.

#### `FixedSlippageModel`

Fixed slippage model - applies a fixed basis point slippage to the price.

A basis point (BPS) is 1/100th of a percent. So 5 BPS = 0.05%.

##### Methods

###### `__init__(slippage_bps=5)`

Initialize with slippage in basis points.

Args:
    slippage_bps: Slippage in basis points (5 = 0.05%)

###### `apply_slippage(price, quantity, direction, bar)`

*Returns:* `float`

No docstring provided.

#### `VolumeBasedSlippageModel`

Volume-based slippage model that scales with order size relative to volume.

Uses a price impact formula based on the square root of the ratio of
order quantity to bar volume, scaled by the price impact parameter.

##### Methods

###### `__init__(price_impact=0.1)`

Initialize with price impact parameter.

Args:
    price_impact: Price impact parameter (higher = more slippage)

###### `apply_slippage(price, quantity, direction, bar)`

*Returns:* `float`

Apply slippage based on volume.

#### `FeeModel`

Base class for fee models.

##### Methods

###### `calculate_fee(quantity, price)`

*Returns:* `float`

Calculate transaction fee.

Args:
    quantity: Order quantity
    price: Execution price
    
Returns:
    float: Fee amount

#### `NoFeeModel`

No fee model - returns zero fees.

##### Methods

###### `calculate_fee(quantity, price)`

*Returns:* `float`

No docstring provided.

#### `FixedFeeModel`

Fixed fee model - applies a fixed basis point fee to the transaction value.

##### Methods

###### `__init__(fee_bps=10)`

Initialize with fee in basis points.

Args:
    fee_bps: Fee in basis points (10 = 0.1%)

###### `calculate_fee(quantity, price)`

*Returns:* `float`

No docstring provided.

#### `TieredFeeModel`

Tiered fee model with different fees based on transaction value.

For example:
- Transactions under $10,000: 0.1%
- Transactions $10,000-$100,000: 0.05%
- Transactions over $100,000: 0.03%

##### Methods

###### `__init__(tier_thresholds=None, tier_fees_bps=None)`

Initialize with tier thresholds and fees.

Args:
    tier_thresholds: List of tier thresholds in ascending order
    tier_fees_bps: List of fees in basis points for each tier

###### `calculate_fee(quantity, price)`

*Returns:* `float`

No docstring provided.

#### `MarketSimulator`

Simulates market effects like slippage, delays, and transaction costs.

##### Methods

###### `__init__(config=None)`

Initialize the market simulator.

Args:
    config: Optional configuration dictionary

###### `_get_slippage_model()`

*Returns:* `SlippageModel`

Create the slippage model based on configuration.

###### `_get_fee_model()`

*Returns:* `FeeModel`

Create the fee model based on configuration.

###### `calculate_execution_price(order, bar)`

*Returns:* `float`

Calculate the execution price including slippage.

Args:
    order: Order object with quantity and direction
    bar: Current bar data (either BarEvent or dict)
    
Returns:
    float: Execution price with slippage

###### `calculate_fees(order, execution_price)`

*Returns:* `float`

Calculate transaction fees.

Args:
    order: Order object with quantity
    execution_price: Price after slippage
    
Returns:
    float: Fee amount

## position_manager

Position Management Module for Trading System

This module provides position sizing, risk management, and allocation strategies
for controlling how signals are converted into actual trading positions.

### Classes

#### `SizingStrategy`

Base class for position sizing strategies.

##### Methods

###### `calculate_size(signal, portfolio)`

Calculate position size based on signal and portfolio.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    
Returns:
    float: Position size (positive for buy, negative for sell)

*Returns:* float: Position size (positive for buy, negative for sell)

#### `FixedSizingStrategy`

Position sizing using a fixed number of units.

##### Methods

###### `__init__(fixed_size=100)`

Initialize with fixed size.

Args:
    fixed_size: Number of units to trade

###### `calculate_size(signal, portfolio)`

Calculate position size.

#### `PercentOfEquitySizing`

Size position as a percentage of portfolio equity.

##### Methods

###### `__init__(percent=0.02)`

Initialize with percent of equity to risk.

Args:
    percent: Percentage of equity to allocate (0.02 = 2%)

###### `calculate_size(signal, portfolio)`

Calculate position size.

#### `VolatilityBasedSizing`

Size positions based on asset volatility.

##### Methods

###### `__init__(risk_pct=0.01, lookback_period=20)`

Initialize with risk percentage and lookback.

Args:
    risk_pct: Percentage of equity to risk per unit of volatility
    lookback_period: Period for calculating volatility

###### `calculate_size(signal, portfolio)`

Calculate position size based on volatility.

#### `KellySizingStrategy`

Position sizing based on the Kelly Criterion.

##### Methods

###### `__init__(win_rate=0.5, win_loss_ratio=1.0, fraction=0.5)`

Initialize with Kelly parameters.

Args:
    win_rate: Historical win rate (0-1)
    win_loss_ratio: Ratio of average win to average loss
    fraction: Fraction of Kelly to use (0-1, lower is more conservative)

###### `update_parameters(trade_history)`

Update win rate and win/loss ratio based on trade history.

Args:
    trade_history: List of trade results

###### `calculate_size(signal, portfolio)`

Calculate position size using Kelly formula.

#### `RiskManager`

Manages risk controls and limits for trading.

##### Methods

###### `__init__(max_position_pct=0.25, max_drawdown_pct=0.1, max_concentration_pct=None, use_stop_loss=False, stop_loss_pct=0.05, use_take_profit=False, take_profit_pct=0.1)`

Initialize with risk parameters.

Args:
    max_position_pct: Maximum position size as percentage of portfolio
    max_drawdown_pct: Maximum allowable drawdown from peak equity
    max_concentration_pct: Maximum allocation to a single instrument
    use_stop_loss: Whether to use stop losses
    stop_loss_pct: Stop loss percentage from entry
    use_take_profit: Whether to use take profits
    take_profit_pct: Take profit percentage from entry

###### `check_signal(signal, portfolio)`

Check if a signal should be acted upon given risk constraints.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    
Returns:
    bool: True if signal passes risk checks, False otherwise

*Returns:* bool: True if signal passes risk checks, False otherwise

###### `adjust_position_size(symbol, size, portfolio)`

Adjust position size to comply with risk limits.

Args:
    symbol: Instrument symbol
    size: Calculated position size
    portfolio: Current portfolio state
    
Returns:
    float: Adjusted position size

*Returns:* float: Adjusted position size

###### `get_risk_metrics(portfolio)`

Get current risk metrics for the portfolio.

Args:
    portfolio: Current portfolio state
    
Returns:
    dict: Risk metrics

*Returns:* dict: Risk metrics

###### `set_stop_loss(symbol, entry_price, direction)`

Set stop loss and take profit levels for a new position.

Args:
    symbol: Instrument symbol
    entry_price: Entry price
    direction: Trade direction (1 for long, -1 for short)

###### `check_stops(symbol, current_price, direction)`

Check if stop loss or take profit levels have been hit.

Args:
    symbol: Instrument symbol
    current_price: Current market price
    direction: Position direction (1 for long, -1 for short)
    
Returns:
    bool: True if stop or take profit hit, False otherwise

*Returns:* bool: True if stop or take profit hit, False otherwise

###### `reset()`

Reset risk manager state.

#### `AllocationStrategy`

Base class for portfolio allocation strategies.

##### Methods

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size based on portfolio allocation constraints.

Args:
    symbol: Instrument symbol
    size: Calculated position size
    portfolio: Current portfolio state
    
Returns:
    float: Adjusted position size

*Returns:* float: Adjusted position size

#### `EqualAllocationStrategy`

Allocate capital equally across instruments.

This strategy ensures that no instrument uses more than its fair share
of the portfolio, based on the maximum number of simultaneous positions.

##### Methods

###### `__init__(max_instruments=10)`

Initialize with maximum number of instruments.

Args:
    max_instruments: Maximum number of simultaneous positions

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size for equal allocation.

#### `VolatilityParityAllocation`

Allocate capital based on relative volatility of instruments.

This strategy allocates more capital to less volatile instruments
and less capital to more volatile ones, targeting equal risk contribution.

##### Methods

###### `__init__(lookback_period=20, target_portfolio_vol=0.01)`

Initialize with volatility parameters.

Args:
    lookback_period: Period for calculating volatility
    target_portfolio_vol: Target portfolio volatility

###### `update_volatility(symbol, price)`

Update volatility estimate for an instrument.

Args:
    symbol: Instrument symbol
    price: Current price

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size for volatility parity.

#### `PositionManager`

Manages position sizing, risk, and allocation decisions.

##### Methods

###### `__init__(sizing_strategy=None, risk_manager=None, allocation_strategy=None)`

Initialize the position manager.

Args:
    sizing_strategy: Strategy for determining position size
    risk_manager: Risk management controls
    allocation_strategy: Portfolio allocation strategy

###### `calculate_position_size(signal, portfolio)`

Calculate the appropriate position size for a signal.

Args:
    signal: The trading signal
    portfolio: Current portfolio state
    
Returns:
    float: Position size (positive for buy, negative for sell, 0 for no trade)

*Returns:* float: Position size (positive for buy, negative for sell, 0 for no trade)

###### `get_risk_metrics(portfolio)`

Get current risk metrics for the portfolio.

Args:
    portfolio: Current portfolio state
    
Returns:
    dict: Risk metrics

*Returns:* dict: Risk metrics

###### `update_stops(bar_data, portfolio)`

Update and check stop losses and take profits.

Args:
    bar_data: Current bar data
    portfolio: Current portfolio state
    
Returns:
    dict: Positions to close due to stops {symbol: reason}

*Returns:* dict: Positions to close due to stops {symbol: reason}

###### `reset()`

Reset position manager state.

#### `DefaultPositionManager`

Default position manager that implements basic functionality.
Used as a fallback if no position manager is provided.

##### Methods

###### `calculate_position_size(signal, portfolio)`

Calculate position size based on signal.

This simple implementation just returns a fixed position size
in the direction of the signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio
    
Returns:
    float: Position size (positive for buy, negative for sell)

*Returns:* float: Position size (positive for buy, negative for sell)

###### `reset()`

Reset the position manager state.

## position_sizers

Position Management Module for Trading System

This module provides position sizing, risk management, and allocation strategies
for controlling how signals are converted into actual trading positions.

### Classes

#### `SizingStrategy`

Base class for position sizing strategies.

##### Methods

###### `calculate_size(signal, portfolio)`

Calculate position size based on signal and portfolio.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    
Returns:
    float: Position size (positive for buy, negative for sell)

*Returns:* float: Position size (positive for buy, negative for sell)

#### `FixedSizingStrategy`

Position sizing using a fixed number of units.

##### Methods

###### `__init__(fixed_size=100)`

Initialize with fixed size.

Args:
    fixed_size: Number of units to trade

###### `calculate_size(signal, portfolio)`

Calculate position size.

#### `PercentOfEquitySizing`

Size position as a percentage of portfolio equity.

##### Methods

###### `__init__(percent=0.02)`

Initialize with percent of equity to risk.

Args:
    percent: Percentage of equity to allocate (0.02 = 2%)

###### `calculate_size(signal, portfolio)`

Calculate position size.

#### `VolatilityBasedSizing`

Size positions based on asset volatility.

##### Methods

###### `__init__(risk_pct=0.01, lookback_period=20)`

Initialize with risk percentage and lookback.

Args:
    risk_pct: Percentage of equity to risk per unit of volatility
    lookback_period: Period for calculating volatility

###### `calculate_size(signal, portfolio)`

Calculate position size based on volatility.

#### `KellySizingStrategy`

Position sizing based on the Kelly Criterion.

##### Methods

###### `__init__(win_rate=0.5, win_loss_ratio=1.0, fraction=0.5)`

Initialize with Kelly parameters.

Args:
    win_rate: Historical win rate (0-1)
    win_loss_ratio: Ratio of average win to average loss
    fraction: Fraction of Kelly to use (0-1, lower is more conservative)

###### `update_parameters(trade_history)`

Update win rate and win/loss ratio based on trade history.

Args:
    trade_history: List of trade results

###### `calculate_size(signal, portfolio)`

Calculate position size using Kelly formula.

#### `RiskManager`

Manages risk controls and limits for trading.

##### Methods

###### `__init__(max_position_pct=0.25, max_drawdown_pct=0.1, max_concentration_pct=None, use_stop_loss=False, stop_loss_pct=0.05, use_take_profit=False, take_profit_pct=0.1)`

Initialize with risk parameters.

Args:
    max_position_pct: Maximum position size as percentage of portfolio
    max_drawdown_pct: Maximum allowable drawdown from peak equity
    max_concentration_pct: Maximum allocation to a single instrument
    use_stop_loss: Whether to use stop losses
    stop_loss_pct: Stop loss percentage from entry
    use_take_profit: Whether to use take profits
    take_profit_pct: Take profit percentage from entry

###### `check_signal(signal, portfolio)`

Check if a signal should be acted upon given risk constraints.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    
Returns:
    bool: True if signal passes risk checks, False otherwise

*Returns:* bool: True if signal passes risk checks, False otherwise

###### `adjust_position_size(symbol, size, portfolio)`

Adjust position size to comply with risk limits.

Args:
    symbol: Instrument symbol
    size: Calculated position size
    portfolio: Current portfolio state
    
Returns:
    float: Adjusted position size

*Returns:* float: Adjusted position size

###### `get_risk_metrics(portfolio)`

Get current risk metrics for the portfolio.

Args:
    portfolio: Current portfolio state
    
Returns:
    dict: Risk metrics

*Returns:* dict: Risk metrics

###### `set_stop_loss(symbol, entry_price, direction)`

Set stop loss and take profit levels for a new position.

Args:
    symbol: Instrument symbol
    entry_price: Entry price
    direction: Trade direction (1 for long, -1 for short)

###### `check_stops(symbol, current_price, direction)`

Check if stop loss or take profit levels have been hit.

Args:
    symbol: Instrument symbol
    current_price: Current market price
    direction: Position direction (1 for long, -1 for short)
    
Returns:
    bool: True if stop or take profit hit, False otherwise

*Returns:* bool: True if stop or take profit hit, False otherwise

###### `reset()`

Reset risk manager state.

#### `AllocationStrategy`

Base class for portfolio allocation strategies.

##### Methods

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size based on portfolio allocation constraints.

Args:
    symbol: Instrument symbol
    size: Calculated position size
    portfolio: Current portfolio state
    
Returns:
    float: Adjusted position size

*Returns:* float: Adjusted position size

#### `EqualAllocationStrategy`

Allocate capital equally across instruments.

This strategy ensures that no instrument uses more than its fair share
of the portfolio, based on the maximum number of simultaneous positions.

##### Methods

###### `__init__(max_instruments=10)`

Initialize with maximum number of instruments.

Args:
    max_instruments: Maximum number of simultaneous positions

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size for equal allocation.

#### `VolatilityParityAllocation`

Allocate capital based on relative volatility of instruments.

This strategy allocates more capital to less volatile instruments
and less capital to more volatile ones, targeting equal risk contribution.

##### Methods

###### `__init__(lookback_period=20, target_portfolio_vol=0.01)`

Initialize with volatility parameters.

Args:
    lookback_period: Period for calculating volatility
    target_portfolio_vol: Target portfolio volatility

###### `update_volatility(symbol, price)`

Update volatility estimate for an instrument.

Args:
    symbol: Instrument symbol
    price: Current price

###### `adjust_allocation(symbol, size, portfolio)`

Adjust position size for volatility parity.

#### `PositionManager`

Manages position sizing, risk, and allocation decisions.

##### Methods

###### `__init__(sizing_strategy=None, risk_manager=None, allocation_strategy=None)`

Initialize the position manager.

Args:
    sizing_strategy: Strategy for determining position size
    risk_manager: Risk management controls
    allocation_strategy: Portfolio allocation strategy

###### `calculate_position_size(signal, portfolio)`

Calculate the appropriate position size for a signal.

Args:
    signal: The trading signal
    portfolio: Current portfolio state
    
Returns:
    float: Position size (positive for buy, negative for sell, 0 for no trade)

*Returns:* float: Position size (positive for buy, negative for sell, 0 for no trade)

###### `get_risk_metrics(portfolio)`

Get current risk metrics for the portfolio.

Args:
    portfolio: Current portfolio state
    
Returns:
    dict: Risk metrics

*Returns:* dict: Risk metrics

###### `update_stops(bar_data, portfolio)`

Update and check stop losses and take profits.

Args:
    bar_data: Current bar data
    portfolio: Current portfolio state
    
Returns:
    dict: Positions to close due to stops {symbol: reason}

*Returns:* dict: Positions to close due to stops {symbol: reason}

###### `reset()`

Reset position manager state.

# Position_management Module

Position Management Package

This package provides position management and portfolio management capabilities.

## Contents

- [allocation](#allocation)
- [portfolio](#portfolio)
- [position](#position)
- [position_manager](#position_manager)
- [position_sizers](#position_sizers)
- [position_utils](#position_utils)

## allocation

Allocation Module

This module provides portfolio allocation strategies for distributing
capital across multiple trading instruments.

### Classes

#### `AllocationStrategy`

Abstract base class for portfolio allocation strategies.

Allocation strategies determine how to distribute capital
across multiple instruments in a portfolio.

##### Methods

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital across instruments.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of allocation weights by symbol (0-1)

#### `EqualWeightAllocation`

Equal weight allocation strategy.

This strategy allocates equal capital to each instrument 
in the portfolio.

##### Methods

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital equally across instruments.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of equal allocation weights by symbol

#### `MarketCapAllocation`

Market capitalization weighted allocation.

This strategy allocates capital based on relative market 
capitalization of each instrument.

##### Methods

###### `__init__(market_caps)`

Initialize market cap allocation strategy.

Args:
    market_caps: Dictionary of market capitalizations by symbol

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital based on market capitalization.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of market cap-weighted allocation weights by symbol

#### `SignalStrengthAllocation`

Signal strength weighted allocation.

This strategy allocates capital based on the relative strength
or confidence of trading signals.

##### Methods

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital based on signal strength.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of signal strength-weighted allocation weights by symbol

#### `VolatilityParityAllocation`

Volatility parity (risk parity) allocation.

This strategy allocates capital to achieve equal risk contribution
from each instrument based on historical volatility.

##### Methods

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital based on volatility parity.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of volatility-parity allocation weights by symbol

#### `MaximumSharpeAllocation`

Maximum Sharpe ratio allocation.

This strategy allocates capital to maximize the portfolio's expected
Sharpe ratio based on expected returns and covariance matrix.

##### Methods

###### `__init__(expected_returns=None, covariance_matrix=None, risk_free_rate=0.0)`

Initialize maximum Sharpe ratio allocation strategy.

Args:
    expected_returns: Dictionary of expected returns by symbol
    covariance_matrix: Dictionary of covariance values between symbols
    risk_free_rate: Risk-free rate for Sharpe ratio calculation

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital to maximize Sharpe ratio.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of allocation weights by symbol

#### `ConstrainedAllocation`

Constrained allocation strategy.

This strategy applies constraints (min/max weights, sector limits)
to an underlying allocation strategy.

##### Methods

###### `__init__(base_strategy, min_weight=0.0, max_weight=0.25, sector_limits=None, sector_mapping=None)`

Initialize constrained allocation strategy.

Args:
    base_strategy: Underlying allocation strategy
    min_weight: Minimum weight for any symbol
    max_weight: Maximum weight for any symbol
    sector_limits: Maximum allocation per sector
    sector_mapping: Mapping of symbols to sectors

###### `allocate(portfolio, signals, prices)`

*Returns:* `Dict[str, float]`

Allocate portfolio capital with constraints.

Args:
    portfolio: Current portfolio state
    signals: Dictionary of signals by symbol
    prices: Dictionary of current prices by symbol
    
Returns:
    Dictionary of constrained allocation weights by symbol

#### `AllocationStrategyFactory`

Factory for creating allocation strategies.

This factory creates different types of allocation strategies
based on configuration parameters.

##### Methods

###### `create_strategy(strategy_type)`

*Returns:* `AllocationStrategy`

Create an allocation strategy of the specified type.

Args:
    strategy_type: Type of allocation strategy to create
    **kwargs: Parameters for the allocation strategy
    
Returns:
    AllocationStrategy instance
    
Raises:
    ValueError: If strategy type is not recognized

###### `create_from_config(config)`

*Returns:* `AllocationStrategy`

Create an allocation strategy from a configuration dictionary.

Args:
    config: Configuration dictionary with 'type' and 'params' keys
    
Returns:
    AllocationStrategy instance
    
Raises:
    ValueError: If configuration is invalid

## portfolio

Event-Based Portfolio Module

This module provides an event-driven portfolio implementation that
communicates through events rather than direct method calls.

### Classes

#### `EventPortfolio`

Event-driven portfolio that manages positions based on events.

##### Methods

###### `__init__(initial_capital, event_bus, portfolio_id=None, name=None, currency='USD', allow_fractional_shares=True, margin_enabled=False, leverage=1.0)`

Initialize event-driven portfolio.

Args:
    initial_capital: Initial account capital
    event_bus: Event bus for communication
    portfolio_id: Unique portfolio ID (generated if None)
    name: Optional portfolio name
    currency: Portfolio base currency
    allow_fractional_shares: Whether fractional shares are allowed
    margin_enabled: Whether margin trading is enabled
    leverage: Maximum allowed leverage

###### `_process_event(event)`

*Returns:* `None`

Process incoming events.

Args:
    event: Event to process

###### `mark_to_market(bar_event)`

Update portfolio positions with current market prices.

Args:
    bar_event: Bar event containing current market data

###### `get_position_snapshot()`

Get a snapshot of all current positions.

Returns:
    Dictionary with position information

*Returns:* Dictionary with position information

###### `_handle_position_action(event)`

*Returns:* `None`

Handle position action events.

Args:
    event: Position action event

###### `_handle_fill(event)`

*Returns:* `None`

Handle fill events.

Args:
    event: Fill event

###### `_open_position(symbol, direction, quantity, entry_price, entry_time, stop_loss=None, take_profit=None, strategy_id=None)`

*Returns:* `Position`

Open a new position.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    quantity: Position size
    entry_price: Entry price
    entry_time: Entry timestamp
    stop_loss: Optional stop loss price
    take_profit: Optional take profit price
    strategy_id: Optional strategy ID
    
Returns:
    Newly created Position

###### `_close_position(position_id, exit_price, exit_time, exit_type)`

*Returns:* `Dict[str, Any]`

Close a position.

Args:
    position_id: ID of position to close
    exit_price: Exit price
    exit_time: Exit timestamp
    exit_type: Type of exit
    
Returns:
    Position summary dictionary

###### `_update_metrics()`

*Returns:* `None`

Update portfolio metrics.

###### `_emit_portfolio_update()`

*Returns:* `None`

Emit portfolio update event.

###### `_emit_position_opened(position)`

*Returns:* `None`

Emit position opened event.

Args:
    position: Position that was opened

###### `_emit_position_closed(position_summary)`

*Returns:* `None`

Emit position closed event.

Args:
    position_summary: Position summary dictionary

###### `get_net_position(symbol)`

Get net position for a symbol.

###### `get_positions_by_symbol(symbol)`

Get positions for a symbol.

###### `update_position(symbol, quantity_delta, price, timestamp)`

Update portfolio with a position change.

Args:
    symbol: Instrument symbol
    quantity_delta: Change in position quantity (positive for buy, negative for sell)
    price: Execution price
    timestamp: Execution timestamp

Returns:
    True if successful, False otherwise

*Returns:* True if successful, False otherwise

###### `on_signal(event)`

Disabled method to prevent portfolio from reacting directly to signals.

Args:
    event: Signal event

## position

Position Module

This module defines the Position class and related utilities for representing
and managing trading positions within the system.

### Classes

#### `PositionStatus`

Enumeration of possible position statuses.

#### `EntryType`

Enumeration of possible position entry types.

#### `ExitType`

Enumeration of possible position exit types.

#### `Position`

Represents a trading position.

A position encapsulates information about an open or closed trading position,
including its size, entry and exit details, profit/loss, and current status.

##### Methods

###### `__init__(symbol, direction, quantity, entry_price, entry_time, entry_type, position_id=None, strategy_id=None, entry_order_id=None, stop_loss=None, take_profit=None, initial_risk=None, metadata=None)`

Initialize a new position.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    quantity: Position size in units
    entry_price: Entry price
    entry_time: Entry timestamp
    entry_type: Type of entry (market, limit, etc.)
    position_id: Unique position ID (generated if None)
    strategy_id: ID of the strategy that created the position
    entry_order_id: ID of the order that opened the position
    stop_loss: Optional stop loss price
    take_profit: Optional take profit price
    initial_risk: Optional initial risk amount
    metadata: Optional additional position metadata

###### `update_price(current_price, timestamp)`

*Returns:* `Dict[str, Any]`

Update position with current price and check for exits.

Args:
    current_price: Current market price
    timestamp: Current timestamp
    
Returns:
    Dictionary with exit information if exit triggered, None otherwise

###### `_check_exits(current_price, timestamp)`

*Returns:* `Optional[Dict[str, Any]]`

Check if any exit conditions are triggered.

Args:
    current_price: Current market price
    timestamp: Current timestamp
    
Returns:
    Dictionary with exit information if exit triggered, None otherwise

###### `close(exit_price, exit_time, exit_type, exit_order_id=None)`

*Returns:* `Dict[str, Any]`

Close the position.

Args:
    exit_price: Exit price
    exit_time: Exit timestamp
    exit_type: Type of exit
    exit_order_id: ID of exit order
    
Returns:
    Dictionary with position summary information

###### `partially_close(quantity, exit_price, exit_time, exit_type, exit_order_id=None)`

*Returns:* `Dict[str, Any]`

Partially close the position.

Args:
    quantity: Quantity to close
    exit_price: Exit price
    exit_time: Exit timestamp
    exit_type: Type of exit
    exit_order_id: ID of exit order
    
Returns:
    Dictionary with partial close information

###### `add(quantity, price, time, order_id=None)`

*Returns:* `Dict[str, Any]`

Add to the position (increase size).

Args:
    quantity: Quantity to add
    price: Price of additional units
    time: Timestamp of addition
    order_id: ID of order that added to position
    
Returns:
    Dictionary with addition information

###### `modify_stop_loss(new_stop_loss)`

*Returns:* `Dict[str, Any]`

Modify the stop loss price.

Args:
    new_stop_loss: New stop loss price
    
Returns:
    Dictionary with modification information

###### `modify_take_profit(new_take_profit)`

*Returns:* `Dict[str, Any]`

Modify the take profit price.

Args:
    new_take_profit: New take profit price
    
Returns:
    Dictionary with modification information

###### `set_trailing_stop(distance)`

*Returns:* `Dict[str, Any]`

Set a trailing stop for the position.

Args:
    distance: Distance from current price for trailing stop
    
Returns:
    Dictionary with trailing stop information

###### `_calculate_unrealized_pnl()`

*Returns:* `float`

Calculate unrealized P&L for the position.

Returns:
    Unrealized P&L

###### `_calculate_realized_pnl(exit_price)`

*Returns:* `float`

Calculate realized P&L for the position.

Args:
    exit_price: Exit price
    
Returns:
    Realized P&L

###### `get_duration()`

*Returns:* `Optional[datetime.timedelta]`

Get the duration of the position.

Returns:
    Timedelta if position is closed, None otherwise

###### `get_current_return()`

*Returns:* `float`

Get the current return percentage for the position.

Returns:
    Current return percentage

###### `get_risk_reward_ratio()`

*Returns:* `Optional[float]`

Get the risk-reward ratio for the position.

Returns:
    Risk-reward ratio if stop loss and take profit are set, None otherwise

###### `get_summary()`

*Returns:* `Dict[str, Any]`

Get a summary of the position.

Returns:
    Dictionary with position summary information

###### `__str__()`

*Returns:* `str`

String representation of the position.

#### `PositionFactory`

Factory for creating Position objects with different configurations.

This factory simplifies position creation with sensible defaults
and provides utility methods for creating positions with specific
risk parameters.

##### Methods

###### `create_position(cls, symbol, direction, quantity, entry_price, entry_time)`

*Returns:* `Position`

Create a basic position with the essential parameters.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    quantity: Position size in units
    entry_price: Entry price
    entry_time: Entry timestamp
    **kwargs: Additional position parameters
    
Returns:
    Position: Newly created position

###### `create_position_with_stops(cls, symbol, direction, quantity, entry_price, entry_time, stop_loss_pct=None, stop_loss_price=None, take_profit_pct=None, take_profit_price=None)`

*Returns:* `Position`

Create a position with stop loss and/or take profit levels.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    quantity: Position size in units
    entry_price: Entry price
    entry_time: Entry timestamp
    stop_loss_pct: Optional stop loss percentage from entry
    stop_loss_price: Optional explicit stop loss price
    take_profit_pct: Optional take profit percentage from entry
    take_profit_price: Optional explicit take profit price
    **kwargs: Additional position parameters
    
Returns:
    Position: Position with stop loss and/or take profit

###### `create_position_with_risk(cls, symbol, direction, entry_price, entry_time, account_size, risk_pct, stop_loss_price, take_profit_price=None, max_position_pct=None)`

*Returns:* `Position`

Create a position sized according to risk percentage of account.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    entry_price: Entry price
    entry_time: Entry timestamp
    account_size: Current account size
    risk_pct: Percentage of account to risk
    stop_loss_price: Stop loss price
    take_profit_price: Optional take profit price
    max_position_pct: Optional maximum position size as percentage of account
    **kwargs: Additional position parameters
    
Returns:
    Position: Risk-sized position

###### `create_position_with_trailing_stop(cls, symbol, direction, quantity, entry_price, entry_time, trailing_stop_distance, activation_pct=None)`

*Returns:* `Position`

Create a position with a trailing stop.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    quantity: Position size in units
    entry_price: Entry price
    entry_time: Entry timestamp
    trailing_stop_distance: Distance for trailing stop
    activation_pct: Optional percentage move to activate trailing stop
    **kwargs: Additional position parameters
    
Returns:
    Position: Position with trailing stop

###### `create_from_signal(cls, signal, account_size, risk_pct=0.01, position_pct=None, use_stops=True)`

*Returns:* `Position`

Create a position from a trading signal.

Args:
    signal: Trading signal dictionary
    account_size: Current account size
    risk_pct: Percentage of account to risk
    position_pct: Optional position size as percentage of account
    use_stops: Whether to use stop loss/take profit from signal
    **kwargs: Additional position parameters
    
Returns:
    Position: Position based on signal
    
Raises:
    ValueError: If signal is missing required fields

#### `PositionManager`

Manages a collection of positions.

This class provides functionality for tracking, updating, and managing
multiple trading positions across different symbols.

##### Methods

###### `__init__()`

Initialize the position manager.

###### `add_position(position)`

*Returns:* `str`

Add a position to the manager.

Args:
    position: Position to add
    
Returns:
    str: Position ID

###### `get_position(position_id)`

*Returns:* `Optional[Position]`

Get a position by ID.

Args:
    position_id: Position ID
    
Returns:
    Position object or None if not found

###### `get_positions_for_symbol(symbol)`

*Returns:* `List[Position]`

Get all positions for a symbol.

Args:
    symbol: Instrument symbol
    
Returns:
    List of Position objects

###### `get_open_positions()`

*Returns:* `List[Position]`

Get all open positions.

Returns:
    List of open Position objects

###### `get_open_positions_for_symbol(symbol)`

*Returns:* `List[Position]`

Get open positions for a symbol.

Args:
    symbol: Instrument symbol
    
Returns:
    List of open Position objects for the symbol

###### `update_positions(symbol, current_price, timestamp)`

*Returns:* `List[Dict[str, Any]]`

Update all positions for a symbol with current price.

Args:
    symbol: Instrument symbol
    current_price: Current market price
    timestamp: Current timestamp
    
Returns:
    List of exit information for any triggered exits

###### `close_position(position_id, exit_price, exit_time, exit_type, exit_order_id=None)`

*Returns:* `Dict[str, Any]`

Close a position.

Args:
    position_id: Position ID
    exit_price: Exit price
    exit_time: Exit timestamp
    exit_type: Type of exit
    exit_order_id: ID of exit order
    
Returns:
    Dictionary with position summary or None if position not found

###### `close_all_positions(exit_price_func, exit_time, exit_type)`

*Returns:* `List[Dict[str, Any]]`

Close all open positions.

Args:
    exit_price_func: Function that takes a position and returns exit price
    exit_time: Exit timestamp
    exit_type: Type of exit
    
Returns:
    List of position summaries

###### `get_portfolio_value(price_func)`

*Returns:* `float`

Calculate total portfolio value.

Args:
    price_func: Function that takes a symbol and returns current price
    
Returns:
    Total portfolio value

###### `get_portfolio_metrics(price_func)`

*Returns:* `Dict[str, Any]`

Calculate portfolio metrics.

Args:
    price_func: Function that takes a symbol and returns current price
    
Returns:
    Dictionary with portfolio metrics

###### `record_transaction_costs(position_id, costs)`

*Returns:* `bool`

Record transaction costs for a position.

Args:
    position_id: Position ID
    costs: Transaction costs
    
Returns:
    True if successful, False if position not found

###### `apply_to_positions(func, filter_func=None)`

*Returns:* `List[Any]`

Apply a function to positions optionally filtered by a filter function.

Args:
    func: Function to apply to each position
    filter_func: Optional function to filter positions
    
Returns:
    List of function results

###### `get_position_summaries()`

*Returns:* `List[Dict[str, Any]]`

Get summaries for all positions.

Returns:
    List of position summaries

## position_manager

Fixed Position Manager Implementation

This module contains the fixed version of the PositionManager class
that properly handles signals and creates trades.

### Classes

#### `PositionManager`

Manages trading positions, sizing, and allocation.

The position manager integrates with risk management to determine appropriate
position sizes based on signals, market conditions, and portfolio state.

##### Methods

###### `__init__(portfolio, position_sizer=None, allocation_strategy=None, risk_manager=None, max_positions=0, event_bus=None)`

Initialize position manager.

Args:
    portfolio: Portfolio to manage
    position_sizer: Strategy for determining position sizes
    allocation_strategy: Strategy for allocating capital across instruments
    risk_manager: Risk management component
    max_positions: Maximum number of positions (0 for unlimited)
    event_bus: Event bus for emitting events

###### `on_signal(event)`

Process a signal event into position actions.

Args:
    event: Event containing a SignalEvent or the SignalEvent directly

Returns:
    List of position actions

*Returns:* List of position actions

###### `_process_signal(signal)`

Process a signal into position actions.

Args:
    signal: SignalEvent to process

Returns:
    List of position action dictionaries

*Returns:* List of position action dictionaries

###### `_calculate_position_size(signal)`

Calculate position size based on signal and portfolio.

Args:
    signal: Trading signal

Returns:
    Position size (positive for buy, negative for sell)

*Returns:* Position size (positive for buy, negative for sell)

###### `execute_position_action(action, current_time=None)`

Execute a position action.

Args:
    action: Position action dictionary
    current_time: Current timestamp (defaults to now)
    
Returns:
    Result dictionary or None if action failed

*Returns:* Result dictionary or None if action failed

## position_sizers

Position Sizers Module

This module provides different position sizing strategies to determine the appropriate
position size based on risk parameters and market conditions.

### Classes

#### `PositionSizer`

Abstract base class for position sizing strategies.

Position sizers determine the appropriate position size based on
risk parameters, portfolio state, and market conditions.

##### Methods

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size (number of units)

#### `FixedSizeSizer`

Position sizer that always returns a fixed size.

This sizer always returns the same position size regardless
of portfolio value or market conditions.

##### Methods

###### `__init__(fixed_size=100)`

Initialize fixed size position sizer.

Args:
    fixed_size: Number of units to trade

###### `calculate_position_size(signal, portfolio, current_price=None)`

Calculate a fixed position size.

Args:
    signal: Trading signal
    portfolio: Portfolio state
    current_price: Optional override for current price

Returns:
    Fixed position size (positive for buy, negative for sell)

*Returns:* Fixed position size (positive for buy, negative for sell)

#### `PercentOfEquitySizer`

Position sizer that sizes based on a percentage of portfolio equity.

This sizer calculates position size to be a specified percentage of
the current portfolio equity.

##### Methods

###### `__init__(percent=2.0, max_pct=25.0)`

Initialize percent of equity position sizer.

Args:
    percent: Percentage of equity to allocate (2.0 = 2%)
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size based on a percentage of equity.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size (positive for buy, negative for sell)

#### `VolatilityPositionSizer`

Position sizer that sizes based on asset volatility.

This sizer calculates position size to risk a specified percentage of
portfolio equity per unit of volatility.

##### Methods

###### `__init__(risk_pct=1.0, atr_multiplier=2.0, lookback_period=20, max_pct=25.0)`

Initialize volatility-based position sizer.

Args:
    risk_pct: Percentage of equity to risk (1.0 = 1%)
    atr_multiplier: Multiplier for ATR to set stop distance
    lookback_period: Period for ATR calculation
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size based on volatility

#### `KellyCriterionSizer`

Position sizer based on the Kelly Criterion formula.

This sizer calculates the optimal position size based on the estimated
win rate and reward-to-risk ratio.

##### Methods

###### `__init__(win_rate=0.5, reward_risk_ratio=1.0, fraction=0.5, max_pct=25.0)`

Initialize Kelly Criterion position sizer.

Args:
    win_rate: Expected win rate (0.5 = 50%)
    reward_risk_ratio: Expected reward-to-risk ratio
    fraction: Fraction of Kelly to use (0.5 = "Half Kelly")
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size based on Kelly Criterion

#### `RiskParityPositionSizer`

Position sizer based on risk parity principles.

This sizer allocates capital to maintain equal risk contribution 
across different assets based on their volatility.

##### Methods

###### `__init__(target_portfolio_volatility=0.1, max_pct=25.0)`

Initialize risk parity position sizer.

Args:
    target_portfolio_volatility: Target annualized portfolio volatility
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size based on risk parity

#### `PSARPositionSizer`

Position sizer based on Parabolic SAR stops.

This sizer calculates position size using Parabolic SAR for stop placement
and a fixed risk percentage of equity.

##### Methods

###### `__init__(risk_pct=1.0, psar_factor=0.02, psar_max=0.2, max_pct=25.0)`

Initialize PSAR-based position sizer.

Args:
    risk_pct: Percentage of equity to risk (1.0 = 1%)
    psar_factor: PSAR acceleration factor
    psar_max: Maximum PSAR acceleration
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size based on PSAR stops

#### `AdaptivePositionSizer`

Position sizer that adapts sizing based on market conditions.

This sizer dynamically adjusts position sizing based on market volatility,
trend strength, and other market conditions.

##### Methods

###### `__init__(base_risk_pct=1.0, volatility_factor=0.5, trend_factor=0.5, max_pct=25.0)`

Initialize adaptive position sizer.

Args:
    base_risk_pct: Base percentage of equity to risk (1.0 = 1%)
    volatility_factor: How much to adjust for volatility (0-1)
    trend_factor: How much to adjust for trend strength (0-1)
    max_pct: Maximum percentage of equity for any one position

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size for a signal.

Args:
    signal: Trading signal
    portfolio: Current portfolio state
    current_price: Current market price
    
Returns:
    Position size based on adaptive factors

#### `AdjustedFixedSizer`

Position sizer that adjusts fixed size based on price and available capital.

##### Methods

###### `calculate_position_size(signal, portfolio, current_price=None)`

Calculate position size adjusted to maximize capital usage.

Args:
    signal: Trading signal
    portfolio: Portfolio state
    current_price: Optional override for current price
    
Returns:
    Position size that fits within available capital

*Returns:* Position size that fits within available capital

#### `PositionSizerFactory`

Factory for creating position sizers.

This factory creates different types of position sizers based on
configuration parameters.

##### Methods

###### `create_sizer(sizer_type)`

*Returns:* `PositionSizer`

Create a position sizer of the specified type.

Args:
    sizer_type: Type of position sizer to create
    **kwargs: Parameters for the position sizer
    
Returns:
    PositionSizer instance
    
Raises:
    ValueError: If sizer type is not recognized

###### `create_from_config(config)`

*Returns:* `PositionSizer`

Create a position sizer from a configuration dictionary.

Args:
    config: Configuration dictionary with 'type' and 'params' keys
    
Returns:
    PositionSizer instance
    
Raises:
    ValueError: If configuration is invalid

## position_utils

Position Management Utilities Module

This module provides utility functions for working with positions and position management
in the trading system. It standardizes position actions and calculations.

### Functions

#### `get_signal_direction(signal_event)`

*Returns:* `int`

Extract direction from a signal event.

Args:
    signal_event: Signal event
    
Returns:
    Direction as an integer (1 for buy/long, -1 for sell/short, 0 for neutral)

#### `create_position_action(action)`

*Returns:* `Dict[str, Any]`

Create a standardized position action dictionary.

Args:
    action: Action type ('entry', 'exit', 'modify')
    **kwargs: Action-specific parameters
    
Returns:
    Standardized position action dictionary

#### `create_entry_action(symbol, direction, size, price, stop_loss=None, take_profit=None, strategy_id=None, entry_type, timestamp=None, metadata=None)`

*Returns:* `Dict[str, Any]`

Create a standardized entry action.

Args:
    symbol: Instrument symbol
    direction: Position direction (1 for long, -1 for short)
    size: Position size
    price: Entry price
    stop_loss: Optional stop loss price
    take_profit: Optional take profit price
    strategy_id: Optional strategy ID
    entry_type: Type of entry
    timestamp: Action timestamp
    metadata: Additional action metadata
    
Returns:
    Entry action dictionary

#### `create_exit_action(position_id, symbol, price, exit_type, reason=None, timestamp=None, metadata=None)`

*Returns:* `Dict[str, Any]`

Create a standardized exit action.

Args:
    position_id: ID of position to exit
    symbol: Instrument symbol
    price: Exit price
    exit_type: Type of exit
    reason: Optional reason for exit
    timestamp: Action timestamp
    metadata: Additional action metadata
    
Returns:
    Exit action dictionary

#### `calculate_position_size(signal_event, portfolio, risk_pct=0.01)`

*Returns:* `float`

Calculate position size based on risk percentage.

Args:
    signal_event: Signal event
    portfolio: Portfolio instance
    risk_pct: Percentage of portfolio to risk (0.01 = 1%)
    
Returns:
    Position size

#### `calculate_risk_reward_ratio(entry_price, stop_loss, take_profit, direction)`

*Returns:* `Optional[float]`

Calculate risk-reward ratio for a position.

Args:
    entry_price: Entry price
    stop_loss: Stop loss price
    take_profit: Take profit price
    direction: Position direction (1 for long, -1 for short)
    
Returns:
    Risk-reward ratio or None if not calculable

#### `signal_to_position_action(signal_event, portfolio)`

*Returns:* `Optional[Dict[str, Any]]`

Convert a signal event to a position action.

Args:
    signal_event: Signal event
    portfolio: Portfolio instance
    
Returns:
    Position action dictionary or None if no action needed

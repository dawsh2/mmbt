# Risk_management Module

Risk management package for algorithmic trading systems.

This package provides a comprehensive framework for implementing advanced risk 
management using MAE, MFE, and ETD analysis to derive data-driven risk parameters.

Main components:
- RiskManager: Applies risk rules to trades
- RiskMetricsCollector: Collects risk metrics from trades
- RiskAnalysisEngine: Analyzes collected metrics
- RiskParameterOptimizer: Derives optimal risk parameters

## Contents

- [analyzer](#analyzer)
- [collector](#collector)
- [parameter_optimizer](#parameter_optimizer)
- [risk_manager](#risk_manager)
- [risk_parameters](#risk_parameters)
- [types](#types)

## analyzer

Analysis engine for risk management metrics.

This module provides tools for analyzing Maximum Adverse Excursion (MAE),
Maximum Favorable Excursion (MFE), and Entry-To-Exit Duration (ETD) metrics
to derive optimal risk management parameters.

### Classes

#### `RiskAnalysisEngine`

Analyzes MAE, MFE, and ETD metrics to derive insights for risk management.

This class processes trade metrics to calculate statistical properties
that can be used to optimize risk management parameters.

##### Methods

###### `__init__(metrics_df)`

Initialize the risk analysis engine.

Args:
    metrics_df: DataFrame containing trade metrics including mae_pct, mfe_pct, etc.

###### `analyze_mae()`

*Returns:* `Dict[str, float]`

Analyze MAE distribution and characteristics.

Returns:
    Dictionary of MAE statistics

###### `analyze_mfe()`

*Returns:* `Dict[str, float]`

Analyze MFE distribution and characteristics.

Returns:
    Dictionary of MFE statistics

###### `analyze_etd()`

*Returns:* `Dict[str, Any]`

Analyze trade duration characteristics.

Returns:
    Dictionary of ETD statistics

###### `analyze_exit_reasons()`

*Returns:* `Dict[str, Any]`

Analyze the distribution and performance of different exit reasons.

Returns:
    Dictionary of exit reason statistics

###### `calculate_mad_ratio()`

*Returns:* `float`

Calculate the MAE/MFE Adjusted Duration (MAD) ratio.

This is a custom metric that evaluates the quality of trades by comparing
how quickly favorable moves happen relative to adverse moves.

Returns:
    MAD ratio value

###### `analyze_all()`

*Returns:* `RiskAnalysisResults`

Run all analyses and return comprehensive results.

Returns:
    RiskAnalysisResults object containing all analysis results

###### `from_collector(cls, collector, clear_price_paths=True)`

Create an analyzer directly from a RiskMetricsCollector.

Args:
    collector: A RiskMetricsCollector instance
    clear_price_paths: Whether to clear price paths to save memory
    
Returns:
    RiskAnalysisEngine instance

*Returns:* RiskAnalysisEngine instance

## collector

Data collection module for risk management metrics.

This module provides tools for tracking and collecting Maximum Adverse Excursion (MAE),
Maximum Favorable Excursion (MFE), and Entry-To-Exit Duration (ETD) metrics.

### Classes

#### `RiskMetricsCollector`

Collects and stores MAE, MFE, and ETD metrics for analyzing trading performance.

This class tracks the complete lifecycle of trades, including the price path
between entry and exit to calculate MAE, MFE, and other risk metrics.

##### Methods

###### `__init__()`

Initialize the risk metrics collector.

###### `start_trade(trade_id, entry_time, entry_price, direction)`

*Returns:* `None`

Start tracking a new trade.

Args:
    trade_id: Unique identifier for the trade
    entry_time: Time of trade entry
    entry_price: Price at entry
    direction: Trade direction ('long' or 'short')

###### `update_price_path(trade_id, bar_data)`

*Returns:* `Optional[Tuple[float, float]]`

Update the price path for a trade with new bar data.

Args:
    trade_id: Unique identifier for the trade
    bar_data: Bar data containing at minimum 'high', 'low', and 'timestamp'
    
Returns:
    Optional tuple of (mae_pct, mfe_pct) for the current trade

###### `end_trade(trade_id, exit_time, exit_price, exit_reason)`

*Returns:* `Optional[TradeMetrics]`

End tracking for a trade and calculate final metrics.

Args:
    trade_id: Unique identifier for the trade
    exit_time: Time of trade exit
    exit_price: Price at exit
    exit_reason: Reason for the exit
    
Returns:
    The completed TradeMetrics object or None if trade_id not found

###### `track_completed_trade(entry_time, entry_price, direction, exit_time, exit_price, price_path=None, exit_reason)`

*Returns:* `TradeMetrics`

Record a completed trade with all information in one call.

Args:
    entry_time: Time of trade entry
    entry_price: Price at entry
    direction: Trade direction ('long' or 'short')
    exit_time: Time of trade exit
    exit_price: Price at exit
    price_path: Optional list of bar data between entry and exit
    exit_reason: Reason for the exit
    
Returns:
    The newly created and stored TradeMetrics object

###### `get_metrics_dataframe()`

*Returns:* `pd.DataFrame`

Convert metrics to pandas DataFrame for analysis.

Returns:
    DataFrame containing all trade metrics

###### `save_to_csv(filepath)`

*Returns:* `None`

Save the collected metrics to a CSV file.

Args:
    filepath: Path to save the CSV file

###### `load_from_csv(filepath)`

*Returns:* `None`

Load metrics from a CSV file.

Args:
    filepath: Path to the CSV file

###### `clear()`

*Returns:* `None`

Clear all stored trade metrics.

## parameter_optimizer

Parameter optimization for risk management rules.

This module derives optimal risk management parameters from analyzed trade metrics
including stop-loss levels, take-profit targets, trailing stops, and time exits.

### Classes

#### `RiskParameterOptimizer`

Derives optimal risk management parameters from analyzed trade metrics.

This class uses the statistical properties of MAE, MFE, and ETD to create
data-driven risk management rules that align with the strategy's characteristics.

##### Methods

###### `__init__(analysis_results)`

Initialize the risk parameter optimizer.

Args:
    analysis_results: Results from the RiskAnalysisEngine

###### `optimize_stop_loss(risk_tolerance='moderate')`

*Returns:* `Dict[str, Any]`

Derive optimal stop-loss parameters based on MAE analysis.

Args:
    risk_tolerance: 'conservative', 'moderate', or 'aggressive'
                    (or corresponding RiskToleranceLevel enum)
    
Returns:
    Dictionary of optimized stop-loss parameters

###### `optimize_take_profit(profit_target_style='moderate')`

*Returns:* `Dict[str, Any]`

Derive optimal take-profit parameters based on MFE analysis.

Args:
    profit_target_style: 'conservative', 'moderate', or 'aggressive'
                         (or corresponding RiskToleranceLevel enum)
    
Returns:
    Dictionary of optimized take-profit parameters

###### `optimize_trailing_stop()`

*Returns:* `Dict[str, float]`

Derive optimal trailing stop parameters based on MAE and MFE analysis.

Returns:
    Dictionary of optimized trailing stop parameters

###### `optimize_time_exit()`

*Returns:* `Dict[str, Any]`

Derive optimal time-based exit parameters based on ETD analysis.

Returns:
    Dictionary of optimized time-based exit parameters

###### `calculate_risk_reward_setups()`

*Returns:* `List[Dict[str, Any]]`

Calculate various risk-reward setups based on the optimized parameters.

Returns:
    List of different risk management setups with expected metrics

###### `get_optimal_parameters(risk_tolerance='moderate', include_trailing_stop=True, include_time_exit=True)`

*Returns:* `RiskParameters`

Get a complete set of optimal risk parameters.

Args:
    risk_tolerance: Overall risk tolerance level
    include_trailing_stop: Whether to include trailing stop parameters
    include_time_exit: Whether to include time-based exit parameters
    
Returns:
    RiskParameters object with optimized values

###### `get_balanced_parameters()`

*Returns:* `RiskParameters`

Get a balanced set of risk parameters that maximizes expectancy.

This method evaluates different combinations and returns the one with
the highest expected value.

Returns:
    RiskParameters object with balanced optimal values

## risk_manager

Risk manager module for implementing MAE, MFE, and ETD based risk management.

This module provides a RiskManager class that can be used to apply data-driven
risk management rules to any trading strategy, using parameters derived from
historical trade analysis.

### Classes

#### `RiskManager`

Applies risk management rules based on MAE, MFE, and ETD analysis.

This class applies stop-loss, take-profit, trailing stop, and time-based
exit rules to trading positions, using parameters derived from historical
trade analysis. Additionally, it handles position sizing through integration
with position sizers.

##### Methods

###### `__init__(risk_params, position_sizer=None, metrics_collector=None, position_size_calculator=None)`

Initialize the risk manager.

Args:
    risk_params: Risk parameters for stop-loss, take-profit, etc.
    position_sizer: Optional position sizer for calculating position sizes
    metrics_collector: Optional collector for tracking trade metrics
    position_size_calculator: Optional function for calculating position sizes (legacy)

###### `calculate_position_size(signal, portfolio, current_price=None)`

*Returns:* `float`

Calculate position size based on risk parameters and signal.

Args:
    signal: Trading signal event
    portfolio: Portfolio or dictionary with equity information
    current_price: Optional current price (uses signal price if None)
    
Returns:
    Position size (positive for buy, negative for sell)

###### `_calculate_stop_price(current_price, signal)`

*Returns:* `Optional[float]`

Calculate stop price based on risk parameters and signal.

Args:
    current_price: Current market price
    signal: Trading signal event
    
Returns:
    Stop price or None if not calculable

###### `_calculate_target_price(current_price, signal)`

*Returns:* `Optional[float]`

Calculate target price based on risk parameters and signal.

Args:
    current_price: Current market price
    signal: Trading signal event
    
Returns:
    Target price or None if not calculable

###### `open_trade(trade_id, direction, entry_price, entry_time, initial_stop_price=None, initial_target_price=None)`

*Returns:* `Dict[str, Any]`

Register a new trade with the risk manager.

Args:
    trade_id: Unique identifier for the trade
    direction: 'long' or 'short'
    entry_price: Price at entry
    entry_time: Time of entry
    initial_stop_price: Optional custom stop-loss price (overrides calculated stop)
    initial_target_price: Optional custom take-profit price (overrides calculated target)
    **kwargs: Additional parameters for trade tracking
    
Returns:
    Dictionary with calculated risk parameters for the trade

## risk_parameters

Type definitions and enums for the risk management module.

### Classes

#### `RiskToleranceLevel`

Enum for different risk tolerance levels.

#### `ExitReason`

Enum for different exit reasons.

#### `TradeMetrics`

Data class for storing trade metrics.

#### `RiskParameters`

Data class for storing risk management parameters.

##### Methods

###### `__post_init__()`

Initialize optional dictionary fields if they are None.

#### `RiskAnalysisResults`

Data class for storing risk analysis results.

## types

Type definitions and enums for the risk management module.

### Classes

#### `RiskToleranceLevel`

Enum for different risk tolerance levels.

#### `ExitReason`

Enum for different exit reasons.

#### `TradeMetrics`

Data class for storing trade metrics.

#### `RiskParameters`

Data class for storing risk management parameters.

#### `RiskAnalysisResults`

Data class for storing risk analysis results.

# Analytics Module

No module overview available.

## Contents

- [metrics](#metrics)
- [visualization](#visualization)

## metrics

Performance Metrics Module for Trading System

This module provides comprehensive performance analysis tools for evaluating
trading strategies, including risk metrics, return metrics, and trade statistics.

### Functions

#### `calculate_metrics_from_trades(trades)`

Calculate key performance metrics from a list of trades.

Args:
    trades: List of trade tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)
    
Returns:
    dict: Dictionary of performance metrics

*Returns:* dict: Dictionary of performance metrics

#### `calculate_max_drawdown(equity_curve)`

Calculate the maximum drawdown from an equity curve.

Args:
    equity_curve: List of equity values
    
Returns:
    float: Maximum drawdown percentage

*Returns:* float: Maximum drawdown percentage

#### `calculate_consecutive_winloss(returns)`

Calculate maximum consecutive winning and losing trades.

Args:
    returns: List of trade returns
    
Returns:
    tuple: (max_consecutive_wins, max_consecutive_losses)

*Returns:* tuple: (max_consecutive_wins, max_consecutive_losses)

#### `calculate_monthly_returns(trades)`

Calculate monthly returns from trades.

Args:
    trades: List of trade tuples
    
Returns:
    dict: Monthly returns mapped by YYYY-MM

*Returns:* dict: Monthly returns mapped by YYYY-MM

#### `calculate_drawdown_periods(equity_curve, threshold=5.0)`

Identify significant drawdown periods in an equity curve.

Args:
    equity_curve: List of equity values
    threshold: Minimum drawdown percentage to track
    
Returns:
    list: List of drawdown periods (dictionaries with start/end/max info)

*Returns:* list: List of drawdown periods (dictionaries with start/end/max info)

#### `calculate_regime_performance(trades, regime_data)`

Calculate performance metrics by market regime.

Args:
    trades: List of trade tuples
    regime_data: Dictionary mapping timestamps to regime types
    
Returns:
    dict: Performance metrics by regime

*Returns:* dict: Performance metrics by regime

#### `analyze_trade_durations(trades)`

Analyze trade durations.

Args:
    trades: List of trade tuples
    
Returns:
    dict: Statistics on trade durations

*Returns:* dict: Statistics on trade durations

#### `process_backtester_results(results)`

Convert backtester results to standard format

Args:
    results: Dictionary of results from Backtester.run()
    
Returns:
    tuple: (trades list, equity curve)

*Returns:* tuple: (trades list, equity curve)

## visualization

Visualization Module for Trading System

This module provides tools for visualizing trading results and performance metrics,
including equity curves, drawdowns, trade distributions, and regime-based analysis.

### Classes

#### `TradeVisualizer`

Class for creating trading system visualizations.

##### Methods

###### `__init__(figsize, style='seaborn-v0_8-darkgrid')`

Initialize with default figure size and style.

Args:
    figsize: Default figure size for plots
    style: Matplotlib style to use

###### `plot_equity_curve(trades, title='Equity Curve', benchmark_data=None, initial_capital=10000)`

Plot the equity curve from trades.

Args:
    trades: List of trade tuples
    title: Plot title
    benchmark_data: Optional benchmark data for comparison
    initial_capital: Initial capital for equity calculation
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_drawdowns(trades, title='Drawdown Analysis', threshold=5.0, initial_capital=10000)`

Plot drawdowns over time.

Args:
    trades: List of trade tuples
    title: Plot title
    threshold: Minimum drawdown percentage to highlight
    initial_capital: Initial capital for equity calculation
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_returns_distribution(trades, title='Trade Returns Distribution')`

Plot the distribution of trade returns.

Args:
    trades: List of trade tuples
    title: Plot title
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_monthly_returns(trades, title='Monthly Returns')`

Plot monthly returns as a heatmap.

Args:
    trades: List of trade tuples
    title: Plot title
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_regime_performance(trades, regime_data, title='Performance by Market Regime')`

Plot performance broken down by market regime.

Args:
    trades: List of trade tuples
    regime_data: Dict mapping timestamps to regime types
    title: Plot title
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_trade_analysis(trades, title='Trade Analysis')`

Create a comprehensive trade analysis with multiple subplots.

Args:
    trades: List of trade tuples
    title: Plot title
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `plot_trade_durations(trades, title='Trade Duration Analysis')`

Analyze and plot trade durations.

Args:
    trades: List of trade tuples
    title: Plot title
    
Returns:
    matplotlib.figure.Figure: The figure object

*Returns:* matplotlib.figure.Figure: The figure object

###### `create_performance_dashboard(trades, regime_data=None, title='Trading Performance Dashboard', initial_capital=10000)`

Create a comprehensive performance dashboard with multiple visualizations.

Args:
    trades: List of trade tuples
    regime_data: Optional dict mapping timestamps to regime types
    title: Dashboard title
    initial_capital: Initial capital for equity calculations
    
Returns:
    List[matplotlib.figure.Figure]: List of figure objects

*Returns:* List[matplotlib.figure.Figure]: List of figure objects

###### `save_dashboard(figures, output_dir='./analysis_output', base_filename='trading_analysis')`

Save all dashboard figures to files.

Args:
    figures: List of figure objects
    output_dir: Directory to save figures
    base_filename: Base filename for saved figures
    
Returns:
    list: List of saved file paths

*Returns:* list: List of saved file paths

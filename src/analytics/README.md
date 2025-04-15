# Analytics Module

The Analytics module provides tools for analyzing and visualizing backtesting results and trading performance metrics. It helps traders evaluate strategies, measure risk-adjusted returns, and gain insights into trading patterns.

## Overview

This module includes functions to:
- Process and standardize backtesting results
- Calculate performance metrics
- Generate visualizations of trading performance 
- Analyze drawdowns and recovery periods
- Compare multiple strategies

## Basic Usage

```python
from analytics import process_backtester_results
from analytics.performance_metrics import calculate_sharpe_ratio, calculate_drawdown

# Process backtest results
results = backtester.run()
trades, equity_curve = process_backtester_results(results)

# Calculate performance metrics
sharpe = calculate_sharpe_ratio(equity_curve)
max_drawdown, max_drawdown_duration = calculate_drawdown(equity_curve)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Maximum Drawdown Duration: {max_drawdown_duration} periods")
```

## Core Functions

### `process_backtester_results(results)`

Converts backtester results to a standardized format.

**Parameters:**
- `results` (dict): Dictionary of results from `Backtester.run()`

**Returns:**
- `tuple`: (trades list, equity curve)

**Example:**
```python
trades, equity_curve = process_backtester_results(results)
```

### `calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`

Calculates the Sharpe ratio of a return series.

**Parameters:**
- `returns` (list or array): Series of returns
- `risk_free_rate` (float): Annual risk-free rate (default: 0.0)
- `periods_per_year` (int): Number of periods in a year (default: 252 for daily returns)

**Returns:**
- `float`: Sharpe ratio

### `calculate_drawdown(equity_curve)`

Calculates maximum drawdown and duration.

**Parameters:**
- `equity_curve` (list or array): Series of equity values

**Returns:**
- `tuple`: (maximum drawdown percentage, maximum drawdown duration)

## Integration with Other Modules

### With Backtester Module

```python
from backtester import Backtester
from data_handler import CSVDataHandler
from strategies import WeightedStrategy
from analytics import process_backtester_results, compare_strategies

# Run backtest
backtester = Backtester(data_handler, strategy)
results = backtester.run()

# Analyze results
trades, equity_curve = process_backtester_results(results)
performance_metrics = calculate_performance_metrics(equity_curve, trades)
```

### With Visualization Components

```python
from analytics.visualization import plot_equity_curve, plot_drawdown, plot_trade_distribution

# Create visualizations
equity_fig = plot_equity_curve(equity_curve)
drawdown_fig = plot_drawdown(equity_curve)
trade_dist_fig = plot_trade_distribution(trades)
```

## Best Practices

1. **Standardize analysis workflow** - Create a consistent process for analyzing all strategies
2. **Compare to benchmarks** - Always compare strategy performance to relevant market benchmarks
3. **Focus on risk-adjusted returns** - Use metrics like Sharpe and Sortino ratios for evaluation
4. **Analyze drawdowns carefully** - Pay special attention to maximum drawdown and recovery time
5. **Consider regime-specific performance** - Evaluate how strategies perform in different market regimes
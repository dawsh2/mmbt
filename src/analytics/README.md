# Analytics Module

The Analytics module provides tools for analyzing and visualizing backtesting results and trading performance metrics. It helps traders evaluate strategies, measure risk-adjusted returns, and gain insights into trading patterns.

## Module Structure

```
src/analytics/
├── __init__.py             # Package exports and fallback definitions
├── metrics.py              # Performance metrics calculation functions
└── visualization.py        # Visualization tools for performance analysis
```

## Core Functionality

- **Performance Metrics Calculation**: Calculate metrics like Sharpe ratio, drawdown, win rate
- **Trade Analysis**: Analyze trade characteristics, durations, and profit distributions
- **Visualization**: Create charts for equity curves, drawdowns, return distributions, and more
- **Regime Analysis**: Compare performance across different market regimes

## Basic Usage

```python
from src.analytics.metrics import calculate_metrics_from_trades, calculate_max_drawdown

# Process backtest results
trades = backtester_results['trades']
equity_curve = backtester_results['equity_curve']

# Calculate comprehensive metrics
metrics = calculate_metrics_from_trades(trades)
max_dd = calculate_max_drawdown(equity_curve)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Maximum Drawdown: {max_dd:.2f}%")
```

## Visualization Usage

```python
from src.analytics.visualization import TradeVisualizer

# Create a visualizer
visualizer = TradeVisualizer()

# Create various plots
equity_fig = visualizer.plot_equity_curve(trades, initial_capital=100000)
drawdown_fig = visualizer.plot_drawdowns(trades, initial_capital=100000)
returns_fig = visualizer.plot_returns_distribution(trades)

# Create a comprehensive dashboard
dashboard_figs = visualizer.create_performance_dashboard(
    trades, 
    regime_data=market_regimes,
    title="Strategy Performance Dashboard"
)

# Save the analysis
visualizer.save_dashboard(dashboard_figs, output_dir="./analysis_results")
```

## Key Metrics Functions

### `calculate_metrics_from_trades(trades)`

Calculate comprehensive performance metrics from a list of trades.

**Parameters:**
- `trades`: List of trade tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)

**Returns:**
- Dictionary of performance metrics including:
  - `total_trades`: Number of trades
  - `win_rate`: Percentage of winning trades
  - `total_return`: Total percentage return
  - `sharpe_ratio`: Sharpe ratio (risk-adjusted return)
  - `sortino_ratio`: Sortino ratio (downside risk-adjusted return)
  - `max_drawdown`: Maximum drawdown percentage
  - `calmar_ratio`: Return divided by maximum drawdown
  - `profit_factor`: Gross profit divided by gross loss
  - And several other metrics

### `calculate_max_drawdown(equity_curve)`

Calculate the maximum drawdown from an equity curve.

**Parameters:**
- `equity_curve`: List or array of equity values

**Returns:**
- Maximum drawdown as a percentage

### `calculate_monthly_returns(trades)`

Calculate monthly returns from trades.

**Parameters:**
- `trades`: List of trade tuples

**Returns:**
- Dictionary mapping month strings (YYYY-MM) to percentage returns

### `analyze_trade_durations(trades)`

Analyze trade durations.

**Parameters:**
- `trades`: List of trade tuples

**Returns:**
- Dictionary with statistics on trade durations

## Visualization Features

The `TradeVisualizer` class provides the following visualization methods:

### `plot_equity_curve(trades, title="Equity Curve", benchmark_data=None, initial_capital=10000)`

Plot the equity curve from trades.

### `plot_drawdowns(trades, title="Drawdown Analysis", threshold=5.0, initial_capital=10000)`

Plot drawdowns over time.

### `plot_returns_distribution(trades, title="Trade Returns Distribution")`

Plot the distribution of trade returns.

### `plot_monthly_returns(trades, title="Monthly Returns")`

Plot monthly returns as a heatmap.

### `plot_regime_performance(trades, regime_data, title="Performance by Market Regime")`

Plot performance broken down by market regime.

### `plot_trade_analysis(trades, title="Trade Analysis")`

Create a comprehensive trade analysis with multiple subplots.

### `plot_trade_durations(trades, title="Trade Duration Analysis")`

Analyze and plot trade durations.

### `create_performance_dashboard(trades, regime_data=None, title="Trading Performance Dashboard", initial_capital=10000)`

Create a comprehensive performance dashboard with multiple visualizations.

## Integration with Other Modules

### With Backtester Module

```python
from src.engine import Backtester
from src.analytics.metrics import calculate_metrics_from_trades

# Run backtest
backtester = Backtester(data_handler, strategy)
results = backtester.run()

# Process and analyze results
trades = results['trades']
metrics = calculate_metrics_from_trades(trades)
```

### With Risk Management Module

The Analytics module can be used to visualize risk metrics collected by the Risk Management module:

```python
from src.risk_management import RiskMetricsCollector
from src.analytics.visualization import TradeVisualizer

# Get metrics from risk collector
metrics_df = risk_metrics_collector.get_metrics_dataframe()

# Create risk visualizations
visualizer = TradeVisualizer()
mae_mfe_fig = visualizer.plot_risk_metrics(metrics_df)
```

## Best Practices

1. **Use standardized trade format** - Ensure trades are in the expected format for metrics functions
2. **Compare to benchmarks** - Use the benchmark_data parameter in visualizations to compare to market indices
3. **Analyze drawdowns carefully** - Pay special attention to maximum drawdown and recovery time
4. **Save analysis results** - Use the save_dashboard method to preserve analysis for later review
5. **Use comprehensive dashboards** - The create_performance_dashboard method provides a complete view of performance

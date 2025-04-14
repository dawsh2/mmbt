# Risk Management Module Documentation

The Risk Management package provides a comprehensive framework for implementing advanced risk management using MAE (Maximum Adverse Excursion), MFE (Maximum Favorable Excursion), and ETD (Entry-To-Exit Duration) analysis to derive data-driven risk parameters for algorithmic trading systems.

## Core Concepts

**RiskManager**: Applies risk rules to trades based on parameters derived from historical analysis.  
**RiskMetricsCollector**: Collects and analyzes trade metrics like MAE, MFE, and trade duration.  
**RiskAnalysisEngine**: Analyzes collected metrics to derive insights.  
**RiskParameterOptimizer**: Derives optimal risk parameters from historical metrics.  
**Exit Reasons**: Enumeration of different exit conditions (stop-loss, take-profit, etc.).

## Basic Usage

```python
from risk_management import RiskManager, RiskMetricsCollector, RiskParameters
from risk_management.types import ExitReason
from datetime import datetime

# Create risk parameters
risk_params = RiskParameters(
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    trailing_stop_activation_pct=3.0,
    trailing_stop_distance_pct=1.5
)

# Create metrics collector
metrics_collector = RiskMetricsCollector()

# Create risk manager with parameters and collector
risk_manager = RiskManager(
    risk_params=risk_params,
    metrics_collector=metrics_collector
)

# Open a trade
trade_details = risk_manager.open_trade(
    trade_id="trade_001",
    direction="LONG",
    entry_price=100.0,
    entry_time=datetime.now()
)

print(f"Stop Loss: {trade_details['stop_price']:.2f}")
print(f"Take Profit: {trade_details['target_price']:.2f}")

# Update with market data
for i in range(10):
    # Simulate price movement
    current_price = 100.0 + i * 0.5
    current_time = datetime.now()
    
    # Process bar data
    bar_data = {
        "High": current_price + 0.2,
        "Low": current_price - 0.2,
        "Close": current_price,
        "timestamp": current_time
    }
    
    # Update risk manager with new price data
    update_result = risk_manager.update_price(
        trade_id="trade_001",
        current_price=current_price,
        current_time=current_time,
        bar_data=bar_data
    )
    
    # Check for exit signals
    if update_result['status'] == 'exit':
        print(f"Exit signal: {update_result['exit_info']['exit_reason']}")
        
        # Close the trade
        close_result = risk_manager.close_trade(
            trade_id="trade_001",
            exit_price=current_price,
            exit_time=current_time,
            exit_reason=update_result['exit_info']['exit_reason']
        )
        
        print(f"Trade closed: Return: {close_result['trade_summary']['return_pct']:.2f}%")
        break

# After multiple trades, analyze metrics
metrics_df = metrics_collector.get_metrics_dataframe()
```

## API Reference

### RiskManager

Applies risk management rules based on MAE, MFE, and ETD analysis.

**Constructor Parameters:**
- `risk_params` (RiskParameters): Risk parameters for stop-loss, take-profit, etc.
- `metrics_collector` (RiskMetricsCollector, optional): Collector for tracking trade metrics
- `position_size_calculator` (callable, optional): Function for calculating position sizes

**Methods:**

#### open_trade(trade_id, direction, entry_price, entry_time, initial_stop_price=None, initial_target_price=None)

Register a new trade with the risk manager.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `direction` (str): 'long' or 'short'
- `entry_price` (float): Trade entry price
- `entry_time` (datetime): Trade entry time
- `initial_stop_price` (float, optional): Custom stop-loss price (overrides calculated stop)
- `initial_target_price` (float, optional): Custom take-profit price (overrides calculated target)

**Returns:**
- `dict`: Trade details with calculated risk parameters

**Example:**
```python
trade_details = risk_manager.open_trade(
    trade_id="trade_001",
    direction="LONG",
    entry_price=100.0,
    entry_time=datetime.now()
)
```

#### update_price(trade_id, current_price, current_time, bar_data=None)

Update trade with current price information and check exit conditions.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `current_price` (float): Current market price
- `current_time` (datetime): Current time
- `bar_data` (dict, optional): Complete bar data for metrics collection

**Returns:**
- `dict`: Update status and any exit signals

**Example:**
```python
update_result = risk_manager.update_price(
    trade_id="trade_001",
    current_price=102.5,
    current_time=datetime.now(),
    bar_data={"High": 102.7, "Low": 102.3, "Close": 102.5}
)

if update_result['status'] == 'exit':
    print(f"Exit signal: {update_result['exit_info']['exit_reason']}")
```

#### close_trade(trade_id, exit_price, exit_time, exit_reason=ExitReason.STRATEGY_EXIT)

Close a trade and record final metrics.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `exit_price` (float): Price at exit
- `exit_time` (datetime): Time of exit
- `exit_reason` (ExitReason): Reason for the exit

**Returns:**
- `dict`: Trade summary information

**Example:**
```python
close_result = risk_manager.close_trade(
    trade_id="trade_001",
    exit_price=102.5,
    exit_time=datetime.now(),
    exit_reason=ExitReason.TAKE_PROFIT
)
```

#### update_risk_parameters(new_params)

Update risk parameters for future trades.

**Parameters:**
- `new_params` (RiskParameters): New risk parameters

**Example:**
```python
new_params = RiskParameters(
    stop_loss_pct=1.5,
    take_profit_pct=3.0
)
risk_manager.update_risk_parameters(new_params)
```

#### get_active_trades()

Get all currently active trades.

**Returns:**
- `dict`: Dictionary of active trades

**Example:**
```python
active_trades = risk_manager.get_active_trades()
```

#### modify_stop_loss(trade_id, new_stop_price)

Modify the stop loss for an active trade.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `new_stop_price` (float): New stop loss price

**Returns:**
- `dict`: Update status

**Example:**
```python
result = risk_manager.modify_stop_loss("trade_001", 98.5)
```

#### modify_take_profit(trade_id, new_target_price)

Modify the take profit for an active trade.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `new_target_price` (float): New take profit price

**Returns:**
- `dict`: Update status

**Example:**
```python
result = risk_manager.modify_take_profit("trade_001", 105.0)
```

#### calculate_expectancy()

Calculate mathematical expectancy based on risk parameters.

**Returns:**
- `float`: Expectancy value

**Example:**
```python
expectancy = risk_manager.calculate_expectancy()
```

#### get_position_size_suggestion(account_size, risk_per_trade_pct=1.0, max_position_pct=5.0)

Get position sizing suggestions based on risk parameters.

**Parameters:**
- `account_size` (float): Current account size
- `risk_per_trade_pct` (float, optional): Percentage of account to risk per trade
- `max_position_pct` (float, optional): Maximum percentage of account for any position

**Returns:**
- `dict`: Position sizing information

**Example:**
```python
sizing = risk_manager.get_position_size_suggestion(
    account_size=100000,
    risk_per_trade_pct=1.0
)
```

### RiskMetricsCollector

Collects and stores MAE, MFE, and ETD metrics for analyzing trading performance.

**Constructor Parameters:**
- None

**Methods:**

#### start_trade(trade_id, entry_time, entry_price, direction)

Start tracking a new trade.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `entry_time` (datetime): Time of trade entry
- `entry_price` (float): Price at entry
- `direction` (str): Trade direction ('long' or 'short')

**Example:**
```python
metrics_collector.start_trade(
    trade_id="trade_001",
    entry_time=datetime.now(),
    entry_price=100.0,
    direction="long"
)
```

#### update_price_path(trade_id, bar_data)

Update the price path for a trade with new bar data.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `bar_data` (dict): Bar data containing at minimum 'high', 'low', and 'timestamp'

**Returns:**
- `tuple` or `None`: (mae_pct, mfe_pct) for the current trade

**Example:**
```python
mae, mfe = metrics_collector.update_price_path(
    trade_id="trade_001",
    bar_data={"high": 102.7, "low": 102.3, "timestamp": datetime.now()}
)
```

#### end_trade(trade_id, exit_time, exit_price, exit_reason=ExitReason.UNKNOWN)

End tracking for a trade and calculate final metrics.

**Parameters:**
- `trade_id` (str): Unique identifier for the trade
- `exit_time` (datetime): Time of trade exit
- `exit_price` (float): Price at exit
- `exit_reason` (ExitReason): Reason for the exit

**Returns:**
- `TradeMetrics` or `None`: The completed TradeMetrics object

**Example:**
```python
trade_metrics = metrics_collector.end_trade(
    trade_id="trade_001",
    exit_time=datetime.now(),
    exit_price=102.5,
    exit_reason=ExitReason.TAKE_PROFIT
)
```

#### track_completed_trade(entry_time, entry_price, direction, exit_time, exit_price, price_path=None, exit_reason=ExitReason.UNKNOWN)

Record a completed trade with all information in one call.

**Parameters:**
- `entry_time` (datetime): Time of trade entry
- `entry_price` (float): Price at entry
- `direction` (str): Trade direction ('long' or 'short')
- `exit_time` (datetime): Time of trade exit
- `exit_price` (float): Price at exit
- `price_path` (list, optional): Optional list of bar data between entry and exit
- `exit_reason` (ExitReason): Reason for the exit

**Returns:**
- `TradeMetrics`: The newly created and stored TradeMetrics object

**Example:**
```python
metrics = metrics_collector.track_completed_trade(
    entry_time=datetime(2023, 1, 1),
    entry_price=100.0,
    direction="long",
    exit_time=datetime(2023, 1, 5),
    exit_price=102.5,
    exit_reason=ExitReason.TAKE_PROFIT
)
```

#### get_metrics_dataframe()

Convert metrics to pandas DataFrame for analysis.

**Returns:**
- `DataFrame`: DataFrame containing all trade metrics

**Example:**
```python
metrics_df = metrics_collector.get_metrics_dataframe()
```

#### save_to_csv(filepath)

Save the collected metrics to a CSV file.

**Parameters:**
- `filepath` (str): Path to save the CSV file

**Example:**
```python
metrics_collector.save_to_csv("trade_metrics.csv")
```

#### load_from_csv(filepath)

Load metrics from a CSV file.

**Parameters:**
- `filepath` (str): Path to the CSV file

**Example:**
```python
metrics_collector.load_from_csv("trade_metrics.csv")
```

#### clear()

Clear all stored trade metrics.

**Example:**
```python
metrics_collector.clear()
```

### RiskAnalysisEngine

Analyzes MAE, MFE, and ETD metrics to derive insights for risk management.

**Constructor Parameters:**
- `metrics_df` (DataFrame): DataFrame containing trade metrics

**Methods:**

#### analyze_mae()

Analyze MAE distribution and characteristics.

**Returns:**
- `dict`: Dictionary of MAE statistics

**Example:**
```python
from risk_management import RiskAnalysisEngine

analysis_engine = RiskAnalysisEngine(metrics_df)
mae_stats = analysis_engine.analyze_mae()
```

#### analyze_mfe()

Analyze MFE distribution and characteristics.

**Returns:**
- `dict`: Dictionary of MFE statistics

**Example:**
```python
mfe_stats = analysis_engine.analyze_mfe()
```

#### analyze_etd()

Analyze trade duration characteristics.

**Returns:**
- `dict`: Dictionary of ETD statistics

**Example:**
```python
etd_stats = analysis_engine.analyze_etd()
```

#### analyze_exit_reasons()

Analyze the distribution and performance of different exit reasons.

**Returns:**
- `dict`: Dictionary of exit reason statistics

**Example:**
```python
exit_stats = analysis_engine.analyze_exit_reasons()
```

#### calculate_mad_ratio()

Calculate the MAE/MFE Adjusted Duration (MAD) ratio.

**Returns:**
- `float`: MAD ratio value

**Example:**
```python
mad_ratio = analysis_engine.calculate_mad_ratio()
```

#### analyze_all()

Run all analyses and return comprehensive results.

**Returns:**
- `RiskAnalysisResults`: Object containing all analysis results

**Example:**
```python
all_results = analysis_engine.analyze_all()
```

#### from_collector(collector, clear_price_paths=True)

Create an analyzer directly from a RiskMetricsCollector.

**Parameters:**
- `collector` (RiskMetricsCollector): A RiskMetricsCollector instance
- `clear_price_paths` (bool, optional): Whether to clear price paths to save memory

**Returns:**
- `RiskAnalysisEngine`: RiskAnalysisEngine instance

**Example:**
```python
analysis_engine = RiskAnalysisEngine.from_collector(metrics_collector)
```

### RiskParameterOptimizer

Derives optimal risk management parameters from analyzed trade metrics.

**Constructor Parameters:**
- `analysis_results` (RiskAnalysisResults): Results from the RiskAnalysisEngine

**Methods:**

#### optimize_stop_loss(risk_tolerance='moderate')

Derive optimal stop-loss parameters based on MAE analysis.

**Parameters:**
- `risk_tolerance` (str or RiskToleranceLevel): 'conservative', 'moderate', or 'aggressive'

**Returns:**
- `dict`: Optimized stop-loss parameters

**Example:**
```python
from risk_management import RiskParameterOptimizer

optimizer = RiskParameterOptimizer(analysis_results)
stop_loss = optimizer.optimize_stop_loss('conservative')
```

#### optimize_take_profit(profit_target_style='moderate')

Derive optimal take-profit parameters based on MFE analysis.

**Parameters:**
- `profit_target_style` (str or RiskToleranceLevel): 'conservative', 'moderate', or 'aggressive'

**Returns:**
- `dict`: Optimized take-profit parameters

**Example:**
```python
take_profit = optimizer.optimize_take_profit('aggressive')
```

#### optimize_trailing_stop()

Derive optimal trailing stop parameters based on MAE and MFE analysis.

**Returns:**
- `dict`: Optimized trailing stop parameters

**Example:**
```python
trailing_stop = optimizer.optimize_trailing_stop()
```

#### optimize_time_exit()

Derive optimal time-based exit parameters based on ETD analysis.

**Returns:**
- `dict`: Optimized time-based exit parameters

**Example:**
```python
time_exit = optimizer.optimize_time_exit()
```

#### calculate_risk_reward_setups()

Calculate various risk-reward setups based on the optimized parameters.

**Returns:**
- `list`: List of different risk management setups with expected metrics

**Example:**
```python
setups = optimizer.calculate_risk_reward_setups()
```

#### get_optimal_parameters(risk_tolerance='moderate', include_trailing_stop=True, include_time_exit=True)

Get a complete set of optimal risk parameters.

**Parameters:**
- `risk_tolerance` (str or RiskToleranceLevel): Overall risk tolerance level
- `include_trailing_stop` (bool): Whether to include trailing stop parameters
- `include_time_exit` (bool): Whether to include time-based exit parameters

**Returns:**
- `RiskParameters`: RiskParameters object with optimized values

**Example:**
```python
optimal_params = optimizer.get_optimal_parameters('moderate')
```

#### get_balanced_parameters()

Get a balanced set of risk parameters that maximizes expectancy.

**Returns:**
- `RiskParameters`: RiskParameters object with balanced optimal values

**Example:**
```python
balanced_params = optimizer.get_balanced_parameters()
```

### Data Classes and Enums

#### RiskParameters

Data class for storing risk management parameters.

**Attributes:**
- `stop_loss_pct` (float): Stop loss percentage
- `take_profit_pct` (float, optional): Take profit percentage
- `trailing_stop_activation_pct` (float, optional): Percentage move to activate trailing stop
- `trailing_stop_distance_pct` (float, optional): Trailing stop distance as percentage
- `max_duration` (int or timedelta, optional): Maximum trade duration
- `risk_reward_ratio` (float, optional): Risk-reward ratio
- `expected_win_rate` (float, optional): Expected win rate
- `risk_tolerance` (RiskToleranceLevel, optional): Risk tolerance level

**Example:**
```python
from risk_management.types import RiskParameters, RiskToleranceLevel

params = RiskParameters(
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    trailing_stop_activation_pct=3.0,
    trailing_stop_distance_pct=1.5,
    risk_tolerance=RiskToleranceLevel.MODERATE
)
```

#### RiskToleranceLevel

Enum for different risk tolerance levels.

```python
from risk_management.types import RiskToleranceLevel

level = RiskToleranceLevel.CONSERVATIVE
# Options: CONSERVATIVE, MODERATE, AGGRESSIVE
```

#### ExitReason

Enum for different exit reasons.

```python
from risk_management.types import ExitReason

reason = ExitReason.STOP_LOSS
# Options: STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, TIME_EXIT, STRATEGY_EXIT, UNKNOWN
```

#### TradeMetrics

Data class for storing trade metrics.

**Attributes:**
- `entry_time` (datetime): Trade entry time
- `entry_price` (float): Trade entry price
- `direction` (str): Trade direction ('long' or 'short')
- `exit_time` (datetime, optional): Trade exit time
- `exit_price` (float, optional): Trade exit price
- `return_pct` (float, optional): Percentage return
- `mae_pct` (float, optional): Maximum Adverse Excursion percentage
- `mfe_pct` (float, optional): Maximum Favorable Excursion percentage
- `duration` (timedelta, optional): Trade duration
- `duration_bars` (int, optional): Trade duration in bars
- `is_winner` (bool, optional): Whether the trade was profitable
- `exit_reason` (ExitReason, optional): Reason for the exit
- `price_path` (list, optional): List of price data during the trade

## Advanced Usage

### Optimizing Risk Parameters Based on Historical Data

```python
from risk_management import RiskMetricsCollector, RiskAnalysisEngine, RiskParameterOptimizer
from risk_management.types import RiskParameters, ExitReason
import pandas as pd

# Load historical trade data
trades_df = pd.read_csv('historical_trades.csv')

# Create metrics collector
collector = RiskMetricsCollector()

# Add each historical trade
for _, trade in trades_df.iterrows():
    collector.track_completed_trade(
        entry_time=pd.to_datetime(trade['entry_time']),
        entry_price=trade['entry_price'],
        direction=trade['direction'],
        exit_time=pd.to_datetime(trade['exit_time']),
        exit_price=trade['exit_price'],
        exit_reason=ExitReason[trade['exit_reason']]
    )

# Create analysis engine from collector
analyzer = RiskAnalysisEngine.from_collector(collector)

# Run analysis
analysis_results = analyzer.analyze_all()

# Create parameter optimizer
optimizer = RiskParameterOptimizer(analysis_results)

# Get different parameter sets
conservative_params = optimizer.get_optimal_parameters('conservative')
moderate_params = optimizer.get_optimal_parameters('moderate')
aggressive_params = optimizer.get_optimal_parameters('aggressive')

# Get balanced parameters (best expectancy)
balanced_params = optimizer.get_balanced_parameters()

# Display results
print("Conservative Parameters:")
print(f"  Stop Loss: {conservative_params.stop_loss_pct:.2f}%")
print(f"  Take Profit: {conservative_params.take_profit_pct:.2f}%")

print("\nAggressive Parameters:")
print(f"  Stop Loss: {aggressive_params.stop_loss_pct:.2f}%")
print(f"  Take Profit: {aggressive_params.take_profit_pct:.2f}%")

print("\nBalanced Parameters (Best Expectancy):")
print(f"  Stop Loss: {balanced_params.stop_loss_pct:.2f}%")
print(f"  Take Profit: {balanced_params.take_profit_pct:.2f}%")
print(f"  Trailing Stop Activation: {balanced_params.trailing_stop_activation_pct:.2f}%")
print(f"  Trailing Stop Distance: {balanced_params.trailing_stop_distance_pct:.2f}%")
print(f"  Expected Win Rate: {balanced_params.expected_win_rate:.2f}")
print(f"  Risk-Reward Ratio: {balanced_params.risk_reward_ratio:.2f}")
```

### Creating a Dynamic Risk Manager with Parameter Updates

```python
from risk_management import RiskManager, RiskMetricsCollector, RiskAnalysisEngine, RiskParameterOptimizer
from risk_management.types import RiskParameters
import pandas as pd

class DynamicRiskManager:
    """Risk manager that periodically updates parameters based on performance."""
    
    def __init__(self, initial_params, update_frequency=50):
        self.metrics_collector = RiskMetricsCollector()
        self.risk_manager = RiskManager(
            risk_params=initial_params,
            metrics_collector=self.metrics_collector
        )
        self.update_frequency = update_frequency
        self.trade_count = 0
        
    def open_trade(self, *args, **kwargs):
        """Open a trade with the risk manager."""
        self.trade_count += 1
        return self.risk_manager.open_trade(*args, **kwargs)
        
    def update_price(self, *args, **kwargs):
        """Update price with the risk manager."""
        return self.risk_manager.update_price(*args, **kwargs)
        
    def close_trade(self, *args, **kwargs):
        """Close a trade with the risk manager."""
        result = self.risk_manager.close_trade(*args, **kwargs)
        
        # Check if it's time to update parameters
        if self.trade_count % self.update_frequency == 0:
            self._update_parameters()
            
        return result
        
    def _update_parameters(self):
        """Update risk parameters based on collected metrics."""
        if len(self.metrics_collector.trade_metrics) < 20:
            # Not enough data to update parameters
            return
            
        # Create analysis engine
        analyzer = RiskAnalysisEngine.from_collector(self.metrics_collector)
        
        # Run analysis
        analysis_results = analyzer.analyze_all()
        
        # Create parameter optimizer
        optimizer = RiskParameterOptimizer(analysis_results)
        
        # Get balanced parameters
        new_params = optimizer.get_balanced_parameters()
        
        # Update risk manager with new parameters
        self.risk_manager.update_risk_parameters(new_params)
        
        print(f"Updated risk parameters based on {len(self.metrics_collector.trade_metrics)} trades")
        print(f"  New Stop Loss: {new_params.stop_loss_pct:.2f}%")
        print(f"  New Take Profit: {new_params.take_profit_pct:.2f}%")
```

### Implementing Custom Position Sizing

```python
from risk_management import RiskManager, RiskParameters
from risk_management.types import ExitReason

def kelly_position_sizer(direction, entry_price, stop_price, target_price, 
                         risk_reward_ratio, expected_win_rate):
    """
    Calculate position size using Kelly Criterion.
    
    Args:
        direction: Trade direction ('long' or 'short')
        entry_price: Entry price
        stop_price: Stop loss price
        target_price: Take profit price
        risk_reward_ratio: Expected risk-reward ratio
        expected_win_rate: Expected win rate
        
    Returns:
        Recommended position size as percentage of portfolio
    """
    # Calculate win probability and payoff ratio
    win_prob = expected_win_rate
    
    # Calculate payoff ratio from prices
    if direction.lower() == 'long':
        loss = entry_price - stop_price
        gain = target_price - entry_price
    else:  # short
        loss = stop_price - entry_price
        gain = entry_price - target_price
        
    # Normalize to make loss = 1
    if loss != 0:
        payoff_ratio = gain / loss
    else:
        payoff_ratio = risk_reward_ratio
    
    # Kelly formula: f* = p - (1-p)/r
    # where p = probability of win, r = payoff ratio
    kelly_pct = win_prob - (1 - win_prob) / payoff_ratio
    
    # Limit Kelly to reasonable range (0-50%)
    kelly_pct = max(0, min(kelly_pct, 0.5))
    
    # Apply half-Kelly for safety
    half_kelly = kelly_pct * 0.5
    
    return half_kelly

# Create risk manager with custom position sizer
risk_params = RiskParameters(
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    expected_win_rate=0.6,
    risk_reward_ratio=2.0
)

risk_manager = RiskManager(
    risk_params=risk_params,
    position_size_calculator=kelly_position_sizer
)

# Open trade with position sizing
trade_details = risk_manager.open_trade(
    trade_id="trade_001",
    direction="LONG",
    entry_price=100.0,
    entry_time=datetime.now()
)

print(f"Recommended position size: {trade_details['position_size']:.2%}")
```

### Analyzing Trade Performance by Exit Reason

```python
from risk_management import RiskMetricsCollector, RiskAnalysisEngine
import pandas as pd
import matplotlib.pyplot as plt

# Assume we have a collector with completed trades
collector = RiskMetricsCollector()
# ...add trades to collector...

# Get metrics DataFrame
metrics_df = collector.get_metrics_dataframe()

# Create analysis engine
analyzer = RiskAnalysisEngine(metrics_df)

# Analyze exit reasons
exit_stats = analyzer.analyze_exit_reasons()

# Plot performance by exit reason
plt.figure(figsize=(12, 8))

# Create subplots
plt.subplot(2, 2, 1)
exit_counts = pd.Series(exit_stats['counts'])
exit_counts.plot(kind='bar')
plt.title("Number of Trades by Exit Reason")
plt.ylabel("Count")

plt.subplot(2, 2, 2)
win_rates = pd.Series(exit_stats['win_rates'])
win_rates.plot(kind='bar')
plt.title("Win Rate by Exit Reason")
plt.ylabel("Win Rate")

plt.subplot(2, 2, 3)
avg_returns = pd.Series(exit_stats['avg_returns'])
avg_returns.plot(kind='bar')
plt.title("Average Return by Exit Reason")
plt.ylabel("Return %")

plt.subplot(2, 2, 4)
avg_mae = pd.Series(exit_stats['avg_mae'])
avg_mfe = pd.Series(exit_stats['avg_mfe'])
pd.DataFrame({'MAE': avg_mae, 'MFE': avg_mfe}).plot(kind='bar')
plt.title("Average MAE/MFE by Exit Reason")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()

# Analyze which exit reasons perform best
exit_performance = pd.DataFrame({
    'count': pd.Series(exit_stats['counts']),
    'win_rate': pd.Series(exit_stats['win_rates']),
    'avg_return': pd.Series(exit_stats['avg_returns']),
    'mae': pd.Series(exit_stats['avg_mae']),
    'mfe': pd.Series(exit_stats['avg_mfe'])
})

print("Exit Reason Performance Summary:")
print(exit_performance)
```

## Best Practices

1. **Start with conservative parameters**: Begin with more conservative risk parameters and adjust based on real performance data.

2. **Collect sufficient data**: Collect metrics from at least 30-50 trades before attempting to optimize parameters.

3. **Regularly update parameters**: Markets change over time, so periodically re-optimize your risk parameters.

4. **Consider regime-specific parameters**: Different market regimes may require different risk parameters.

5. **Verify parameter stability**: Ensure that optimized parameters are stable across different time periods.

6. **Balance theory and empirics**: Combine theoretical risk models with empirical results from your trading data.

7. **Use proper position sizing**: Always incorporate position sizing that accounts for risk parameters and account size.

8. **Monitor MAE and MFE**: Continuously analyze MAE and MFE to identify potential improvements to risk rules.

9. **Evaluate expectancy**: Focus on maximizing the mathematical expectancy of your trading system.

10. **Maintain risk discipline**: Never override risk parameters based on subjective judgment or emotions.
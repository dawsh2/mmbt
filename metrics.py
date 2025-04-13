"""
Performance metrics calculation for the backtesting engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def convert_trades_to_returns(trades):
    """
    Convert trade tuples/dicts to a numpy array of returns.
    
    Args:
        trades: List of trade tuples or dictionaries
        
    Returns:
        numpy.ndarray: Array of log returns
    """
    returns = []
    
    for trade in trades:
        if isinstance(trade, (tuple, list)) and len(trade) >= 6:
            # Trade tuple format: (entry_time, direction, entry_price, exit_time, exit_price, log_return, ...)
            log_return = trade[5]
            returns.append(log_return)
        elif isinstance(trade, dict) and 'log_return' in trade:
            # Trade dictionary format
            log_return = trade['log_return']
            returns.append(log_return)
    
    return np.array(returns)

def calculate_time_weighted_metrics(trades, metrics=None, timeframe='minute', 
                                   trading_hours_start=9.5, trading_hours_end=16, 
                                   trading_days_per_year=252, risk_free_rate=0):
    """
    Calculate time-weighted metrics that account for periods of inactivity.

    This method creates a series of returns for every time period in the backtest,
    properly accounting for periods with no trades.

    Args:
        trades: List of trade tuples or dictionaries
        metrics: List of metrics to calculate (default: ['sharpe', 'sortino'])
        timeframe: Time frame for analysis ('minute', 'hour', 'day')
        trading_hours_start: Start of trading hours (e.g., 9.5 for 9:30 AM)
        trading_hours_end: End of trading hours (e.g., 16 for 4:00 PM)
        trading_days_per_year: Number of trading days per year (default: 252)
        risk_free_rate: Annual risk-free rate

    Returns:
        dict: Dictionary of calculated metrics
    """
    if not trades:
        return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0}

    if metrics is None:
        metrics = ['sharpe', 'sortino']

    # First, determine the start and end times of your backtest
    start_time = trades[0][0]  # First trade entry time
    end_time = trades[-1][3]   # Last trade exit time

    from datetime import datetime, timedelta

    # Convert timestamps to datetime objects if they're strings
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    # Set time delta based on timeframe
    if timeframe == 'minute':
        delta = timedelta(minutes=1)
        periods_per_day = (trading_hours_end - trading_hours_start) * 60
    elif timeframe == 'hour':
        delta = timedelta(hours=1)
        periods_per_day = trading_hours_end - trading_hours_start
    elif timeframe == 'day':
        delta = timedelta(days=1)
        periods_per_day = 1
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Generate a list of all timestamps in the backtest period
    current_time = start_time
    all_periods = []
    
    while current_time <= end_time:
        # Only include timestamps during trading hours
        hour = current_time.hour + current_time.minute / 60.0
        
        if timeframe == 'day' or (trading_hours_start <= hour < trading_hours_end):
            # Skip weekends for intraday timeframes
            if timeframe == 'day' or current_time.weekday() < 5:  # Monday=0, Sunday=6
                all_periods.append(current_time)
        
        current_time += delta

    # Initialize equity curve with starting value
    initial_equity = 10000
    equity_curve = {period: initial_equity for period in all_periods}

    # Update equity curve at trade entry/exit points
    current_equity = initial_equity
    for trade in trades:
        entry_time = trade[0]
        exit_time = trade[3]
        log_return = trade[5]

        # Convert timestamps if needed
        if isinstance(entry_time, str):
            entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        if isinstance(exit_time, str):
            exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

        # Calculate new equity after this trade
        current_equity *= np.exp(log_return)

        # Update equity value at exit time and all subsequent periods
        for period in all_periods:
            if period >= exit_time:
                equity_curve[period] = current_equity

    # Calculate period-by-period returns
    period_returns = []
    prev_equity = None

    for period, equity in sorted(equity_curve.items()):
        if prev_equity is not None:
            period_return = np.log(equity / prev_equity)
            period_returns.append(period_return)
        prev_equity = equity

    # Calculate metrics from period returns
    if len(period_returns) < 2:
        return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0}

    period_returns = np.array(period_returns)
    avg_period_return = np.mean(period_returns)
    std_period_return = np.std(period_returns)

    # Calculate annualization factor
    periods_per_year = periods_per_day * trading_days_per_year
    
    # Convert annual risk-free rate to per-period rate
    period_risk_free_rate = risk_free_rate / periods_per_year

    # Calculate metrics
    result = {}
    
    if 'sharpe' in metrics and std_period_return > 0:
        sharpe = (avg_period_return - period_risk_free_rate) / std_period_return
        result['sharpe_ratio'] = sharpe * np.sqrt(periods_per_year)
    else:
        result['sharpe_ratio'] = 0.0
        
    if 'sortino' in metrics:
        # Calculate downside deviation (only negative returns)
        downside_returns = period_returns[period_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                sortino = (avg_period_return - period_risk_free_rate) / downside_deviation
                result['sortino_ratio'] = sortino * np.sqrt(periods_per_year)
            else:
                result['sortino_ratio'] = float('inf') if avg_period_return > period_risk_free_rate else 0.0
        else:
            result['sortino_ratio'] = float('inf') if avg_period_return > period_risk_free_rate else 0.0
    
    if 'calmar' in metrics:
        # Calculate maximum drawdown
        cumulative_returns = np.cumsum(period_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        if max_drawdown > 0:
            # Annualize the average return
            annual_return = avg_period_return * periods_per_year
            result['calmar_ratio'] = annual_return / max_drawdown
        else:
            result['calmar_ratio'] = float('inf') if avg_period_return > 0 else 0.0
    
    return result


# Also add a function to calculate metrics directly from trades
def calculate_metrics_from_trades(trades):
    """
    Calculate performance metrics directly from a list of trades.
    
    Args:
        trades: List of trade tuples or dictionaries
        
    Returns:
        dict: Performance metrics
    """
    returns_array = convert_trades_to_returns(trades)
    metrics = calculate_metrics(returns_array)
    
    # Update number of trades
    metrics['number_of_trades'] = len(trades)
    
    return metrics

# In metrics.py, add this function:
def calculate_sharpe(trades, risk_free_rate=0):
    """
    Calculate Sharpe ratio using minute-by-minute returns from trades.

    This method creates a series of returns for every minute in the backtest
    period, properly accounting for periods with no trades.

    Args:
        trades: List of trade tuples or dictionaries
        risk_free_rate: Annual risk-free rate

    Returns:
        float: Annualized Sharpe ratio
    """
    if not trades:
        return 0.0

    # First, determine the start and end times of your backtest
    start_time = trades[0][0]  # First trade entry time
    end_time = trades[-1][3]   # Last trade exit time

    # Create a dictionary to store minute-by-minute equity values
    from datetime import datetime, timedelta

    # Convert timestamps to datetime objects if they're strings
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    # Generate a list of all minute timestamps in the backtest period
    current_time = start_time
    all_minutes = []
    while current_time <= end_time:
        # Only include timestamps during trading hours (e.g., 9:30 AM to 4:00 PM)
        hour = current_time.hour
        minute = current_time.minute
        if (9 <= hour < 16) or (hour == 16 and minute == 0):
            # Skip weekends
            if current_time.weekday() < 5:  # Monday=0, Sunday=6
                all_minutes.append(current_time)
        current_time += timedelta(minutes=1)

    # Initialize equity curve with starting value (e.g., $10,000)
    initial_equity = 10000
    equity_curve = {minute: initial_equity for minute in all_minutes}

    # Update equity curve at trade entry/exit points
    current_equity = initial_equity
    for trade in trades:
        entry_time = trade[0]
        exit_time = trade[3]
        log_return = trade[5]

        # Convert timestamps if needed
        if isinstance(entry_time, str):
            entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        if isinstance(exit_time, str):
            exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

        # Calculate new equity after this trade
        current_equity *= np.exp(log_return)

        # Update equity value at exit time and all subsequent minutes
        for minute in all_minutes:
            if minute >= exit_time:
                equity_curve[minute] = current_equity

    # Calculate minute-by-minute returns
    minute_returns = []
    prev_equity = None

    for minute, equity in sorted(equity_curve.items()):
        if prev_equity is not None:
            minute_return = np.log(equity / prev_equity)
            minute_returns.append(minute_return)
        prev_equity = equity

    # Calculate Sharpe ratio using minute returns
    if len(minute_returns) < 2:
        return 0.0

    avg_minute_return = np.mean(minute_returns)
    std_minute_return = np.std(minute_returns)

    if std_minute_return == 0:
        return 0.0

    # Annualization factor for minutes (assuming 6.5 trading hours per day)
    minutes_per_day = 6.5 * 60  # 390 minutes in a typical trading day
    trading_days_per_year = 252  # Standard trading days per year
    minutes_per_year = minutes_per_day * trading_days_per_year

    # Convert annual risk-free rate to per-minute rate
    minute_risk_free_rate = risk_free_rate / minutes_per_year

    # Calculate Sharpe and annualize
    sharpe = (avg_minute_return - minute_risk_free_rate) / std_minute_return
    annualized_sharpe = sharpe * np.sqrt(minutes_per_year)

    return annualized_sharpe


def calculate_sortino(returns_array, target_return=0, annualization_factor=252):
    """Calculate Sortino ratio based on an array of returns.
    
    Args:
        returns_array: Array of returns
        target_return: Minimum acceptable return (default 0)
        annualization_factor: Factor for annualizing (default 252 trading days)
        
    Returns:
        float: Annualized Sortino ratio
    """
    if len(returns_array) < 2:
        return 0.0
    
    avg_return = np.mean(returns_array)
    
    # Calculate downside deviation (only returns below target)
    downside_returns = returns_array[returns_array < target_return]
    
    # If no downside returns, return inf or a large number for positive avg return
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if avg_return > target_return else 0
    
    downside_deviation = np.std(downside_returns)
    
    # Calculate Sortino and annualize
    sortino = (avg_return - target_return) / downside_deviation
    annualized_sortino = sortino * np.sqrt(annualization_factor)
    
    return annualized_sortino


def calculate_returns(signals: pd.Series, returns: pd.Series) -> pd.Series:
    """Calculate strategy returns based on signals and asset returns.
    
    Args:
        signals: Series of signals (-1, 0, 1)
        returns: Series of log returns
        
    Returns:
        Series of strategy returns
    """
    return signals * returns


def calculate_metrics(strategy_returns) -> Dict[str, float]:
    """Calculate performance metrics for a strategy."""
    # Ensure we have numeric data
    if isinstance(strategy_returns, pd.Series):
        strategy_returns = strategy_returns.astype(float)
        returns_array = strategy_returns.values
    else:
        try:
            returns_array = np.array(strategy_returns, dtype=float)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'number_of_trades': 0
            }
    
    # Check if array is empty or has only zeros
    if len(returns_array) == 0 or np.all(returns_array == 0):
        print("WARNING: Empty or all-zero returns array")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'number_of_trades': 0
        }
    
    # Calculate metrics
    try:
        # Debug statistics about returns
        print(f"DEBUG - Returns statistics: count={len(returns_array)}, " +
              f"mean={np.mean(returns_array):.6f}, sum={np.sum(returns_array):.6f}")
        
        # Convert log returns to simple returns
        simple_returns = np.exp(returns_array) - 1
        
        # Calculate total return
        total_return = np.exp(np.sum(returns_array)) - 1
        
        # Calculate annualized metrics (assuming 252 trading days per year)
        annualized_return = np.exp(np.mean(returns_array) * 252) - 1
        annualized_volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        sortino_ratio = calculate_sortino_ratio(returns_array)
        
        # Calculate drawdowns
        cum_returns = np.cumprod(1 + simple_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns / peak) - 1
        max_drawdown = np.min(drawdowns)
        
        # Calculate win rate
        win_rate = np.sum(returns_array > 0) / len(returns_array)
        
        # Calculate profit factor
        winning_trades = np.sum(returns_array[returns_array > 0])
        losing_trades = abs(np.sum(returns_array[returns_array < 0]))
        profit_factor = winning_trades / losing_trades if losing_trades != 0 else float('inf')
        
        # Debug the calculated metrics
        print(f"DEBUG - Calculated metrics: TR={total_return:.6f}, AR={annualized_return:.6f}, " +
              f"Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.6f}")
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,  # Add this line
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'number_of_trades': 0  # Placeholder, updated later
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'number_of_trades': 0
        }


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """Print performance metrics in a formatted way."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Check if metrics dictionary is empty or missing keys
    if not metrics:
        print("No metrics available")
        return
    
    # Helper function to safely convert NumPy types to Python types
    def safe_float(value):
        try:
            return float(value)
        except:
            return 0.0
    
    # Print each metric, safely handling NumPy types
    print(f"Total Return: {safe_float(metrics.get('total_return', 0.0)):.2%}")
    print(f"Annualized Return: {safe_float(metrics.get('annualized_return', 0.0)):.2%}")
    print(f"Annualized Volatility: {safe_float(metrics.get('annualized_volatility', 0.0)):.2%}")
    print(f"Sharpe Ratio: {safe_float(metrics.get('sharpe_ratio', 0.0)):.2f}")
    print(f"Maximum Drawdown: {safe_float(metrics.get('max_drawdown', 0.0)):.2%}")
    print(f"Win Rate: {safe_float(metrics.get('win_rate', 0.0)):.2%}")
    print(f"Profit Factor: {safe_float(metrics.get('profit_factor', 0.0)):.2f}")
    print(f"Number of Trades: {int(safe_float(metrics.get('number_of_trades', 0)))}")

def compare_strategies(metrics_list: List[Dict[str, float]], names: List[str]):
    """Compare performance metrics for multiple strategies.
    
    Args:
        metrics_list: List of metrics dictionaries
        names: List of strategy names
    """
    print("\nStrategy Comparison")
    print("=" * 80)
    
    # Print headers
    header = "Metric"
    for name in names:
        header += f" | {name:>15}"
    print(header)
    print("-" * len(header))
    
    # Print metrics
    metrics_to_display = [
        ('total_return', 'Total Return', lambda x: f"{x:.2%}"),
        ('annualized_return', 'Ann. Return', lambda x: f"{x:.2%}"),
        ('annualized_volatility', 'Ann. Volatility', lambda x: f"{x:.2%}"),
        ('sharpe_ratio', 'Sharpe Ratio', lambda x: f"{x:.2f}"),
        ('max_drawdown', 'Max Drawdown', lambda x: f"{x:.2%}"),
        ('win_rate', 'Win Rate', lambda x: f"{x:.2%}"),
        ('profit_factor', 'Profit Factor', lambda x: f"{x:.2f}"),
        ('number_of_trades', 'Trades', lambda x: f"{int(x)}")
    ]
    
    for key, label, formatter in metrics_to_display:
        row = f"{label:<15}"
        for metrics in metrics_list:
            row += f" | {formatter(metrics[key]):>15}"
        print(row)


class MetricsRegistry:
        _metrics = {}

        @classmethod
        def register(cls, name, calculation_func):
            cls._metrics[name] = calculation_func

        @classmethod
        def calculate(cls, name, trade_data):
            if name in cls._metrics:
                return cls._metrics[name](trade_data)
            raise ValueError(f"Unknown metric: {name}")

# Register metrics
MetricsRegistry.register('sharpe', calculate_sharpe)
MetricsRegistry.register('sortino', calculate_sortino)        

"""
Performance metrics calculation for the backtesting engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

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
    # Print type and value for debugging
    print(f"DEBUG - Type of strategy_returns: {type(strategy_returns)}")
    
    # Ensure we have numeric data
    if isinstance(strategy_returns, pd.Series):
        # Convert to float values - this is critical!
        strategy_returns = strategy_returns.astype(float)
        returns_array = strategy_returns.values
    else:
        # Try to convert to numpy array of floats
        try:
            returns_array = np.array(strategy_returns, dtype=float)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            # Return placeholder metrics
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
    
    # Check if array is empty
    if len(returns_array) == 0:
        print("WARNING: Empty returns array")
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
    
    # Calculate metrics with safe operations
    try:
        # Convert log returns to simple returns
        simple_returns = np.exp(returns_array) - 1
        
        # Calculate total return
        total_return = np.exp(np.sum(returns_array)) - 1
        
        # Calculate annualized metrics
        annualized_return = np.exp(np.mean(returns_array) * 252) - 1
        annualized_volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
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
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'number_of_trades': 0  # Placeholder, updated later
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
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
        
    # Define metrics to display with default values
    metrics_to_display = [
        ('total_return', 'Total Return', 0.0),
        ('annualized_return', 'Annualized Return', 0.0),
        ('annualized_volatility', 'Annualized Volatility', 0.0),
        ('sharpe_ratio', 'Sharpe Ratio', 0.0),
        ('max_drawdown', 'Maximum Drawdown', 0.0),
        ('win_rate', 'Win Rate', 0.0),
        ('profit_factor', 'Profit Factor', 0.0),
        ('number_of_trades', 'Number of Trades', 0)
    ]
    
    # Print each metric with a default value if not found
    for key, label, default in metrics_to_display:
        value = metrics.get(key, default)
        
        # Format based on type
        if key in ['sharpe_ratio', 'profit_factor']:
            print(f"{label}: {value:.2f}")
        elif key == 'number_of_trades':
            print(f"{label}: {int(value)}")
        else:
            # Percentage format
            print(f"{label}: {value:.2%}")    

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

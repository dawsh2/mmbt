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


def calculate_metrics(strategy_returns, signals=None) -> Dict[str, float]:
    """Calculate performance metrics for a strategy."""
    if isinstance(strategy_returns, pd.Series):
        strategy_returns = strategy_returns.astype(float)
        returns_array = strategy_returns.values
    else:
        try:
            returns_array = np.array(strategy_returns, dtype=float)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            return {}

    if len(returns_array) == 0 or np.all(np.isnan(returns_array)):
        print("No usable returns — skipping metrics.")
        return {}

    # Basic metrics
    total_return = np.nansum(returns_array)
    annualized_return = np.nanmean(returns_array) * 252
    annualized_volatility = np.nanstd(returns_array) * np.sqrt(252)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0 else np.nan
    )

    max_drawdown = 0
    cumulative = np.cumsum(returns_array)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown = np.max(drawdown)

    # Win/loss stats
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # ✅ Count trades from signals
    number_of_trades = (signals.fillna(0) != signals.fillna(0).shift(1)).sum() if signals is not None else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'number_of_trades': number_of_trades,
    }


# # In metrics.py, update the calculate_metrics function
# def calculate_metrics(strategy_returns, signals=None) -> Dict[str, float]:

#     """Calculate performance metrics for a strategy."""
#     # Ensure we have numeric data
#     if isinstance(strategy_returns, pd.Series):
#         strategy_returns = strategy_returns.astype(float)
#         returns_array = strategy_returns.values
#     else:
#         try:
#             returns_array = np.array(strategy_returns, dtype=float)
#         except Exception as e:
#             print(f"Error converting to numpy array: {e}")
#             return {
#                 'total_return': 0.0,
#                 'annualized_return': 0.0,
#                 'annualized_volatility': 0.0,
#                 'sharpe_ratio': 0.0,
#                 'max_drawdown': 0.0,
#                 'win_rate': 0.0,
#                 'profit_factor': 0.0,
#                 'number_of_trades': 0
#             }
    
#     # Check if array is empty or has only zeros
#     if len(returns_array) == 0 or np.all(returns_array == 0):
#         print("WARNING: Empty or all-zero returns array")
#         return {
#             'total_return': 0.0,
#             'annualized_return': 0.0,
#             'annualized_volatility': 0.0,
#             'sharpe_ratio': 0.0,
#             'max_drawdown': 0.0,
#             'win_rate': 0.0,
#             'profit_factor': 0.0,
#             'number_of_trades': 0
#         }
    
#     # Calculate metrics
#     try:
#         # Debug statistics about returns
#         print(f"DEBUG - Returns statistics: count={len(returns_array)}, " +
#               f"mean={np.mean(returns_array):.6f}, sum={np.sum(returns_array):.6f}")
        
#         # Convert log returns to simple returns
#         simple_returns = np.exp(returns_array) - 1
        
#         # Calculate total return
#         total_return = np.exp(np.sum(returns_array)) - 1
        
#         # Calculate annualized metrics (assuming 252 trading days per year)
#         annualized_return = np.exp(np.mean(returns_array) * 252) - 1
#         annualized_volatility = np.std(returns_array) * np.sqrt(252)
#         sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
#         # Calculate drawdowns
#         cum_returns = np.cumprod(1 + simple_returns)
#         peak = np.maximum.accumulate(cum_returns)
#         drawdowns = (cum_returns / peak) - 1
#         max_drawdown = np.min(drawdowns)
        
#         # Calculate win rate
#         win_rate = np.sum(returns_array > 0) / len(returns_array)
        
#         # Calculate profit factor
#         winning_trades = np.sum(returns_array[returns_array > 0])
#         losing_trades = abs(np.sum(returns_array[returns_array < 0]))
#         profit_factor = winning_trades / losing_trades if losing_trades != 0 else float('inf')

#         if signals is not None:
#             number_of_trades = int((signals != signals.shift(1)).sum())
#         else:
#             number_of_trades = 0
        
#         # Debug the calculated metrics
#         print(f"DEBUG - Calculated metrics: TR={total_return:.6f}, AR={annualized_return:.6f}, " +
#               f"Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.6f}")
        
#         return {
#             'total_return': total_return,
#             'annualized_return': annualized_return,
#             'annualized_volatility': annualized_volatility,
#             'sharpe_ratio': sharpe_ratio,
#             'max_drawdown': max_drawdown,
#             'win_rate': win_rate,
#             'profit_factor': profit_factor,
#             'number_of_trades': 0  # Placeholder, updated later
#         }
        
#     except Exception as e:
#         print(f"Error calculating metrics: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             'total_return': 0.0,
#             'annualized_return': 0.0,
#             'annualized_volatility': 0.0,
#             'sharpe_ratio': 0.0,
#             'max_drawdown': 0.0,
#             'win_rate': 0.0,
#             'profit_factor': 0.0,
#             'number_of_trades': 0
#         }


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

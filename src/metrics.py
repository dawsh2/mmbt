"""
Performance metrics calculation for the backtesting engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional


def calculate_returns(prices: pd.Series, signals: pd.Series) -> pd.Series:
    """
    Calculate strategy returns based on signals and price data.
    
    Args:
        prices: Series of close prices
        signals: Series of trading signals (-1, 0, 1)
        
    Returns:
        Series of strategy returns
    """
    # Ensure we're working with Series
    prices = pd.Series(prices)
    signals = pd.Series(signals)
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    
    # Apply signals (shifted to represent next-bar execution)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    return strategy_returns

def calculate_metrics(strategy_returns: pd.Series, signals: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate performance metrics for a strategy.
    
    Args:
        strategy_returns: Series of strategy returns
        signals: Optional Series of trading signals for trade counting
        
    Returns:
        Dictionary of performance metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Clean inputs
    strategy_returns = pd.Series(strategy_returns).fillna(0)
    
    # Calculate number of trades if signals are provided
    if signals is not None:
        signals = pd.Series(signals).fillna(0)
        # Count trade entries as signal changes from 0 to non-zero or sign changes
        number_of_trades = int((signals != signals.shift(1)).sum())
        metrics['number_of_trades'] = number_of_trades
    else:
        # Default to 0 if no signals provided
        metrics['number_of_trades'] = 0
    
    # Skip remaining calculations if we have no trades
    if metrics['number_of_trades'] == 0:
        metrics['total_return'] = np.nan
        metrics['sharpe_ratio'] = np.nan
        metrics['max_drawdown'] = np.nan
        return metrics
    
    # Calculate total return (cumulative)
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    metrics['total_return'] = float(cumulative_returns.iloc[-1])
    
    # Calculate annualized Sharpe ratio (assuming daily data)
    annual_factor = 252  # Trading days in a year
    sharpe_ratio = np.sqrt(annual_factor) * (strategy_returns.mean() / strategy_returns.std())
    metrics['sharpe_ratio'] = float(sharpe_ratio)
    
    # Calculate maximum drawdown
    rolling_max = (1 + cumulative_returns).cummax()
    drawdowns = (1 + cumulative_returns) / rolling_max - 1
    metrics['max_drawdown'] = float(drawdowns.min())
    
    return metrics


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

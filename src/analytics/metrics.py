"""
Performance Metrics Module for Trading System

This module provides comprehensive performance analysis tools for evaluating
trading strategies, including risk metrics, return metrics, and trade statistics.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict


def calculate_metrics_from_trades(trades):
    """
    Calculate key performance metrics from a list of trades.
    
    Args:
        trades: List of trade tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)
        
    Returns:
        dict: Dictionary of performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'log_return': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'profit_factor': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    # Extract returns
    returns = [trade[5] for trade in trades]
    
    # Basic metrics
    win_count = sum(1 for r in returns if r > 0)
    loss_count = len(returns) - win_count
    win_rate = win_count / len(returns) if len(returns) > 0 else 0
    
    # Return metrics
    total_log_return = sum(returns)
    total_return = (np.exp(total_log_return) - 1) * 100  # Convert to percentage
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if len(returns) > 1 else 0
    
    # Calculate equity curve for drawdown analysis
    equity_curve = [1.0]
    for r in returns:
        equity_curve.append(equity_curve[-1] * np.exp(r))
    
    # Max drawdown calculation
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Calmar ratio (return divided by max drawdown)
    calmar_ratio = total_return / max_dd if max_dd > 0 else float('inf')
    
    # Sharpe ratio (annualized)
    # Assuming trades represent a full year's worth of trading
    sharpe_ratio = (avg_return / std_return) * np.sqrt(252 / len(returns)) if std_return > 0 else 0
    
    # Profit factor (sum of profits divided by sum of losses)
    profits = sum(r for r in returns if r > 0)
    losses = sum(abs(r) for r in returns if r < 0)
    profit_factor = profits / losses if losses > 0 else float('inf')
    
    # Max consecutive wins/losses
    max_consec_wins, max_consec_losses = calculate_consecutive_winloss(returns)
    
    # Sortino ratio (penalizes only downside deviation)
    downside_returns = [r for r in returns if r < 0]
    downside_deviation = np.std(downside_returns) if downside_returns and len(downside_returns) > 1 else 0.0001
    sortino_ratio = (avg_return / downside_deviation) * np.sqrt(252 / len(returns)) if downside_deviation > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'win_count': win_count,
        'loss_count': loss_count,
        'total_return': total_return,
        'total_log_return': total_log_return,
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar_ratio,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consec_wins,
        'max_consecutive_losses': max_consec_losses
    }


def calculate_max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown from an equity curve.
    
    Args:
        equity_curve: List of equity values
        
    Returns:
        float: Maximum drawdown percentage
    """
    max_dd = 0
    peak = equity_curve[0]
    
    for value in equity_curve:
        if value > peak:
            peak = value
        else:
            dd = (peak - value) / peak * 100  # Convert to percentage
            max_dd = max(max_dd, dd)
    
    return max_dd


def calculate_consecutive_winloss(returns):
    """
    Calculate maximum consecutive winning and losing trades.
    
    Args:
        returns: List of trade returns
        
    Returns:
        tuple: (max_consecutive_wins, max_consecutive_losses)
    """
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for r in returns:
        if r > 0:  # Win
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:  # Loss
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    return max_wins, max_losses


def calculate_monthly_returns(trades):
    """
    Calculate monthly returns from trades.
    
    Args:
        trades: List of trade tuples
        
    Returns:
        dict: Monthly returns mapped by YYYY-MM
    """
    if not trades:
        return {}
    
    # Group trades by month
    monthly_trades = defaultdict(list)
    
    for trade in trades:
        exit_time = trade[3]
        log_return = trade[5]
        
        # Convert to datetime if string
        if isinstance(exit_time, str):
            exit_time = pd.to_datetime(exit_time)
        
        # Extract year-month key
        if isinstance(exit_time, datetime):
            month_key = f"{exit_time.year}-{exit_time.month:02d}"
            monthly_trades[month_key].append(log_return)
    
    # Calculate returns for each month
    monthly_returns = {}
    for month, returns in monthly_trades.items():
        compound_return = np.exp(sum(returns)) - 1
        monthly_returns[month] = compound_return * 100  # Convert to percentage
    
    return monthly_returns


def calculate_drawdown_periods(equity_curve, threshold=5.0):
    """
    Identify significant drawdown periods in an equity curve.
    
    Args:
        equity_curve: List of equity values
        threshold: Minimum drawdown percentage to track
        
    Returns:
        list: List of drawdown periods (dictionaries with start/end/max info)
    """
    drawdown_periods = []
    
    peak_idx = 0
    peak_value = equity_curve[0]
    in_drawdown = False
    current_drawdown = None
    
    for i, value in enumerate(equity_curve):
        if value > peak_value:
            # New peak
            peak_idx = i
            peak_value = value
            
            # End any current drawdown
            if in_drawdown and current_drawdown:
                current_drawdown['end_idx'] = i
                current_drawdown['recovery_idx'] = i
                drawdown_periods.append(current_drawdown)
                current_drawdown = None
                in_drawdown = False
        elif value < peak_value:
            # Calculate drawdown percentage
            dd_pct = (peak_value - value) / peak_value * 100
            
            if dd_pct >= threshold and not in_drawdown:
                # Start new drawdown
                in_drawdown = True
                current_drawdown = {
                    'start_idx': peak_idx,
                    'max_dd_idx': i,
                    'max_dd': dd_pct,
                    'peak_value': peak_value,
                    'trough_value': value
                }
            elif in_drawdown and dd_pct > current_drawdown['max_dd']:
                # Update existing drawdown if deeper
                current_drawdown['max_dd_idx'] = i
                current_drawdown['max_dd'] = dd_pct
                current_drawdown['trough_value'] = value
    
    # Handle any unfinished drawdown
    if in_drawdown and current_drawdown:
        current_drawdown['end_idx'] = len(equity_curve) - 1
        current_drawdown['recovery_idx'] = None  # No recovery yet
        drawdown_periods.append(current_drawdown)
    
    return drawdown_periods


def calculate_regime_performance(trades, regime_data):
    """
    Calculate performance metrics by market regime.
    
    Args:
        trades: List of trade tuples
        regime_data: Dictionary mapping timestamps to regime types
        
    Returns:
        dict: Performance metrics by regime
    """
    if not trades or not regime_data:
        return {}
    
    # Group trades by regime
    trades_by_regime = defaultdict(list)
    
    for trade in trades:
        entry_time = trade[0]
        log_return = trade[5]
        
        # Convert to string format if datetime
        if isinstance(entry_time, datetime):
            entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Find the regime at entry time
        regime = regime_data.get(entry_time, 'Unknown')
        trades_by_regime[regime].append(log_return)
    
    # Calculate metrics for each regime
    regime_metrics = {}
    
    for regime, returns in trades_by_regime.items():
        win_count = sum(1 for r in returns if r > 0)
        
        # Create a mini equity curve for this regime
        regime_equity = [1.0]
        for r in returns:
            regime_equity.append(regime_equity[-1] * np.exp(r))
        
        # Calculate max drawdown for this regime
        max_dd = calculate_max_drawdown(regime_equity)
        
        regime_metrics[regime] = {
            'num_trades': len(returns),
            'win_rate': win_count / len(returns) if returns else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'std_return': np.std(returns) if len(returns) > 1 else 0,
            'total_return': (regime_equity[-1] / regime_equity[0] - 1) * 100,  # Percentage
            'max_drawdown': max_dd
        }
    
    return regime_metrics


def analyze_trade_durations(trades):
    """
    Analyze trade durations.
    
    Args:
        trades: List of trade tuples
        
    Returns:
        dict: Statistics on trade durations
    """
    if not trades:
        return {
            'avg_duration': 0,
            'median_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'duration_bins': [],
            'duration_counts': []
        }
    
    # Calculate durations
    durations = []
    for trade in trades:
        if len(trade) >= 4:
            entry_time = trade[0]
            exit_time = trade[3]
            
            # Convert to datetime if they're strings
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            
            if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
                duration = (exit_time - entry_time).total_seconds() / (24 * 3600)  # in days
                durations.append(duration)
    
    if not durations:
        return {
            'avg_duration': 0,
            'median_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'duration_bins': [],
            'duration_counts': []
        }
    
    # Calculate statistics
    avg_duration = np.mean(durations)
    median_duration = np.median(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    # Create histogram data for durations
    if max_duration > 0:
        # Determine appropriate bin edges
        if max_duration <= 1:  # Less than a day
            bins = np.linspace(0, max_duration, min(10, len(durations)))
        elif max_duration <= 7:  # Less than a week
            bins = np.linspace(0, 7, min(8, len(durations)))
        elif max_duration <= 30:  # Less than a month
            bins = np.array([0, 1, 2, 3, 5, 7, 14, 21, 30])
            bins = bins[bins <= max_duration]
        else:
            bins = np.array([0, 1, 7, 14, 30, 60, 90, 180, 365])
            bins = bins[bins <= max_duration * 1.1]
        
        # Calculate histogram
        counts, bin_edges = np.histogram(durations, bins=bins)
        
        # Convert to lists for JSON serialization
        duration_bins = list(bin_edges[:-1])
        duration_counts = list(counts)
    else:
        duration_bins = []
        duration_counts = []
    
    return {
        'avg_duration': avg_duration,
        'median_duration': median_duration,
        'min_duration': min_duration,
        'max_duration': max_duration,
        'duration_bins': duration_bins,
        'duration_counts': duration_counts
    }

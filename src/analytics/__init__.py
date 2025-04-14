#!/usr/bin/env python
# analytics/__init__.py - Analytics module initialization

# Import metrics functions
try:
    from .metrics import (
        process_backtester_results,
        calculate_metrics_from_trades,
        calculate_max_drawdown,
        calculate_consecutive_winloss,
        calculate_monthly_returns,
        calculate_drawdown_periods,
        calculate_regime_performance,
        analyze_trade_durations
    )
except ImportError as e:
    # Fallback if metrics module is not available or missing functions
    print(f"Warning: Could not import from metrics module: {e}")
    
    # Define a minimal version of process_backtester_results if it's not available
    def process_backtester_results(results):
        """Minimal version of process_backtester_results"""
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        
        # If equity curve not provided, calculate it
        if not equity_curve and trades:
            initial_capital = results.get('initial_capital', 10000)
            equity = [initial_capital]
            for trade in trades:
                equity.append(equity[-1] * (1 + trade[5]))
            equity_curve = equity
            
        return trades, equity_curve

# Import visualization classes and functions
try:
    from .visualization import TradeVisualizer
except ImportError as e:
    # Fallback if visualization module is not available
    print(f"Warning: Could not import from visualization module: {e}")

# Define which symbols should be available when using 'from analytics import *'
__all__ = [
    'process_backtester_results',
    
    # Metrics functions
    'calculate_metrics_from_trades',
    'calculate_max_drawdown',
    'calculate_consecutive_winloss', 
    'calculate_monthly_returns',
    'calculate_drawdown_periods',
    'calculate_regime_performance',
    'analyze_trade_durations',
    
    # Visualization classes
    'TradeVisualizer'
]

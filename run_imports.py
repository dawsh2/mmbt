#!/usr/bin/env python
"""
Run script that properly uses your trading system modules.
This script should be placed in your project root directory.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import your actual modules
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler
from src.rules import SMARule, RSIRule, create_composite_rule
from src.optimization import OptimizerManager, OptimizationMethod
from src.strategies import WeightedStrategy
from src.engine import Backtester
from src.config import ConfigManager

def main():
    print("Trading System Optimization Using Actual Modules")
    print("==============================================")
    
    # Create configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    config.set('backtester.market_simulation.fee_bps', 10)
    
    # Path to your actual data file
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(data_dir, 'data', 'data.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    print(f"\nUsing data file: {data_file}")
    
    try:
        # Create a data source that points to your CSV file
        # Use your actual CSVDataSource class with the right parameters
        data_source = CSVDataSource(
            base_path=os.path.dirname(data_file),
            filename_pattern="{symbol}.csv"  # This will use the file name as the symbol
        )
        
        # Create the data handler
        data_handler = DataHandler(data_source, train_fraction=0.8)
        
        # Get symbol from filename
        symbol = os.path.splitext(os.path.basename(data_file))[0]
        print(f"Using symbol: {symbol}")
        
        # Load the data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Use 2 years of data to keep it manageable
        
        # Set proper time format and timezone handling for timestamps
        # This addresses the timezone issue by making timestamps timezone-aware
        import pytz
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)
        
        # Load data with proper timeframe
        data_handler.load_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"  # Start with daily data for simplicity
        )
        
        # Create optimizer
        print("\nInitializing optimizer...")
        optimizer = OptimizerManager(data_handler)
        
        # Register rules to optimize
        print("Registering rules for optimization...")
        
        # SMA rule parameters
        sma_params = {
            "fast_window": [5, 10, 20], 
            "slow_window": [30, 50, 100]
        }
        
        # RSI rule parameters
        rsi_params = {
            "period": [7, 14, 21], 
            "overbought": [70, 75, 80], 
            "oversold": [20, 25, 30]
        }
        
        # Register the rules with the optimizer
        optimizer.register_rule("sma_rule", SMARule, sma_params)
        optimizer.register_rule("rsi_rule", RSIRule, rsi_params)
        
        # Run optimization
        print("\nRunning optimization with grid search...")
        optimized_rules = optimizer.optimize(
            component_type='rule',
            method=OptimizationMethod.GRID_SEARCH,
            metrics='sharpe',
            verbose=True
        )
        
        # Create a composite rule from optimized rules
        print("\nCreating strategy from optimized rules...")
        rule_objects = list(optimized_rules.values())
        
        for name, rule in optimized_rules.items():
            print(f"  Optimized {name}: {rule.__class__.__name__} with parameters:")
            for param, value in rule.params.items():
                print(f"    {param}: {value}")
        
        # Create a weighted strategy with optimized rules
        strategy = WeightedStrategy(
            rules=rule_objects,
            weights=np.array([0.6, 0.4]),  # Weights for SMA and RSI
            buy_threshold=0.3,
            sell_threshold=-0.3,
            name="OptimizedStrategy"
        )
        
        # Run backtest with optimized strategy
        print("\nRunning backtest with optimized strategy...")
        backtester = Backtester(config, data_handler, strategy)
        results = backtester.run()
        
        # Display results
        print("\nBacktest Results:")
        
        # Handle results based on what's available
        if isinstance(results, dict):
            # Extract metrics from results dictionary
            print(f"  Total Return: {results.get('total_percent_return', 0):.2f}%")
            print(f"  Number of Trades: {results.get('num_trades', 0)}")
            
            # Calculate Sharpe ratio if available
            try:
                sharpe = backtester.calculate_sharpe()
                print(f"  Sharpe Ratio: {sharpe:.4f}")
            except Exception as e:
                print(f"  Error calculating Sharpe ratio: {e}")
                
            # Display equity curve if available
            if 'portfolio_history' in results:
                print_equity_curve(results['portfolio_history'])
        else:
            # If results are not in expected format
            print("  Results not in expected format. Raw results:")
            print(results)
        
        print("\nOptimization and backtest complete!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

def print_equity_curve(portfolio_history):
    """Print equity curve data."""
    if not portfolio_history or len(portfolio_history) == 0:
        print("No portfolio history data available.")
        return
    
    # Extract equity values based on structure
    try:
        if isinstance(portfolio_history[0], dict):
            equity = [p.get('equity', 0) for p in portfolio_history]
        else:
            equity = portfolio_history
    except:
        print("Could not extract equity values from portfolio history.")
        return
    
    # Calculate statistics
    initial = equity[0] if equity else 0
    final = equity[-1] if equity else 0
    change = final - initial
    percent = (change / initial * 100) if initial else 0
    
    print("\nEquity Curve Summary:")
    print(f"  Starting: ${initial:,.2f}")
    print(f"  Ending:   ${final:,.2f}")
    print(f"  Change:   ${change:,.2f} ({percent:.2f}%)")
    
    # Print first and last points
    print("\nEquity Curve Data (first 5 and last 5 points):")
    print("Point | Equity Value")
    print("-" * 25)
    
    # Print first 5 points
    for i in range(min(5, len(equity))):
        print(f"{i+1:5d} | ${equity[i]:,.2f}")
    
    if len(equity) > 10:
        print("...")
    
    # Print last 5 points
    for i in range(max(5, len(equity)-5), len(equity)):
        print(f"{i+1:5d} | ${equity[i]:,.2f}")

if __name__ == "__main__":
    main()

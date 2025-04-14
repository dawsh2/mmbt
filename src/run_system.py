#!/usr/bin/env python
"""
This is a wrapper script to run optimization with your actual data.
This script should be placed in the root directory of your project (~/mmbt/),
not in the src/ directory.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

# Now we can import modules from src
from data.data_sources import CSVDataSource
from data.data_handler import DataHandler
from rules import SMARule, RSIRule, create_composite_rule
from optimization import OptimizerManager, OptimizationMethod
from strategies import WeightedStrategy
from engine import Backtester
from config import ConfigManager

def main():
    print("Trading System Optimization with Real Data")
    print("=========================================")
    
    # Path to your actual data file
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    print(f"\nUsing data file: {data_file}")
    
    # Create configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    config.set('backtester.market_simulation.fee_bps', 10)
    
    # Set up custom data source and handler
    print("\nSetting up data handler...")
    
    try:
        # Read the CSV data
        df = pd.read_csv(data_file)
        print(f"Loaded data with {len(df)} rows")
        
        # Check and display basic information about the data
        print("\nData sample:")
        print(df.head())
        
        print("\nData columns:")
        print(df.columns.tolist())
        
        # Prepare data handler
        # Note: Adjust these parameters based on your actual data structure
        # This is a simplified example - you may need to adapt to your data format
        
        # Convert to expected OHLCV format if needed
        # The code below assumes your CSV has standard columns
        if 'Date' in df.columns and not 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        
        # Ensure required columns are present
        expected_cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            print("Attempting to adapt data format...")
            
            # Simple column mapping for common variants
            column_map = {
                'date': 'timestamp',
                'time': 'timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }
            
            # Try to map columns
            for old_col, new_col in column_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Check again for missing columns
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Cannot proceed without required columns: {missing_cols}")
                return
        
        # Create a temporary CSV in the expected format
        temp_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_data')
        os.makedirs(temp_data_dir, exist_ok=True)
        temp_csv_path = os.path.join(temp_data_dir, 'formatted_data.csv')
        
        # Save the formatted data
        df.to_csv(temp_csv_path, index=False)
        
        # Create data source and handler
        data_source = CSVDataSource(temp_data_dir)
        data_handler = DataHandler(data_source, train_fraction=0.8)
        
        # Extract symbol name from filename if not provided
        symbol = os.path.splitext(os.path.basename(data_file))[0]
        
        # Load data
        data_handler.load_data(
            symbols=[symbol],
            # If your data has timestamps, you can use them for filtering
            # Otherwise, use None to load all data
            start_date=None,
            end_date=None,
            timeframe="1d"  # Adjust based on your data timeframe
        )
        
        # Create optimizer
        print("\nInitializing optimizer...")
        optimizer = OptimizerManager(data_handler)
        
        # Register rules to optimize
        print("Registering rules for optimization...")
        optimizer.register_rule("sma_rule", SMARule, {
            "fast_window": [5, 10, 20], 
            "slow_window": [30, 50, 100]
        })
        
        optimizer.register_rule("rsi_rule", RSIRule, {
            "period": [7, 14, 21], 
            "overbought": [70, 75, 80], 
            "oversold": [20, 25, 30]
        })
        
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
            weights=np.array([0.6, 0.4]),  # Arbitrary weights for demonstration
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
        print(f"  Total Return: {results.get('total_percent_return', 0):.2f}%")
        print(f"  Number of Trades: {results.get('num_trades', 0)}")
        
        # Additional metrics if available
        if 'win_rate' in results:
            print(f"  Win Rate: {results.get('win_rate', 0):.2f}%")
        
        # Try to calculate Sharpe ratio
        try:
            sharpe = backtester.calculate_sharpe()
            print(f"  Sharpe Ratio: {sharpe:.4f}")
        except Exception as e:
            print(f"  Error calculating Sharpe ratio: {e}")
        
        # Display equity curve data
        print_equity_curve(results.get('portfolio_history', []))
        
        print("\nOptimization and backtest complete!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

def print_equity_curve(portfolio_history):
    """Print equity curve data in text format."""
    if not portfolio_history:
        print("No portfolio history to plot.")
        return
    
    # Extract equity values - adjust based on your actual portfolio history structure
    try:
        if isinstance(portfolio_history[0], dict):
            equity = [p.get('equity', 0) for p in portfolio_history]
        else:
            equity = portfolio_history  # Fallback if structure is different
    except:
        print("Could not extract equity values from portfolio history.")
        return
    
    print("\nEquity Curve Data (first 5 and last 5 points):")
    print("Bar Number | Equity Value")
    print("-" * 30)
    
    # Print first 5 points
    for i in range(min(5, len(equity))):
        print(f"{i+1:10d} | ${equity[i]:,.2f}")
    
    if len(equity) > 10:
        print("...")
    
    # Print last 5 points
    for i in range(max(5, len(equity)-5), len(equity)):
        print(f"{i+1:10d} | ${equity[i]:,.2f}")
    
    # Calculate stats
    initial = equity[0] if equity else 0
    final = equity[-1] if equity else 0
    change = final - initial
    percent = (change / initial * 100) if initial else 0
    
    print("\nEquity Summary:")
    print(f"Starting: ${initial:,.2f}")
    print(f"Ending:   ${final:,.2f}")
    print(f"Change:   ${change:,.2f} ({percent:.2f}%)")

if __name__ == "__main__":
    main()

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

import sys
import os

# Remove any path that might include the custom logging module
custom_logging_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'logging')
sys.path = [p for p in sys.path if custom_logging_path not in p]

# Then add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import your actual modules
from data.data_sources import CSVDataSource
from data.data_handler import DataHandler
from rules import SMARule, RSIRule, create_composite_rule
from optimization import OptimizerManager, OptimizationMethod
from strategies import WeightedStrategy
from engine import Backtester
from config import ConfigManager

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
        # Setup a custom data source for your CSV
        class SingleFileCSVSource(CSVDataSource):
            """Custom data source that loads a single CSV file."""
            
            def __init__(self, filepath):
                self.filepath = filepath
                self.data = None
                self._load_data()
                
            def _load_data(self):
                """Load data from CSV."""
                self.data = pd.read_csv(self.filepath)
                
                # Ensure timestamp column is properly formatted
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                
                print(f"Loaded {len(self.data)} rows from {self.filepath}")
                
            def get_data(self, symbol, start_date=None, end_date=None, timeframe=None):
                """Get data for the specified symbol and date range."""
                # Filter by date if requested
                if start_date or end_date:
                    df = self.data.copy()
                    
                    if start_date:
                        df = df[df['timestamp'] >= start_date]
                    
                    if end_date:
                        df = df[df['timestamp'] <= end_date]
                    
                    return df
                
                return self.data
                
            def get_symbols(self):
                """Get available symbols."""
                # Use filename as symbol
                return [os.path.splitext(os.path.basename(self.filepath))[0]]
        
        # Create the data source
        data_source = SingleFileCSVSource(data_file)
        
        # Create the data handler
        data_handler = DataHandler(data_source, train_fraction=0.8)
        
        # Get the symbol
        symbols = data_source.get_symbols()
        if not symbols:
            symbols = ['data']  # Default if can't determine from filename
        
        print(f"Using symbol: {symbols[0]}")
        
        # Load the data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # Use up to 5 years of data
        
        data_handler.load_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
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

#!/usr/bin/env python
"""
This script uses a different import approach that should work with your codebase.
Place this script in your project root directory (~/mmbt/).
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to Python path to enable imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import strategy components directly - avoiding problematic imports
# This approach bypasses the rules and signals modules that have relative imports
from config.config_manager import ConfigManager

def main():
    print("Trading System with Direct Data Analysis")
    print("=======================================")
    
    # Path to your actual data file
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    print(f"\nUsing data file: {data_file}")
    
    # Create configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    
    # Load and analyze the data
    try:
        # Read the CSV data
        df = pd.read_csv(data_file)
        print(f"Loaded data with {len(df)} rows")
        
        # Display basic information about the data
        print("\nData sample:")
        print(df.head())
        
        print("\nData columns:")
        print(df.columns.tolist())
        
        # Basic data analysis
        print("\nData Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(df[numeric_cols].describe())
        
        # Perform a simple moving average crossover analysis directly without using the rules module
        print("\nPerforming Simple Moving Average Crossover Analysis:")
        # Identify potential price column
        price_col = None
        for col_name in ['Close', 'close', 'Price', 'price']:
            if col_name in df.columns:
                price_col = col_name
                break
        
        if not price_col:
            print("Could not identify price column in data")
            return
        
        # Calculate different moving averages
        fast_windows = [5, 10, 15]
        slow_windows = [20, 30, 50]
        
        results = []
        
        for fast in fast_windows:
            for slow in slow_windows:
                if fast >= slow:
                    continue
                
                # Calculate moving averages
                df[f'MA_{fast}'] = df[price_col].rolling(window=fast).mean()
                df[f'MA_{slow}'] = df[price_col].rolling(window=slow).mean()
                
                # Identify crossovers
                df['signal'] = 0
                df.loc[df[f'MA_{fast}'] > df[f'MA_{slow}'], 'signal'] = 1  # Buy signal
                df.loc[df[f'MA_{fast}'] < df[f'MA_{slow}'], 'signal'] = -1  # Sell signal
                
                # Calculate returns based on signals (simple approach)
                df['returns'] = df[price_col].pct_change()
                df['strategy_returns'] = df['signal'].shift(1) * df['returns']
                
                # Calculate performance metrics
                total_return = (df['strategy_returns'].dropna() + 1).prod() - 1
                sharpe = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
                
                results.append({
                    'fast_window': fast,
                    'slow_window': slow,
                    'total_return': total_return * 100,  # Convert to percentage
                    'sharpe_ratio': sharpe
                })
                
                print(f"  SMA({fast}, {slow}): Return: {total_return*100:.2f}%, Sharpe: {sharpe:.4f}")
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['sharpe_ratio'])
        print("\nBest SMA Parameters:")
        print(f"  Fast Window: {best_result['fast_window']}")
        print(f"  Slow Window: {best_result['slow_window']}")
        print(f"  Return: {best_result['total_return']:.2f}%")
        print(f"  Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
        
        # Perform simple RSI analysis similarly
        print("\nPerforming RSI Analysis:")
        # Calculate RSI for different periods
        rsi_periods = [7, 14, 21]
        rsi_results = []
        
        for period in rsi_periods:
            # Calculate RSI (simplified)
            delta = df[price_col].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Generate signals based on RSI
            df['rsi_signal'] = 0
            df.loc[df[f'RSI_{period}'] < 30, 'rsi_signal'] = 1  # Buy when oversold
            df.loc[df[f'RSI_{period}'] > 70, 'rsi_signal'] = -1  # Sell when overbought
            
            # Calculate returns
            df['rsi_strategy_returns'] = df['rsi_signal'].shift(1) * df['returns']
            
            # Calculate performance metrics
            rsi_total_return = (df['rsi_strategy_returns'].dropna() + 1).prod() - 1
            rsi_sharpe = np.sqrt(252) * df['rsi_strategy_returns'].mean() / df['rsi_strategy_returns'].std()
            
            rsi_results.append({
                'period': period,
                'total_return': rsi_total_return * 100,
                'sharpe_ratio': rsi_sharpe
            })
            
            print(f"  RSI({period}): Return: {rsi_total_return*100:.2f}%, Sharpe: {rsi_sharpe:.4f}")
        
        # Find best RSI parameters
        best_rsi = max(rsi_results, key=lambda x: x['sharpe_ratio'])
        print("\nBest RSI Parameters:")
        print(f"  Period: {best_rsi['period']}")
        print(f"  Return: {best_rsi['total_return']:.2f}%")
        print(f"  Sharpe Ratio: {best_rsi['sharpe_ratio']:.4f}")
        
        # Create a combined strategy with best SMA and RSI parameters
        print("\nCombined Strategy Backtest:")
        
        # Apply the best parameters
        fast = best_result['fast_window']
        slow = best_result['slow_window']
        rsi_period = best_rsi['period']
        
        # Calculate indicators with best parameters
        df[f'MA_{fast}'] = df[price_col].rolling(window=fast).mean()
        df[f'MA_{slow}'] = df[price_col].rolling(window=slow).mean()
        
        delta = df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # Generate combined signals
        df['sma_signal'] = 0
        df.loc[df[f'MA_{fast}'] > df[f'MA_{slow}'], 'sma_signal'] = 1
        df.loc[df[f'MA_{fast}'] < df[f'MA_{slow}'], 'sma_signal'] = -1
        
        df['rsi_signal'] = 0
        df.loc[df[f'RSI_{rsi_period}'] < 30, 'rsi_signal'] = 1
        df.loc[df[f'RSI_{rsi_period}'] > 70, 'rsi_signal'] = -1
        
        # Combined signal (simple majority vote)
        df['combined_signal'] = 0
        df.loc[(df['sma_signal'] + df['rsi_signal']) > 0, 'combined_signal'] = 1
        df.loc[(df['sma_signal'] + df['rsi_signal']) < 0, 'combined_signal'] = -1
        
        # Calculate combined strategy returns
        df['combined_returns'] = df['combined_signal'].shift(1) * df['returns']
        
        # Calculate performance metrics
        combined_total_return = (df['combined_returns'].dropna() + 1).prod() - 1
        combined_sharpe = np.sqrt(252) * df['combined_returns'].mean() / df['combined_returns'].std()
        
        # Calculate equity curve
        initial_capital = 100000
        df['equity'] = initial_capital * (1 + df['combined_returns'].fillna(0)).cumprod()
        
        print("\nCombined Strategy Results:")
        print(f"  Total Return: {combined_total_return*100:.2f}%")
        print(f"  Sharpe Ratio: {combined_sharpe:.4f}")
        
        # Calculate additional metrics
        # Drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min() * 100
        
        # Win rate (approximate)
        win_rate = (df['combined_returns'] > 0).sum() / (df['combined_returns'] != 0).sum() * 100
        
        print(f"  Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"  Win Rate: {win_rate:.2f}%")
        
        # Display equity curve
        print_equity_curve(df['equity'].tolist())
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def print_equity_curve(equity_curve):
    """Print equity curve data in text format."""
    if not equity_curve:
        print("No equity curve data available.")
        return
    
    # Remove NaN values if any
    equity_curve = [e for e in equity_curve if isinstance(e, (int, float)) and not np.isnan(e)]
    
    print("\nEquity Curve Data (first 5 and last 5 points):")
    print("Day Number | Equity Value")
    print("-" * 30)
    
    # Print first 5 points
    for i in range(min(5, len(equity_curve))):
        print(f"{i+1:10d} | ${equity_curve[i]:,.2f}")
    
    if len(equity_curve) > 10:
        print("...")
    
    # Print last 5 points
    for i in range(max(5, len(equity_curve)-5), len(equity_curve)):
        print(f"{i+1:10d} | ${equity_curve[i]:,.2f}")
    
    # Calculate stats
    initial = equity_curve[0] if equity_curve else 0
    final = equity_curve[-1] if equity_curve else 0
    change = final - initial
    percent = (change / initial * 100) if initial else 0
    
    print("\nEquity Summary:")
    print(f"Starting: ${initial:,.2f}")
    print(f"Ending:   ${final:,.2f}")
    print(f"Change:   ${change:,.2f} ({percent:.2f}%)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Fixed version of train/test splitting script.
Place this script in your project root directory (~/mmbt/).
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def main():
    print("Trading System with Train/Test Split")
    print("===================================")
    
    # Path to your actual data file
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    print(f"\nUsing data file: {data_file}")
    
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
        
        # Make sure timestamp column is properly formatted
        if 'timestamp' in df.columns:
            if isinstance(df['timestamp'][0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort data by timestamp to ensure chronological order
            df = df.sort_values('timestamp')
            print(f"\nData timeframe: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Identify price column
        price_col = None
        for col_name in ['Close', 'close', 'Price', 'price']:
            if col_name in df.columns:
                price_col = col_name
                break
        
        if not price_col:
            print("Could not identify price column in data")
            return
        
        # Split data into training and testing sets (80% train, 20% test)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        print(f"\nSplit data into:")
        print(f"  Training set: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
        print(f"  Testing set:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
        
        # Step 1: Optimize parameters on training data
        print("\n=============================================")
        print("OPTIMIZATION PHASE (Using Training Data Only)")
        print("=============================================")
        
        # Moving Average Crossover optimization on training data
        print("\nPerforming Simple Moving Average Crossover Analysis (Training Data):")
        fast_windows = [5, 10, 15]
        slow_windows = [20, 30, 50]
        
        sma_results = []
        
        for fast in fast_windows:
            for slow in slow_windows:
                if fast >= slow:
                    continue
                
                # Calculate moving averages
                train_df[f'MA_{fast}'] = train_df[price_col].rolling(window=fast).mean()
                train_df[f'MA_{slow}'] = train_df[price_col].rolling(window=slow).mean()
                
                # Identify crossovers
                train_df[f'sma_signal_{fast}_{slow}'] = 0
                train_df.loc[train_df[f'MA_{fast}'] > train_df[f'MA_{slow}'], f'sma_signal_{fast}_{slow}'] = 1  # Buy signal
                train_df.loc[train_df[f'MA_{fast}'] < train_df[f'MA_{slow}'], f'sma_signal_{fast}_{slow}'] = -1  # Sell signal
                
                # Calculate returns based on signals
                train_df['returns'] = train_df[price_col].pct_change()
                train_df[f'sma_returns_{fast}_{slow}'] = train_df[f'sma_signal_{fast}_{slow}'].shift(1) * train_df['returns']
                
                # Calculate performance metrics
                # Skip NaN values at beginning due to rolling windows
                valid_returns = train_df[f'sma_returns_{fast}_{slow}'].dropna()
                if len(valid_returns) > 0:
                    total_return = (valid_returns + 1).prod() - 1
                    sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
                else:
                    total_return = 0
                    sharpe = 0
                
                sma_results.append({
                    'fast_window': fast,
                    'slow_window': slow,
                    'total_return': total_return * 100,  # Convert to percentage
                    'sharpe_ratio': sharpe
                })
                
                print(f"  SMA({fast}, {slow}): Return: {total_return*100:.2f}%, Sharpe: {sharpe:.4f}")
        
        # Find best SMA parameters
        if sma_results:
            best_sma = max(sma_results, key=lambda x: x['sharpe_ratio'] if not np.isnan(x['sharpe_ratio']) else -float('inf'))
            print("\nBest SMA Parameters (Training Data):")
            print(f"  Fast Window: {best_sma['fast_window']}")
            print(f"  Slow Window: {best_sma['slow_window']}")
            print(f"  Return: {best_sma['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {best_sma['sharpe_ratio']:.4f}")
            best_fast = best_sma['fast_window']
            best_slow = best_sma['slow_window']
        else:
            print("No valid SMA results found")
            best_fast = 5
            best_slow = 30
        
        # Store the best SMA signal column name
        best_sma_signal = f'sma_signal_{best_fast}_{best_slow}'
        
        # RSI optimization on training data
        print("\nPerforming RSI Analysis (Training Data):")
        rsi_periods = [7, 14, 21]
        rsi_results = []
        
        for period in rsi_periods:
            # Calculate RSI
            delta = train_df[price_col].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Avoid division by zero
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            train_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Generate signals based on RSI
            train_df[f'rsi_signal_{period}'] = 0
            train_df.loc[train_df[f'RSI_{period}'] < 30, f'rsi_signal_{period}'] = 1  # Buy when oversold
            train_df.loc[train_df[f'RSI_{period}'] > 70, f'rsi_signal_{period}'] = -1  # Sell when overbought
            
            # Calculate returns
            train_df[f'rsi_returns_{period}'] = train_df[f'rsi_signal_{period}'].shift(1) * train_df['returns']
            
            # Calculate performance metrics
            valid_returns = train_df[f'rsi_returns_{period}'].dropna()
            if len(valid_returns) > 0:
                rsi_total_return = (valid_returns + 1).prod() - 1
                rsi_sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
            else:
                rsi_total_return = 0
                rsi_sharpe = 0
            
            rsi_results.append({
                'period': period,
                'total_return': rsi_total_return * 100,
                'sharpe_ratio': rsi_sharpe
            })
            
            print(f"  RSI({period}): Return: {rsi_total_return*100:.2f}%, Sharpe: {rsi_sharpe:.4f}")
        
        # Find best RSI parameters
        if rsi_results:
            best_rsi = max(rsi_results, key=lambda x: x['sharpe_ratio'] if not np.isnan(x['sharpe_ratio']) else -float('inf'))
            print("\nBest RSI Parameters (Training Data):")
            print(f"  Period: {best_rsi['period']}")
            print(f"  Return: {best_rsi['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {best_rsi['sharpe_ratio']:.4f}")
            best_rsi_period = best_rsi['period']
        else:
            print("No valid RSI results found")
            best_rsi_period = 14
        
        # Store the best RSI signal column name
        best_rsi_signal = f'rsi_signal_{best_rsi_period}'
        
        # Create combined strategy signal on training data
        train_df['combined_signal'] = 0
        train_df.loc[(train_df[best_sma_signal] + train_df[best_rsi_signal]) > 0, 'combined_signal'] = 1
        train_df.loc[(train_df[best_sma_signal] + train_df[best_rsi_signal]) < 0, 'combined_signal'] = -1
        train_df['combined_returns'] = train_df['combined_signal'].shift(1) * train_df['returns']
        
        # Calculate performance metrics for combined strategy on training data
        valid_returns = train_df['combined_returns'].dropna()
        if len(valid_returns) > 0:
            combined_return = (valid_returns + 1).prod() - 1
            combined_sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
            
            print("\nCombined Strategy (Training Data):")
            print(f"  Return: {combined_return*100:.2f}%")
            print(f"  Sharpe Ratio: {combined_sharpe:.4f}")
        else:
            print("\nNo valid returns for combined strategy on training data")
        
        # Step 2: Test the optimized parameters on the test data
        print("\n=================================================")
        print("TESTING PHASE (Using Testing Data with Best Params)")
        print("=================================================")
        
        # Apply best SMA parameters to test data
        print(f"\nApplying Best Parameters to Test Data:")
        print(f"  SMA Fast Window: {best_fast}")
        print(f"  SMA Slow Window: {best_slow}")
        print(f"  RSI Period: {best_rsi_period}")
        
        # Calculate indicators with best parameters on test data
        test_df[f'MA_{best_fast}'] = test_df[price_col].rolling(window=best_fast).mean()
        test_df[f'MA_{best_slow}'] = test_df[price_col].rolling(window=best_slow).mean()
        
        # Calculate RSI on test data
        delta = test_df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=best_rsi_period).mean()
        avg_loss = loss.rolling(window=best_rsi_period).mean()
        
        # Avoid division by zero
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        test_df[f'RSI_{best_rsi_period}'] = 100 - (100 / (1 + rs))
        
        # Generate signals on test data
        test_df[best_sma_signal] = 0
        test_df.loc[test_df[f'MA_{best_fast}'] > test_df[f'MA_{best_slow}'], best_sma_signal] = 1
        test_df.loc[test_df[f'MA_{best_fast}'] < test_df[f'MA_{best_slow}'], best_sma_signal] = -1
        
        test_df[best_rsi_signal] = 0
        test_df.loc[test_df[f'RSI_{best_rsi_period}'] < 30, best_rsi_signal] = 1
        test_df.loc[test_df[f'RSI_{best_rsi_period}'] > 70, best_rsi_signal] = -1
        
        # Combined signal (simple majority vote)
        test_df['combined_signal'] = 0
        test_df.loc[(test_df[best_sma_signal] + test_df[best_rsi_signal]) > 0, 'combined_signal'] = 1
        test_df.loc[(test_df[best_sma_signal] + test_df[best_rsi_signal]) < 0, 'combined_signal'] = -1
        
        # Calculate returns for each strategy on test data
        test_df['returns'] = test_df[price_col].pct_change()
        test_df[f'sma_returns_{best_fast}_{best_slow}'] = test_df[best_sma_signal].shift(1) * test_df['returns']
        test_df[f'rsi_returns_{best_rsi_period}'] = test_df[best_rsi_signal].shift(1) * test_df['returns']
        test_df['combined_returns'] = test_df['combined_signal'].shift(1) * test_df['returns']
        
        # Calculate performance metrics on test data
        strategies = {
            'SMA Strategy': f'sma_returns_{best_fast}_{best_slow}',
            'RSI Strategy': f'rsi_returns_{best_rsi_period}',
            'Combined Strategy': 'combined_returns'
        }
        
        print("\nOut-of-Sample Strategy Performance (Test Data):")
        print("------------------------------------------------")
        
        for strategy_name, returns_col in strategies.items():
            valid_returns = test_df[returns_col].dropna()
            
            if len(valid_returns) > 0:
                # Calculate metrics
                total_return = (valid_returns + 1).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(valid_returns)) - 1
                sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
                
                # Calculate drawdown
                equity_curve = (1 + valid_returns).cumprod()
                running_max = equity_curve.cummax()
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                # Win rate
                win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                
                print(f"\n{strategy_name}:")
                print(f"  Total Return: {total_return*100:.2f}%")
                print(f"  Annualized Return: {annual_return*100:.2f}%")
                print(f"  Sharpe Ratio: {sharpe:.4f}")
                print(f"  Maximum Drawdown: {max_drawdown:.2f}%")
                print(f"  Win Rate: {win_rate:.2f}%")
                
                # Generate equity curve for the combined strategy
                if returns_col == 'combined_returns':
                    # Calculate equity curve
                    initial_capital = 100000
                    test_df['equity'] = initial_capital * (1 + test_df[returns_col].fillna(0)).cumprod()
                    print_equity_curve(test_df['equity'].tolist())
                    
                    # Calculate trade statistics
                    signal_changes = test_df['combined_signal'].diff().abs()
                    num_trades = signal_changes[signal_changes > 0].sum() / 2  # Divide by 2 since each trade involves entry and exit
                    print(f"  Approximate Number of Trades: {int(num_trades)}")
                    
                    # Calculate average trade duration
                    # Get indices where signal changes (excluding 0 to non-zero)
                    signal_change_indices = test_df.index[
                        (test_df['combined_signal'] != 0) & 
                        (test_df['combined_signal'].shift(1) != test_df['combined_signal'])
                    ].tolist()
                    
                    if len(signal_change_indices) > 1 and 'timestamp' in test_df.columns:
                        try:
                            # Calculate durations between signals
                            durations = []
                            for i in range(0, len(signal_change_indices) - 1, 2):
                                if i + 1 < len(signal_change_indices):
                                    entry_time = test_df.iloc[signal_change_indices[i]]['timestamp']
                                    exit_time = test_df.iloc[signal_change_indices[i+1]]['timestamp']
                                    if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                                        duration = exit_time - entry_time
                                        durations.append(duration)
                            
                            if durations:
                                avg_duration = sum(durations, timedelta(0)) / len(durations)
                                print(f"  Average Trade Duration: {avg_duration}")
                        except Exception as e:
                            print(f"  Could not calculate average trade duration: {e}")
            else:
                print(f"\n{strategy_name}: No valid returns calculated")
        
        # Compare training vs testing performance
        print("\n=========================================================")
        print("COMPARISON: Training vs Testing Performance (Combined Strategy)")
        print("=========================================================")
        
        # Calculate metrics on training data
        train_returns = train_df['combined_returns'].dropna()
        if len(train_returns) > 0:
            train_total_return = (train_returns + 1).prod() - 1
            train_sharpe = np.sqrt(252) * train_returns.mean() / train_returns.std() if train_returns.std() != 0 else 0
        else:
            train_total_return = 0
            train_sharpe = 0
        
        # Calculate metrics on test data
        test_returns = test_df['combined_returns'].dropna()
        if len(test_returns) > 0:
            test_total_return = (test_returns + 1).prod() - 1
            test_sharpe = np.sqrt(252) * test_returns.mean() / test_returns.std() if test_returns.std() != 0 else 0
        else:
            test_total_return = 0
            test_sharpe = 0
        
        print("\nCombined Strategy Performance Comparison:")
        print("------------------------------------------")
        print(f"  Training Return: {train_total_return*100:.2f}%")
        print(f"  Testing Return:  {test_total_return*100:.2f}%")
        print(f"  Training Sharpe: {train_sharpe:.4f}")
        print(f"  Testing Sharpe:  {test_sharpe:.4f}")
        
        # Determine if there's evidence of overfitting
        return_diff = abs(train_total_return - test_total_return)
        sharpe_diff = abs(train_sharpe - test_sharpe)
        
        print("\nOverfitting Analysis:")
        if return_diff > 0.1:  # 10% difference
            print("  Warning: Significant difference between training and testing returns (potential overfitting)")
        else:
            print("  Returns are consistent between training and testing periods")
            
        if sharpe_diff > 0.5:
            print("  Warning: Significant difference between training and testing Sharpe ratios (potential overfitting)")
        else:
            print("  Sharpe ratios are consistent between training and testing periods")
        
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
    percent = (change / initial * 100) if initial > 0 else 0
    
    print("\nEquity Summary:")
    print(f"Starting: ${initial:,.2f}")
    print(f"Ending:   ${final:,.2f}")
    print(f"Change:   ${change:,.2f} ({percent:.2f}%)")

if __name__ == "__main__":
    main()

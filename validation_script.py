"""
Backtesting Validation Script

This script validates the backtesting engine by:
1. Generating synthetic price data with test_data_generator.py
2. Running a simple ThresholdRule through the backtester
3. Comparing actual vs expected results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from test_data_generator import generate_test_data
from backtester import Backtester, BarEvent
from signals import Signal, SignalType
from threshold_rule import ThresholdRule  # Import our custom threshold rule

# Parameters for our threshold rule
UPPER_THRESHOLD = 110.0
LOWER_THRESHOLD = 90.0

def plot_backtest_results(df, trades, title="Backtest Results"):
    """Plot the price data with entry/exit points."""
    plt.figure(figsize=(15, 8))
    
    # Plot price data
    plt.plot(df['timestamp'], df['Close'], label='Close Price')
    
    # Add threshold lines
    plt.axhline(y=UPPER_THRESHOLD, color='green', linestyle='--', 
                label=f'Upper Threshold ({UPPER_THRESHOLD})')
    plt.axhline(y=LOWER_THRESHOLD, color='red', linestyle='--', 
                label=f'Lower Threshold ({LOWER_THRESHOLD})')
    
    # Plot entry and exit points
    entries_long = [(t[0], t[2]) for t in trades if t[1] == 'long']
    exits_long = [(t[3], t[4]) for t in trades if t[1] == 'long']
    
    entries_short = [(t[0], t[2]) for t in trades if t[1] == 'short']
    exits_short = [(t[3], t[4]) for t in trades if t[1] == 'short']
    
    if entries_long:
        entry_times, entry_prices = zip(*entries_long)
        plt.scatter(entry_times, entry_prices, color='green', marker='^', s=100, label='Long Entry')
    
    if exits_long:
        exit_times, exit_prices = zip(*exits_long)
        plt.scatter(exit_times, exit_prices, color='blue', marker='v', s=100, label='Long Exit')
    
    if entries_short:
        entry_times, entry_prices = zip(*entries_short)
        plt.scatter(entry_times, entry_prices, color='red', marker='v', s=100, label='Short Entry')
    
    if exits_short:
        exit_times, exit_prices = zip(*exits_short)
        plt.scatter(exit_times, exit_prices, color='orange', marker='^', s=100, label='Short Exit')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('backtest_validation_results.png')
    plt.close()


def calculate_expected_results(df):
    """Calculate expected signals and returns based on threshold logic without using Signal objects."""
    current_position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = None
    entry_time = None
    trades = []
    
    # Constants for signal values (to avoid using SignalType enum)
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    # Identify signals and calculate expected trades
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        close = row['Close']
        
        # Determine signal based on price thresholds
        if close > UPPER_THRESHOLD:
            signal_value = BUY
        elif close < LOWER_THRESHOLD:
            signal_value = SELL
        else:
            signal_value = NEUTRAL
        
        # Process the signal to generate trades
        if current_position == 0:  # Not in a position
            if signal_value == BUY:
                # Enter long position
                current_position = 1
                entry_price = close
                entry_time = timestamp
            elif signal_value == SELL:
                # Enter short position
                current_position = -1
                entry_price = close
                entry_time = timestamp
        elif current_position == 1:  # In long position
            if signal_value == SELL or signal_value == NEUTRAL:
                # Exit long position
                log_return = np.log(close / entry_price)
                trades.append((
                    entry_time,
                    'long',
                    entry_price,
                    timestamp,
                    close,
                    log_return
                ))
                current_position = 0
                entry_price = None
                entry_time = None
                
                # If sell signal, enter short position
                if signal_value == SELL:
                    current_position = -1
                    entry_price = close
                    entry_time = timestamp
        elif current_position == -1:  # In short position
            if signal_value == BUY or signal_value == NEUTRAL:
                # Exit short position
                log_return = np.log(entry_price / close)
                trades.append((
                    entry_time,
                    'short',
                    entry_price,
                    timestamp,
                    close,
                    log_return
                ))
                current_position = 0
                entry_price = None
                entry_time = None
                
                # If buy signal, enter long position
                if signal_value == BUY:
                    current_position = 1
                    entry_price = close
                    entry_time = timestamp
    
    # Calculate expected performance metrics
    if trades:
        total_log_return = sum(trade[5] for trade in trades)
        total_return = (np.exp(total_log_return) - 1) * 100  # Convert to percentage
        avg_log_return = total_log_return / len(trades)
        win_count = sum(1 for trade in trades if trade[5] > 0)
        win_rate = win_count / len(trades) if trades else 0
    else:
        total_log_return = 0
        total_return = 0
        avg_log_return = 0
        win_rate = 0
    
    expected_results = {
        'total_log_return': total_log_return,
        'total_percent_return': total_return,
        'average_log_return': avg_log_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }
    
    return expected_results    

class SimpleStrategy:
    """Simple strategy that uses a single rule"""
    def __init__(self, rule):
        self.rule = rule
        
    def on_bar(self, event):
        return self.rule.on_bar(event.bar)
        
    def reset(self):
        self.rule.reset()

class SimpleDataHandler:
    """Simple data handler for testing"""
    def __init__(self, data_df):
        self.data = data_df
        self.current_train_index = 0
        self.current_test_index = 0
        
    def get_next_train_bar(self):
        if self.current_train_index < len(self.data):
            bar = self.data.iloc[self.current_train_index].to_dict()
            self.current_train_index += 1
            return bar
        return None
        
    def get_next_test_bar(self):
        if self.current_test_index < len(self.data):
            bar = self.data.iloc[self.current_test_index].to_dict()
            self.current_test_index += 1
            return bar
        return None
        
    def reset_train(self):
        self.current_train_index = 0
        
    def reset_test(self):
        self.current_test_index = 0

def run_backtest_validation():
    """Run the backtest validation"""
    print("="*80)
    print("BACKTEST VALIDATION")
    print("="*80)
    
    # 1. Generate or load test data
    print("\nGenerating test data...")
    df = generate_test_data()
    print(f"Generated {len(df)} bars of test data")
    
    # 2. Calculate expected results based on our rule logic
    print("\nCalculating expected results...")
    expected_results = calculate_expected_results(df)
    print(f"Expected trades: {expected_results['num_trades']}")
    print(f"Expected total return: {expected_results['total_percent_return']:.2f}%")
    
    # 3. Set up the backtester with our threshold rule
    print("\nSetting up backtester...")
    threshold_rule = ThresholdRule({
        'upper_threshold': UPPER_THRESHOLD,
        'lower_threshold': LOWER_THRESHOLD
    })
    strategy = SimpleStrategy(threshold_rule)
    data_handler = SimpleDataHandler(df)
    
    # 4. Run the backtest
    print("\nRunning backtest...")
    backtester = Backtester(data_handler, strategy)
    actual_results = backtester.run(use_test_data=True)
    
    print(f"Actual trades: {actual_results['num_trades']}")
    print(f"Actual total return: {actual_results['total_percent_return']:.2f}%")
    
    # 5. Compare expected vs actual results
    print("\nComparing results:")
    print(f"{'Metric':<25} {'Expected':<15} {'Actual':<15} {'Match?':<10}")
    print("-"*70)
    
    metrics = [
        ('Number of trades', expected_results['num_trades'], actual_results['num_trades']),
        ('Total log return', round(expected_results['total_log_return'], 4), 
         round(actual_results['total_log_return'], 4)),
        ('Total percent return', round(expected_results['total_percent_return'], 2), 
         round(actual_results['total_percent_return'], 2)),
        ('Average log return', round(expected_results['average_log_return'], 4), 
         round(actual_results['average_log_return'], 4))
    ]
    
    for metric_name, expected, actual in metrics:
        # Determine if they match within a small tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            tolerance = 0.0001 if 'log' in metric_name.lower() else 0.01
            matches = abs(expected - actual) <= tolerance
        else:
            matches = expected == actual
            
        match_str = "✓" if matches else "✗"
        print(f"{metric_name:<25} {expected:<15} {actual:<15} {match_str:<10}")
    
    # Check if number of trades match
    if expected_results['num_trades'] == actual_results['num_trades']:
        print("\nTrade by trade comparison:")
        print(f"{'#':<5} {'Entry Date':<12} {'Direction':<10} {'Entry':<8} {'Exit':<8} {'Return %':<10} {'Match?':<10}")
        print("-"*70)
        
        for i, (expected_trade, actual_trade) in enumerate(zip(expected_results['trades'], actual_results['trades'])):
            # Check if trade details match
            matches = (
                expected_trade[0] == actual_trade[0] and  # Entry time
                expected_trade[1] == actual_trade[1] and  # Direction
                abs(expected_trade[2] - actual_trade[2]) < 0.01 and  # Entry price
                abs(expected_trade[4] - actual_trade[4]) < 0.01 and  # Exit price
                abs(expected_trade[5] - actual_trade[5]) < 0.0001  # Log return
            )
            
            match_str = "✓" if matches else "✗"
            exp_pct = (np.exp(expected_trade[5]) - 1) * 100
            
            print(f"{i+1:<5} {expected_trade[0]:<12} {expected_trade[1]:<10} "
                  f"{expected_trade[2]:<8.2f} {expected_trade[4]:<8.2f} "
                  f"{exp_pct:<10.2f} {match_str:<10}")
    
    # 6. Plot the results
    plot_backtest_results(df, actual_results['trades'], 
                         title=f"Validation Results: {actual_results['num_trades']} trades, "
                               f"{actual_results['total_percent_return']:.2f}% return")
    
    # 7. Return overall validation result
    validation_passed = all(
        abs(metrics[i][1] - metrics[i][2]) < 0.01 for i in range(len(metrics))
    )
    
    print("\n" + "="*80)
    if validation_passed:
        print("✅ VALIDATION PASSED: Backtest results match expected results")
    else:
        print("❌ VALIDATION FAILED: Backtest results don't match expected results")
    print("="*80)
    
    return validation_passed

if __name__ == "__main__":
    run_backtest_validation()

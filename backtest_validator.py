"""
Simplified Backtest Validator

This script provides a simpler validation test that will work with your backtest system.
It focuses on identifying issues with signal handling and return calculation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# In backtest_validator.py
from improved_threshold_rule import ImprovedThresholdRule



# First generate test data if it doesn't exist
def ensure_test_data_exists():
    """Make sure test data exists, generating it if needed."""
    if not os.path.exists('test_data/test_ohlc_data.csv'):
        print("Test data not found, generating it...")
        from test_data_generator import create_test_rule_class
        import test_data_generator
        test_data_generator.generate_test_data()
        test_data_generator.calculate_expected_signals_and_returns(
            pd.read_csv('test_data/test_ohlc_data.csv')
        )
        print("Test data generated.")
    else:
        print("Using existing test data.")

# Import your trading components
from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent

# Define a compatible test rule that works with your system
class TestThresholdRule:
    """
    Simple test rule that buys above threshold and sells below.
    Designed to be compatible with your signal handling system.
    """
    def __init__(self, params=None):
        self.threshold = params.get('threshold', 100.0) if params else 100.0
        self.rule_id = "TestThresholdRule"


    # Gemini 
    # def on_bar(self, bar):
    #     """Generate signals based on price relative to threshold."""
    #     # Extract price from the bar, regardless of whether it's an event or dict
    #     if isinstance(bar, dict):
    #         close = bar['Close']
    #     elif hasattr(bar, 'Close'):
    #         close = bar.Close
    #     elif hasattr(bar, 'bar') and isinstance(bar.bar, dict):
    #         close = bar.bar['Close']
    #     else:
    #         print(f"Warning: Unrecognized bar format: {type(bar)}")
    #         if hasattr(bar, '__dict__'):
    #             print(f"Bar attributes: {bar.__dict__}")
    #         return 0  # Default to neutral

    #     # Generate signal
    #     signal_value = 1 if close > self.threshold else -1
    #     print(f"[Rule] Bar close: {close}, Threshold: {self.threshold}, Signal: {signal_value}")
    #     return signal_value
        
        
    # def on_bar(self, bar):
    #     """Generate signals based on price relative to threshold."""
    #     # Extract price from the bar, regardless of whether it's an event or dict
    #     if isinstance(bar, dict):
    #         close = bar['Close']
    #     elif hasattr(bar, 'Close'):
    #         close = bar.Close
    #     elif hasattr(bar, 'bar') and isinstance(bar.bar, dict):
    #         close = bar.bar['Close']
    #     else:
    #         print(f"Warning: Unrecognized bar format: {type(bar)}")
    #         if hasattr(bar, '__dict__'):
    #             print(f"Bar attributes: {bar.__dict__}")
    #         return 0  # Default to neutral

    #     # Generate signal
    #     signal_value = 1 if close > self.threshold else -1
    #     print(f"Bar close: {close}, Threshold: {self.threshold}, Signal: {signal_value}")
    #     return signal_value
        
    
    def reset(self):
        """Reset rule state."""
        pass

# Define a simple strategy using the rule
class SimpleStrategy:
    """Wrapper strategy for the test rule."""
    def __init__(self, rule):
        self.rule = rule
    
    def on_bar(self, event):
        """Process a bar event using the rule."""
        # The backtester calls this with an event, but we need to extract the bar
        # to pass to the rule. We support multiple formats to be compatible.
        if hasattr(event, 'bar'):
            return self.rule.on_bar(event.bar)
        else:
            # If event is the bar itself
            return self.rule.on_bar(event)
    
    def reset(self):
        """Reset the strategy state."""
        self.rule.reset()

def run_validation_test():
    """Run a simplified validation test."""
    print("\n=== SIMPLIFIED VALIDATION TEST ===")
    
    # Ensure test data exists
    ensure_test_data_exists()
    
    # Load test data
    data_path = 'test_data/test_ohlc_data.csv'
    data_handler = CSVDataHandler(data_path, train_fraction=0.7)
    
    # Create test rule and strategy
    # Replace the existing test rule creation with:
    test_rule = ImprovedThresholdRule()
    #test_rule = TestThresholdRule()
    strategy = SimpleStrategy(test_rule)
    
    # Run backtest
    print("Running backtest with TestThresholdRule...")
    backtester = Backtester(data_handler, strategy)
    results = backtester.run(use_test_data=True)
    
    # Print actual results
    print("\nActual Results:")
    print(f"Total Return: {results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Load expected results for comparison
    try:
        expected_trades_df = pd.read_csv('test_data/expected_trades.csv')
        expected_log_return = sum(expected_trades_df['log_return'])
        expected_return = (np.exp(expected_log_return) - 1) * 100
        
        print("\nExpected Results:")
        print(f"Expected Return: {expected_return:.2f}%")
        print(f"Expected Trades: {len(expected_trades_df)}")
        
        # Compare trade counts
        if results['num_trades'] != len(expected_trades_df):
            print(f"\n⚠️ Trade count mismatch: Expected {len(expected_trades_df)}, got {results['num_trades']}")
            
            # Give more detailed info about the differences
            print("\nPossible causes for trade count difference:")
            print("1. Signal conversion issues: Check how your backtester converts rule signals to trade decisions")
            print("2. Entry/exit logic: Check if your backtester uses different logic for entering/exiting trades")
            print("3. Signal format: Check if your rule returns signals in the format your backtester expects")
    except Exception as e:
        print(f"Error comparing with expected results: {str(e)}")
    
    # Create a visualization of the trades
    try:
        print("\nGenerating trade visualization...")
        df = pd.read_csv(data_path)
        
        plt.figure(figsize=(15, 8))
        
        # Plot price
        plt.plot(df['timestamp'], df['Close'], label='Close Price')
        
        # Add threshold line
        plt.axhline(y=100.0, color='r', linestyle='--', label='Threshold (100.0)')
        
        # Add trade markers if available
        if 'trades' in results and results['trades']:
            # Check trade format
            trade = results['trades'][0]
            if isinstance(trade, (list, tuple)):
                # Tuple format: (entry_time, direction, entry_price, exit_time, exit_price, log_return)
                entries = [(t[0], t[2]) for t in results['trades']]
                exits = [(t[3], t[4]) for t in results['trades']]
            elif isinstance(trade, dict):
                # Dict format
                entries = [(t['entry_time'], t['entry_price']) for t in results['trades']]
                exits = [(t['exit_time'], t['exit_price']) for t in results['trades']]
            
            # Plot entries and exits
            entry_times = [e[0] for e in entries]
            entry_prices = [e[1] for e in entries]
            exit_times = [e[0] for e in exits]
            exit_prices = [e[1] for e in exits]
            
            plt.scatter(entry_times, entry_prices, color='green', marker='^', s=100, label='Entry')
            plt.scatter(exit_times, exit_prices, color='red', marker='v', s=100, label='Exit')
        
        plt.title('Test Data with Actual Trades')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('actual_trades_visualization.png')
        plt.close()
        print("Visualization saved to 'actual_trades_visualization.png'")
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
    
    return results

if __name__ == "__main__":
    run_validation_test()

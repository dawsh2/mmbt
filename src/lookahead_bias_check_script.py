import os
import sys

# Set working dir to src/ regardless of where script is located
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)  # Ensure src/ is first in import path

"""
Lookahead Bias Validation Script for Backtesting Engine.

This script performs automated tests to detect possible lookahead bias
in the trading strategy backtester.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add project directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import backtester modules
from config import Config
from data import DataHandler
from strategy import StrategyFactory
from backtester import Backtester
import ga

def create_test_config():
    """Create a test configuration."""
    config = Config()
    config.data_file = '../data/data.csv'  # Update this to your data path
    config.train_size = 0.7
    config.top_n = 5
    config.use_weights = True
    config.train = True
    config.test = True
    return config

def run_zero_returns_test(data_file):
    """
    Zero Returns Test: Replace all returns with zeros and verify 
    the strategy doesn't generate positive returns.
    
    If the strategy still produces positive returns with zero actual returns,
    it's likely using future information.
    """
    print("\n=== Running Zero Returns Test ===")
    
    # Create configuration
    config = create_test_config()
    config.data_file = data_file
    
    # Create data handler
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    
    # Create a copy of the data with zeroed returns
    original_data = data_handler.data.copy()
    
    # Save original Close prices
    original_close = original_data['Close'].copy()
    
    # Modify Close prices to create zero returns
    # First value stays the same, all others set to create zero returns
    modified_close = original_close.copy()
    for i in range(1, len(modified_close)):
        modified_close.iloc[i] = modified_close.iloc[i-1]
    
    # Apply modified Close prices
    data_handler.data['Close'] = modified_close
    
    # Recalculate log returns
    data_handler.data['LogReturn'] = np.log(data_handler.data['Close'] / 
                                        data_handler.data['Close'].shift(1)).fillna(0)
    
    # Split data
    data_handler.split_data()
    
    # Create and run backtester with zeroed returns
    strategy = StrategyFactory.create_strategy(config)
    backtester = Backtester(config, data_handler, strategy)
    
    # Run backtest
    backtester.run(ga_module=ga)
    
    # Get results
    zero_returns_results = backtester.results.get('performance', {}).get('strategy', {})
    
    # Print results
    print("\nZero Returns Test Results:")
    print(f"Total Return: {zero_returns_results.get('total_return', 0):.4f}")
    print(f"Sharpe Ratio: {zero_returns_results.get('sharpe_ratio', 0):.4f}")
    
    # Restore original data
    data_handler.data = original_data
    
    # Check if strategy generated positive returns with zero actual returns
    has_lookahead_bias = zero_returns_results.get('total_return', 0) > 0.01
    print(f"Lookahead bias detected: {has_lookahead_bias}")
    
    return has_lookahead_bias, zero_returns_results

def run_signal_delay_test(data_file, delay=2):
    """
    Signal Delay Test: Add extra signal delays and verify performance degrades.
    
    If adding extra delays doesn't significantly degrade performance,
    it suggests the strategy might already have lookahead bias.
    """
    print(f"\n=== Running Signal Delay Test (Delay={delay}) ===")
    
    # Create configuration
    config = create_test_config()
    config.data_file = data_file
    
    # First, run normal backtest
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    data_handler.split_data()
    
    strategy = StrategyFactory.create_strategy(config)
    backtester = Backtester(config, data_handler, strategy)
    backtester.run(ga_module=ga)
    
    normal_results = backtester.results.get('performance', {}).get('strategy', {})
    
    # Now, monkey patch the signal application logic to add extra delay
    original_calculate_performance = backtester._calculate_performance
    
    def delayed_calculate_performance(signals_df):
        # Add extra delay to signals
        signals_df['Signal'] = signals_df['Signal'].shift(delay)
        signals_df['Signal'] = signals_df['Signal'].fillna(0)
        
        # Call original method
        return original_calculate_performance(signals_df)
    
    # Replace the method
    backtester._calculate_performance = delayed_calculate_performance.__get__(
        backtester, Backtester)
    
    # Re-run with delayed signals
    strategy = StrategyFactory.create_strategy(config)
    backtester = Backtester(config, data_handler, strategy)
    backtester.run(ga_module=ga)
    
    delayed_results = backtester.results.get('performance', {}).get('strategy', {})
    
    # Print results
    print("\nNormal vs. Delayed Signal Results:")
    print(f"Normal Total Return: {normal_results.get('total_return', 0):.4f}")
    print(f"Delayed Total Return: {delayed_results.get('total_return', 0):.4f}")
    print(f"Performance Change: {delayed_results.get('total_return', 0) - normal_results.get('total_return', 0):.4f}")
    
    # If delayed performance isn't significantly worse, it suggests lookahead bias
    # Allow some margin for random luck
    has_lookahead_bias = delayed_results.get('total_return', 0) > normal_results.get('total_return', 0) * 0.7
    print(f"Lookahead bias detected: {has_lookahead_bias}")
    
    return has_lookahead_bias, normal_results, delayed_results

def run_data_subset_test(data_file, num_subsets=2):
    """
    Data Subset Test: Run on different data subsets and verify consistent performance.
    
    If performance varies wildly across subsets, it suggests the strategy
    might be overfitted or have data leakage.
    """
    print(f"\n=== Running Data Subset Test (Subsets={num_subsets}) ===")
    
    # Create configuration
    config = create_test_config()
    config.data_file = data_file
    
    # Load full dataset
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    
    full_data = data_handler.data.copy()
    subset_size = len(full_data) // num_subsets
    
    results = []
    
    for i in range(num_subsets):
        print(f"\nTesting Subset {i+1}/{num_subsets}")
        
        # Create subset
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < num_subsets - 1 else len(full_data)
        
        data_handler.data = full_data.iloc[start_idx:end_idx].copy()
        
        # Split data
        data_handler.split_data()
        
        # Create and run backtester
        strategy = StrategyFactory.create_strategy(config)
        backtester = Backtester(config, data_handler, strategy)
        backtester.run(ga_module=ga)
        
        # Get results
        subset_results = backtester.results.get('performance', {}).get('strategy', {})
        results.append(subset_results)
        
        print(f"Subset {i+1} Total Return: {subset_results.get('total_return', 0):.4f}")
    
    # Calculate consistency metrics
    returns = [r.get('total_return', 0) for r in results]
    sharpes = [r.get('sharpe_ratio', 0) for r in results]
    
    return_std = np.std(returns)
    return_mean = np.mean(returns)
    
    # High coefficient of variation suggests inconsistent performance
    return_cv = return_std / return_mean if return_mean != 0 else float('inf')
    
    print("\nConsistency Metrics:")
    print(f"Return Mean: {return_mean:.4f}")
    print(f"Return Std Dev: {return_std:.4f}")
    print(f"Return Coefficient of Variation: {return_cv:.4f}")
    
    # If CV > 1.0, performance is highly variable
    has_consistency_issue = return_cv > 1.0
    print(f"Consistency issue detected: {has_consistency_issue}")
    
    return has_consistency_issue, results

def run_randomized_returns_test(data_file, num_trials=3):
    """
    Randomized Returns Test: Shuffle returns while preserving signals
    to test for spurious correlations.
    
    If the strategy still produces positive returns with randomized returns,
    it suggests data leakage or overfitting.
    """
    print(f"\n=== Running Randomized Returns Test (Trials={num_trials}) ===")
    
    # Create configuration
    config = create_test_config()
    config.data_file = data_file
    
    # Load data
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    
    original_data = data_handler.data.copy()
    
    trial_results = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")
        
        # Create a copy of data
        data_handler.data = original_data.copy()
        
        # Get the Close prices
        close_prices = data_handler.data['Close'].copy()
        
        # Randomize the order of Close prices (but keep first one)
        first_price = close_prices.iloc[0]
        shuffled_prices = close_prices.iloc[1:].sample(frac=1).reset_index(drop=True)
        
        # Reconstruct the Close series
        new_close = pd.Series([first_price], index=[close_prices.index[0]])
        new_close = pd.concat([new_close, pd.Series(
            shuffled_prices.values, index=close_prices.index[1:])])
        
        # Apply modified Close prices
        data_handler.data['Close'] = new_close
        
        # Recalculate log returns
        data_handler.data['LogReturn'] = np.log(data_handler.data['Close'] / 
                                           data_handler.data['Close'].shift(1)).fillna(0)
        
        # Split data
        data_handler.split_data()
        
        # Create and run backtester
        strategy = StrategyFactory.create_strategy(config)
        backtester = Backtester(config, data_handler, strategy)
        backtester.run(ga_module=ga)
        
        # Get results
        rand_results = backtester.results.get('performance', {}).get('strategy', {})
        trial_results.append(rand_results)
        
        print(f"Trial {trial+1} Total Return: {rand_results.get('total_return', 0):.4f}")
    
    # Calculate average performance
    avg_return = np.mean([r.get('total_return', 0) for r in trial_results])
    avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in trial_results])
    
    print("\nRandomized Returns Results:")
    print(f"Average Total Return: {avg_return:.4f}")
    print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    
    # If average return is still significantly positive, suggests issue
    has_leakage = avg_return > 0.1
    print(f"Data leakage detected: {has_leakage}")
    
    return has_leakage, trial_results


def generate_report(tests_results):
    """Generate an HTML report with test results."""
    report_file = f"lookahead_bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lookahead Bias Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .test {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Lookahead Bias Test Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add test results
    for test_name, test_data in tests_results.items():
        html += f"""
        <div class="test">
            <h2>{test_name}</h2>
            <p class="{'fail' if test_data.get('bias_detected', False) else 'pass'}">
                Bias Detected: {test_data.get('bias_detected', False)}
            </p>
            <h3>Test Details:</h3>
            <pre>{test_data.get('details', '')}</pre>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"\nReport generated: {report_file}")
    return report_file

def main():
    """Run all tests and generate report."""
    print("=== Lookahead Bias Validation Script ===")
    
    # Set data file path
    data_file = 'data/data.csv'  # Update this to your data path
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    print(f"Using data file: {data_file}")
    
    # Run tests
    test_results = {}
    
    # Zero Returns Test
    try:
        bias_detected, results = run_zero_returns_test(data_file)
        test_results['Zero Returns Test'] = {
            'bias_detected': bias_detected,
            'details': str(results)
        }
    except Exception as e:
        print(f"Error in Zero Returns Test: {e}")
        test_results['Zero Returns Test'] = {
            'bias_detected': "Error",
            'details': str(e)
        }
    
    # Signal Delay Test
    try:
        bias_detected, normal, delayed = run_signal_delay_test(data_file)
        test_results['Signal Delay Test'] = {
            'bias_detected': bias_detected,
            'details': f"Normal: {normal}\nDelayed: {delayed}"
        }
    except Exception as e:
        print(f"Error in Signal Delay Test: {e}")
        test_results['Signal Delay Test'] = {
            'bias_detected': "Error",
            'details': str(e)
        }
    
    # Data Subset Test
    try:
        consistency_issue, results = run_data_subset_test(data_file)
        test_results['Data Subset Test'] = {
            'bias_detected': consistency_issue,
            'details': str(results)
        }
    except Exception as e:
        print(f"Error in Data Subset Test: {e}")
        test_results['Data Subset Test'] = {
            'bias_detected': "Error",
            'details': str(e)
        }
    
    # Randomized Returns Test
    try:
        leakage_detected, results = run_randomized_returns_test(data_file)
        test_results['Randomized Returns Test'] = {
            'bias_detected': leakage_detected,
            'details': str(results)
        }
    except Exception as e:
        print(f"Error in Randomized Returns Test: {e}")
        test_results['Randomized Returns Test'] = {
            'bias_detected': "Error",
            'details': str(e)
        }
    
    # Generate report
    report_file = generate_report(test_results)
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, test_data in test_results.items():
        bias_status = test_data.get('bias_detected', False)
        status_text = "FAILED" if bias_status else "PASSED"
        print(f"{test_name}: {status_text}")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()

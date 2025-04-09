"""
Script to debug signal generation in the trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import components from the backtesting engine
from config import Config
from data import DataHandler
from rules import TradingRules

def debug_signal_generation(data_file='data.csv', params_file='params.json', sample_size=100):
    """
    Debug signal generation by examining a sample of data.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
        sample_size: Number of data points to examine (from end of test data)
    """
    print(f"Debugging signal generation using {params_file}")
    
    # Create a default configuration
    config = Config()
    config.data_file = data_file
    config.params_file = params_file
    
    # Load data
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    data_handler.split_data()
    
    # Load rule parameters
    rules = TradingRules()
    rules.load_params(params_file)
    
    # Get testing data
    test_ohlc = data_handler.get_ohlc(train=False)
    
    # Extract log returns and identify top rules
    _, _, _, test_close = test_ohlc
    test_logr = np.log(test_close/test_close.shift(1))
    
    if hasattr(rules, 'rule_indices'):
        top_n = 3  # Use top-3 rules as in the results.json
        top_indices = rules.rule_indices[:top_n]
        top_rules = [i+1 for i in top_indices]
        print(f"Top {top_n} rules: {top_rules}")
        
        # Generate individual rule signals
        rule_signals = {}
        for i in top_indices:
            rule_func = rules.rule_functions[i]
            rule_name = f"Rule{i+1}"
            try:
                _, signal = rule_func(rules.rule_params[i], test_ohlc)
                rule_signals[rule_name] = signal
                print(f"Generated signals for {rule_name}")
            except Exception as e:
                print(f"Error generating signals for {rule_name}: {str(e)}")
        
        # Generate combined signals
        try:
            combined_signals_df = rules.generate_signals(test_ohlc, rules.rule_params, top_n=top_n)
            print(f"Generated combined signals for top {top_n} rules")
            
            # Create DataFrame with all signals for comparison
            signal_df = pd.DataFrame(index=test_close.index)
            signal_df['LogReturn'] = test_logr
            
            for rule_name, signal in rule_signals.items():
                signal_df[rule_name] = signal
            
            signal_df['Combined'] = combined_signals_df['Signal']
            
            # Make sure all signal data is aligned
            signal_df.dropna(inplace=True)
            
            # Check if the combined signal matches what we'd expect from individual rules
            # For unweighted strategies, we'd expect the combined signal to be the sign of the sum
            signal_df['ExpectedCombined'] = np.sign(sum(signal_df[f"Rule{i+1}"] for i in top_indices))
            
            # Check for discrepancies
            discrepancies = (signal_df['Combined'] != signal_df['ExpectedCombined']).sum()
            total_signals = len(signal_df)
            
            print(f"\nSignal discrepancies: {discrepancies} out of {total_signals} " +
                  f"({discrepancies/total_signals*100:.2f}%)")
            
            if discrepancies > 0:
                print("\nPossible reasons for discrepancies:")
                print("1. NaN handling different between individual and combined signals")
                print("2. Different delay application (shift) in signal generation")
                print("3. Logic error in rules.generate_signals method")
                print("4. Signal combination method different from simple sum")
                
                # Show sample of discrepancies
                discrepancy_rows = signal_df[signal_df['Combined'] != signal_df['ExpectedCombined']]
                if len(discrepancy_rows) > 0:
                    print("\nSample of discrepancies:")
                    pd.set_option('display.max_columns', None)
                    print(discrepancy_rows.head(min(5, len(discrepancy_rows))))
            
            # Show a sample of recent data
            print(f"\nSample of recent {sample_size} data points:")
            sample = signal_df.tail(sample_size)
            
            # Check performance on this sample
            sample_strategy_returns = sample['Combined'].shift(1) * sample['LogReturn']
            sample_rule_returns = {}
            
            for rule_name in rule_signals.keys():
                sample_rule_returns[rule_name] = sample[rule_name].shift(1) * sample['LogReturn']
            
            # Calculate cumulative returns
            cum_strategy_returns = (1 + sample_strategy_returns.cumsum().fillna(0))
            cum_rule_returns = {rule_name: (1 + returns.cumsum().fillna(0)) 
                               for rule_name, returns in sample_rule_returns.items()}
            
            # Plot
            plt.figure(figsize=(14, 7))
            
            # Plot individual rule returns
            for rule_name, returns in cum_rule_returns.items():
                plt.plot(returns.index, returns, label=rule_name, alpha=0.7)
            
            # Plot combined strategy returns
            plt.plot(cum_strategy_returns.index, cum_strategy_returns, 
                    'k-', linewidth=2, label='Combined Strategy')
            
            plt.title(f'Cumulative Returns - Last {sample_size} Data Points')
            plt.legend()
            plt.grid(True)
            plt.savefig('signal_comparison.png')
            print("Signal comparison plot saved to signal_comparison.png")
            
            # Check signal distribution
            print("\nSignal distribution in sample:")
            for col in sample.columns:
                if col not in ['LogReturn', 'ExpectedCombined']:
                    buy = (sample[col] == 1).sum()
                    sell = (sample[col] == -1).sum()
                    neutral = (sample[col] == 0).sum()
                    total = buy + sell + neutral
                    
                    print(f"{col}: Buy={buy} ({buy/total*100:.1f}%), " +
                          f"Sell={sell} ({sell/total*100:.1f}%), " +
                          f"Neutral={neutral} ({neutral/total*100:.1f}%)")
            
            # Create cross-correlation matrix of signals
            signal_cols = [col for col in sample.columns 
                          if col not in ['LogReturn', 'ExpectedCombined']]
            if len(signal_cols) > 1:
                corr_matrix = sample[signal_cols].corr()
                print("\nSignal correlation matrix:")
                print(corr_matrix)
                
                # Check for highly correlated signals
                high_corr = False
                for i in range(len(signal_cols)):
                    for j in range(i+1, len(signal_cols)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            if not high_corr:
                                print("\nHighly correlated signals detected:")
                                high_corr = True
                            print(f"{signal_cols[i]} and {signal_cols[j]}: {corr_matrix.iloc[i, j]:.2f}")
                
                if not high_corr:
                    print("\nNo highly correlated signals detected.")
            
        except Exception as e:
            print(f"Error during signal analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Rule indices not available - cannot identify top rules.")

if __name__ == "__main__":
    debug_signal_generation()

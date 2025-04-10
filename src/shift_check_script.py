#!/usr/bin/env python
"""
Fixed shift validation script that tests individual rules and manually combines signals
to work around the 'weights' AttributeError.
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if current_dir.endswith('src'):
    src_dir = current_dir
    params_path = os.path.join(os.path.dirname(current_dir), 'params.json')
else:
    params_path = os.path.join(current_dir, 'params.json')
sys.path.insert(0, src_dir)

from config import Config
from data import DataHandler
import rules

# Try multiple possible locations for the data file
possible_data_paths = [
    'data.csv',
    '../data/data.csv',
    'data/data.csv',
    os.path.join(os.path.dirname(current_dir), 'data', 'data.csv')
]

def load_rule_params():
    """Load rule parameters from params.json"""
    try:
        if os.path.exists(params_path):
            print(f"Loading rule parameters from {params_path}")
            with open(params_path, 'r') as f:
                params_data = json.load(f)
                return params_data['params'], params_data['indices'], params_data.get('scores', [])
        else:
            print(f"Could not find params.json at {params_path}")
            return None, None, None
    except Exception as e:
        print(f"Error loading params.json: {e}")
        return None, None, None

def get_rule_function(rule_index):
    """Get the rule function based on index"""
    rule_functions = [
        rules.Rule0, rules.Rule1, rules.Rule2, rules.Rule3, rules.Rule4,
        rules.Rule5, rules.Rule6, rules.Rule7, rules.Rule8, rules.Rule9,
        rules.Rule10, rules.Rule11, rules.Rule12, rules.Rule13, rules.Rule14, rules.Rule15
    ]
    if 0 <= rule_index < len(rule_functions):
        return rule_functions[rule_index]
    return None

def test_individual_rules(sample_data, rule_params, rule_indices):
    """Test individual rules with the specified parameters"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL RULES")
    print("="*80)
    
    rule_signals_list = []
    rule_returns_list = []
    
    for param, rule_idx in zip(rule_params, rule_indices):
        rule_func = get_rule_function(rule_idx)
        if rule_func is None:
            print(f"Rule {rule_idx} not found, skipping")
            continue
            
        rule_name = rule_func.__name__
        print(f"\nTesting {rule_name} with params: {param}")
        
        # Generate signals
        try:
            score, signals = rule_func(param, sample_data)
            
            # Calculate returns
            log_returns = np.log(sample_data['Close'] / sample_data['Close'].shift(1)).fillna(0)
            
            # Calculate returns with proper shifting
            shifted_signals = signals.shift(1).fillna(0)
            rule_returns = shifted_signals * log_returns
            
            # Store results
            rule_signals_list.append((rule_name, signals))
            rule_returns_list.append((rule_name, rule_returns))
            
            # Print statistics
            print(f"Signal value counts: {signals.value_counts().to_dict()}")
            print(f"Signal mean: {signals.mean():.4f}")
            print(f"Returns sum: {rule_returns.sum():.6f}")
            
            # Check for NaN values
            if signals.isna().any():
                print("WARNING: NaN values detected in signals")
            
            # Check for lookahead sensitivity
            unshifted_returns = signals * log_returns
            unshifted_sum = unshifted_returns.sum()
            shifted_sum = rule_returns.sum()
            
            if abs(unshifted_sum - shifted_sum) > 0.0001:
                print(f"Lookahead sensitivity detected!")
                print(f"Unshifted returns: {unshifted_sum:.6f}")
                print(f"Shifted returns: {shifted_sum:.6f}")
                print(f"Difference: {unshifted_sum - shifted_sum:.6f}")
            else:
                print("No lookahead sensitivity detected in this sample")
        
        except Exception as e:
            print(f"Error testing {rule_name}: {e}")
    
    return rule_signals_list, rule_returns_list

def manually_combine_signals(rule_signals, rule_scores=None):
    """
    Manually combine signals from multiple rules using a weighted approach.
    
    Args:
        rule_signals: List of (rule_name, signals) tuples
        rule_scores: List of scores (weights) for each rule
    
    Returns:
        Combined signals
    """
    if not rule_signals:
        return None
    
    # Create a DataFrame with all signals
    signals_df = pd.DataFrame()
    for i, (rule_name, signals) in enumerate(rule_signals):
        signals_df[rule_name] = signals
    
    # If no scores provided, use equal weights
    if rule_scores is None or len(rule_scores) != len(rule_signals):
        weights = [1.0 / len(rule_signals)] * len(rule_signals)
    else:
        # Normalize weights
        total = sum(rule_scores)
        weights = [score / total for score in rule_scores] if total > 0 else [1.0 / len(rule_signals)] * len(rule_signals)
    
    # Apply weights
    weighted_signals = pd.DataFrame()
    for i, (rule_name, _) in enumerate(rule_signals):
        weighted_signals[rule_name] = signals_df[rule_name] * weights[i]
    
    # Combine weighted signals
    combined = weighted_signals.sum(axis=1)
    
    # Convert to -1/0/1 based on threshold
    threshold = 0.2
    final_signals = pd.Series(0, index=combined.index)
    final_signals[combined > threshold] = 1
    final_signals[combined < -threshold] = -1
    
    return final_signals

# Try to load data
config = Config()
data_handler = DataHandler(config)

# Try each possible data path
data_loaded = False
for data_path in possible_data_paths:
    config.data_file = data_path
    if data_handler.load_data():
        data_loaded = True
        print(f"Successfully loaded data from {data_path}")
        break

if not data_loaded:
    print("Could not load data from any expected location.")
    sys.exit(1)

# Preprocess and split data
data_handler.preprocess()
data_handler.split_data()

# Get a sample of test data (last 100 rows to have more signals)
sample_data = data_handler.test_data.iloc[-100:].copy() if data_handler.test_data is not None else data_handler.data.iloc[-100:].copy()

# Load rule parameters from params.json
rule_params, rule_indices, rule_scores = load_rule_params()

if rule_params and rule_indices:
    # Test each individual rule
    rule_signals, rule_returns = test_individual_rules(sample_data, rule_params, rule_indices)
    
    # Manually combine signals to simulate the strategy
    print("\n" + "="*80)
    print("SIMULATING STRATEGY WITH MANUALLY COMBINED SIGNALS")
    print("="*80)
    
    combined_signals = manually_combine_signals(rule_signals, rule_scores)
    
    if combined_signals is not None:
        print("\nManually combined signals statistics:")
        print(f"Signal value counts: {combined_signals.value_counts().to_dict()}")
        print(f"Signal mean: {combined_signals.mean():.4f}")
        
        # Calculate log returns
        log_returns = np.log(sample_data['Close'] / sample_data['Close'].shift(1)).fillna(0)
        
        # Calculate strategy returns with shifting
        shifted_signals = combined_signals.shift(1).fillna(0)
        strategy_returns = shifted_signals * log_returns
        
        # Calculate without shifting (would have lookahead bias)
        unshifted_returns = combined_signals * log_returns
        
        print("\nSTRATEGY RETURNS COMPARISON (WITH vs WITHOUT SHIFTING):")
        shifted_sum = strategy_returns.sum()
        unshifted_sum = unshifted_returns.sum()
        print(f"With proper shifting: {shifted_sum:.6f}")
        print(f"Without shifting (biased): {unshifted_sum:.6f}")
        print(f"Difference: {unshifted_sum - shifted_sum:.6f}")
        
        # Calculate difference percentage
        if abs(shifted_sum) > 0.0001:
            diff_pct = abs((unshifted_sum - shifted_sum) / shifted_sum) * 100
            print(f"Difference percentage: {diff_pct:.2f}%")
            
            if diff_pct > 5.0:
                print("\n*** SIGNIFICANT LOOKAHEAD EFFECT DETECTED ***")
                print("This confirms that shifting matters for your strategy.")
            else:
                print("\nMinimal lookahead effect detected with this data sample.")
        
        # Compare individual rule returns with strategy returns
        print("\nCOMPARING INDIVIDUAL RULES TO STRATEGY:")
        for rule_name, rule_return in rule_returns:
            rule_sum = rule_return.sum()
            print(f"{rule_name} returns: {rule_sum:.6f}")
        
        print(f"Combined strategy returns: {shifted_sum:.6f}")
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.title('Closing Price')
        plt.plot(sample_data.index, sample_data['Close'])
        plt.grid(True)
        
        # Plot strategy signals vs individual rule signals
        plt.subplot(3, 1, 2)
        plt.title('Signals')
        plt.plot(combined_signals.index, combined_signals, label='Combined', linewidth=2, color='black')
        
        # Add up to 5 individual rule signals
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, (rule_name, rule_signal) in enumerate(rule_signals[:5]):  # Limit to 5 rules for clarity
            plt.plot(rule_signal.index, rule_signal, label=rule_name, color=colors[i % len(colors)], alpha=0.6)
        
        plt.legend()
        plt.grid(True)
        
        # Plot cumulative returns
        plt.subplot(3, 1, 3)
        plt.title('Cumulative Returns')
        plt.plot(strategy_returns.index, strategy_returns.cumsum(), label='Strategy (shifted)', linewidth=2, color='black')
        plt.plot(unshifted_returns.index, unshifted_returns.cumsum(), label='Strategy (unshifted)', linewidth=2, color='gray', linestyle='--')
        
        # Add individual rule returns
        for i, (rule_name, rule_return) in enumerate(rule_returns[:5]):  # Limit to 5 rules for clarity
            plt.plot(rule_return.index, rule_return.cumsum(), label=rule_name, color=colors[i % len(colors)], alpha=0.6)
        
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png')
        print("\nVisualization saved to 'strategy_comparison.png'")
        
        # Conclusion
        print("\nCONCLUSION:")
        if abs(unshifted_sum - shifted_sum) > 0.0001:
            print("1. Shifting signals is NECESSARY to prevent lookahead bias")
            print("2. Your system requires proper shifting to avoid using future information")
            print("3. The lookahead bias tests should pass if all signals are properly shifted")
            print("\nRECOMMENDATION:")
            print("Make sure all signal shifting happens consistently in your system.")
            
            # Check if any rule has particularly high lookahead sensitivity
            print("\nLOOKAHEAD SENSITIVITY BY RULE:")
            for rule_name, rule_signals_data in rule_signals:
                unshifted = (rule_signals_data * log_returns).sum()
                shifted = (rule_signals_data.shift(1).fillna(0) * log_returns).sum()
                diff = abs(unshifted - shifted)
                print(f"{rule_name}: {diff:.6f}")
        else:
            print("1. Shifting signals had minimal impact on this data sample")
            print("2. This could be because the sample period has consistent trends")
            print("3. Try testing with a more volatile period to better validate signal shifting")
        
    else:
        print("No valid signals generated by any rules.")
        
else:
    print("Could not load rule parameters. Testing with default rule parameters.")

#!/usr/bin/env python
"""
Signal Tracer Script

This script traces a single trade signal through the entire backtesting system,
printing the DataFrame at each critical processing step to help diagnose
lookahead bias issues.
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

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
from rules import RuleSystem
from strategy import StrategyFactory
from backtester import Backtester

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

def print_df_snapshot(df, title, max_rows=5):
    """Print a clean snapshot of a DataFrame"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    if df is None:
        print("DataFrame is None")
        return
    
    # Get shape and column info
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index type: {type(df.index)}")
    
    # Check if the index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"Index range: {df.index.min()} to {df.index.max()}")
    
    # Print first few rows
    print("\nFirst rows:")
    print(df.head(max_rows))
    
    # Check for specific columns of interest
    for col in ['Signal', 'LogReturn']:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())
            print(f"{col} statistics:")
            print(df[col].describe())
    
    # Check for NaN values
    if df.isna().any().any():
        print("\nWarning: NaN values detected")
        print(df.isna().sum())

def trace_rule_execution(rule_func, params, data, rule_name=None):
    """Trace the execution of a single rule"""
    if rule_name is None:
        rule_name = rule_func.__name__
    
    print("\n" + "="*80)
    print(f"TRACING RULE EXECUTION: {rule_name}")
    print("="*80)
    
    # 1. Print input data sample
    print_df_snapshot(data, "1. INPUT DATA SAMPLE TO RULE", max_rows=3)
    
    # 2. Generate signals
    print("\n2. GENERATING SIGNALS...")
    try:
        score, signals = rule_func(params, data)
        print(f"Signal generated successfully. Score: {score}")
        
        # Print signals
        print_df_snapshot(pd.DataFrame({'Signal': signals}), "2.1 RAW SIGNALS FROM RULE")
        
        # 3. Shift signals as would happen in strategy
        shifted_signals = signals.shift(1).fillna(0)
        print_df_snapshot(pd.DataFrame({'Original': signals, 'Shifted': shifted_signals}), 
                           "3. SIGNALS AFTER SHIFTING")
        
        # 4. Apply to returns
        # First, check if LogReturn exists in data
        if 'LogReturn' in data.columns:
            log_returns = data['LogReturn']
            print("\nUsing pre-calculated LogReturn from data")
        else:
            log_returns = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
            print("\nCalculated LogReturn on the fly")
        
        # Calculate strategy returns with and without shifting
        unshifted_returns = signals * log_returns
        shifted_returns = shifted_signals * log_returns
        
        returns_df = pd.DataFrame({
            'LogReturn': log_returns,
            'Signal': signals,
            'Shifted_Signal': shifted_signals,
            'Unshifted_Return': unshifted_returns,
            'Shifted_Return': shifted_returns
        })
        
        print_df_snapshot(returns_df, "4. RETURNS CALCULATION")
        
        # 5. Print performance comparison
        unshifted_sum = unshifted_returns.sum()
        shifted_sum = shifted_returns.sum()
        
        print("\n5. PERFORMANCE COMPARISON")
        print(f"Unshifted returns sum: {unshifted_sum:.8f}")
        print(f"Shifted returns sum: {shifted_sum:.8f}")
        print(f"Difference: {unshifted_sum - shifted_sum:.8f}")
        
        if abs(unshifted_sum - shifted_sum) > 0.0001:
            print("*** LOOKAHEAD EFFECT DETECTED ***")
            
        return signals, log_returns
    
    except Exception as e:
        print(f"Error in rule execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def trace_strategy_execution(strategy, data):
    """Trace execution through the strategy component"""
    print("\n" + "="*80)
    print(f"TRACING STRATEGY EXECUTION: {strategy}")
    print("="*80)
    
    # 1. Print input data sample
    print_df_snapshot(data, "1. INPUT DATA SAMPLE TO STRATEGY", max_rows=3)
    
    # 2. Generate signals
    print("\n2. GENERATING STRATEGY SIGNALS...")
    try:
        signals_df = strategy.generate_signals(data)
        
        # Print strategy signals
        print_df_snapshot(signals_df, "2.1 SIGNALS FROM STRATEGY")
        
        # Extract signals and returns
        if isinstance(signals_df, pd.DataFrame):
            if 'Signal' in signals_df.columns:
                signals = signals_df['Signal']
            else:
                signals = signals_df.iloc[:, 0]  # Assume first column is signals
                
            if 'LogReturn' in signals_df.columns:
                log_returns = signals_df['LogReturn']
            else:
                log_returns = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
        else:
            signals = signals_df  # Assume it's a Series
            log_returns = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
        
        # 3. Shift signals as would happen in backtester
        shifted_signals = signals.shift(1).fillna(0)
        print_df_snapshot(pd.DataFrame({'Original': signals, 'Shifted': shifted_signals}), 
                          "3. SIGNALS AFTER SHIFTING")
        
        # 4. Apply to returns
        unshifted_returns = signals * log_returns
        shifted_returns = shifted_signals * log_returns
        
        returns_df = pd.DataFrame({
            'LogReturn': log_returns,
            'Signal': signals,
            'Shifted_Signal': shifted_signals,
            'Unshifted_Return': unshifted_returns,
            'Shifted_Return': shifted_returns
        })
        
        print_df_snapshot(returns_df, "4. RETURNS CALCULATION")
        
        # 5. Print performance comparison
        unshifted_sum = unshifted_returns.sum()
        shifted_sum = shifted_returns.sum()
        
        print("\n5. PERFORMANCE COMPARISON")
        print(f"Unshifted returns sum: {unshifted_sum:.8f}")
        print(f"Shifted returns sum: {shifted_sum:.8f}")
        print(f"Difference: {unshifted_sum - shifted_sum:.8f}")
        
        if abs(unshifted_sum - shifted_sum) > 0.0001:
            print("*** LOOKAHEAD EFFECT DETECTED ***")
            
        return signals_df
    
    except Exception as e:
        print(f"Error in strategy execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def trace_backtester_execution(config, data_handler, strategy):
    """Trace execution through the backtester component"""
    print("\n" + "="*80)
    print(f"TRACING BACKTESTER EXECUTION")
    print("="*80)
    
    try:
        # Create backtester
        backtester = Backtester(config, data_handler, strategy)
        
        # Get test data
        test_data = data_handler.test_data
        print_df_snapshot(test_data, "1. TEST DATA SAMPLE", max_rows=3)
        
        # Generate signals using strategy
        print("\n2. GENERATING SIGNALS IN BACKTESTER...")
        signals_df = strategy.generate_signals(
            test_data, 
            strategy.rules.rule_params if hasattr(strategy.rules, 'rule_params') else None,
            filter_regime=config.filter_regime
        )
        
        print_df_snapshot(signals_df, "2.1 SIGNALS FROM STRATEGY IN BACKTESTER")
        
        # Simulate the _calculate_performance method
        print("\n3. SIMULATING _calculate_performance...")
        returns = pd.to_numeric(signals_df['LogReturn'], errors='coerce')
        signals = pd.to_numeric(signals_df['Signal'], errors='coerce')
        
        # Check the implementation in backtester.py
        print("\nChecking backtester._calculate_performance implementation...")
        calc_perf_code = inspect.getsource(backtester._calculate_performance)
        print(calc_perf_code)
        
        # Apply shifting as in backtester code
        print("\n4. APPLYING SIGNAL SHIFTING...")
        shifted_signals = signals.shift(1).fillna(0)
        strategy_returns = shifted_signals * returns
        
        returns_df = pd.DataFrame({
            'LogReturn': returns,
            'Signal': signals,
            'Shifted_Signal': shifted_signals,
            'Strategy_Return': strategy_returns
        })
        
        print_df_snapshot(returns_df, "4.1 RETURNS CALCULATION IN BACKTESTER")
        
        # Calculate performance metrics
        total_return = strategy_returns.sum()
        print(f"\n5. Total return in backtester: {total_return:.8f}")
        
        return signals_df, returns_df
        
    except Exception as e:
        print(f"Error in backtester execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_signal_trace_visualization(rule_signals, strategy_signals, backtester_signals, log_returns):
    """Create visualization of signals and returns at each stage"""
    plt.figure(figsize=(12, 12))
    
    # Plot rule signals
    plt.subplot(4, 1, 1)
    plt.title('Rule Signals')
    plt.plot(rule_signals.index, rule_signals, label='Rule Signal')
    plt.plot(rule_signals.shift(1).fillna(0).index, rule_signals.shift(1).fillna(0), 
             label='Shifted Rule Signal', linestyle='--')
    plt.legend()
    plt.grid(True)
    
    # Plot strategy signals
    plt.subplot(4, 1, 2)
    plt.title('Strategy Signals')
    if isinstance(strategy_signals, pd.DataFrame) and 'Signal' in strategy_signals.columns:
        strat_signal = strategy_signals['Signal']
    else:
        strat_signal = strategy_signals
    plt.plot(strat_signal.index, strat_signal, label='Strategy Signal')
    plt.plot(strat_signal.shift(1).fillna(0).index, strat_signal.shift(1).fillna(0), 
             label='Shifted Strategy Signal', linestyle='--')
    plt.legend()
    plt.grid(True)
    
    # Plot backtester signals if available
    plt.subplot(4, 1, 3)
    plt.title('Backtester Signals')
    if backtester_signals is not None:
        if isinstance(backtester_signals, pd.DataFrame) and 'Signal' in backtester_signals.columns:
            back_signal = backtester_signals['Signal']
        else:
            back_signal = backtester_signals
        plt.plot(back_signal.index, back_signal, label='Backtester Signal')
        plt.plot(back_signal.shift(1).fillna(0).index, back_signal.shift(1).fillna(0), 
                 label='Shifted Backtester Signal', linestyle='--')
    else:
        plt.text(0.5, 0.5, 'No backtester signals available', 
                 horizontalalignment='center', verticalalignment='center')
    plt.legend()
    plt.grid(True)
    
    # Plot returns
    plt.subplot(4, 1, 4)
    plt.title('Signal Application to Returns')
    
    # Rule returns (both shifted and unshifted)
    rule_unshifted = (rule_signals * log_returns).cumsum()
    rule_shifted = (rule_signals.shift(1).fillna(0) * log_returns).cumsum()
    plt.plot(rule_unshifted.index, rule_unshifted, label='Rule Unshifted (Biased)')
    plt.plot(rule_shifted.index, rule_shifted, label='Rule Shifted (Correct)')
    
    # Strategy returns if available
    if isinstance(strategy_signals, pd.DataFrame) and 'Signal' in strategy_signals.columns:
        strat_signal = strategy_signals['Signal']
        strat_shifted = (strat_signal.shift(1).fillna(0) * log_returns).cumsum()
        plt.plot(strat_shifted.index, strat_shifted, label='Strategy Returns', linestyle='-.')
    
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('signal_trace_visualization.png')
    print("\nVisualization saved to 'signal_trace_visualization.png'")

# Import inspection tools
import inspect

# Main execution
def main():
    # Load configuration
    config = Config()
    data_handler = DataHandler(config)

    # Try to load data
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

    # Load rule parameters
    rule_params, rule_indices, rule_scores = load_rule_params()

    if not rule_params or not rule_indices:
        print("Could not load rule parameters. Using default parameters.")
        # Default to Rule0 with basic parameters
        rule_params = [(5, 50)]
        rule_indices = [0]

    # Select a specific rule to trace
    rule_idx = rule_indices[0]  # Use the first rule
    rule_param = rule_params[0]
    
    # Get the rule function
    rule_functions = [
        rules.Rule0, rules.Rule1, rules.Rule2, rules.Rule3, rules.Rule4,
        rules.Rule5, rules.Rule6, rules.Rule7, rules.Rule8, rules.Rule9,
        rules.Rule10, rules.Rule11, rules.Rule12, rules.Rule13, rules.Rule14, rules.Rule15
    ]
    
    if 0 <= rule_idx < len(rule_functions):
        rule_func = rule_functions[rule_idx]
    else:
        print(f"Invalid rule index {rule_idx}. Using Rule0.")
        rule_func = rules.Rule0
        rule_param = (5, 50)
    
    # Get a sample of test data (last 30 days for better visualization)
    sample_data = data_handler.test_data.iloc[-30:].copy()
    
    # 1. Trace rule execution
    rule_signals, log_returns = trace_rule_execution(rule_func, rule_param, sample_data)
    
    # 2. Create and initialize strategy
    config.use_weights = True  # Set to match your actual configuration
    strategy = StrategyFactory.create_strategy(config)
    
    # Load parameters into strategy's rule system
    if hasattr(strategy, 'rules'):
        strategy.rules.best_params = rule_params
        strategy.rules.best_indices = rule_indices
        if hasattr(strategy.rules, 'weights') and rule_scores:
            strategy.rules.weights = rule_scores
        
        # Add a default weights attribute if missing and needed
        if not hasattr(strategy.rules, 'weights'):
            setattr(strategy.rules, 'weights', [1.0 / len(rule_params)] * len(rule_params))
    
    # 3. Trace strategy execution
    strategy_signals = trace_strategy_execution(strategy, sample_data)
    
    # 4. Trace backtester execution
    backtester_signals, returns_df = trace_backtester_execution(config, data_handler, strategy)
    
    # 5. Create visualization
    if rule_signals is not None and isinstance(rule_signals, (pd.Series, np.ndarray)):
        create_signal_trace_visualization(rule_signals, strategy_signals, backtester_signals, log_returns)
    
    # 6. Analyze findings
    print("\n" + "="*80)
    print("SIGNAL TRACING ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nKey files and their signal shifting implementation:")
    
    # Check rules.py implementation
    print("\n1. rules.py - Signal Generation:")
    with open(os.path.join(src_dir, 'rules.py'), 'r') as f:
        code = f.read()
        
    # Look for signal shifting patterns
    rule_shifts = code.count('signals.shift(1)')
    log_return_calcs = code.count('np.log(') + code.count('log_returns =')
    
    print(f"  - Signal shifts: {rule_shifts}")
    print(f"  - Log return calculations: {log_return_calcs}")
    
    # Check generate_signals in rules.py specifically
    try:
        rules_code = inspect.getsource(RuleSystem.generate_signals)
        print("\nRuleSystem.generate_signals implementation:")
        print(rules_code)
    except:
        print("Could not inspect RuleSystem.generate_signals")
    
    # Check backtester.py implementation
    print("\n2. backtester.py - Performance Calculation:")
    with open(os.path.join(src_dir, 'backtester.py'), 'r') as f:
        code = f.read()
        
    # Look for signal shifting patterns
    backtester_shifts = code.count('signals.shift(1)')
    backtester_shifting_blocks = code.count('# Shift signals')
    
    print(f"  - Signal shifts: {backtester_shifts}")
    print(f"  - Signal shifting blocks: {backtester_shifting_blocks}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print("\nBased on this trace analysis:")
    
    # Look for double-shifting
    if rule_shifts > 0 and backtester_shifts > 0:
        print("1. POTENTIAL ISSUE DETECTED: Signals may be shifted in both rules.py and backtester.py")
        print("   This could cause double-shifting and affect your lookahead bias tests.")
        print("\n   Recommendation: Ensure signals are shifted exactly once before being applied to returns.")
    
    # Check log return calculations
    if log_return_calcs > rule_shifts:
        print("2. Log returns are calculated multiple times in different places.")
        print("   This might lead to inconsistencies if calculations differ.")
        print("\n   Recommendation: Standardize log return calculation.")
    
    print("\nNext steps:")
    print("1. Check for internal shifting within each rule function")
    print("2. Verify that backtester._calculate_performance properly shifts signals")
    print("3. Check that generated signals are not pre-shifted when passed to performance calculation")
    
    print("\nRemember: The goal is to ensure signals from time t are only applied to returns at time t+1.")

if __name__ == "__main__":
    main()

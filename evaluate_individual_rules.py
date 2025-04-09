"""
Script to evaluate the performance of individual trading rules.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import components from the backtesting engine
from config import Config
from data import DataHandler
from rules import TradingRules
from metrics import calculate_metrics, print_metrics

def evaluate_individual_rules(data_file='data.csv', params_file='params.json'):
    """
    Evaluate the performance of each individual trading rule using the trained parameters.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
    """
    print(f"Evaluating individual rule performance using {params_file}")
    
    # Create a default configuration
    config = Config()
    config.data_file = data_file
    config.params_file = params_file
    
    # Load data
    data_handler = DataHandler(config)
    success = data_handler.load_data()
    if not success:
        print(f"Failed to load data from {data_file}")
        return
    
    data_handler.preprocess()
    data_handler.split_data()
    
    # Load rule parameters
    rules = TradingRules()
    success = rules.load_params(params_file)
    if not success:
        print(f"Failed to load parameters from {params_file}")
        return
    
    # Get test data
    test_ohlc = data_handler.get_ohlc(train=False)
    
    # Extract returns for buy and hold comparison
    _, _, _, close = test_ohlc
    logr = np.log(close/close.shift(1))
    
    # Calculate buy and hold performance
    bh_returns = logr.dropna()
    bh_metrics = calculate_metrics(bh_returns)
    print_metrics(bh_metrics, "Buy and Hold Performance")
    
    # Evaluate each rule individually
    rule_metrics = []
    rule_names = []
    
    print("\nEvaluating individual rules:")
    print("=" * 50)
    
    for i, rule_func in enumerate(rules.rule_functions):
        rule_name = f"Rule{i+1}"
        print(f"\nEvaluating {rule_name}...")
        
        try:
            # Get signal from this rule only
            _, signal = rule_func(rules.rule_params[i], test_ohlc)
            
            # Create a signals dataframe
            signals_df = pd.DataFrame(index=close.index)
            signals_df['LogReturn'] = logr
            signals_df['Signal'] = signal
            
            # Clean up NaNs
            signals_df.dropna(inplace=True)
            
            # Calculate strategy returns (with 1-day delay to avoid lookahead bias)
            strategy_returns = signals_df['Signal'].shift(1) * signals_df['LogReturn']
            strategy_returns = strategy_returns.dropna()
            
            # Calculate trades
            trades = (signals_df['Signal'].diff() != 0).sum() / 2
            
            # Calculate metrics
            metrics = calculate_metrics(strategy_returns)
            metrics['number_of_trades'] = trades
            
            # Print metrics
            print_metrics(metrics, f"{rule_name} Performance")
            
            # Store for later comparison
            rule_metrics.append(metrics)
            rule_names.append(rule_name)
            
        except Exception as e:
            print(f"Error evaluating {rule_name}: {str(e)}")
    
    # Compare all rules
    from metrics import compare_strategies
    compare_strategies(rule_metrics, rule_names)
    
    # Also calculate combined strategy metrics based on top rules
    top_n = 3  # Use top-3 rules as in the results.json
    
    if hasattr(rules, 'rule_indices'):
        top_indices = rules.rule_indices[:top_n]
        print(f"\nEvaluating Top-{top_n} Rules Strategy (Rules {[i+1 for i in top_indices]})")
        
        # Generate signals for top rules
        signals_df = rules.generate_signals(test_ohlc, rules.rule_params, top_n=top_n)
        
        # Calculate strategy returns (with 1-day delay)
        strategy_returns = signals_df['Signal'].shift(1) * signals_df['LogReturn']
        strategy_returns = strategy_returns.dropna()
        
        # Calculate metrics
        metrics = calculate_metrics(strategy_returns)
        metrics['number_of_trades'] = (signals_df['Signal'].diff() != 0).sum() / 2
        
        # Print metrics
        print_metrics(metrics, f"Top-{top_n} Rules Strategy Performance")
    
if __name__ == "__main__":
    evaluate_individual_rules()

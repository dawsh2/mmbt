"""
Simple script to print metrics for each individual trading rule.
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

def print_rule_metrics(data_file='data.csv', params_file='params.json'):
    """
    Print performance metrics for each individual rule.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
    """
    print(f"Analyzing rule metrics from {params_file}")
    
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
    
    # Print rule parameters and scores
    print("\nRule parameters and scores from params.json:")
    print("=" * 60)
    
    # Access rule parameters and scores
    rule_params = rules.rule_params
    rule_scores = rules.rule_scores if hasattr(rules, 'rule_scores') else None
    rule_indices = rules.rule_indices.tolist() if hasattr(rules, 'rule_indices') else None
    
    # Print parameters and scores
    for i, params in enumerate(rule_params):
        rule_name = f"Rule{i+1}"
        score = rule_scores[i] if rule_scores is not None else "unknown"
        rank = rule_indices.index(i) + 1 if rule_indices is not None else "unknown"
        
        print(f"{rule_name}: params={params}, score={score:.6f}, rank={rank}")
    
    # Find which rules are used in the top-N strategy
    if rule_indices is not None:
        top_n = 3  # Based on your results.json
        top_rules = [f"Rule{i+1}" for i in rule_indices[:top_n]]
        print(f"\nTop {top_n} rules used in strategy: {top_rules}")
    
    # Get test data
    test_ohlc = data_handler.get_ohlc(train=False)
    
    # Calculate individual rule signals
    print("\nRule signal distribution:")
    print("=" * 60)
    
    for i, rule_func in enumerate(rules.rule_functions):
        rule_name = f"Rule{i+1}"
        
        try:
            # Get signal from this rule
            _, signal = rule_func(rule_params[i], test_ohlc)
            
            # Calculate signal distribution
            if isinstance(signal, pd.Series):
                buy_signals = (signal == 1).sum()
                sell_signals = (signal == -1).sum()
                neutral_signals = (signal == 0).sum()
                total_signals = len(signal.dropna())
                
                # Print distribution
                print(f"{rule_name}: Buy={buy_signals} ({buy_signals/total_signals*100:.1f}%), "
                      f"Sell={sell_signals} ({sell_signals/total_signals*100:.1f}%), "
                      f"Neutral={neutral_signals} ({neutral_signals/total_signals*100:.1f}%)")
        except Exception as e:
            print(f"{rule_name}: Error calculating signals - {str(e)}")
    
    # Load results from results.json if available
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
            
        print("\nStrategy performance from results.json:")
        print("=" * 60)
        print(f"Strategy: {results['strategy']}")
        
        # Print strategy metrics
        strategy_metrics = results['performance']['strategy']
        for metric, value in strategy_metrics.items():
            if metric in ['win_rate', 'total_return', 'annualized_return', 'max_drawdown']:
                print(f"{metric}: {value*100:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
        
        # Print buy and hold metrics
        bh_metrics = results['performance']['buy_and_hold']
        print("\nBuy and Hold metrics from results.json:")
        print("=" * 60)
        
        for metric, value in bh_metrics.items():
            if metric in ['win_rate', 'total_return', 'annualized_return', 'max_drawdown']:
                print(f"{metric}: {value*100:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
                
    except Exception as e:
        print(f"\nCould not load results.json: {str(e)}")

if __name__ == "__main__":
    print_rule_metrics()

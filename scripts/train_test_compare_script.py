"""
Script to compare training and testing performance of individual rules vs. top-N strategy.
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

def compare_train_test_performance(data_file='data.csv', params_file='params.json'):
    """
    Compare training and testing performance of individual rules vs. top-N strategy.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
    """
    print(f"Comparing training and testing performance using {params_file}")
    
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
    
    # Get training and testing data
    train_ohlc = data_handler.get_ohlc(train=True)
    test_ohlc = data_handler.get_ohlc(train=False)
    
    # Extract log returns
    _, _, _, train_close = train_ohlc
    _, _, _, test_close = test_ohlc
    train_logr = np.log(train_close/train_close.shift(1))
    test_logr = np.log(test_close/test_close.shift(1))
    
    # Create results container
    results = {
        'training': {
            'individual_rules': [],
            'top_n': None,
            'buy_hold': None
        },
        'testing': {
            'individual_rules': [],
            'top_n': None,
            'buy_hold': None
        }
    }
    
    # Calculate buy and hold performance
    train_bh = train_logr.dropna()
    test_bh = test_logr.dropna()
    
    results['training']['buy_hold'] = calculate_metrics(train_bh)
    results['testing']['buy_hold'] = calculate_metrics(test_bh)
    
    print("\nBuy and Hold Performance:")
    print("-" * 50)
    print("Training: Total Return: {:.2%}, Sharpe: {:.4f}".format(
        results['training']['buy_hold']['total_return'],
        results['training']['buy_hold']['sharpe_ratio']
    ))
    print("Testing: Total Return: {:.2%}, Sharpe: {:.4f}".format(
        results['testing']['buy_hold']['total_return'],
        results['testing']['buy_hold']['sharpe_ratio']
    ))
    
    # Evaluate each rule individually
    for i, rule_func in enumerate(rules.rule_functions):
        rule_name = f"Rule{i+1}"
        print(f"\nEvaluating {rule_name}...")
        
        rule_results = {'name': rule_name, 'train': None, 'test': None}
        
        try:
            # Training performance
            _, train_signal = rule_func(rules.rule_params[i], train_ohlc)
            
            train_signal_df = pd.DataFrame(index=train_close.index)
            train_signal_df['LogReturn'] = train_logr
            train_signal_df['Signal'] = train_signal
            train_signal_df.dropna(inplace=True)
            
            train_strategy_returns = train_signal_df['Signal'].shift(1) * train_signal_df['LogReturn']
            train_strategy_returns = train_strategy_returns.dropna()
            
            # Calculate metrics for training
            train_metrics = calculate_metrics(train_strategy_returns)
            results['training']['individual_rules'].append({
                'name': rule_name,
                'metrics': train_metrics
            })
            
            # Testing performance
            _, test_signal = rule_func(rules.rule_params[i], test_ohlc)
            
            test_signal_df = pd.DataFrame(index=test_close.index)
            test_signal_df['LogReturn'] = test_logr
            test_signal_df['Signal'] = test_signal
            test_signal_df.dropna(inplace=True)
            
            test_strategy_returns = test_signal_df['Signal'].shift(1) * test_signal_df['LogReturn']
            test_strategy_returns = test_strategy_returns.dropna()
            
            # Calculate metrics for testing
            test_metrics = calculate_metrics(test_strategy_returns)
            results['testing']['individual_rules'].append({
                'name': rule_name,
                'metrics': test_metrics
            })
            
            # Print key metrics
            print("  Training: Total Return: {:.2%}, Sharpe: {:.4f}".format(
                train_metrics['total_return'],
                train_metrics['sharpe_ratio']
            ))
            print("  Testing: Total Return: {:.2%}, Sharpe: {:.4f}".format(
                test_metrics['total_return'],
                test_metrics['sharpe_ratio']
            ))
            
        except Exception as e:
            print(f"  Error evaluating {rule_name}: {str(e)}")
    
    # Calculate top-N strategy performance
    try:
        top_n = 3  # Use top-3 rules as in the results.json
        
        if hasattr(rules, 'rule_indices'):
            top_indices = rules.rule_indices[:top_n]
            top_rules = [i+1 for i in top_indices]
            print(f"\nEvaluating Top-{top_n} Rules Strategy (Rules {top_rules})...")
            
            # Training performance
            train_signals_df = rules.generate_signals(train_ohlc, rules.rule_params, top_n=top_n)
            train_strategy_returns = train_signals_df['Signal'].shift(1) * train_signals_df['LogReturn']
            train_strategy_returns = train_strategy_returns.dropna()
            
            # Calculate metrics for training
            train_metrics = calculate_metrics(train_strategy_returns)
            results['training']['top_n'] = train_metrics
            
            # Testing performance
            test_signals_df = rules.generate_signals(test_ohlc, rules.rule_params, top_n=top_n)
            test_strategy_returns = test_signals_df['Signal'].shift(1) * test_signals_df['LogReturn']
            test_strategy_returns = test_strategy_returns.dropna()
            
            # Calculate metrics for testing
            test_metrics = calculate_metrics(test_strategy_returns)
            results['testing']['top_n'] = test_metrics
            
            # Print key metrics
            print("  Training: Total Return: {:.2%}, Sharpe: {:.4f}".format(
                train_metrics['total_return'],
                train_metrics['sharpe_ratio']
            ))
            print("  Testing: Total Return: {:.2%}, Sharpe: {:.4f}".format(
                test_metrics['total_return'],
                test_metrics['sharpe_ratio']
            ))
            
            # Check if individual rules outperform the combined strategy
            print("\nDo individual rules outperform the combined strategy?")
            print("-" * 60)
            
            # Training outperformers
            train_outperformers = []
            for rule in results['training']['individual_rules']:
                if rule['metrics']['total_return'] > train_metrics['total_return']:
                    train_outperformers.append({
                        'name': rule['name'],
                        'return': rule['metrics']['total_return'],
                        'outperformance': rule['metrics']['total_return'] - train_metrics['total_return']
                    })
            
            if train_outperformers:
                print("Rules outperforming combined strategy on TRAINING data:")
                for rule in sorted(train_outperformers, key=lambda x: x['outperformance'], reverse=True):
                    print(f"  {rule['name']}: {rule['return']:.2%} (better by {rule['outperformance']:.2%})")
            else:
                print("No individual rules outperform the combined strategy on training data.")
                
            # Testing outperformers
            test_outperformers = []
            for rule in results['testing']['individual_rules']:
                if rule['metrics']['total_return'] > test_metrics['total_return']:
                    test_outperformers.append({
                        'name': rule['name'],
                        'return': rule['metrics']['total_return'],
                        'outperformance': rule['metrics']['total_return'] - test_metrics['total_return']
                    })
            
            if test_outperformers:
                print("\nRules outperforming combined strategy on TESTING data:")
                for rule in sorted(test_outperformers, key=lambda x: x['outperformance'], reverse=True):
                    print(f"  {rule['name']}: {rule['return']:.2%} (better by {rule['outperformance']:.2%})")
            else:
                print("\nNo individual rules outperform the combined strategy on testing data.")
            
            # Check which rules in top N are underperforming
            top_rule_names = [f"Rule{i}" for i in top_rules]
            print("\nPerformance of selected top rules individually:")
            for rule in results['testing']['individual_rules']:
                if rule['name'] in top_rule_names:
                    diff = rule['metrics']['total_return'] - test_metrics['total_return']
                    print(f"  {rule['name']}: {rule['metrics']['total_return']:.2%} " + 
                         (f"(worse by {-diff:.2%})" if diff < 0 else f"(better by {diff:.2%})"))
                    
            # Check potential issues in the implementation
            print("\nPotential implementation issues to check:")
            print("-" * 60)
            print("1. Signal combination method (check rules.py, generate_signals method)")
            print("2. Top-N rule selection logic (check if the right rules are being used)")
            print("3. Signal application timing (ensure signals are applied with correct delay)")
            print("4. NaN handling in signal generation and performance calculation")
            
        else:
            print("Rule indices not available - cannot identify top rules.")
    
    except Exception as e:
        print(f"Error evaluating Top-N strategy: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    compare_train_test_performance()


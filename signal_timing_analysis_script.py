"""
Script to analyze signal timing issues in the trading strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import components from the backtesting engine
from config import Config
from data import DataHandler
from rules import TradingRules
from metrics import calculate_metrics

def analyze_signal_timing(data_file='data.csv', params_file='params.json'):
    """
    Analyze potential signal timing issues in the strategy.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
    """
    print(f"Analyzing signal timing using {params_file}")
    
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
    
    # Extract log returns
    _, _, _, test_close = test_ohlc
    test_logr = np.log(test_close/test_close.shift(1))
    
    # Identify top rules
    if hasattr(rules, 'rule_indices'):
        top_n = 3  # Use top-3 rules as in the results.json
        top_indices = rules.rule_indices[:top_n]
        top_rules = [i+1 for i in top_indices]
        print(f"Top {top_n} rules: {top_rules}")
        
        # Generate combined signals using standard method
        try:
            combined_signals_df = rules.generate_signals(test_ohlc, rules.rule_params, top_n=top_n)
            print(f"Generated combined signals for top {top_n} rules")
            
            # Calculate strategy returns with different delay settings
            
            # Approach 1: Standard approach (with 1-day delay)
            print("\nTesting different signal timing approaches:")
            print("-" * 50)
            
            # First, drop any NaN values
            clean_signals_df = combined_signals_df.dropna()
            
            # Standard approach: apply signals with 1-day delay
            standard_returns = clean_signals_df['Signal'].shift(1) * clean_signals_df['LogReturn']
            standard_returns = standard_returns.dropna()
            standard_metrics = calculate_metrics(standard_returns)
            
            print("1. Standard approach (1-day delay):")
            print(f"   Total Return: {standard_metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {standard_metrics['sharpe_ratio']:.4f}")
            
            # Approach 2: No delay (lookahead bias, for debugging)
            debug_returns = clean_signals_df['Signal'] * clean_signals_df['LogReturn']
            debug_metrics = calculate_metrics(debug_returns)
            
            print("\n2. No delay (has lookahead bias, for debugging only):")
            print(f"   Total Return: {debug_metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {debug_metrics['sharpe_ratio']:.4f}")
            
            # Approach 3: Generate signals for each rule first, then combine and apply delay
            print("\n3. Generate individual rule signals first, then combine with delay:")
            
            # Create a DataFrame to hold individual rule signals
            rule_signals_df = pd.DataFrame(index=test_close.index)
            rule_signals_df['LogReturn'] = test_logr
            
            # Get signals from each top rule
            for i in top_indices:
                rule_func = rules.rule_functions[i]
                rule_name = f"Rule{i+1}"
                _, signal = rule_func(rules.rule_params[i], test_ohlc)
                rule_signals_df[rule_name] = signal
            
            # Drop NaN values first
            rule_signals_df.dropna(inplace=True)
            
            # Combine signals manually - properly handling pandas Series
            combined_signal = pd.Series(0, index=rule_signals_df.index)
            for i in top_indices:
                combined_signal += rule_signals_df[f"Rule{i+1}"]
            
            rule_signals_df['CombinedSignal'] = np.sign(combined_signal)
            
            # Apply signals with 1-day delay
            manual_returns = rule_signals_df['CombinedSignal'].shift(1) * rule_signals_df['LogReturn']
            manual_returns = manual_returns.dropna()
            manual_metrics = calculate_metrics(manual_returns)
            
            print(f"   Total Return: {manual_metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {manual_metrics['sharpe_ratio']:.4f}")
            
            # Individual rule performance
            print("\nIndividual rule performance:")
            print("-" * 50)
            
            # Print performance of each rule independently
            for i in top_indices:
                rule_name = f"Rule{i+1}"
                single_returns = rule_signals_df[rule_name].shift(1) * rule_signals_df['LogReturn']
                single_returns = single_returns.dropna()
                single_metrics = calculate_metrics(single_returns)
                
                print(f"{rule_name}:")
                print(f"   Total Return: {single_metrics['total_return']:.2%}")
                print(f"   Sharpe Ratio: {single_metrics['sharpe_ratio']:.4f}")
                
                # Compare with combined strategy
                if single_metrics['total_return'] > manual_metrics['total_return']:
                    print(f"   OUTPERFORMS combined strategy by {single_metrics['total_return'] - manual_metrics['total_return']:.2%}")
                else:
                    print(f"   UNDERPERFORMS combined strategy by {manual_metrics['total_return'] - single_metrics['total_return']:.2%}")
            
            # Print performance of all other rules for comparison
            print("\nPerformance of other rules:")
            print("-" * 50)
            
            for i in range(len(rules.rule_functions)):
                if i not in top_indices:
                    rule_func = rules.rule_functions[i]
                    rule_name = f"Rule{i+1}"
                    try:
                        _, signal = rule_func(rules.rule_params[i], test_ohlc)
                        other_df = pd.DataFrame(index=test_close.index)
                        other_df['LogReturn'] = test_logr
                        other_df['Signal'] = signal
                        other_df.dropna(inplace=True)
                        
                        other_returns = other_df['Signal'].shift(1) * other_df['LogReturn']
                        other_returns = other_returns.dropna()
                        other_metrics = calculate_metrics(other_returns)
                        
                        print(f"{rule_name}:")
                        print(f"   Total Return: {other_metrics['total_return']:.2%}")
                        print(f"   Sharpe Ratio: {other_metrics['sharpe_ratio']:.4f}")
                        
                        # Compare with combined strategy
                        if other_metrics['total_return'] > manual_metrics['total_return']:
                            print(f"   OUTPERFORMS combined strategy by {other_metrics['total_return'] - manual_metrics['total_return']:.2%}")
                        else:
                            print(f"   UNDERPERFORMS combined strategy by {manual_metrics['total_return'] - other_metrics['total_return']:.2%}")
                    except Exception as e:
                        print(f"{rule_name}: Error calculating performance - {str(e)}")
            
            # Compare signal distributions
            print("\nComparing signal distributions:")
            print("-" * 50)
            
            # Distribution in standard approach
            std_buy = (clean_signals_df['Signal'] == 1).sum()
            std_sell = (clean_signals_df['Signal'] == -1).sum()
            std_neutral = (clean_signals_df['Signal'] == 0).sum()
            std_total = std_buy + std_sell + std_neutral
            
            print("Standard combined signals:")
            print(f"Buy: {std_buy} ({std_buy/std_total*100:.1f}%), " +
                  f"Sell: {std_sell} ({std_sell/std_total*100:.1f}%), " +
                  f"Neutral: {std_neutral} ({std_neutral/std_total*100:.1f}%)")
            
            # Distribution in manual approach
            man_buy = (rule_signals_df['CombinedSignal'] == 1).sum()
            man_sell = (rule_signals_df['CombinedSignal'] == -1).sum()
            man_neutral = (rule_signals_df['CombinedSignal'] == 0).sum()
            man_total = man_buy + man_sell + man_neutral
            
            print("\nManually combined signals:")
            print(f"Buy: {man_buy} ({man_buy/man_total*100:.1f}%), " +
                  f"Sell: {man_sell} ({man_sell/man_total*100:.1f}%), " +
                  f"Neutral: {man_neutral} ({man_neutral/man_total*100:.1f}%)")
            
            # Calculate confusion matrix between methods
            common_index = rule_signals_df.index.intersection(clean_signals_df.index)
            if len(common_index) > 0:
                std_signals = clean_signals_df.loc[common_index, 'Signal']
                man_signals = rule_signals_df.loc[common_index, 'CombinedSignal']
                
                agreement = (std_signals == man_signals).sum()
                disagreement = (std_signals != man_signals).sum()
                agreement_pct = agreement / (agreement + disagreement) * 100
                
                print(f"\nSignal agreement between methods: {agreement_pct:.1f}%")
                if agreement_pct < 100:
                    print("This suggests a difference in how signals are combined.")
                    
                    # Show sample of disagreements
                    if disagreement > 0:
                        disagree_df = pd.DataFrame({
                            'Standard': std_signals,
                            'Manual': man_signals
                        })
                        disagree_df = disagree_df[disagree_df['Standard'] != disagree_df['Manual']]
                        print("\nSample of disagreements:")
                        print(disagree_df.head(5))
                        
                        # Check if some rules dominate others consistently
                        print("\nChecking for rule domination patterns:")
                        # Get individual rule signals for the disagreement cases
                        rule_signals = pd.DataFrame(index=disagree_df.index)
                        for i in top_indices:
                            rule_name = f"Rule{i+1}"
                            rule_signals[rule_name] = rule_signals_df.loc[disagree_df.index, rule_name]
                        
                        # Join with disagree_df
                        analysis_df = pd.concat([disagree_df, rule_signals], axis=1)
                        
                        # Print first few rows for examination
                        print(analysis_df.head(5))
                        
                        # Look for patterns where one rule consistently overrides others
                        for i in top_indices:
                            rule_name = f"Rule{i+1}"
                            agreement_with_rule = (analysis_df['Manual'] == analysis_df[rule_name]).sum()
                            agreement_pct = agreement_with_rule / len(analysis_df) * 100
                            print(f"{rule_name} agrees with manual combined signal: {agreement_pct:.1f}%")
            
            # Check for issues with rules.generate_signals method
            print("\nPotential issues to investigate:")
            print("-" * 50)
            print("1. Check if the correct rule indices are being used in top_n selection")
            print("2. Review how signals are combined in the rules.generate_signals method")
            print("3. Check if there are timing differences in when signals are generated vs applied")
            print("4. Verify that the weights or voting mechanism is working correctly")
            
        except Exception as e:
            print(f"Error during signal timing analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Rule indices not available - cannot identify top rules.")

if __name__ == "__main__":
    analyze_signal_timing()


"""
Script to visualize the performance of individual trading rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import components from the backtesting engine
from config import Config
from data import DataHandler
from rules import TradingRules
from metrics import calculate_metrics, print_metrics

def visualize_rule_performance(data_file='data.csv', params_file='params.json', output_file='rule_performance.png'):
    """
    Visualize the cumulative performance of each individual trading rule.
    
    Args:
        data_file: Path to the CSV data file
        params_file: Path to the JSON file containing trained parameters
        output_file: Path to save the output visualization
    """
    print(f"Visualizing rule performance using {params_file}")
    
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
    
    # Create dataframe to store cumulative returns
    perf_df = pd.DataFrame(index=close.index)
    
    # Add buy and hold performance
    perf_df['Buy_and_Hold'] = np.exp(logr.cumsum()) - 1
    
    # Calculate top-n strategy performance
    top_n = 3  # Use top-3 rules as in the results.json
    
    if hasattr(rules, 'rule_indices'):
        top_indices = rules.rule_indices[:top_n]
        print(f"Top-{top_n} Rules: {[i+1 for i in top_indices]}")
        
        # Generate signals for top rules
        signals_df = rules.generate_signals(test_ohlc, rules.rule_params, top_n=top_n)
        
        # Calculate strategy returns (with 1-day delay)
        strategy_returns = signals_df['Signal'].shift(1) * signals_df['LogReturn']
        perf_df['Top_N_Strategy'] = np.exp(strategy_returns.cumsum()) - 1
    
    # Evaluate each rule individually
    for i, rule_func in enumerate(rules.rule_functions):
        rule_name = f"Rule{i+1}"
        print(f"Processing {rule_name}...")
        
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
            
            # Calculate cumulative returns
            perf_df[rule_name] = np.exp(strategy_returns.cumsum()) - 1
            
        except Exception as e:
            print(f"Error processing {rule_name}: {str(e)}")
    
    # Drop NaN values
    perf_df = perf_df.dropna()
    
    # Calculate correlations between rule returns
    correlation_matrix = perf_df.pct_change().corr()
    print("\nCorrelation matrix between rules:")
    print(correlation_matrix)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot buy and hold
    plt.plot(perf_df.index, perf_df['Buy_and_Hold'] * 100, 'k-', linewidth=2, label='Buy and Hold')
    
    # Plot individual rules
    colors = plt.cm.jet(np.linspace(0, 1, len(rules.rule_functions)))
    
    for i, rule_name in enumerate([f"Rule{i+1}" for i in range(len(rules.rule_functions))]):
        if rule_name in perf_df.columns:
            plt.plot(perf_df.index, perf_df[rule_name] * 100, '-', color=colors[i], alpha=0.5, linewidth=1, label=rule_name)
    
    # Plot top-n strategy
    if 'Top_N_Strategy' in perf_df.columns:
        plt.plot(perf_df.index, perf_df['Top_N_Strategy'] * 100, 'g-', linewidth=3, label=f'Top-{top_n} Strategy')
    
    plt.title('Cumulative Returns by Trading Rule (%)')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title('Correlation Between Rule Returns')
    plt.tight_layout()
    plt.savefig('rule_correlations.png')
    print("Correlation matrix visualization saved to rule_correlations.png")
    
if __name__ == "__main__":
    visualize_rule_performance()

"""
Validation script to verify return calculations in the trading system.
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def validate_returns(trades):
    """
    Validates that return calculations are correct.
    
    Args:
        trades: List of trade tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)
        
    Returns:
        bool: True if all return calculations are correct, False otherwise
    """
    all_correct = True
    
    for i, trade in enumerate(trades):
        entry_time, direction, entry_price, exit_time, exit_price, reported_log_return = trade
        
        # Calculate expected log return
        if direction == "BUY":
            expected_log_return = math.log(exit_price / entry_price)
        else:  # "SELL"
            expected_log_return = math.log(entry_price / exit_price)
        
        # Compare with reported log return
        if abs(expected_log_return - reported_log_return) > 1e-10:
            print(f"Error in trade {i}:")
            print(f"  Reported log return: {reported_log_return}")
            print(f"  Expected log return: {expected_log_return}")
            print(f"  Difference: {abs(expected_log_return - reported_log_return)}")
            all_correct = False
            
    if all_correct:
        print("All return calculations are correct!")
    
    return all_correct

def validate_total_returns(trades):
    """
    Validates that total return calculations are correct.
    
    Args:
        trades: List of trade tuples
        
    Returns:
        tuple: (reported_total_log_return, calculated_total_log_return, 
                reported_total_percent_return, calculated_total_percent_return)
    """
    # Extract individual log returns
    log_returns = [t[5] for t in trades]
    
    # Calculate total log return
    calculated_total_log_return = sum(log_returns)
    
    # Calculate total percent return
    calculated_total_percent_return = (math.exp(calculated_total_log_return) - 1) * 100
    
    return (calculated_total_log_return, calculated_total_percent_return)

def analyze_trade_distribution(trades):
    """
    Analyzes the distribution of trade returns.
    
    Args:
        trades: List of trade tuples
    """
    log_returns = [t[5] for t in trades]
    
    plt.figure(figsize=(10, 6))
    plt.hist(log_returns, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Calculate statistics
    win_rate = sum(1 for r in log_returns if r > 0) / len(log_returns) if log_returns else 0
    max_win = max(log_returns) if log_returns else 0
    max_loss = min(log_returns) if log_returns else 0
    avg_win = np.mean([r for r in log_returns if r > 0]) if any(r > 0 for r in log_returns) else 0
    avg_loss = np.mean([r for r in log_returns if r < 0]) if any(r < 0 for r in log_returns) else 0
    
    print(f"Number of trades: {len(log_returns)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average winning trade: {avg_win:.4f} ({(math.exp(avg_win) - 1) * 100:.2f}%)")
    print(f"Average losing trade: {avg_loss:.4f} ({(math.exp(avg_loss) - 1) * 100:.2f}%)")
    print(f"Largest winner: {max_win:.4f} ({(math.exp(max_win) - 1) * 100:.2f}%)")
    print(f"Largest loser: {max_loss:.4f} ({(math.exp(max_loss) - 1) * 100:.2f}%)")
    print(f"Profit factor: {abs(sum(r for r in log_returns if r > 0) / sum(r for r in log_returns if r < 0)) if sum(r for r in log_returns if r < 0) else 'Infinite'}")
    
    plt.show()

def analyze_equity_curve(trades):
    """
    Creates an equity curve from the trades.
    
    Args:
        trades: List of trade tuples
    """
    equity = [1.0]  # Start with $1
    dates = []
    
    if not trades:
        return
    
    for trade in trades:
        log_return = trade[5]
        equity.append(equity[-1] * math.exp(log_return))
        dates.append(trade[3])  # Exit date
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(equity)), equity)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_validation(results):
    """
    Run comprehensive validation on backtest results.
    
    Args:
        results: Results dictionary from backtester
    """
    trades = results["trades"]
    
    print("Validating individual trade returns...")
    validate_returns(trades)
    
    print("\nValidating total returns...")
    total_log_return, total_percent_return = validate_total_returns(trades)
    print(f"Calculated total log return: {total_log_return:.4f}")
    print(f"Reported total log return: {results['total_log_return']:.4f}")
    print(f"Calculated total percent return: {total_percent_return:.2f}%")
    print(f"Reported total percent return: {results['total_percent_return']:.2f}%")
    
    print("\nAnalyzing trade distribution...")
    analyze_trade_distribution(trades)
    
    print("\nCreating equity curve...")
    analyze_equity_curve(trades)

# Example usage (this would go in your main.py):
"""
# After running backtest:
results_oos = out_of_sample_backtester.run(use_test_data=True)

# Import and run validation
from validation import run_validation
run_validation(results_oos)
"""

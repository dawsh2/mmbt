#!/usr/bin/env python3
"""
Optimization test script that works with the fixed Rule base class.
"""

import datetime
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Import system components
from src.events import Event, EventType
from src.config import ConfigManager
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource
from src.rules import create_rule, Rule
from src.rules.rule_factory import RuleFactory
from src.optimization import OptimizerManager, OptimizationMethod
from src.signals import Signal, SignalType
from src.position_management.portfolio import Portfolio


# Import backtester components
from src.engine import Backtester
from src.engine.market_simulator import MarketSimulator
from src.strategies import WeightedStrategy, TopNStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None, days=365):
    """Create synthetic price data for testing."""
    # Create a synthetic dataset
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    # Create a price series with some randomness and a trend
    base_price = 100
    prices = [base_price]
    
    # Add a trend with noise
    for i in range(1, len(dates)):
        # Random daily change between -1% and +1% with a slight upward bias
        daily_change = np.random.normal(0.0005, 0.01) 
        
        # Add some regime changes to test rules
        if i % 60 == 0:  # Every ~2 months, change trend
            daily_change = -0.02 if prices[-1] > base_price * 1.1 else 0.02
            
        new_price = prices[-1] * (1 + daily_change)
        prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(100000, 20000)) for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} bars, saved to {filename}")
    
    return df

def plot_equity_curves(results, title):
    """
    Plot equity curves for multiple backtest results.
    
    Args:
        results: List of (name, results_dict, params) tuples
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for name, result, params in results:
        # Extract equity curve if available
        if 'equity_curve' in result:
            equity = result['equity_curve']
            plt.plot(equity, label=f"{name} ({result['total_percent_return']:.2f}%)")
        else:
            # If no equity curve, create a simple line from initial to final equity
            initial_equity = 100000  # Default from most configs
            final_equity = initial_equity * (1 + result.get('total_percent_return', 0) / 100)
            plt.plot([0, 1], [initial_equity, final_equity], label=f"{name} ({result.get('total_percent_return', 0):.2f}%)")
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("equity_curves.png")
    logger.info("Saved equity curves plot to equity_curves.png")

def debug_rsi_rule_behavior(data_handler, param_sets=None):
    """
    Debug RSI rule behavior with different parameter sets.
    
    Args:
        data_handler: DataHandler with loaded data
        param_sets: List of parameter dictionaries to test
    """
    if param_sets is None:
        # Test a few diverse parameter sets
        param_sets = [
            {'rsi_period': 7, 'overbought': 70, 'oversold': 30, 'signal_type': 'levels'},
            {'rsi_period': 14, 'overbought': 70, 'oversold': 30, 'signal_type': 'levels'},
            {'rsi_period': 21, 'overbought': 80, 'oversold': 20, 'signal_type': 'levels'}
        ]
    
    # Create a dictionary to store results for each parameter set
    results = {}
    
    # Test each parameter set
    for params in param_sets:
        param_key = f"RSI-{params['rsi_period']}-{params['overbought']}-{params['oversold']}"
        
        # Create the rule with these parameters
        rule = create_rule('RSIRule', params)
        
        # Create a strategy with just this rule
        strategy = TopNStrategy(rule_objects=[rule])
        
        # Create market simulator
        market_simulator = MarketSimulator({
            'slippage_model': 'fixed',
            'slippage_bps': 5
        })
        
        # Create config for the backtester
        config = ConfigManager()
        config.set('backtester.initial_capital', 100000)
        
        # Create backtester with all required parameters
        backtester = Backtester(config, data_handler, strategy)
        backtester.market_simulator = market_simulator
        
        # Run backtest
        backtest_results = backtester.run()
        
        # Calculate metrics
        sharpe = backtester.calculate_sharpe()
        
        # Store in results
        results[param_key] = {
            'params': params,
            'backtest_results': backtest_results,
            'sharpe': sharpe,
            'total_return': backtest_results['total_log_return'] * 100
        }
    
    # Print summary
    print("\n=== RSI RULE BEHAVIOR DEBUG ===")
    print(f"Tested {len(param_sets)} parameter sets")
    
    for key, result in results.items():
        params = result['params']
        print(f"\nParameters: period={params['rsi_period']}, overbought={params['overbought']}, oversold={params['oversold']}")
        print(f"  - Total Return: {result['total_return']:.2f}%")
        print(f"  - Sharpe Ratio: {result['sharpe']:.2f}")
        print(f"  - Number of Trades: {result['backtest_results']['num_trades']}")
    
    return results

def run_backtest(rule, data_handler, test_data=False):
    """
    Run a backtest for a single rule using the system's Backtester.
    
    Args:
        rule: Rule instance to test
        data_handler: DataHandler with loaded data
        test_data: Whether to use test data (True) or training data (False)
        
    Returns:
        dict: Results dictionary with performance metrics
    """
    # Print rule information for debugging
    print(f"\nBacktesting rule: {rule.name} with params: {rule.params}")
    
    # Create a strategy with just this rule
    strategy = TopNStrategy(rule_objects=[rule])
    
    # Create market simulator with standard settings
    market_simulator = MarketSimulator({
        'slippage_model': 'fixed',
        'slippage_bps': 5
    })
    
    # Create config for backtester
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    
    # Create backtester with all required parameters
    backtester = Backtester(config, data_handler, strategy)
    backtester.market_simulator = market_simulator
    
    # Run backtest
    results = backtester.run(use_test_data=test_data)
    
    # Calculate additional metrics
    sharpe = backtester.calculate_sharpe()
    
    # Convert log returns to percentage returns for easier reading
    total_percent_return = (np.exp(results['total_log_return']) - 1) * 100
    
    # Add additional metrics to results
    results['sharpe_ratio'] = sharpe
    results['total_percent_return'] = total_percent_return
    
    # Print summary
    print(f"Backtest results for {rule.name}:")
    print(f"  - Total Return: {total_percent_return:.2f}%")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")
    print(f"  - Number of Trades: {results['num_trades']}")
    
    # Get trade statistics
    if results['num_trades'] > 0:
        win_count = sum(1 for trade in results['trades'] if trade[5] > 0)
        win_rate = win_count / results['num_trades'] * 100
        print(f"  - Win Rate: {win_rate:.2f}%")
    
    return results

def main():
    """Main function to demonstrate rule usage and parameter optimization."""
    logger.info("Starting rule testing and optimization")
    
    # 1. Create or load synthetic data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    
    # Create filename following the convention
    standard_filename = f"{symbol}_{timeframe}.csv"
    
    # Create synthetic data if it doesn't exist
    if not os.path.exists(standard_filename):
        create_synthetic_data(symbol, timeframe, standard_filename, days=500)
    
    # 2. Create configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    
    # 3. Set up data sources and handler
    data_source = CSVDataSource(".")  # Look for CSV files in current directory
    data_handler = DataHandler(data_source)
    
    # 4. Load market data
    logger.info("Loading market data")
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)  # Adjust based on your data
    
    # Load data
    data_handler.load_data(
        symbols=[symbol],  # DataHandler expects a list
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    logger.info(f"Successfully loaded {len(data_handler.train_data)} bars of data")
    
    # 5. Split data into training and testing sets (80/20 split)
    train_size = int(len(data_handler.train_data) * 0.8)
    
    # Modify data_handler to split the data
    test_data = data_handler.train_data.iloc[train_size:].copy()
    data_handler.train_data = data_handler.train_data.iloc[:train_size].copy()
    data_handler.test_data = test_data
    
    logger.info(f"Split data into {len(data_handler.train_data)} training bars and {len(data_handler.test_data)} testing bars")
    
    # 6. Create rules to test
    logger.info("Creating test rules")
    
    # A. SMA Crossover rule with default parameters
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 10,
        'slow_window': 30,
        'smooth_signals': True
    })
    
    # B. RSI rule with default parameters
    rsi_rule = create_rule('RSIRule', {
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30,
        'signal_type': 'levels'  # Make sure to include this
    })
    
    # 7. Debug RSI rule behavior with different parameters
    logger.info("Debugging RSI rule behavior with different parameters...")
    rsi_debug_results = debug_rsi_rule_behavior(data_handler)
    
    # 8. Run backtests for default parameters using proper Backtester
    logger.info("Running backtests with default parameters")
    
    # SMA backtest
    sma_results = run_backtest(sma_rule, data_handler)
    
    # RSI backtest
    rsi_results = run_backtest(rsi_rule, data_handler)
    
    # 9. Set up optimizer
    logger.info("Setting up optimizer")
    optimizer = OptimizerManager(data_handler)
    
    # 10. Register SMA rule for optimization
    optimizer.register_rule(
        "sma_rule",
        sma_rule.__class__,  # Use the class of the rule
        {
            'fast_window': [5, 10, 15],
            'slow_window': [20, 30, 50],
            'smooth_signals': [True, False]
        }
    )
    
    # 11. Register RSI rule for optimization
    optimizer.register_rule(
        "rsi_rule",
        rsi_rule.__class__,  # Use the class of the rule
        {
            'rsi_period': [7, 14, 21],
            'overbought': [65, 70, 75, 80],
            'oversold': [20, 25, 30, 35],
            'signal_type': ['levels']
        }
    )
    
    # 12. Run optimization for both rules
    logger.info("Optimizing rules")
    
    # Run grid search
    optimized_rules = optimizer.optimize(
        component_type='rule',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='sharpe',  # Using Sharpe ratio as optimization metric
        verbose=True
    )
    
    # 13. Get the optimized rules and their parameters
    optimized_sma = None
    optimized_rsi = None
    
    # Extract the optimized rules
    for rule in optimized_rules.values():
        if isinstance(rule, sma_rule.__class__):
            optimized_sma = rule
        elif isinstance(rule, rsi_rule.__class__):
            optimized_rsi = rule
    
    # 14. Debug optimized RSI rule behavior if found
    if optimized_rsi:
        logger.info("Debugging optimized RSI rule behavior...")
        optimized_params = optimized_rsi.params
        debug_rsi_rule_behavior(data_handler, [optimized_params])
    else:
        logger.warning("Optimization did not return an RSI rule")
    
    # Check if optimization returned both rules
    if not optimized_sma:
        logger.warning("Optimization did not return an SMA rule")
        # Fall back to default parameters
        optimized_sma = sma_rule
    
    if not optimized_rsi:
        logger.warning("Optimization did not return an RSI rule")
        # Fall back to default parameters
        optimized_rsi = rsi_rule
    
    # 15. Run backtests on train data with optimized parameters
    logger.info("Running backtests with optimized parameters on training data")
    
    # SMA backtest on train data
    optimized_sma_train_results = run_backtest(optimized_sma, data_handler, test_data=False)
    
    # RSI backtest on train data
    optimized_rsi_train_results = run_backtest(optimized_rsi, data_handler, test_data=False)
    
    # 16. Run backtests on test data with optimized parameters
    logger.info("Running backtests with optimized parameters on test data")
    
    # SMA backtest on test data
    optimized_sma_test_results = run_backtest(optimized_sma, data_handler, test_data=True)
    
    # RSI backtest on test data
    optimized_rsi_test_results = run_backtest(optimized_rsi, data_handler, test_data=True)
    
    # 17. Plot equity curves for comparison
    train_results = [
        (f"SMA Default (Train)", sma_results, sma_rule.params),
        (f"RSI Default (Train)", rsi_results, rsi_rule.params),
        (f"SMA Optimized (Train)", optimized_sma_train_results, optimized_sma.params),
        (f"RSI Optimized (Train)", optimized_rsi_train_results, optimized_rsi.params)
    ]
    
    test_results = [
        (f"SMA Optimized (Test)", optimized_sma_test_results, optimized_sma.params),
        (f"RSI Optimized (Test)", optimized_rsi_test_results, optimized_rsi.params)
    ]
    
    # Plot training results
    plot_equity_curves(train_results, "Training Data Comparison")
    
    # Plot test results
    plot_equity_curves(test_results, "Test Data Performance")
    
    # 18. Print summary
    logger.info("\nParameter Optimization Summary:")
    logger.info("-" * 50)
    
    logger.info("SMA Crossover Rule:")
    logger.info(f"  Default parameters: {sma_rule.params}")
    logger.info(f"  Default performance (Train): Return={sma_results['total_percent_return']:.2f}%, Sharpe={sma_results['sharpe_ratio']:.2f}")
    logger.info(f"  Optimized parameters: {optimized_sma.params}")
    logger.info(f"  Optimized performance (Train): Return={optimized_sma_train_results['total_percent_return']:.2f}%, Sharpe={optimized_sma_train_results['sharpe_ratio']:.2f}")
    logger.info(f"  Improvement (Train): Return +{optimized_sma_train_results['total_percent_return'] - sma_results['total_percent_return']:.2f}%, Sharpe +{optimized_sma_train_results['sharpe_ratio'] - sma_results['sharpe_ratio']:.2f}")
    logger.info(f"  Out-of-sample performance (Test): Return={optimized_sma_test_results['total_percent_return']:.2f}%, Sharpe={optimized_sma_test_results['sharpe_ratio']:.2f}")
    
    logger.info("\nRSI Rule:")
    logger.info(f"  Default parameters: {rsi_rule.params}")
    logger.info(f"  Default performance (Train): Return={rsi_results['total_percent_return']:.2f}%, Sharpe={rsi_results['sharpe_ratio']:.2f}")
    logger.info(f"  Optimized parameters: {optimized_rsi.params}")
    logger.info(f"  Optimized performance (Train): Return={optimized_rsi_train_results['total_percent_return']:.2f}%, Sharpe={optimized_rsi_train_results['sharpe_ratio']:.2f}")
    logger.info(f"  Improvement (Train): Return +{optimized_rsi_train_results['total_percent_return'] - rsi_results['total_percent_return']:.2f}%, Sharpe +{optimized_rsi_train_results['sharpe_ratio'] - rsi_results['sharpe_ratio']:.2f}")
    logger.info(f"  Out-of-sample performance (Test): Return={optimized_rsi_test_results['total_percent_return']:.2f}%, Sharpe={optimized_rsi_test_results['sharpe_ratio']:.2f}")
    
    return {
        "default_results": {
            "sma": {"params": sma_rule.params, "results": sma_results},
            "rsi": {"params": rsi_rule.params, "results": rsi_results}
        },
        "optimized_results": {
            "sma": {
                "params": optimized_sma.params,
                "train_results": optimized_sma_train_results,
                "test_results": optimized_sma_test_results
            },
            "rsi": {
                "params": optimized_rsi.params,
                "train_results": optimized_rsi_train_results,
                "test_results": optimized_rsi_test_results
            }
        }
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\nScript executed successfully!")
        
        # Print best parameters
        print("\nBest SMA Parameters:")
        for param, value in results["optimized_results"]["sma"]["params"].items():
            print(f"  {param}: {value}")
        
        print("\nBest RSI Parameters:")
        for param, value in results["optimized_results"]["rsi"]["params"].items():
            print(f"  {param}: {value}")
        
        print("\nCheck equity_curves.png for a visual comparison")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

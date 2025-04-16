#!/usr/bin/env python3
"""
Grid search optimization and backtest using the system's optimization module.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import system components
from src.events import EventBus, Event, EventType
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler
from src.config import ConfigManager

# Import rules explicitly - needed for optimization
from src.rules.crossover_rules import SMAcrossoverRule
from src.rules.volatility_rules import BollingerBandRule

# Import optimization components
from src.optimization.optimizer_manager import OptimizerManager
from src.optimization import OptimizationMethod

# Import backtesting components
from src.engine.backtester import Backtester
from src.position_management.position_manager import PositionManager
from src.position_management.position_sizers import PercentOfEquitySizer
from src.position_management.portfolio import Portfolio
from src.strategies import WeightedStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None, days=365):
    """Create synthetic price data for testing with more realistic patterns."""
    # Create a synthetic dataset with realistic patterns
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    # Create a price series with clear patterns for rule testing
    prices = []
    
    # Initial price
    base_price = 100
    
    # Create trends, reversals, and volatility clusters
    trend_direction = 1
    volatility = 0.01
    
    for i in range(days):
        # Every 40 days, create a new major trend
        if i % 40 == 0:
            trend_direction = -trend_direction  # Reverse trend
            volatility = 0.015  # Higher volatility during trend changes
        
        # Every 10 days, create minor adjustments
        elif i % 10 == 0:
            trend_direction *= 0.8  # Trend weakens
            volatility = max(0.005, volatility * 0.9)  # Volatility changes
            
        # Add daily price change with momentum and mean reversion
        momentum = 0.3 * trend_direction * 0.01  # Momentum component
        
        # Mean reversion when price strays too far from base
        if prices and abs(prices[-1] / base_price - 1) > 0.3:
            mean_reversion = -0.001 * (prices[-1] / base_price - 1)
        else:
            mean_reversion = 0
            
        # Combine components
        daily_change = momentum + mean_reversion + np.random.normal(0, volatility)
        
        # Calculate new price
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_change)
            
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(100000, 20000)) for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} bars, saved to {filename}")
    
    return df

def setup_data_handler(filename):
    """Setup data handler based on available classes and required parameters."""
    try:
        # Create the data source
        data_source = CSVDataSource(".")  # Look in current directory
        
        # Create the data handler with the data source
        data_handler = DataHandler(data_source)
        
        # Parse dates from the filename or use default
        df = pd.read_csv(filename)
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract start and end dates from data
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
        else:
            # Default dates if no timestamp column
            start_date = datetime(2022, 1, 1)
            end_date = start_date + timedelta(days=365)
        
        # Load data with required parameters
        data_handler.load_data(
            symbols=["SYNTHETIC"],
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        return data_handler
    
    except Exception as e:
        logger.error(f"Error setting up data handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to set up data handler: {e}")

def run_backtest_with_strategy(data_handler, strategy, use_test_data=True):
    """
    Run a backtest with a specific strategy.
    
    Args:
        data_handler: Data handler with data to test
        strategy: Strategy to use for backtesting
        use_test_data: Whether to use test data (True) or training data (False)
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running backtest with strategy: {strategy.name}")
    
    # Configure backtester
    config = ConfigManager()
    config.set('backtester.initial_capital', 10000)
    
    # Create portfolio - needed by position manager
    portfolio = Portfolio(initial_capital=10000)
    
    # Create position manager with portfolio
    position_manager = PositionManager(
        portfolio=portfolio,  # Required portfolio parameter
        position_sizer=PercentOfEquitySizer(percent=0.95)
    )
    
    # Create and run backtester
    backtester = Backtester(config, data_handler, strategy, position_manager)
    result = backtester.run(use_test_data=use_test_data)
    
    # Calculate metrics
    sharpe = backtester.calculate_sharpe()
    max_drawdown = backtester.calculate_max_drawdown()
    
    logger.info(f"Backtest complete on {'test' if use_test_data else 'train'} data:")
    logger.info(f"Total Return: {result['total_percent_return']:.2f}%")
    logger.info(f"Number of Trades: {result['num_trades']}")
    logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}")
    
    return result, sharpe, max_drawdown

def main():
    """Main function for optimization and backtesting."""
    # Setup output directory
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}.csv"
    data_df = create_synthetic_data(symbol, timeframe, filename)
    
    logger.info(f"Created synthetic data file: {filename}")
    
    # Setup data handler
    try:
        data_handler = setup_data_handler(filename)
        logger.info("Data handler setup successful")
    except Exception as e:
        logger.error(f"Failed to set up data handler: {e}")
        return {"error": f"Data handler setup failed: {e}"}
    
    # Create optimizer manager
    optimizer = OptimizerManager(data_handler)
    
    # Define rules to optimize with parameter ranges - using actual rule classes not string names
    rule_optimizations = [
        {
            'rule_class': SMAcrossoverRule,  # Actual class, not string
            'param_grid': {
                'fast_window': [5, 10, 15, 20],
                'slow_window': [30, 40, 50, 60],
                'smooth_signals': [True, False]
            }
        },
        {
            'rule_class': BollingerBandRule,  # Actual class, not string
            'param_grid': {
                'period': [10, 20, 30],
                'std_dev': [1.5, 2.0, 2.5],
                # Fix: Use breakout_type instead of signal_type for BollingerBandRule
                'breakout_type': ['upper', 'lower', 'both']
            }
        }
    ]
    
    # Register rules for optimization - now using actual class and name
    for rule_config in rule_optimizations:
        rule_class = rule_config['rule_class']
        param_grid = rule_config['param_grid']
        rule_name = rule_class.__name__
        
        optimizer.register_rule(rule_name, rule_class, param_grid)
        
    # Perform optimization for each rule
    optimization_results = {}
    
    for rule_config in rule_optimizations:
        rule_class = rule_config['rule_class']
        rule_name = rule_class.__name__
        
        logger.info(f"\n{'='*50}\nOptimizing {rule_name}\n{'='*50}")
        
        try:
            # Run grid search optimization using the OptimizerManager
            optimized_rules = optimizer.optimize(
                component_type='rule',
                method=OptimizationMethod.GRID_SEARCH,
                components=[rule_name],  # Use the rule name
                metrics='sharpe',
                verbose=True,
                top_n=3  # Get top 3 parameter sets
            )
            
            # Get the best rule (first in optimized dictionary)
            if optimized_rules:
                best_rule = list(optimized_rules.values())[0]
                logger.info(f"Best parameters for {rule_name}: {best_rule.params}")
                
                # Store optimized rule
                optimization_results[rule_name] = {
                    'best_rule': best_rule,
                    'best_params': best_rule.params
                }
                
                # Create strategy with optimized rule
                strategy = WeightedStrategy(
                    components=[best_rule],
                    weights=[1.0],
                    buy_threshold=0.5,
                    sell_threshold=-0.5,
                    name=f"Optimized_{rule_name}"
                )
                
                # Run in-sample backtest with optimized strategy
                in_sample_result, in_sample_sharpe, in_sample_drawdown = run_backtest_with_strategy(
                    data_handler, strategy, use_test_data=False
                )
                
                # Run out-of-sample backtest with optimized strategy
                out_sample_result, out_sample_sharpe, out_sample_drawdown = run_backtest_with_strategy(
                    data_handler, strategy, use_test_data=True
                )
                
                # Store results
                optimization_results[rule_name]['in_sample_result'] = in_sample_result
                optimization_results[rule_name]['in_sample_sharpe'] = in_sample_sharpe
                optimization_results[rule_name]['in_sample_drawdown'] = in_sample_drawdown
                
                optimization_results[rule_name]['out_sample_result'] = out_sample_result
                optimization_results[rule_name]['out_sample_sharpe'] = out_sample_sharpe
                optimization_results[rule_name]['out_sample_drawdown'] = out_sample_drawdown
                
            else:
                logger.warning(f"No optimized rules found for {rule_name}")
                
        except Exception as e:
            logger.error(f"Error during optimization of {rule_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            optimization_results[rule_name] = {'error': str(e)}
    
    # Print summary of all results
    logger.info(f"\n{'='*50}\nOptimization and Backtest Summary\n{'='*50}")
    
    for rule_name, results in optimization_results.items():
        if 'error' in results:
            logger.error(f"\n{rule_name}: Error - {results['error']}")
        elif 'in_sample_result' in results:
            logger.info(f"\n{rule_name}:")
            logger.info(f"Best Parameters: {results['best_params']}")
            logger.info(f"In-Sample Sharpe: {results['in_sample_sharpe']:.4f}")
            logger.info(f"In-Sample Return: {results['in_sample_result']['total_percent_return']:.2f}%")
            logger.info(f"In-Sample Trades: {results['in_sample_result']['num_trades']}")
            logger.info(f"In-Sample Drawdown: {results['in_sample_drawdown']:.2f}")
            
            logger.info(f"Out-of-Sample Sharpe: {results['out_sample_sharpe']:.4f}")
            logger.info(f"Out-of-Sample Return: {results['out_sample_result']['total_percent_return']:.2f}%")
            logger.info(f"Out-of-Sample Trades: {results['out_sample_result']['num_trades']}")
            logger.info(f"Out-of-Sample Drawdown: {results['out_sample_drawdown']:.2f}")
    
    return optimization_results

if __name__ == "__main__":
    try:
        results = main()
        
        print("\n" + "=" * 50)
        print("OPTIMIZATION AND BACKTEST RESULTS SUMMARY")
        print("=" * 50)
        
        # Check if results contain an error
        if isinstance(results, dict) and 'error' in results:
            print(f"\nError: {results['error']}")
        else:
            # Print a tabular summary of results
            print("\nRule Type                | In-Sample Return | Out-of-Sample Return | IS/OOS Ratio")
            print("-" * 75)
            
            for rule_name, rule_results in results.items():
                if 'error' in rule_results:
                    print(f"{rule_name:25} | ERROR: {rule_results['error']}")
                elif 'in_sample_result' in rule_results and 'out_sample_result' in rule_results:
                    in_sample = rule_results['in_sample_result']['total_percent_return']
                    out_sample = rule_results['out_sample_result']['total_percent_return']
                    ratio = in_sample / out_sample if out_sample != 0 else float('inf')
                    
                    print(f"{rule_name:25} | {in_sample:15.2f}% | {out_sample:19.2f}% | {ratio:10.2f}")
            
            print("\nNote: An IS/OOS ratio close to 1.0 indicates robust parameters with less overfitting.")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

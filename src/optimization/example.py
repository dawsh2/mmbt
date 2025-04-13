"""
Example usage of the unified optimization framework.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_handler import CSVDataHandler
from optimizer_manager import OptimizerManager, OptimizationMethod, OptimizationSequence
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5
from regime_detection import TrendStrengthRegimeDetector, VolatilityRegimeDetector

def main():
    # Setup output directory for saving analysis charts
    output_dir = "optimization_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load data
    filepath = os.path.expanduser("~/data/price_data.csv")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    print("==== Unified Optimization Framework Example ====")
    
    # Create optimizer manager
    optimizer = OptimizerManager(data_handler)
    
    # ----- Example 1: Simple Rule Optimization -----
    print("\n1. Simple Rule Optimization")
    
    # Register rules with parameter ranges for optimization
    optimizer.register_rule("sma_crossover", Rule0, 
                           {'fast_window': [5, 10, 15], 'slow_window': [20, 30, 50]})
    
    optimizer.register_rule("ma_crossover", Rule1, 
                           {'ma1': [10, 20], 'ma2': [30, 50]})
    
    optimizer.register_rule("ema_ma_crossover", Rule2, 
                           {'ema1_period': [10, 20], 'ma2_period': [30, 50]})
    
    # Run grid search optimization
    print("Running grid search optimization...")
    optimized_rules = optimizer.optimize(
        component_type='rule',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='sharpe',
        verbose=True
    )
    
    print(f"Optimized {len(optimized_rules)} rules")
    
    # ----- Example 2: Regime Detector Optimization -----
    print("\n2. Regime Detector Optimization")
    
    # Register regime detectors with parameter ranges
    optimizer.register_regime_detector("trend_strength", TrendStrengthRegimeDetector,
                                      {'adx_period': [10, 14, 20], 'adx_threshold': [20, 25, 30]})
    
    optimizer.register_regime_detector("volatility", VolatilityRegimeDetector,
                                      {'lookback_period': [15, 20, 25], 'volatility_threshold': [0.01, 0.015, 0.02]})
    
    # Run regime detector optimization
    print("Optimizing regime detectors...")
    optimized_detectors = optimizer.optimize(
        component_type='regime_detector',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='stability',
        verbose=True
    )
    
    print(f"Optimized {len(optimized_detectors)} regime detectors")
    
    # ----- Example 3: Complete Optimization Sequence -----
    print("\n3. Complete Optimization Sequence")
    
    # Get the best detector
    if optimized_detectors:
        best_detector = list(optimized_detectors.values())[0]
        print(f"Using optimized detector: {best_detector.__class__.__name__}")
    else:
        # Fallback to default detector
        best_detector = TrendStrengthRegimeDetector()
        print("Using default regime detector")
    
    # Register additional rules
    optimizer.register_rule("ema_ema_crossover", Rule3, 
                           {'ema1_period': [10, 20], 'ema2_period': [30, 50]})
    
    optimizer.register_rule("dema_ma_crossover", Rule4, 
                           {'dema1_period': [10, 20], 'ma2_period': [30, 50]})
    
    optimizer.register_rule("dema_dema_crossover", Rule5, 
                           {'dema1_period': [10, 15], 'dema2_period': [25, 30]})
    
    # Run rules-first optimization sequence
    print("Running rules-first optimization sequence...")
    strategy = optimizer.optimize(
        component_type='rule',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='sharpe',
        regime_detector=best_detector,
        sequence=OptimizationSequence.RULES_FIRST,
        verbose=True
    )
    
    # ----- Example 4: Backtest Optimized Strategy -----
    print("\n4. Backtesting Optimized Strategy")
    
    from backtester import Backtester
    
    # Run backtest on out-of-sample data
    backtester = Backtester(data_handler, strategy)
    results = backtester.run(use_test_data=True)
    
    # Print results
    print("\nOut-of-Sample Backtest Results:")
    print(f"Total Return: {results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Sharpe Ratio: {backtester.calculate_sharpe():.4f}")
    
    # Calculate win rate
    if results['trades']:
        win_count = sum(1 for trade in results['trades'] if trade[5] > 0)
        win_rate = win_count / len(results['trades'])
        print(f"Win Rate: {win_rate:.2%}")
    
    # ----- Example 5: Plot Results -----
    print("\n5. Plotting Results")
    
    # Plot equity curve
    if results['trades']:
        equity = [10000]  # Start with initial capital
        for trade in results['trades']:
            equity.append(equity[-1] * np.exp(trade[5]))
            
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title("Out-of-Sample Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "equity_curve.png"))
        plt.close()
        
        print(f"Equity curve saved to {output_dir}/equity_curve.png")
    
    print("\nOptimization examples completed.")

if __name__ == "__main__":
    main()

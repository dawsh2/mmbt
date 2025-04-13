"""
Comprehensive Trading System Testing Script

This script implements a wide parameter search space with multiple optimization 
approaches including genetic algorithm, regime-based optimization, and walk-forward
validation to test for robustness.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from regime_detection import RegimeType, TrendStrengthRegimeDetector, VolatilityRegimeDetector, RegimeManager
from optimizer_manager import OptimizerManager, OptimizationMethod, OptimizationSequence
from validator import WalkForwardValidator, CrossValidator
from trade_analyzer import TradeAnalyzer
from trade_visualizer import TradeVisualizer

# ----- Configuration Settings -----
DATA_FILE = os.path.expanduser("~/mmbt/data/data.csv")
RESULTS_DIR = "optimization_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wide parameter search space
EXPANDED_RULES_CONFIG = [
    # Rule0: Simple Moving Average Crossover
    (Rule0, {'fast_window': list(range(2, 51, 3)), 'slow_window': list(range(10, 201, 10))}),
    
    # Rule1: Simple Moving Average Crossover with MA1 and MA2
    (Rule1, {'ma1': list(range(3, 51, 3)), 'ma2': list(range(20, 201, 10))}),
    
    # Rule2: EMA and MA Crossover
    (Rule2, {'ema1_period': list(range(3, 41, 2)), 'ma2_period': list(range(20, 201, 10))}),
    
    # Rule3: EMA and EMA Crossover
    (Rule3, {'ema1_period': list(range(3, 41, 2)), 'ema2_period': list(range(10, 101, 5))}),
    
    # Rule4: DEMA and MA Crossover
    (Rule4, {'dema1_period': list(range(3, 31, 2)), 'ma2_period': list(range(20, 151, 10))}),
    
    # Rule5: DEMA and DEMA Crossover
    (Rule5, {'dema1_period': list(range(3, 31, 2)), 'dema2_period': list(range(10, 101, 5))}),
    
    # Rule6: TEMA and MA Crossover
    (Rule6, {'tema1_period': list(range(3, 31, 2)), 'ma2_period': list(range(20, 151, 10))}),
    
    # Rule7: Stochastic Oscillator
    (Rule7, {'stoch1_period': list(range(5, 31, 2)), 'stochma2_period': list(range(2, 11, 1))}),
    
    # Rule8: Vortex Indicator
    (Rule8, {'vortex1_period': list(range(5, 31, 2)), 'vortex2_period': list(range(5, 31, 2))}),
    
    # Rule9: Ichimoku Cloud
    (Rule9, {'p1': list(range(5, 21, 2)), 'p2': list(range(20, 101, 5))}),
    
    # Rule10: RSI Overbought/Oversold
    (Rule10, {'rsi1_period': list(range(5, 31, 2)), 'c2_threshold': list(range(20, 81, 5))}),
    
    # Rule11: CCI Overbought/Oversold
    (Rule11, {'cci1_period': list(range(5, 31, 2)), 'c2_threshold': list(range(50, 201, 10))}),
    
    # Rule12: RSI-based strategy
    (Rule12, {'rsi_period': list(range(5, 31, 2)), 'overbought': list(range(60, 91, 5)), 'oversold': list(range(10, 41, 5))}),
    
    # Rule13: Stochastic Oscillator strategy
    (Rule13, {'stoch_period': list(range(5, 31, 2)), 'stoch_d_period': list(range(2, 11, 1))}),
    
    # Rule14: ATR Trailing Stop
    (Rule14, {'atr_period': list(range(5, 31, 2)), 'atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]}),
    
    # Rule15: Bollinger Bands strategy
    (Rule15, {'bb_period': list(range(10, 51, 3)), 'bb_std_dev_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0]})
]

# ----- Ensemble Weighting Methods -----
def equal_weights(rule_objects):
    """Simple 1/N weighting strategy."""
    n = len(rule_objects)
    return np.ones(n) / n

def performance_based_weights(rule_objects, data_handler, metric='sharpe'):
    """Weight rules based on their individual performance."""
    performances = []
    
    for rule in rule_objects:
        # Create single-rule strategy
        strategy = WeightedRuleStrategy(rule_objects=[rule], weights=[1.0])
        
        # Backtest
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=False)  # Use training data
        
        if metric == 'sharpe':
            performance = backtester.calculate_sharpe()
        elif metric == 'return':
            performance = results['total_log_return']
        elif metric == 'win_rate':
            performance = results['win_rate'] if 'win_rate' in results else 0
        else:
            performance = results['total_log_return']
        
        # Store performance (ensure it's at least slightly positive for weighting)
        performances.append(max(performance, 0.01))
    
    # Convert to weights (normalize so they sum to 1)
    weights = np.array(performances) / sum(performances)
    return weights

def inverse_volatility_weights(rule_objects, data_handler, lookback=50):
    """Risk parity approach - weight inversely to signal volatility."""
    volatilities = []
    
    for rule in rule_objects:
        signals = []
        data_handler.reset_train()
        
        # Collect signals from each rule
        while True:
            bar = data_handler.get_next_train_bar()
            if bar is None:
                break
            
            signal = rule.on_bar(bar)
            if signal and hasattr(signal, 'signal_type'):
                signals.append(signal.signal_type.value)  # Convert to numeric (-1, 0, 1)
            else:
                signals.append(0)
        
        # Calculate volatility of signals
        if len(signals) > lookback:
            # Use rolling volatility of recent signals
            rolling_vol = pd.Series(signals).rolling(lookback).std()
            avg_vol = rolling_vol.mean()
        else:
            avg_vol = np.std(signals) if len(signals) > 1 else 1.0
        
        volatilities.append(max(avg_vol, 0.01))  # Ensure non-zero
    
    # Inverse volatility weighting
    inv_vols = 1.0 / np.array(volatilities)
    weights = inv_vols / sum(inv_vols)
    
    return weights

# ----- Main Execution Functions -----
def run_individual_rule_analysis(data_handler, rules_config):
    """Test each rule individually with expanded parameter search."""
    print("\n=== Individual Rule Performance Analysis ===")
    rule_performances = {}
    
    for i, (rule_class, params) in enumerate(rules_config):
        print(f"\nAnalyzing {rule_class.__name__} ({i+1}/{len(rules_config)})")
        
        # Train rule with expanded parameters
        rule_system = EventDrivenRuleSystem(rules_config=[(rule_class, params)], top_n=1)
        rule_system.train_rules(data_handler)
        
        if not rule_system.trained_rule_objects:
            print(f"No valid parameters found for {rule_class.__name__}")
            continue
            
        rule_object = list(rule_system.trained_rule_objects.values())[0]
        
        # Test on out-of-sample data
        strategy = WeightedRuleStrategy(rule_objects=[rule_object], weights=[1.0])
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=True)
        
        sharpe = backtester.calculate_sharpe()
        
        print(f"  Best Parameters: {rule_system.best_params}")
        print(f"  Out-of-Sample Results:")
        print(f"    Return: {results['total_percent_return']:.2f}%")
        print(f"    Trades: {results['num_trades']}")
        print(f"    Sharpe: {sharpe:.4f}")
        
        rule_performances[rule_class.__name__] = {
            'return': results['total_percent_return'],
            'sharpe': sharpe,
            'num_trades': results['num_trades'],
            'best_params': rule_system.best_params
        }
    
    # Rank rules by performance
    sorted_rules = sorted(rule_performances.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print("\n=== Individual Rule Ranking (by Sharpe) ===")
    for i, (rule_name, perf) in enumerate(sorted_rules):
        print(f"{i+1}. {rule_name}: Sharpe={perf['sharpe']:.4f}, Return={perf['return']:.2f}%, Trades={perf['num_trades']}")
    
    return rule_performances

def run_ensemble_weight_comparison(data_handler, rule_objects):
    """Compare different ensemble weighting methods."""
    print("\n=== Ensemble Weighting Method Comparison ===")
    
    # Methods to test
    weighting_methods = {
        'Equal Weights (1/N)': equal_weights,
        'Performance-Based': performance_based_weights,
        'Inverse Volatility': inverse_volatility_weights
    }
    
    results = {}
    
    for name, method in weighting_methods.items():
        print(f"\nTesting {name} method")
        
        # Get weights
        if name == 'Equal Weights (1/N)':
            weights = method(rule_objects)
        else:
            weights = method(rule_objects, data_handler)
        
        print(f"  Generated weights: {weights}")
        
        # Create strategy with these weights
        strategy = WeightedRuleStrategy(rule_objects=rule_objects, weights=weights)
        
        # Backtest on out-of-sample data
        backtester = Backtester(data_handler, strategy)
        backtest_results = backtester.run(use_test_data=True)
        
        sharpe = backtester.calculate_sharpe()
        
        print(f"  Out-of-Sample Results:")
        print(f"    Return: {backtest_results['total_percent_return']:.2f}%")
        print(f"    Trades: {backtest_results['num_trades']}")
        print(f"    Sharpe: {sharpe:.4f}")
        
        results[name] = {
            'weights': weights,
            'return': backtest_results['total_percent_return'],
            'sharpe': sharpe,
            'num_trades': backtest_results['num_trades'],
            'trades': backtest_results['trades']
        }
    
    # Compare methods
    print("\n=== Ensemble Weighting Methods Comparison ===")
    print(f"{'Method':<25} {'Return':<10} {'Trades':<10} {'Sharpe':<10}")
    print("-" * 55)
    
    for name, res in results.items():
        print(f"{name:<25} {res['return']:>8.2f}% {res['num_trades']:>10} {res['sharpe']:>9.4f}")
    
    # Visualize results
    visualizer = TradeVisualizer()
    fig = visualizer.create_comparison_chart(
        {name: {'total_return': res['return'], 'trades': res['trades']} 
         for name, res in results.items()},
        title="Ensemble Weighting Methods Comparison"
    )
    
    plt.savefig(os.path.join(RESULTS_DIR, "ensemble_weights_comparison.png"))
    
    return results

def run_optimization_sequence_comparison(data_handler, rule_objects):
    """Compare different optimization sequences."""
    print("\n=== Optimization Sequence Comparison ===")
    
    # Create regime detector
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    
    # Configure genetic optimization parameters
    genetic_params = {
        'genetic': {
            'population_size': 30,
            'num_generations': 50,
            'mutation_rate': 0.1
        }
    }
    
    # Run different optimization sequences
    results = {}
    
    # 1. Rules-First Optimization
    print("\n=== Rules-First Optimization ===")
    optimizer_rules_first = OptimizerManager(
        data_handler=data_handler,
        rule_objects=rule_objects
    )
    
    rules_first_results = optimizer_rules_first.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.RULES_FIRST,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=genetic_params,
        verbose=True
    )
    
    rules_first_strategy = optimizer_rules_first.get_optimized_strategy()
    
    # 2. Regimes-First Optimization
    print("\n=== Regimes-First Optimization ===")
    optimizer_regimes_first = OptimizerManager(
        data_handler=data_handler,
        rule_objects=rule_objects
    )
    
    regimes_first_results = optimizer_regimes_first.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.REGIMES_FIRST,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=genetic_params,
        verbose=True
    )
    
    regimes_first_strategy = optimizer_regimes_first.get_optimized_strategy()
    
    # 3. Iterative Optimization
    print("\n=== Iterative Optimization ===")
    optimizer_iterative = OptimizerManager(
        data_handler=data_handler,
        rule_objects=rule_objects
    )
    
    iterative_params = {
        'genetic': {
            'population_size': 25,
            'num_generations': 30
        },
        'iterations': 3  # Number of iterations between rule and regime optimization
    }
    
    iterative_results = optimizer_iterative.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.ITERATIVE,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=iterative_params,
        verbose=True
    )
    
    iterative_strategy = optimizer_iterative.get_optimized_strategy()
    
    # Backtest all strategies on test data
    strategies = {
        "Rules-First": rules_first_strategy,
        "Regimes-First": regimes_first_strategy,
        "Iterative": iterative_strategy
    }
    
    comparison_results = {}
    
    for name, strategy in strategies.items():
        backtester = Backtester(data_handler, strategy)
        backtest = backtester.run(use_test_data=True)
        sharpe = backtester.calculate_sharpe()
        
        comparison_results[name] = {
            'return': backtest['total_percent_return'],
            'sharpe': sharpe,
            'num_trades': backtest['num_trades'],
            'trades': backtest['trades']
        }
    
    # Compare methods
    print("\n=== Optimization Sequence Comparison ===")
    print(f"{'Method':<20} {'Return':<10} {'Trades':<10} {'Sharpe':<10}")
    print("-" * 50)
    
    for name, res in comparison_results.items():
        print(f"{name:<20} {res['return']:>8.2f}% {res['num_trades']:>10} {res['sharpe']:>9.4f}")
    
    # Visualize results
    visualizer = TradeVisualizer()
    fig = visualizer.create_comparison_chart(
        {name: {'total_return': res['return'], 'trades': res['trades']} 
         for name, res in comparison_results.items()},
        title="Optimization Sequence Comparison"
    )
    
    plt.savefig(os.path.join(RESULTS_DIR, "optimization_sequence_comparison.png"))
    
    return comparison_results

def run_walk_forward_validation(data_handler, rules_config, top_n=8):
    """Run walk-forward validation to test strategy robustness."""
    print("\n=== Walk-Forward Validation ===")
    
    validator = WalkForwardValidator(
        data_filepath=DATA_FILE,
        rules_config=rules_config,
        window_size=252,  # 1 year of training days
        step_size=63,     # 3 months forward testing
        train_pct=0.7,    # 70% training, 30% testing within each window
        top_n=top_n,
        optimization_method='genetic',
        optimization_metric='sharpe'
    )
    
    results = validator.run_validation(verbose=True, plot_results=True)
    
    # Save walk-forward equity curve
    plt.savefig(os.path.join(RESULTS_DIR, "walk_forward_equity.png"))
    
    # Save window returns chart
    plt.figure(figsize=(12, 6))
    windows = [r['window'] for r in results['window_results']]
    returns = [r['total_return'] for r in results['window_results']]
    plt.bar(windows, returns)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Returns by Walk-Forward Window')
    plt.xlabel('Window')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "walk_forward_returns.png"))
    plt.close()
    
    return results

def run_market_regime_analysis(data_handler, rule_objects):
    """Analyze performance across different market regimes."""
    print("\n=== Market Regime Analysis ===")
    
    # Create regime detector
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    
    # Create baseline equal-weight strategy
    baseline_strategy = WeightedRuleStrategy(
        rule_objects=rule_objects,
        weights=np.ones(len(rule_objects)) / len(rule_objects)
    )
    
    # Create regime-specific optimizer
    regime_optimizer = OptimizerManager(
        data_handler=data_handler,
        rule_objects=rule_objects
    )
    
    # Configure genetic optimization
    genetic_params = {
        'genetic': {
            'population_size': 25,
            'num_generations': 30,
            'mutation_rate': 0.1
        }
    }
    
    # Optimize with regime focus
    regime_results = regime_optimizer.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.REGIMES_FIRST,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=genetic_params,
        verbose=True
    )
    
    regime_strategy = regime_optimizer.get_optimized_strategy()
    
    # Backtest both strategies
    baseline_backtester = Backtester(data_handler, baseline_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    
    regime_backtester = Backtester(data_handler, regime_strategy)
    regime_test_results = regime_backtester.run(use_test_data=True)
    
    # Analyze trades by regime
    analyzer = TradeAnalyzer(regime_test_results)
    
    # Identify regimes in test data
    data_handler.reset_test()
    trend_detector.reset()
    regime_data = {}
    
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
        
        regime = trend_detector.detect_regime(bar)
        regime_data[bar['timestamp']] = regime.name
    
    # Analyze performance by regime
    regime_analysis = analyzer.analyze_by_regime(regime_data)
    
    # Print regime analysis
    print("\n=== Performance by Market Regime ===")
    if 'regime_metrics' in regime_analysis:
        for regime, metrics in regime_analysis['regime_metrics'].items():
            print(f"\nRegime: {regime}")
            print(f"  Total Return: {metrics['total_return']:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Trades: {metrics['total_trades']}")
    
    # Create visualization
    visualizer = TradeVisualizer()
    
    # Extract price data from test data
    data_handler.reset_test()
    price_data = []
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
        price_data.append({'timestamp': bar['timestamp'], 'price': bar['Close']})
    
    price_df = pd.DataFrame(price_data)
    
    # Convert regime_data to DataFrame
    regime_df = pd.DataFrame.from_dict(regime_data, orient='index', columns=['regime'])
    regime_df.index.name = 'timestamp'
    
    # Create regime chart
    fig = visualizer.create_regime_chart(
        price_data=price_df,
        regime_data=regime_df,
        trades=regime_test_results['trades'],
        title="Market Regimes and Trades"
    )
    
    plt.savefig(os.path.join(RESULTS_DIR, "regime_performance.png"))
    
    # Also create regime performance chart
    if 'regime_metrics' in regime_analysis:
        fig2 = visualizer.create_regime_performance_chart(
            regime_metrics=regime_analysis,
            title="Strategy Performance by Market Regime"
        )
        plt.savefig(os.path.join(RESULTS_DIR, "regime_metrics.png"))
    
    return {
        'baseline': baseline_results,
        'regime': regime_test_results,
        'regime_analysis': regime_analysis
    }

def run_comprehensive_analysis():
    """Run comprehensive trading system analysis."""
    print("=== Starting Comprehensive Trading System Analysis ===")
    start_time = time.time()
    
    # Load data
    data_handler = CSVDataHandler(DATA_FILE, train_fraction=0.8)
    
    # 1. First, analyze individual rules with expanded parameter space
    print("\n=== Phase 1: Individual Rule Analysis ===")
    rule_performances = run_individual_rule_analysis(data_handler, EXPANDED_RULES_CONFIG)
    
    # Select top rules based on performance
    TOP_N = 8
    top_rules = sorted(rule_performances.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:TOP_N]
    top_rule_names = [rule[0] for rule in top_rules]
    
    print(f"\nSelected top {TOP_N} rules: {', '.join(top_rule_names)}")
    
    # 2. Train rule system with top rules
    print(f"\n=== Phase 2: Training Rule System with Top {TOP_N} Rules ===")
    
    # Filter original rules_config to include only top rules
    top_rules_config = []
    rule_class_map = {
        'Rule0': Rule0, 'Rule1': Rule1, 'Rule2': Rule2, 'Rule3': Rule3, 
        'Rule4': Rule4, 'Rule5': Rule5, 'Rule6': Rule6, 'Rule7': Rule7, 
        'Rule8': Rule8, 'Rule9': Rule9, 'Rule10': Rule10, 'Rule11': Rule11, 
        'Rule12': Rule12, 'Rule13': Rule13, 'Rule14': Rule14, 'Rule15': Rule15
    }
    
    for rule_name in top_rule_names:
        rule_class = rule_class_map[rule_name]
        for r_class, params in EXPANDED_RULES_CONFIG:
            if r_class == rule_class:
                top_rules_config.append((r_class, params))
                break
    
    # Train rule system
    rule_system = EventDrivenRuleSystem(rules_config=top_rules_config, top_n=TOP_N)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    # 3. Compare different ensemble weighting methods
    print("\n=== Phase 3: Ensemble Weighting Methods Comparison ===")
    weight_results = run_ensemble_weight_comparison(data_handler, top_rule_objects)
    
    # 4. Compare different optimization sequences
    print("\n=== Phase 4: Optimization Sequence Comparison ===")
    optimization_results = run_optimization_sequence_comparison(data_handler, top_rule_objects)
    
    # 5. Run walk-forward validation
    print("\n=== Phase 5: Walk-Forward Validation ===")
    wf_results = run_walk_forward_validation(data_handler, top_rules_config, TOP_N)
    
    # 6. Market regime analysis
    print("\n=== Phase 6: Market Regime Analysis ===")
    regime_results = run_market_regime_analysis(data_handler, top_rule_objects)
    
    # 7. Compile and report final results
    print("\n=== Final Results ===")
    
    # Determine the best approach
    approaches = {
        "Equal Weights (1/N)": weight_results['Equal Weights (1/N)']['sharpe'],
        "Performance-Based": weight_results['Performance-Based']['sharpe'],
        "Inverse Volatility": weight_results['Inverse Volatility']['sharpe'],
        "Rules-First Opt": optimization_results['Rules-First']['sharpe'],
        "Regimes-First Opt": optimization_results['Regimes-First']['sharpe'],
        "Iterative Opt": optimization_results['Iterative']['sharpe']
    }
    
    best_approach = max(approaches.items(), key=lambda x: x[1])
    
    print(f"\nBest overall approach: {best_approach[0]} (Sharpe: {best_approach[1]:.4f})")
    print(f"Walk-Forward Validation Results:")
    print(f"  Average Window Return: {wf_results['summary']['avg_return']:.2f}%")
    print(f"  Median Window Return: {wf_results['summary']['median_return']:.2f}%")
    print(f"  Profitable Windows: {wf_results['summary']['pct_profitable_windows']:.1f}%")
    
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")
    
    # Create comprehensive report
    create_final_report(
        rule_performances=rule_performances,
        weight_results=weight_results,
        optimization_results=optimization_results,
        wf_results=wf_results,
        regime_results=regime_results,
        best_approach=best_approach
    )

def create_final_report(rule_performances, weight_results, optimization_results, 
                        wf_results, regime_results, best_approach):
    """Create a comprehensive final report."""
    report_file = os.path.join(RESULTS_DIR, "trading_system_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("===============================================\n")
        f.write("COMPREHENSIVE TRADING SYSTEM ANALYSIS REPORT\n")
        f.write("===============================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data File: {DATA_FILE}\n\n")
        
        f.write("1. INDIVIDUAL RULE PERFORMANCE\n")
        f.write("------------------------------\n")
        sorted_rules = sorted(rule_performances.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        for i, (rule_name, perf) in enumerate(sorted_rules):
            f.write(f"#{i+1}: {rule_name}\n")
            f.write(f"    Sharpe Ratio: {perf['sharpe']:.4f}\n")
            f.write(f"    Return: {perf['return']:.2f}%\n")
            f.write(f"    Trades: {perf['num_trades']}\n")
            f.write(f"    Best Parameters: {perf['best_params']}\n\n")
        
        f.write("\n2. ENSEMBLE WEIGHTING METHODS\n")
        f.write("-----------------------------\n")
        for name, res in weight_results.items():
            f.write(f"{name}:\n")
            f.write(f"    Return: {res['return']:.2f}%\n")
            f.write(f"    Sharpe: {res['sharpe']:.4f}\n")
            f.write(f"    Trades: {res['num_trades']}\n")
            f.write(f"    Weights: {res['weights']}\n\n")
        
        f.write("\n3. OPTIMIZATION SEQUENCES\n")
        f.write("-------------------------\n")
        for name, res in optimization_results.items():
            f.write(f"{name}:\n")
            f.write(f"    Return: {res['return']:.2f}%\n")
            f.write(f"    Sharpe: {res['sharpe']:.4f}\n")
            f.write(f"    Trades: {res['num_trades']}\n\n")
        
        f.write("\n4. WALK-FORWARD VALIDATION\n")
        f.write("--------------------------\n")
        f.write(f"Number of Windows: {wf_results['summary']['num_windows']}\n")
        f.write(f"Average Window Return: {wf_results['summary']['avg_return']:.2f}%\n")
        f.write(f"Median Window Return: {wf_results['summary']['median_return']:.2f}%\n")
        f.write(f"Average Sharpe Ratio: {wf_results['summary']['avg_sharpe']:.4f}\n")
        f.write(f"Profitable Windows: {wf_results['summary']['pct_profitable_windows']:.1f}%\n")
        f.write(f"Total Trades: {wf_results['summary']['total_trades']}\n\n")
        
        # Include window-by-window results
        f.write("Window-by-Window Results:\n")
        for window in wf_results['window_results']:
            f.write(f"  Window {window['window']}: Return: {window['total_return']:.2f}%, ")
            f.write(f"Sharpe: {window['sharpe']:.2f}, Trades: {window['num_trades']}\n")
        
        f.write("\n5. MARKET REGIME ANALYSIS\n")
        f.write("-------------------------\n")
        f.write(f"Baseline Strategy Return: {regime_results['baseline']['total_percent_return']:.2f}%\n")
        f.write(f"Regime-Based Strategy Return: {regime_results['regime']['total_percent_return']:.2f}%\n\n")
        
        if 'regime_metrics' in regime_results['regime_analysis']:
            f.write("Performance by Regime:\n")
            for regime, metrics in regime_results['regime_analysis']['regime_metrics'].items():
                f.write(f"  {regime}:\n")
                f.write(f"    Return: {metrics['total_return']:.2f}%\n")
                f.write(f"    Win Rate: {metrics['win_rate']:.2%}\n")
                f.write(f"    Trades: {metrics['total_trades']}\n\n")
        
        f.write("\n6. CONCLUSION\n")
        f.write("-------------\n")
        f.write(f"Best Approach: {best_approach[0]} (Sharpe: {best_approach[1]:.4f})\n\n")
        
        # Add explanation of 1/N (equal weights) vs optimized approaches
        f.write("Analysis of Equal Weights (1/N) vs Optimized Approaches:\n")
        ew_sharpe = weight_results['Equal Weights (1/N)']['sharpe']
        best_opt_name = max(optimization_results.items(), key=lambda x: x[1]['sharpe'])[0]
        best_opt_sharpe = optimization_results[best_opt_name]['sharpe']
        
        if ew_sharpe > best_opt_sharpe:
            f.write("The Equal Weights (1/N) approach outperformed all optimization methods, ")
            f.write("suggesting that simple, robust approaches can work better than complex optimization ")
            f.write("due to their resilience to overfitting and market changes.\n")
        else:
            f.write(f"The {best_opt_name} optimization approach outperformed Equal Weights, ")
            f.write("indicating that there is value in more sophisticated optimization techniques ")
            f.write("for this particular trading system and market environment.\n")
        
        f.write("\nThe Walk-Forward Validation results provide the most realistic assessment of ")
        f.write("how the system would perform in live trading, as they account for ")
        f.write("the adaptability of the system over time and changing market conditions.\n")
    
    print(f"\nFinal report saved to {report_file}")

# Execute the comprehensive analysis
if __name__ == "__main__":
    run_comprehensive_analysis()

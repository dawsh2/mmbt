"""
Test script for validating genetic algorithm optimization with high-return rules.
This tests whether the GA can identify and weight high-performing rules appropriately.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import time

# Import your actual components
from backtester import Backtester, BarEvent
from signals import Signal, SignalType
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from strategy import Rule0, Rule1, Rule2, Rule3  # Import some of your existing rules


class AlternatingTestRule:
    """
    Test rule that buys when price is 1 and sells when price is 100.
    Should be the best performer in our test dataset.
    """
    def __init__(self, params=None):
        self.rule_id = "AlternatingTestRule"
        self.last_price = None

    def on_bar(self, bar):
        price = bar["Close"]
        timestamp = bar["timestamp"]
        
        # Buy at 1, sell at 100
        if price == 1:
            signal_type = SignalType.BUY
        elif price == 100:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create and return a proper Signal object
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={}
        )
    
    def reset(self):
        self.last_price = None


class RandomTestRule:
    """
    Test rule that generates random signals at a specified frequency.
    Used as a baseline comparison.
    """
    def __init__(self, params=None):
        self.rule_id = "RandomTestRule"
        self.signal_frequency = params.get('signal_frequency', 0.2) if params else 0.2
        np.random.seed(42)  # Fixed seed for reproducibility

    def on_bar(self, bar):
        timestamp = bar["timestamp"]
        price = bar["Close"]
        
        # Generate random signal
        rand = np.random.random()
        if rand < self.signal_frequency / 2:
            signal_type = SignalType.BUY
        elif rand < self.signal_frequency:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={}
        )
    
    def reset(self):
        pass


class InverseTestRule:
    """
    Test rule that does the opposite of what would work - 
    buys at 100 and sells at 1. Should perform poorly.
    """
    def __init__(self, params=None):
        self.rule_id = "InverseTestRule"

    def on_bar(self, bar):
        price = bar["Close"]
        timestamp = bar["timestamp"]
        
        # Opposite of optimal - buy high, sell low
        if price == 100:
            signal_type = SignalType.BUY
        elif price == 1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={}
        )
    
    def reset(self):
        pass


class TestDataHandler:
    """
    Test data handler that generates alternating prices (1, 100, 1, 100, ...).
    """
    def __init__(self, num_bars=100, alternate=True, train_fraction=0.6):
        """
        Initialize with specified number of bars and whether prices should alternate.
        """
        self.num_bars = num_bars
        self.alternate = alternate
        self.train_fraction = train_fraction
        self._generate_data()
        self.current_train_index = 0
        self.current_test_index = 0
        
    def _generate_data(self):
        """Generate the test data with alternating or random prices."""
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.num_bars)]
        
        if self.alternate:
            # Create alternating prices (1, 100, 1, 100, ...)
            prices = []
            for i in range(self.num_bars):
                prices.append(1 if i % 2 == 0 else 100)
        else:
            # Create random prices between 1 and 100
            prices = np.random.uniform(1, 100, self.num_bars)
            
        # Create the data frames
        self.data = []
        for i in range(self.num_bars):
            self.data.append({
                'timestamp': dates[i],
                'Open': prices[i],
                'High': prices[i] * 1.01,  # Add small variations
                'Low': prices[i] * 0.99,
                'Close': prices[i],
                'Volume': 1000
            })
            
        # Create DataFrame versions for easier manipulation
        self.full_df = pd.DataFrame(self.data)
        
        # Split into train/test
        train_size = int(self.train_fraction * len(self.data))
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]
        
        self.train_df = self.full_df.iloc[:train_size]
        self.test_df = self.full_df.iloc[train_size:]
            
    def get_next_train_bar(self):
        """Get the next bar from training data."""
        if self.current_train_index < len(self.train_data):
            bar = self.train_data[self.current_train_index]
            self.current_train_index += 1
            return bar
        return None
            
    def get_next_test_bar(self):
        """Get the next bar from test data."""
        if self.current_test_index < len(self.test_data):
            bar = self.test_data[self.current_test_index]
            self.current_test_index += 1
            return bar
        return None
    
    def reset_train(self):
        """Reset the training data iterator."""
        self.current_train_index = 0
        
    def reset_test(self):
        """Reset the test data iterator."""
        self.current_test_index = 0


def test_rules_individually(data_handler, rule_objects):
    """
    Test each rule individually on the training data to see its performance.
    
    Args:
        data_handler: The test data handler
        rule_objects: List of rule objects to test
        
    Returns:
        dict: Results for each rule
    """
    results = {}
    
    for i, rule in enumerate(rule_objects):
        rule_name = rule.__class__.__name__
        print(f"\nTesting {rule_name} performance...")
        
        # Create strategy with just this rule
        strategy = WeightedRuleStrategy(
            rule_objects=[rule],
            weights=[1.0]
        )
        
        # Run backtest
        backtester = Backtester(data_handler, strategy)
        result = backtester.run(use_test_data=False)  # Use training data
        
        # Calculate metrics
        sharpe = backtester.calculate_sharpe()
        
        # Store results
        results[rule_name] = {
            'return': result['total_percent_return'],
            'trades': result['num_trades'],
            'sharpe': sharpe,
            'log_return': result['total_log_return'],
            'win_rate': sum(1 for t in result['trades'] if t[5] > 0) / result['num_trades'] if result['num_trades'] > 0 else 0
        }
        
        # Print summary
        print(f"  Total Return: {result['total_percent_return']:.2f}%")
        print(f"  Number of Trades: {result['num_trades']}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
    
    return results


def run_ga_optimization_test():
    """Test the genetic algorithm optimization with contrasting rules."""
    print("=== Starting Genetic Algorithm Optimization Test ===")
    
    # Create test data with alternating prices
    data_handler = TestDataHandler(num_bars=200, alternate=True)
    
    # Create rules including our high-return alternating rule
    rule_objects = [
        AlternatingTestRule(),    # Should be the best performer
        RandomTestRule(),         # Mid-range performer
        InverseTestRule(),        # Worst performer
        # Add some of your standard rules with default parameters
        Rule0({'fast_window': 5, 'slow_window': 20}),
        Rule1({'ma1': 10, 'ma2': 30})
    ]
    
    # First, test each rule individually to establish baseline performance
    print("\nTesting individual rule performance on training data...")
    individual_results = test_rules_individually(data_handler, rule_objects)
    
    # Sort rules by performance
    sorted_rules = sorted(individual_results.items(), 
                         key=lambda x: x[1]['sharpe'], 
                         reverse=True)
    
    print("\nIndividual Rule Performance Summary (by Sharpe ratio):")
    print(f"{'Rule':<20} {'Return':<10} {'Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 60)
    for rule_name, metrics in sorted_rules:
        print(f"{rule_name:<20} {metrics['return']:>8.2f}% {metrics['trades']:>10} {metrics['sharpe']:>9.4f} {metrics['win_rate']:>9.2f}")
    
    # Now run genetic optimization
    print("\nRunning genetic algorithm optimization...")
    start_time = time.time()
    
    genetic_optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=rule_objects,
        population_size=20,
        num_generations=30,
        optimization_metric='sharpe'
    )
    
    # Run optimization
    optimal_weights = genetic_optimizer.optimize(verbose=True)
    
    # Print optimization results
    optimization_time = time.time() - start_time
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print("Optimal weights:")
    for i, weight in enumerate(optimal_weights):
        rule_name = rule_objects[i].__class__.__name__
        print(f"  {rule_name}: {weight:.4f}")

    # Get sharpe ratio of best individual rule for comparison
    best_individual_sharpe = sorted_rules[0][1]['sharpe']
    best_individual_rule = sorted_rules[0][0]
    
    # Create weighted strategy with optimal weights
    weighted_strategy = WeightedRuleStrategy(
        rule_objects=rule_objects,
        weights=optimal_weights
    )
    
    # Test the optimized strategy
    print("\nTesting optimized strategy...")
    backtester = Backtester(data_handler, weighted_strategy)
    
    # Test on training data
    train_results = backtester.run(use_test_data=False)
    train_sharpe = backtester.calculate_sharpe()
    
    print(f"\nTraining Results:")
    print(f"Total Return: {train_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {train_results['num_trades']}")
    print(f"Sharpe Ratio: {train_sharpe:.4f}")
    print(f"Win Rate: {sum(1 for t in train_results['trades'] if t[5] > 0) / train_results['num_trades'] if train_results['num_trades'] > 0 else 0:.2f}")
    
    # Test on out-of-sample data
    test_results = backtester.run(use_test_data=True)
    test_sharpe = backtester.calculate_sharpe()
    
    print(f"\nOut-of-Sample Results:")
    print(f"Total Return: {test_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {test_results['num_trades']}")
    print(f"Sharpe Ratio: {test_sharpe:.4f}")
    print(f"Win Rate: {sum(1 for t in test_results['trades'] if t[5] > 0) / test_results['num_trades'] if test_results['num_trades'] > 0 else 0:.2f}")
    
    # Compare with best individual rule
    print(f"\nGA Improvement over best individual rule ({best_individual_rule}):")
    print(f"  Best Individual Sharpe: {best_individual_sharpe:.4f}")
    print(f"  GA Optimized Sharpe: {train_sharpe:.4f}")
    print(f"  Improvement: {((train_sharpe / best_individual_sharpe) - 1) * 100:.2f}%")
    
    # Plot optimized results
    plt.figure(figsize=(10, 6))
    plt.plot(genetic_optimizer.fitness_history)
    plt.title('Genetic Algorithm Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Sharpe Ratio)')
    plt.grid(True)
    plt.savefig("GA_Optimization_Progress.png")
    plt.show()
    
    # Plot equity curves for comparison
    plot_comparison(data_handler, rule_objects, optimal_weights)
    
    # Run an additional test with the AlternatingTestRule having a very small weight
    # to verify GA is actually working and not just using the rule ordering
    print("\nRunning control test with fixed weights...")
    control_weights = np.ones(len(rule_objects)) / len(rule_objects)  # Equal weights
    control_strategy = WeightedRuleStrategy(
        rule_objects=rule_objects,
        weights=control_weights
    )
    
    control_backtester = Backtester(data_handler, control_strategy)
    control_results = control_backtester.run(use_test_data=False)
    control_sharpe = control_backtester.calculate_sharpe()
    
    print(f"\nControl Test Results (Equal Weights):")
    print(f"Total Return: {control_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {control_results['num_trades']}")
    print(f"Sharpe Ratio: {control_sharpe:.4f}")
    
    # GA optimization should significantly outperform equal weights
    improvement = ((train_sharpe / control_sharpe) - 1) * 100 if control_sharpe > 0 else float('inf')
    print(f"GA improvement over equal weights: {improvement:.2f}%")

    return {
        'individual_results': individual_results,
        'optimal_weights': optimal_weights,
        'train_results': train_results,
        'test_results': test_results,
        'best_individual_rule': best_individual_rule,
        'best_individual_sharpe': best_individual_sharpe,
        'optimized_sharpe': train_sharpe,
        'control_sharpe': control_sharpe
    }


def plot_comparison(data_handler, rule_objects, optimal_weights):
    """
    Plot equity curves for individual rules and the optimized ensemble.
    
    Args:
        data_handler: The test data handler
        rule_objects: List of rule objects
        optimal_weights: Optimized weights from GA
    """
    # Create strategies for each rule and the optimized ensemble
    strategies = {}
    
    # Add individual rule strategies
    for rule in rule_objects:
        rule_name = rule.__class__.__name__
        strategies[rule_name] = WeightedRuleStrategy(
            rule_objects=[rule],
            weights=[1.0]
        )
    
    # Add optimized strategy
    strategies["GA Optimized"] = WeightedRuleStrategy(
        rule_objects=rule_objects,
        weights=optimal_weights
    )
    
    # Run backtests and collect equity curves
    equity_curves = {}
    initial_equity = 10000
    
    for name, strategy in strategies.items():
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=True)  # Use test data
        
        if results['num_trades'] > 0:
            # Calculate equity curve
            equity = [initial_equity]
            for trade in results['trades']:
                equity.append(equity[-1] * np.exp(trade[5]))
            
            equity_curves[name] = equity
    
    # Plot equity curves
    plt.figure(figsize=(12, 8))
    
    for name, equity in equity_curves.items():
        # Use dashed line for individual rules, solid for optimized
        linestyle = '-' if name == "GA Optimized" else '--'
        linewidth = 2 if name == "GA Optimized" else 1
        
        plt.plot(equity, label=name, linestyle=linestyle, linewidth=linewidth)
    
    plt.title('Strategy Comparison - Equity Curves')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig("Strategy_Comparison.png")
    plt.show()


if __name__ == "__main__":
    results = run_ga_optimization_test()

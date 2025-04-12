#!/usr/bin/env python3
"""
Minimal backtest script to verify that the updated signal handling code works correctly
and includes running the Genetic Algorithm for memory debugging.

Usage:
    python run_minimal_backtest.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import time
import memory_profiler

# Import the components from your trading system
from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent
from signals import Signal, SignalType, SignalCollection, SignalRouter

# Import the rules for testing
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15, TopNStrategy

# Import the Genetic Optimizer
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def run_minimal_backtest():
    """Run a minimal backtest and the GA optimization."""
    print_separator("MINIMAL BACKTEST")

    # 1. Load a small subset of data
    filepath = os.path.expanduser("~/mmbt/data/data.csv")

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        alternative_path = "data/data.csv"
        if os.path.exists(alternative_path):
            filepath = alternative_path
            print(f"Using alternative path: {filepath}")
        else:
            print("Please provide the correct path to your data file")
            return

    print(f"Loading data from {filepath}")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)

    # 2. Create rule instances for testing
    print("\nCreating rules...")

    # Create rules directly with explicit parameters
    rule0 = Rule0({'fast_window': 10, 'slow_window': 30})
    rule1 = Rule1({'ma1': 15, 'ma2': 45})
    rule2 = Rule2({'ema1_period': 12, 'ma2_period': 26})
    rule3 = Rule3({'ema1_period': 10, 'ema2_period': 30})
    rule4 = Rule4({'dema1_period': 10, 'ma2_period': 30})
    rule5 = Rule5({'dema1_period': 10, 'dema2_period': 30})
    rule6 = Rule6({'tema1_period': 10, 'ma2_period': 30})
    rule7 = Rule7({'stoch1_period': 10, 'stochma2_period': 3})
    rule8 = Rule8({'vortex1_period': 10, 'vortex2_period': 10})
    rule9 = Rule9({'p1': 9, 'p2': 26})
    rule10 = Rule10({'rsi1_period': 10, 'c2_threshold': 30})
    rule11 = Rule11({'cci1_period': 14, 'c2_threshold': 100})
    rule12 = Rule12({'rsi_period': 10, 'hl_threshold': 70, 'll_threshold': 30})
    rule13 = Rule13({'stoch_period': 10, 'cci1_period': 14, 'hl_threshold': 80, 'll_threshold': 20})
    rule14 = Rule14({'atr_period': 14})
    rule15 = Rule15({'bb_period': 20})

    # Debug: Check if rules are properly instantiated
    print(f"Rule0 type: {type(rule0)}")
    print(f"Rule1 type: {type(rule1)}")
    print(f"Rule2 type: {type(rule2)}")
    print(f"Rule3 type: {type(rule3)}")
    print(f"Rule4 type: {type(rule4)}")
    print(f"Rule5 type: {type(rule5)}")
    print(f"Rule6 type: {type(rule6)}")
    print(f"Rule7 type: {type(rule7)}")
    print(f"Rule8 type: {type(rule8)}")
    print(f"Rule9 type: {type(rule9)}")
    print(f"Rule10 type: {type(rule10)}")
    print(f"Rule11 type: {type(rule11)}")
    print(f"Rule12 type: {type(rule12)}")
    print(f"Rule13 type: {type(rule13)}")
    print(f"Rule14 type: {type(rule14)}")
    print(f"Rule15 type: {type(rule15)}")

    # Collect rules into a list and verify each element
    rules = [rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15]

    # Debug: Check rule list contents
    print("\nChecking rule list...")
    for i, rule in enumerate(rules):
        print(f"Rule {i}: {type(rule)}")
        if isinstance(rule, tuple):
            print(f"WARNING: Rule {i} is a tuple: {rule}")

    # 3. Run Genetic Algorithm Optimization
    print_separator("GENETIC ALGORITHM OPTIMIZATION")

    # Instantiate the Genetic Optimizer
    optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=rules,
        population_size=10,  # Reduced for a quicker test
        num_parents=4,
        num_generations=5,   # Reduced for a quicker test
        mutation_rate=0.1,
        optimization_metric='sharpe',
        random_seed=42,
        deterministic=True
    )

    # Run the optimization and profile memory
    print("\nRunning Genetic Algorithm Optimization with memory profiling...")
    try:
        # Use memory_profiler to track memory usage of the optimize function
        mem_usage = memory_profiler.memory_usage((optimizer.optimize, (True, 2, 0.0001)), interval=0.1)

        best_weights = optimizer.best_weights
        best_fitness = optimizer.best_fitness

        print("\nGenetic Algorithm Optimization Results:")
        if best_weights is not None:
            print(f"Best Weights: {best_weights}")
            print(f"Best Fitness: {best_fitness:.4f}")
        else:
            print("Optimization did not complete or no best weights found.")

        # Analyze memory usage
        print("\nMemory Usage During Optimization:")
        max_memory = max(mem_usage)
        print(f"Maximum memory usage: {max_memory:.2f} MB")

        # You can further analyze the 'mem_usage' list to see how memory changed over time

    except Exception as e:
        print(f"Error during Genetic Algorithm Optimization: {str(e)}")

    # 4. Create manual test for SignalRouter (moved after GA for clarity)
    print_separator("SIGNAL ROUTER TEST")

    # Create a simple test bar
    test_bar = {
        "timestamp": datetime.now(),
        "Open": 100.0,
        "High": 101.0,
        "Low": 99.0,
        "Close": 100.5,
        "Volume": 1000
    }

    # Test each rule individually first
    print("\nTesting individual rules:")

    for i, rule in enumerate(rules):
        try:
            result = rule.on_bar(test_bar)
            print(f"Rule {i} returned: {result}")
        except Exception as e:
            print(f"Error with Rule {i}: {str(e)}")

    # If individual rules work, create a router with them
    rule_ids = [f"Rule{i}" for i in range(len(rules))]
    try:
        router = SignalRouter(rules, rule_ids=rule_ids)
        print("\nSignalRouter created successfully")

        # Test the router with a sample bar
        event = BarEvent(test_bar)
        router_output = router.on_bar(event)
        print(f"Router output keys: {router_output.keys()}")
        print(f"Number of signals in collection: {len(router_output['signals'])}")

    except Exception as e:
        print(f"Error creating or using SignalRouter: {str(e)}")
        print("Trying manual signal generation...")

        # Skip the router and manually create signals
        manual_signals = []
        for i, rule in enumerate(rules):
            try:
                signal_value = 0
                if hasattr(rule, 'on_bar'):
                    signal_output = rule.on_bar(test_bar)
                    if isinstance(signal_output, Signal):
                        signal = signal_output
                    elif isinstance(signal_output, int):
                        signal_type = SignalType.BUY if signal_output == 1 else SignalType.SELL if signal_output == -1 else SignalType.NEUTRAL
                        signal = Signal(
                            timestamp=test_bar["timestamp"],
                            type=signal_type,
                            price=test_bar["Close"],
                            rule_id=rule_ids[i],
                            confidence=1.0,
                            metadata={}
                        )
                    elif isinstance(signal_output, dict) and 'signal' in signal_output:
                        signal_type = SignalType.BUY if signal_output['signal'] == 1 else SignalType.SELL if signal_output['signal'] == -1 else SignalType.NEUTRAL
                        signal = Signal(
                            timestamp=test_bar["timestamp"],
                            type=signal_type,
                            price=test_bar["Close"],
                            rule_id=rule_ids[i],
                            confidence=1.0,
                            metadata={}
                        )
                    else:
                        signal = Signal(
                            timestamp=test_bar["timestamp"],
                            type=SignalType.NEUTRAL,
                            price=test_bar["Close"],
                            rule_id=rule_ids[i],
                            confidence=1.0,
                            metadata={}
                        )
                    manual_signals.append(signal)
                    print(f"Manually created signal for Rule {i}: {signal}")
                else:
                    print(f"Rule {i} does not have an on_bar method.")
            except Exception as e:
                print(f"Error creating manual signal for Rule {i}: {str(e)}")

    print_separator("VERIFICATION COMPLETE")
    print("Check the output to see if signal generation and GA optimization are working correctly.")
    print("Pay attention to the memory usage during the GA optimization.")
    print("If you see errors, you may need to fix your rule or GA implementations.")

if __name__ == "__main__":
    run_minimal_backtest()

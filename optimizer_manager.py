"""
Optimizer Manager for coordinating different optimization approaches and sequences.

This module provides a framework for orchestrating multiple optimization methods
in different sequences and combinations, including genetic algorithms, grid search,
Bayesian optimization, and regime-specific optimizations.
"""

import numpy as np
import time
from enum import Enum, auto
from backtester import Backtester
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from regime_detection import RegimeManager, RegimeType


class OptimizationMethod(Enum):
    """Enumeration of supported optimization methods."""
    GENETIC = auto()
    GRID_SEARCH = auto()
    BAYESIAN = auto()
    RANDOM_SEARCH = auto()
    JOINT = auto()


class OptimizationSequence(Enum):
    """Enumeration of optimization sequencing strategies."""
    RULES_FIRST = auto()  # First optimize rules, then regimes
    REGIMES_FIRST = auto()  # First identify regimes, then optimize rules per regime
    JOINT = auto()         # Optimize rules and regime detection jointly
    ITERATIVE = auto()     # Alternate between rule and regime optimization


class OptimizerManager:
    """
    Manages different optimization methods and sequences for trading systems.
    
    This class serves as a coordinator for various optimization approaches,
    allowing flexible configuration of which optimizers to use and in what order.
    """
    
    def __init__(self, data_handler, rule_objects=None, rule_classes=None, rule_params=None):
        """
        Initialize the optimizer manager.
        
        Args:
            data_handler: The data handler for accessing market data
            rule_objects: Optional list of pre-initialized rule objects
            rule_classes: Optional list of rule classes for creating rules
            rule_params: Optional parameters for rule initialization
        """
        self.data_handler = data_handler
        self.rule_objects = rule_objects or []
        self.rule_classes = rule_classes or []
        self.rule_params = rule_params or {}
        
        # Store optimization results
        self.optimized_rule_weights = None
        self.optimized_rule_objects = None
        self.optimized_regime_weights = None
        self.optimized_regime_detector = None
        self.regime_manager = None
        
        # Performance metrics
        self.optimization_metrics = {}
        
    def optimize(self, 
                 method=OptimizationMethod.GENETIC,
                 sequence=OptimizationSequence.RULES_FIRST,
                 metrics='sharpe',
                 regime_detector=None,
                 optimization_params=None,
                 verbose=True):
        """
        Run the optimization process.
        
        Args:
            method: The optimization method to use
            sequence: The sequence of optimization steps
            metrics: Performance metric(s) to optimize for
            regime_detector: Optional regime detector to use
            optimization_params: Additional parameters for the optimizer
            verbose: Whether to print progress information
            
        Returns:
            dict: Results of the optimization process
        """
        start_time = time.time()
        optimization_params = optimization_params or {}
        
        if verbose:
            print(f"\n=== Starting Optimization: {method.name}, Sequence: {sequence.name} ===")
        
        # Initialize rule objects if needed
        if not self.rule_objects and self.rule_classes:
            self._initialize_rule_objects()
        
        # Run the appropriate optimization sequence
        if sequence == OptimizationSequence.RULES_FIRST:
            self._optimize_rules_first(method, metrics, regime_detector, optimization_params, verbose)
        elif sequence == OptimizationSequence.REGIMES_FIRST:
            self._optimize_regimes_first(method, metrics, regime_detector, optimization_params, verbose)
        elif sequence == OptimizationSequence.JOINT:
            self._optimize_joint(method, metrics, regime_detector, optimization_params, verbose)
        elif sequence == OptimizationSequence.ITERATIVE:
            self._optimize_iterative(method, metrics, regime_detector, optimization_params, verbose)
        
        # Calculate final performance metrics
        results = self._evaluate_final_strategy(metrics)
        
        # Add timing information
        total_time = time.time() - start_time
        results['optimization_time'] = total_time
        
        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(f"Optimized performance:")
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not metric.startswith('_'):
                    print(f"  {metric}: {value}")
        
        return results
    
    def _initialize_rule_objects(self):
        """Initialize rule objects from rule classes and parameters."""
        self.rule_objects = []
        for rule_class in self.rule_classes:
            # Get parameters for this rule class or use default
            params = self.rule_params.get(rule_class.__name__, {})
            # Initialize with the first set of parameters
            if isinstance(params, list):
                self.rule_objects.append(rule_class(params[0]))
            else:
                self.rule_objects.append(rule_class(params))
    
    def _optimize_rules_first(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Optimize rule weights first, then optimize for regimes.
        
        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if verbose:
            print("Step 1: Optimizing rule weights...")
        
        # Optimize rule weights using the selected method
        self.optimized_rule_weights = self._optimize_rule_weights(
            method, metrics, optimization_params, verbose
        )
        
        # Create optimized rule strategy
        optimized_strategy = WeightedRuleStrategy(
            rule_objects=self.rule_objects,
            weights=self.optimized_rule_weights
        )
        
        if regime_detector:
            if verbose:
                print("\nStep 2: Optimizing regime-specific strategies...")
            
            # Create and optimize regime manager
            self.regime_manager = RegimeManager(
                regime_detector=regime_detector,
                rule_objects=self.rule_objects,
                data_handler=self.data_handler
            )
            
            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)
            
            # Set the default strategy to the optimized weighted strategy
            self.regime_manager.default_strategy = optimized_strategy
    
    def _optimize_regimes_first(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Identify and optimize for regimes first, then optimize rule weights within each regime.
        
        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if not regime_detector:
            raise ValueError("Regime detector must be provided for regimes-first optimization")
        
        if verbose:
            print("Step 1: Identifying market regimes...")
        
        # Create regime manager
        self.regime_manager = RegimeManager(
            regime_detector=regime_detector,
            rule_objects=self.rule_objects,
            data_handler=self.data_handler
        )
        
        # Identify regime-specific bars
        regime_bars = self.regime_manager._identify_regime_bars()
        
        if verbose:
            print("Detected regimes:")
            for regime, bars in regime_bars.items():
                print(f"  {regime.name}: {len(bars)} bars")
        
        # Optimize rule weights for each regime separately
        if verbose:
            print("\nStep 2: Optimizing rule weights for each regime...")
        
        for regime, bars in regime_bars.items():
            if len(bars) >= 50:  # Need enough data for meaningful optimization
                if verbose:
                    print(f"\nOptimizing for {regime.name} regime...")
                
                # Create data handler specific to this regime
                regime_data = self.regime_manager._create_regime_specific_data(bars)
                
                # Optimize weights for this regime
                weights = self._optimize_rule_weights(
                    method, metrics, optimization_params, verbose,
                    data_handler=regime_data
                )
                
                # Create strategy with optimized weights
                strategy = WeightedRuleStrategy(
                    rule_objects=self.rule_objects,
                    weights=weights
                )
                
                # Store in regime manager
                self.regime_manager.regime_strategies[regime] = strategy
                
                if verbose:
                    print(f"Optimized weights for {regime.name}: {weights}")
            else:
                if verbose:
                    print(f"Insufficient data for {regime.name}, using default strategy")
    
    def _optimize_joint(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Jointly optimize rule weights and regime parameters.
        
        This is a more complex approach that optimizes both rule weights and
        regime detection parameters together.
        
        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if verbose:
            print("Joint optimization of rules and regimes")
            print("Note: This is an advanced method that may take significant time")
        
        # Currently only genetic algorithm is supported for joint optimization
        if method != OptimizationMethod.GENETIC and method != OptimizationMethod.JOINT:
            raise ValueError("Joint optimization currently only supports genetic algorithm")
        
        # This would implement a more complex genetic algorithm that encodes both
        # rule weights and regime parameters in the same chromosome
        
        # For now, just use a simple implementation that optimizes rule weights
        self.optimized_rule_weights = self._optimize_rule_weights(
            method, metrics, optimization_params, verbose
        )
        
        if regime_detector:
            self.regime_manager = RegimeManager(
                regime_detector=regime_detector,
                rule_objects=self.rule_objects,
                data_handler=self.data_handler
            )
            self.regime_manager.optimize_regime_strategies(verbose=verbose)
    
    def _optimize_iterative(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Iteratively optimize rules and regimes in multiple passes.
        
        This approach alternates between optimizing rule weights and regime-specific
        strategies, potentially converging to a better solution.
        
        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if not regime_detector:
            raise ValueError("Regime detector must be provided for iterative optimization")
        
        iterations = optimization_params.get('iterations', 3)
        
        for i in range(iterations):
            if verbose:
                print(f"\nIteration {i+1}/{iterations}")
                print("Step 1: Optimizing rule weights...")
            
            # Optimize rule weights
            self.optimized_rule_weights = self._optimize_rule_weights(
                method, metrics, optimization_params, verbose
            )
            
            # Create weighted strategy
            optimized_strategy = WeightedRuleStrategy(
                rule_objects=self.rule_objects,
                weights=self.optimized_rule_weights
            )
            
            if verbose:
                print("Step 2: Optimizing regime-specific strategies...")
            
            # Create regime manager if needed
            if self.regime_manager is None:
                self.regime_manager = RegimeManager(
                    regime_detector=regime_detector,
                    rule_objects=self.rule_objects,
                    data_handler=self.data_handler
                )
            
            # Set default strategy to current optimized strategy
            self.regime_manager.default_strategy = optimized_strategy
            
            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)
            
            # Evaluate current performance
            results = self._evaluate_final_strategy(metrics)
            
            if verbose:
                print(f"Current performance after iteration {i+1}:")
                if 'sharpe' in results:
                    print(f"  Sharpe ratio: {results['sharpe']:.4f}")
                if 'total_return' in results:
                    print(f"  Total return: {results['total_return']:.2f}%")

    def _optimize_rule_weights(self, method, metrics, optimization_params, verbose, data_handler=None):
        """
        Optimize rule weights using the specified method.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            optimization_params: Additional parameters
            verbose: Whether to print progress
            data_handler: Optional specific data handler to use

        Returns:
            numpy.ndarray: Optimized weights
        """
        data_handler = data_handler or self.data_handler

        if method == OptimizationMethod.GENETIC:
            # Configure genetic optimizer
            genetic_params = optimization_params.get('genetic', {})
            optimizer = GeneticOptimizer(
                data_handler=data_handler,
                rule_objects=self.rule_objects,
                population_size=genetic_params.get('population_size', 20),
                num_parents=genetic_params.get('num_parents', 8),
                num_generations=genetic_params.get('num_generations', 50),
                mutation_rate=genetic_params.get('mutation_rate', 0.1),
                optimization_metric=metrics
            )

            # Run optimization
            return optimizer.optimize(verbose=verbose)

 
        elif method == OptimizationMethod.GRID_SEARCH:
            # Placeholder for grid search implementation
            if verbose:
                print("Grid search not fully implemented yet, using defaults")
            return np.ones(len(self.rule_objects)) / len(self.rule_objects)
            
        elif method == OptimizationMethod.BAYESIAN:
            # Placeholder for Bayesian optimization implementation
            if verbose:
                print("Bayesian optimization not fully implemented yet, using defaults")
            return np.ones(len(self.rule_objects)) / len(self.rule_objects)
            
        elif method == OptimizationMethod.RANDOM_SEARCH:
            # Placeholder for random search implementation
            if verbose:
                print("Random search not fully implemented yet, using defaults")
            return np.ones(len(self.rule_objects)) / len(self.rule_objects)
            
        else:
            # Default to equal weights
            return np.ones(len(self.rule_objects)) / len(self.rule_objects)
    
    def _evaluate_final_strategy(self, metrics):
        """
        Evaluate the final optimized strategy.
        
        Args:
            metrics: Performance metric(s) to calculate
            
        Returns:
            dict: Performance metrics
        """
        results = {}
        
        # Determine which strategy to evaluate
        if self.regime_manager:
            strategy = self.regime_manager
        elif self.optimized_rule_weights is not None:
            strategy = WeightedRuleStrategy(
                rule_objects=self.rule_objects,
                weights=self.optimized_rule_weights
            )
        else:
            # Default to equal-weighted strategy
            strategy = WeightedRuleStrategy(rule_objects=self.rule_objects)
        
        # Run backtest
        backtester = Backtester(self.data_handler, strategy)
        backtest_results = backtester.run(use_test_data=True)  # Use test data for evaluation
        
        # Extract metrics
        results['num_trades'] = backtest_results['num_trades']
        results['total_return'] = backtest_results['total_percent_return']
        results['average_return'] = backtest_results['average_log_return']
        results['trades'] = backtest_results['trades']
        
        # Calculate Sharpe ratio
        results['sharpe'] = backtester.calculate_sharpe()
        
        return results
    
    def get_optimized_strategy(self):
        """
        Get the final optimized strategy.
        
        Returns:
            object: The optimized strategy (RegimeManager or WeightedRuleStrategy)
        """
        if self.regime_manager:
            return self.regime_manager
        elif self.optimized_rule_weights is not None:
            return WeightedRuleStrategy(
                rule_objects=self.rule_objects,
                weights=self.optimized_rule_weights
            )
        else:
            # Default to equal-weighted strategy
            return WeightedRuleStrategy(rule_objects=self.rule_objects)


# Additional optimization implementations could be added as needed:

class GridSearchOptimizer:
    """
    Grid search optimizer for rule weights.
    
    This class performs an exhaustive search over specified parameter values.
    Note: For high-dimensional problems, this approach may be computationally expensive.
    """
    
    def __init__(self, data_handler, rule_objects, param_grid=None):
        """
        Initialize the grid search optimizer.
        
        Args:
            data_handler: The data handler for accessing market data
            rule_objects: List of rule objects to optimize
            param_grid: Parameter grid to search (weights discretization)
        """
        self.data_handler = data_handler
        self.rule_objects = rule_objects
        self.param_grid = param_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.best_weights = None
        self.best_score = float('-inf')
    
    def optimize(self, metric='sharpe', verbose=False):
        """
        Perform grid search to find optimal weights.
        
        Args:
            metric: Performance metric to optimize
            verbose: Whether to print progress
            
        Returns:
            numpy.ndarray: Optimized weights
        """
        # Implementation would go here
        # This would generate combinations of weights from the param_grid
        # and evaluate each combination
        
        return np.ones(len(self.rule_objects)) / len(self.rule_objects)


class BayesianOptimizer:
    """
    Bayesian optimization for rule weights.
    
    This class uses Bayesian optimization to efficiently search the parameter space.
    It balances exploration and exploitation to find good solutions with fewer evaluations.
    """
    
    def __init__(self, data_handler, rule_objects, n_iterations=50):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            data_handler: The data handler for accessing market data
            rule_objects: List of rule objects to optimize
            n_iterations: Number of iterations for optimization
        """
        self.data_handler = data_handler
        self.rule_objects = rule_objects
        self.n_iterations = n_iterations
        self.best_weights = None
        self.best_score = float('-inf')
    
    def optimize(self, metric='sharpe', verbose=False):
        """
        Perform Bayesian optimization to find optimal weights.
        
        Args:
            metric: Performance metric to optimize
            verbose: Whether to print progress
            
        Returns:
            numpy.ndarray: Optimized weights
        """
        # Implementation would go here
        # This would use a Gaussian Process to model the objective function
        # and iteratively select promising points to evaluate
        
        return np.ones(len(self.rule_objects)) / len(self.rule_objects)

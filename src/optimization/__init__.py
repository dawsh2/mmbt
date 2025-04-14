"""
Unified optimization framework for trading systems.

This framework provides a modular approach to optimizing different components
of a trading system, including rules, regime detectors, and strategies. It supports
multiple optimization methods and sequences, allowing for flexible and comprehensive
optimization with robust validation capabilities.

Key components:
- OptimizerManager: Coordinates the optimization process
- GridOptimizer: Implements grid search optimization
- GeneticOptimizer: Implements genetic algorithm optimization
- ComponentFactory: Creates component instances for optimization
- Evaluators: Evaluate component performance
- Validators: Validate optimization results with walk-forward, cross-validation, etc.

Example usage:
    # Create optimizer manager
    optimizer = OptimizerManager(data_handler)
    
    # Register components with parameter ranges
    optimizer.register_rule("sma_crossover", Rule0, 
                           {'fast_window': [5, 10, 15], 'slow_window': [20, 30, 50]})
    
    # Run optimization
    optimized_rules = optimizer.optimize(
        component_type='rule',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='sharpe',
        verbose=True
    )
    
    # Run walk-forward validation
    validation_results = optimizer.validate(
        validation_method='walk_forward',
        component_type='rule',
        method=OptimizationMethod.GENETIC,
        metrics='sharpe',
        validation_params={'window_size': 252, 'step_size': 63}
    )
"""

from enum import Enum, auto

# Import main components for easier access
from src.optimization.components import (
    OptimizableComponent, ComponentFactory,
    RuleFactory, RegimeDetectorFactory, StrategyFactory,
    WeightedStrategyFactory
)

from src.optimization.evaluators import (
    RuleEvaluator, RegimeDetectorEvaluator, StrategyEvaluator
)

from src.optimization.grid_search import GridOptimizer
from src.optimization.genetic_search import GeneticOptimizer
from src.optimization.strategies import WeightedComponentStrategy

# Import from optimizer_manager
from src.optimization.optimizer_manager import OptimizationMethod

# Import validation components
from src.optimization.validation import (
    Validator,
    WalkForwardValidator,
    CrossValidator,
    NestedCrossValidator
)

# Create OptimizationSequence enum if not already defined
class OptimizationSequence(Enum):
    """Enumeration of optimization sequencing strategies."""
    RULES_FIRST = auto()  # First optimize rules, then regimes
    REGIMES_FIRST = auto()  # First identify regimes, then optimize rules per regime
    JOINT = auto()         # Optimize rules and regime detection jointly
    ITERATIVE = auto()     # Alternate between rule and regime optimization

# Define validation methods enum
class ValidationMethod(Enum):
    """Enumeration of validation methods."""
    WALK_FORWARD = auto()  # Walk-forward validation
    CROSS_VALIDATION = auto()  # K-fold cross-validation
    NESTED_CV = auto()  # Nested cross-validation


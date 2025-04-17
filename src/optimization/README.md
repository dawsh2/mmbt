# Optimization Module

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

## Contents

- [components](#components)
- [evaluators](#evaluators)
- [example](#example)
- [factory_adapter](#factory_adapter)
- [genetic_optimizer](#genetic_optimizer)
- [genetic_search](#genetic_search)
- [grid_search](#grid_search)
- [optimizer_manager](#optimizer_manager)
- [param_utils](#param_utils)
- [strategies](#strategies)
- [base](#base)
- [cross_val](#cross_val)
- [nested_cv](#nested_cv)
- [utils](#utils)
- [walk_forward](#walk_forward)

## components

Component interfaces and factories for the optimization framework.

### Classes

#### `OptimizableComponent`

Abstract base class for any component that can be optimized.

##### Methods

###### `evaluate(data_handler, metric='return')`

Evaluate component performance using specified metric.

###### `reset()`

Reset the component's state.

#### `ComponentFactory`

Factory for creating instances of optimizable components.

##### Methods

###### `create(component_class, params)`

Create a component instance with the given parameters.

#### `RuleFactory`

Factory for creating rule instances.

##### Methods

###### `create(rule_class, params)`

Create a rule instance with the given parameters.

#### `RegimeDetectorFactory`

Factory for creating regime detector instances.

##### Methods

###### `create(detector_class, params)`

Create a regime detector instance with the given parameters.

#### `StrategyFactory`

Factory for creating strategy instances.

##### Methods

###### `create(strategy_class, params)`

Create a strategy instance with the given parameters.

#### `WeightedStrategyFactory`

Factory for creating weighted strategies.

##### Methods

###### `create(component_class, params)`

Create a weighted strategy with the given components and parameters.

Args:
    component_class: Strategy class to create
    params: Dictionary of parameters including components and weights
    
Returns:
    WeightedComponentStrategy: A weighted strategy

*Returns:* WeightedComponentStrategy: A weighted strategy

## evaluators

Evaluator classes for different component types in the optimization framework.

### Classes

#### `RuleEvaluator`

Evaluator for trading rules.

##### Methods

###### `evaluate(rule, data_handler, metric='return')`

Evaluate a rule's performance on historical data.

Args:
    rule: The rule to evaluate
    data_handler: Data handler providing market data
    metric: Performance metric ('return', 'sharpe', 'win_rate')
    
Returns:
    float: Evaluation score

*Returns:* float: Evaluation score

#### `RegimeDetectorEvaluator`

Evaluator for regime detectors.

##### Methods

###### `evaluate(detector, data_handler, metric='stability')`

Evaluate a regime detector's performance.

Args:
    detector: The regime detector to evaluate
    data_handler: Data handler providing market data
    metric: Performance metric ('stability', 'accuracy', 'strategy')
    
Returns:
    float: Evaluation score

*Returns:* float: Evaluation score

#### `StrategyEvaluator`

Evaluator for complete trading strategies.

##### Methods

###### `evaluate(strategy, data_handler, metric='sharpe')`

Evaluate a complete strategy's performance.

Args:
    strategy: The strategy to evaluate
    data_handler: Data handler providing market data
    metric: Performance metric
    
Returns:
    float: Evaluation score

*Returns:* float: Evaluation score

## example

Example usage of the unified optimization framework.

### Functions

#### `main()`

No docstring provided.

## factory_adapter

### Classes

#### `RuleFactoryAdapter`

No docstring provided.

##### Methods

###### `__init__()`

No docstring provided.

###### `create(rule_class, params)`

No docstring provided.

## genetic_optimizer

Genetic optimization module for trading system components.

### Classes

#### `GeneticOptimizer`

Optimizes components using a genetic algorithm approach with regularization
and cross-validation to reduce overfitting.

This is a refactored version of the original GeneticOptimizer adapted to work with
the component-based optimization framework.

##### Methods

###### `__init__(component_factory, evaluation_method, top_n=5, population_size=20, num_parents=8, num_generations=50, mutation_rate=0.1, random_seed=None, deterministic=False, batch_size=None, cv_folds=3, regularization_factor=0.2, balance_factor=0.3, max_weight_ratio=3.0, optimize_thresholds=True)`

Initialize the genetic optimizer.

Args:
    component_factory: Factory for creating component instances
    evaluation_method: Function to evaluate a component
    top_n: Number of top components to select
    population_size: Number of chromosomes in the population
    num_parents: Number of parents to select for mating
    num_generations: Number of generations to run the optimization
    mutation_rate: Rate of mutation in the genetic algorithm
    random_seed: Optional seed for random number generator
    deterministic: If True, ensures deterministic behavior
    batch_size: Optional batch size for fitness calculations
    cv_folds: Number of cross-validation folds
    regularization_factor: Weight given to regularization term
    balance_factor: Weight given to balancing toward equal weights
    max_weight_ratio: Maximum allowed ratio between weights
    optimize_thresholds: Whether to optimize threshold parameters

###### `_set_random_seed()`

Set the random seed if specified for reproducible results.

###### `optimize(configs, data_handler, metric='return', verbose=True)`

Optimize components using genetic algorithm.

Args:
    configs: List of (ComponentClass, param_ranges) tuples
    data_handler: Data handler providing market data
    metric: Metric to optimize for ('return', 'sharpe', etc.)
    verbose: Whether to print progress
    
Returns:
    dict: Mapping of component indices to optimized instances

*Returns:* dict: Mapping of component indices to optimized instances

###### `_run_genetic_algorithm(verbose=True)`

Run the genetic algorithm to find optimal weights.

Args:
    verbose: Whether to print progress
    
Returns:
    numpy.ndarray: Optimal weights

*Returns:* numpy.ndarray: Optimal weights

###### `plot_fitness_history()`

Plot the evolution of fitness over generations.

## genetic_search

Genetic optimization module for trading system components.

### Classes

#### `GeneticOptimizer`

Optimizes components using a genetic algorithm approach with regularization
and cross-validation to reduce overfitting.

This is a refactored version of the original GeneticOptimizer adapted to work with
the component-based optimization framework.

##### Methods

###### `__init__(component_factory, evaluation_method, top_n=5, population_size=20, num_parents=8, num_generations=50, mutation_rate=0.1, random_seed=None, deterministic=False, batch_size=None, cv_folds=3, regularization_factor=0.2, balance_factor=0.3, max_weight_ratio=3.0, optimize_thresholds=True)`

Initialize the genetic optimizer.

Args:
    component_factory: Factory for creating component instances
    evaluation_method: Function to evaluate a component
    top_n: Number of top components to select
    population_size: Number of chromosomes in the population
    num_parents: Number of parents to select for mating
    num_generations: Number of generations to run the optimization
    mutation_rate: Rate of mutation in the genetic algorithm
    random_seed: Optional seed for random number generator
    deterministic: If True, ensures deterministic behavior
    batch_size: Optional batch size for fitness calculations
    cv_folds: Number of cross-validation folds
    regularization_factor: Weight given to regularization term
    balance_factor: Weight given to balancing toward equal weights
    max_weight_ratio: Maximum allowed ratio between weights
    optimize_thresholds: Whether to optimize threshold parameters

###### `_set_random_seed()`

Set the random seed if specified for reproducible results.

###### `optimize(configs, data_handler, metric='return', verbose=True)`

Optimize components using genetic algorithm.

Args:
    configs: List of (ComponentClass, param_ranges) tuples
    data_handler: Data handler providing market data
    metric: Metric to optimize for ('return', 'sharpe', etc.)
    verbose: Whether to print progress
    
Returns:
    dict: Mapping of component indices to optimized instances

*Returns:* dict: Mapping of component indices to optimized instances

###### `_run_genetic_algorithm(verbose=True)`

Run the genetic algorithm to find optimal weights.

Args:
    verbose: Whether to print progress
    
Returns:
    numpy.ndarray: Optimal weights

*Returns:* numpy.ndarray: Optimal weights

###### `plot_fitness_history()`

Plot the evolution of fitness over generations.

## grid_search

Grid search optimization module for trading system components.

### Classes

#### `GridOptimizer`

General-purpose grid search optimizer for any component.

##### Methods

###### `__init__(component_factory, evaluation_method, top_n=5)`

Initialize the grid optimizer.

Args:
    component_factory: Factory to create component instances
    evaluation_method: Function to evaluate a component
    top_n: Number of top components to select

###### `optimize(configs, data_handler, metric='return', verbose=True)`

Optimize components using grid search.

Args:
    configs: List of (ComponentClass, param_ranges) tuples
    data_handler: Data handler providing market data
    metric: Metric to optimize for ('return', 'sharpe', etc.)
    verbose: Whether to print progress

Returns:
    dict: Mapping of component indices to optimized instances

*Returns:* dict: Mapping of component indices to optimized instances

###### `_expand_param_grid(configs, verbose=False)`

Expand parameter ranges into a grid of parameter sets.

## optimizer_manager

Enhanced OptimizerManager with integrated grid search capabilities.

### Classes

#### `OptimizerManager`

Enhanced manager for coordinating different optimization approaches.

##### Methods

###### `__init__(data_handler, rule_objects=None)`

Initialize the optimizer manager.

Args:
    data_handler: The data handler for accessing market data
    rule_objects: Optional list of pre-initialized rule objects

###### `register_component(name, component_type, component_class, params_range=None, instance=None)`

Register a component for optimization.

Args:
    name: Unique identifier for the component
    component_type: Type of component ('rule', 'regime_detector', etc.)
    component_class: Class of the component
    params_range: Optional parameter ranges for optimization
    instance: Optional pre-initialized instance

###### `register_rule(name, rule_class, params_range=None, instance=None)`

Register a trading rule.

###### `register_regime_detector(name, detector_class, params_range=None, instance=None)`

Register a regime detector.

###### `optimize(component_type, method, components=None, metrics='return', verbose=True)`

Optimize components of a specific type.

Args:
    component_type: Type of component to optimize ('rule', 'regime_detector', etc.)
    method: Optimization method to use
    components: List of component names to optimize (or None for all registered)
    metrics: Performance metric(s) to optimize for
    verbose: Whether to print progress information
    **kwargs: Additional parameters for specific optimization methods
        - top_n: Number of top components to select
        - genetic: Dictionary of genetic algorithm parameters
            - population_size: Size of population
            - num_generations: Number of generations to run
            - mutation_rate: Rate of mutation
            - num_parents: Number of parents to select
            - cv_folds: Number of cross-validation folds
            - regularization_factor: Strength of regularization
            - optimize_thresholds: Whether to optimize thresholds
        - sequence: Optional optimization sequence to use
        - regime_detector: Optional regime detector for regime-based optimization

Returns:
    dict or object: Optimized components or strategy depending on method and component type

*Returns:* dict or object: Optimized components or strategy depending on method and component type

###### `get_optimized_components(component_type)`

Get optimized components of a specific type.

###### `optimize_regime_specific_rules(regime_detector, optimization_method, optimization_metric='return', verbose=True)`

Optimize rules specifically for different market regimes.

Args:
    regime_detector: The regime detector to use
    optimization_method: Method to use for optimization
    optimization_metric: Metric to optimize for
    verbose: Whether to print progress
    
Returns:
    dict: Mapping from regime to optimized rules

*Returns:* dict: Mapping from regime to optimized rules

###### `_identify_regime_bars(regime_detector)`

Identify which bars belong to each regime.

###### `_create_regime_specific_data(regime_bars)`

Create a data handler with only bars from a specific regime.

###### `_optimize_rules_first(method, metrics, regime_detector, optimization_params, verbose)`

Optimize rule weights first, then optimize for regimes.

Args:
    method: Optimization method
    metrics: Performance metric to optimize
    regime_detector: Regime detector to use
    optimization_params: Additional parameters
    verbose: Whether to print progress

###### `_optimize_regimes_first(method, metrics, regime_detector, optimization_params, verbose)`

Optimize for regimes first, then optimize rule weights for each regime.

Args:
    method: Optimization method
    metrics: Performance metric to optimize
    regime_detector: Regime detector to use
    optimization_params: Additional parameters
    verbose: Whether to print progress

###### `_optimize_iterative(method, metrics, regime_detector, optimization_params, verbose)`

Iteratively optimize rules and regimes in multiple passes.

Args:
    method: Optimization method
    metrics: Performance metric to optimize
    regime_detector: Regime detector to use
    optimization_params: Additional parameters
    verbose: Whether to print progress

###### `_optimize_joint(method, metrics, regime_detector, optimization_params, verbose)`

Jointly optimize rule weights and regime parameters.

Args:
    method: Optimization method
    metrics: Performance metric to optimize
    regime_detector: Regime detector to use
    optimization_params: Additional parameters
    verbose: Whether to print progress

###### `_optimize_joint(method, metrics, regime_detector, optimization_params, verbose)`

Jointly optimize rule weights and regime parameters.

Args:
    method: Optimization method
    metrics: Performance metric to optimize
    regime_detector: Regime detector to use
    optimization_params: Additional parameters
    verbose: Whether to print progress

###### `validate(validation_method, component_type='rule', method, components=None, metrics='sharpe', verbose=True)`

Validate optimization of components using cross-validation or walk-forward.

Args:
    validation_method: Validation method to use ('cross_validation', 'walk_forward', etc.)
    component_type: Type of component to optimize ('rule', 'regime_detector', etc.)
    method: Optimization method to use
    components: List of component names to optimize (or None for all registered)
    metrics: Performance metric(s) to optimize for
    verbose: Whether to print progress information
    **kwargs: Additional parameters
        - validation_params: Dictionary of validation-specific parameters
            - window_size: Size of windows for walk-forward (default: 252)
            - step_size: Step size for walk-forward (default: 63)
            - train_pct: Train percentage for walk-forward (default: 0.7)
            - n_folds: Number of folds for cross-validation (default: 5)
        - ... (other optimization parameters)

Returns:
    dict: Validation results

*Returns:* dict: Validation results

## param_utils

### Functions

#### `validate_parameters(params, required_params)`

Validate that all required parameters are present

#### `ensure_rule_parameters(rule_class_name, params)`

Add missing parameters with default values for known rule types

## strategies

Strategy implementations for the optimization framework.

This module is maintained for backward compatibility.
The main implementation is now in strategies/weighted_strategy.py

## base

Base validator interface for the optimization framework.

### Classes

#### `Validator`

Base abstract class for validation components.

##### Methods

###### `validate(component_factory, optimization_method, data_handler, configs=None, metric='sharpe', verbose=True)`

Validate a component or strategy using the specified method.

Args:
    component_factory: Factory for creating component instances
    optimization_method: Method to use for optimization
    data_handler: Data handler providing market data
    configs: Component configuration for optimization
    metric: Performance metric to optimize
    verbose: Whether to print progress information
    **kwargs: Additional parameters
    
Returns:
    dict: Validation results

*Returns:* dict: Validation results

## cross_val

Cross-validation implementation for trading system optimization.

This module provides k-fold cross-validation to assess strategy robustness
by dividing the dataset into k folds and using each fold as a test set.

### Classes

#### `CrossValidator`

Cross-Validation for trading strategies.

This class performs k-fold cross-validation to assess strategy robustness
by dividing the dataset into k folds and using each fold as a test set.

##### Methods

###### `__init__(n_folds=5, top_n=5, plot_results=True)`

Initialize the cross-validator.

Args:
    n_folds: Number of folds for cross-validation
    top_n: Number of top components to select
    plot_results: Whether to plot results after validation

###### `validate(component_factory, optimization_method, data_handler, configs=None, metric='sharpe', verbose=True)`

Run cross-validation on components.

Args:
    component_factory: Factory for creating component instances
    optimization_method: Method to use for optimization
    data_handler: Data handler providing market data
    configs: Component configuration for optimization
    metric: Performance metric to optimize
    verbose: Whether to print progress information
    **kwargs: Additional parameters for optimization
    
Returns:
    dict: Summary of validation results

*Returns:* dict: Summary of validation results

###### `_load_full_data(data_handler)`

Load the full dataset from the data handler.

Args:
    data_handler: The data handler
    
Returns:
    list: All data points

*Returns:* list: All data points

###### `_create_folds(data)`

Create k-folds from the data.

Args:
    data: The full dataset
    
Returns:
    list: List of (train_data, test_data) tuples

*Returns:* list: List of (train_data, test_data) tuples

###### `_calculate_summary(fold_results)`

Calculate summary statistics from fold results.

Args:
    fold_results: List of result dictionaries for each fold
    
Returns:
    dict: Summary statistics

*Returns:* dict: Summary statistics

###### `_plot_results(fold_results, all_trades)`

Plot the cross-validation results.

Args:
    fold_results: List of result dictionaries for each fold
    all_trades: List of all trades across folds

## nested_cv

Nested cross-validation implementation for trading system optimization.

This module provides nested cross-validation with an inner loop for
hyperparameter optimization and an outer loop for performance evaluation.

### Classes

#### `NestedCrossValidator`

Nested Cross-Validation for more robust evaluation of trading strategies.

This class performs nested cross-validation with an inner loop for
hyperparameter optimization and an outer loop for performance evaluation.

##### Methods

###### `__init__(outer_folds=5, inner_folds=3, top_n=5, optimization_methods=None, plot_results=True)`

Initialize the nested cross-validator.

Args:
    outer_folds: Number of outer folds for final evaluation
    inner_folds: Number of inner folds for hyperparameter optimization
    top_n: Number of top components to select
    optimization_methods: List of optimization methods to compare
    plot_results: Whether to plot results after validation

###### `validate(component_factory, optimization_method, data_handler, configs=None, metric='sharpe', verbose=True)`

Run nested cross-validation on components.

Args:
    component_factory: Factory for creating component instances
    optimization_method: Method to use for optimization
    data_handler: Data handler providing market data
    configs: Component configuration for optimization
    metric: Performance metric to optimize
    verbose: Whether to print progress information
    **kwargs: Additional parameters for optimization
    
Returns:
    dict: Summary of validation results

*Returns:* dict: Summary of validation results

###### `_load_full_data(data_handler)`

Load the full dataset from the data handler.

Args:
    data_handler: The data handler
    
Returns:
    list: All data points

*Returns:* list: All data points

###### `_create_outer_folds(data)`

Create outer folds for nested cross-validation.

Args:
    data: The full dataset
    
Returns:
    list: List of (train_data, test_data) tuples

*Returns:* list: List of (train_data, test_data) tuples

###### `_run_inner_cv(train_data, component_factory, configs, metric, verbose)`

Run inner cross-validation to select the best method.

Args:
    train_data: Training data for the current outer fold
    component_factory: Factory for creating component instances
    configs: Component configuration for optimization
    metric: Performance metric to optimize
    verbose: Whether to print progress information
    **kwargs: Additional parameters for optimization
    
Returns:
    dict: Results for each optimization method

*Returns:* dict: Results for each optimization method

###### `_calculate_summary(fold_results)`

Calculate summary statistics from fold results.

Args:
    fold_results: List of result dictionaries
    
Returns:
    dict: Summary statistics

*Returns:* dict: Summary statistics

###### `_plot_results(method_results, best_method_results, all_trades)`

Plot the nested cross-validation results.

Args:
    method_results: Dictionary of results for each method
    best_method_results: Results using the best method selection
    all_trades: List of all trades across folds

## utils

Utility functions and classes for validation components.

### Functions

#### `create_train_test_windows(data, window_size, step_size, train_pct=0.7)`

Create training and testing windows for walk-forward validation.

Args:
    data: Full dataset
    window_size: Size of each window
    step_size: Number of steps to roll forward between windows
    train_pct: Percentage of window to use for training
    
Returns:
    list: List of (train_window, test_window) tuples

*Returns:* list: List of (train_window, test_window) tuples

### Classes

#### `WindowDataHandler`

Data handler for a specific window or fold.

##### Methods

###### `__init__(train_data, test_data)`

Initialize the window data handler.

Args:
    train_data: Training data for this window
    test_data: Testing data for this window

###### `get_next_train_bar()`

Get the next training bar.

###### `get_next_test_bar()`

Get the next testing bar.

###### `reset_train()`

Reset the training data pointer.

###### `reset_test()`

Reset the testing data pointer.

## walk_forward

### Classes

#### `WalkForwardValidator`

Walk-Forward Validation for trading strategies.

This class performs rolling walk-forward validation to test strategy robustness
by repeatedly training on in-sample data and testing on out-of-sample data.

##### Methods

###### `__init__(window_size=252, step_size=63, train_pct=0.7, top_n=5, plot_results=True)`

Initialize the walk-forward validator.

Args:
    window_size: Size of each window in trading days
    step_size: Number of days to roll forward between windows
    train_pct: Percentage of window to use for training
    top_n: Number of top rules to select
    plot_results: Whether to plot results after validation

###### `validate(component_factory, optimization_method, data_handler, configs=None, metric='sharpe', verbose=True)`

Run walk-forward validation.

Args:
    component_factory: Factory for creating component instances
    optimization_method: Method to use for optimization
    data_handler: Data handler providing market data
    configs: Component configuration for optimization
    metric: Performance metric to optimize
    verbose: Whether to print progress information
    **kwargs: Additional parameters for optimization
    
Returns:
    dict: Summary of validation results

*Returns:* dict: Summary of validation results

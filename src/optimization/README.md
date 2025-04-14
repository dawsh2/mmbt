# Optimization Module Documentation

The Optimization module provides a modular framework for optimizing trading strategy components, finding optimal parameters, and validating results. It supports multiple optimization methods including grid search, genetic algorithms, and walk-forward validation.

## Core Concepts

**OptimizerManager**: Central coordinator that manages different optimization approaches.  
**Component Factory**: Creates component instances with specific parameters for evaluation.  
**Evaluators**: Assess the performance of components using specified metrics.  
**Optimization Methods**: Different algorithms for parameter optimization (grid search, genetic).  
**Validators**: Cross-validation and walk-forward approaches to ensure robustness.

## Basic Usage

```python
from optimization import OptimizerManager, OptimizationMethod
from rules import SMARule, RSIRule
from data_handler import CSVDataHandler

# Create data handler
data_handler = CSVDataHandler("path/to/data.csv")

# Create optimizer
optimizer = OptimizerManager(data_handler)

# Register components with parameter ranges to optimize
optimizer.register_rule("sma_rule", SMARule, 
                       {"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]})

optimizer.register_rule("rsi_rule", RSIRule,
                       {"period": [7, 14, 21], "overbought": [70, 80], "oversold": [20, 30]})

# Optimize using grid search
optimized_rules = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GRID_SEARCH,
    metrics='sharpe',
    verbose=True
)

# Use optimized components
for name, rule in optimized_rules.items():
    print(f"Optimized {name}: {rule}")
```

## API Reference

### OptimizerManager

Central manager for coordinating different optimization approaches.

**Constructor Parameters:**
- `data_handler` (DataHandler): The data handler for accessing market data
- `rule_objects` (list, optional): Optional list of pre-initialized rule objects

**Methods:**

#### register_component(name, component_type, component_class, params_range=None, instance=None)

Register a component for optimization.

**Parameters:**
- `name` (str): Unique identifier for the component
- `component_type` (str): Type of component ('rule', 'regime_detector', etc.)
- `component_class` (class): Class of the component
- `params_range` (dict, optional): Parameter ranges for optimization
- `instance` (object, optional): Pre-initialized instance

**Example:**
```python
optimizer.register_component(
    name="sma_rule",
    component_type="rule",
    component_class=SMARule,
    params_range={"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]}
)
```

#### optimize(component_type, method=OptimizationMethod.GRID_SEARCH, components=None, metrics='return', verbose=True, **kwargs)

Optimize components of a specific type.

**Parameters:**
- `component_type` (str): Type of component to optimize ('rule', 'regime_detector', etc.)
- `method` (OptimizationMethod): Optimization method to use
- `components` (list, optional): List of component names to optimize (or None for all registered)
- `metrics` (str): Performance metric to optimize for
- `verbose` (bool): Whether to print progress information
- `**kwargs`: Additional parameters for specific optimization methods

**Returns:**
- `dict` or `object`: Optimized components or strategy

**Example:**
```python
optimized_rules = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GENETIC,
    metrics='sharpe',
    genetic={
        'population_size': 50,
        'num_generations': 30,
        'mutation_rate': 0.1
    }
)
```

#### validate(validation_method, component_type='rule', method=OptimizationMethod.GENETIC, components=None, metrics='sharpe', verbose=True, **kwargs)

Validate optimization of components using cross-validation or walk-forward.

**Parameters:**
- `validation_method` (str): Validation method ('cross_validation', 'walk_forward', etc.)
- `component_type` (str): Type of component to optimize
- `method` (OptimizationMethod): Optimization method to use
- `components` (list, optional): List of component names to optimize
- `metrics` (str): Performance metric to optimize for
- `verbose` (bool): Whether to print progress information
- `**kwargs`: Additional parameters including validation_params

**Returns:**
- `dict`: Validation results

**Example:**
```python
validation_results = optimizer.validate(
    validation_method='walk_forward',
    component_type='rule',
    method=OptimizationMethod.GRID_SEARCH,
    metrics='sharpe',
    validation_params={
        'window_size': 252,  # One year of trading days
        'step_size': 63      # Quarterly steps
    }
)
```

### GridOptimizer

General-purpose grid search optimizer for any component.

**Constructor Parameters:**
- `component_factory` (ComponentFactory): Factory to create component instances
- `evaluation_method` (callable): Function to evaluate a component
- `top_n` (int, optional): Number of top components to select (default: 5)

**Methods:**

#### optimize(configs, data_handler, metric='return', verbose=True)

Optimize components using grid search.

**Parameters:**
- `configs` (list): List of (ComponentClass, param_ranges) tuples
- `data_handler` (DataHandler): Data handler providing market data
- `metric` (str): Metric to optimize for ('return', 'sharpe', etc.)
- `verbose` (bool): Whether to print progress

**Returns:**
- `dict`: Mapping of component indices to optimized instances

### GeneticOptimizer

Optimizes components using a genetic algorithm approach.

**Constructor Parameters:**
- `component_factory` (ComponentFactory): Factory for creating component instances
- `evaluation_method` (callable): Function to evaluate a component
- `top_n` (int, optional): Number of top components to select (default: 5)
- `population_size` (int, optional): Size of the population (default: 20)
- `num_generations` (int, optional): Number of generations to run (default: 50)
- `mutation_rate` (float, optional): Rate of mutation (default: 0.1)
- `random_seed` (int, optional): Seed for reproducibility
- `cv_folds` (int, optional): Number of cross-validation folds (default: 3)
- `regularization_factor` (float, optional): Weight for regularization (default: 0.2)

**Methods:**

#### optimize(configs, data_handler, metric='return', verbose=True)

Optimize components using genetic algorithm.

**Parameters:**
- `configs` (list): List of (ComponentClass, param_ranges) tuples
- `data_handler` (DataHandler): Data handler providing market data
- `metric` (str): Metric to optimize for ('return', 'sharpe', etc.)
- `verbose` (bool): Whether to print progress

**Returns:**
- `dict`: Mapping of component indices to optimized instances

### WeightedComponentStrategy

Strategy that combines signals from multiple components using weights.

**Constructor Parameters:**
- `components` (list): List of component objects
- `weights` (numpy.ndarray, optional): List of weights for each component
- `buy_threshold` (float, optional): Threshold above which to generate a buy signal (default: 0.5)
- `sell_threshold` (float, optional): Threshold below which to generate a sell signal (default: -0.5)

**Methods:**

#### on_bar(event)

Process a bar event and generate a weighted signal.

**Parameters:**
- `event`: Bar event containing market data

**Returns:**
- `Signal`: Combined signal based on weighted components

### Validation Module

The validation submodule provides tools for ensuring robustness of optimized strategies.

#### Validator (Base Class)

Abstract base class for validation components.

**Methods:**

#### validate(component_factory, optimization_method, data_handler, configs=None, metric='sharpe', verbose=True, **kwargs)

Validate a component or strategy using the specified method.

**Parameters:**
- `component_factory` (ComponentFactory): Factory for creating component instances
- `optimization_method` (OptimizationMethod): Method to use for optimization
- `data_handler` (DataHandler): Data handler providing market data
- `configs` (list, optional): Component configurations for optimization
- `metric` (str): Performance metric to optimize
- `verbose` (bool): Whether to print progress information
- `**kwargs`: Additional parameters

**Returns:**
- `dict`: Validation results

#### WalkForwardValidator

Performs rolling walk-forward validation to test strategy robustness.

**Constructor Parameters:**
- `window_size` (int, optional): Size of each window in trading days (default: 252)
- `step_size` (int, optional): Number of days to roll forward between windows (default: 63)
- `train_pct` (float, optional): Percentage of window to use for training (default: 0.7)
- `top_n` (int, optional): Number of top rules to select (default: 5)
- `plot_results` (bool, optional): Whether to plot results (default: True)

**Example:**
```python
from optimization.validation import WalkForwardValidator

validator = WalkForwardValidator(
    window_size=252,  # One year of trading days
    step_size=63,     # Quarterly steps
    train_pct=0.7     # 70% for training, 30% for testing
)

results = validator.validate(
    component_factory=factory,
    optimization_method=OptimizationMethod.GRID_SEARCH,
    data_handler=data_handler,
    configs=rule_configs,
    metric='sharpe'
)
```

#### CrossValidator

Performs k-fold cross-validation to assess strategy robustness.

**Constructor Parameters:**
- `n_folds` (int, optional): Number of folds for cross-validation (default: 5)
- `top_n` (int, optional): Number of top components to select (default: 5)
- `plot_results` (bool, optional): Whether to plot results (default: True)

**Example:**
```python
from optimization.validation import CrossValidator

validator = CrossValidator(n_folds=5)

results = validator.validate(
    component_factory=factory,
    optimization_method=OptimizationMethod.GENETIC,
    data_handler=data_handler,
    configs=rule_configs,
    metric='sharpe'
)
```

#### NestedCrossValidator

Performs nested cross-validation with an inner loop for hyperparameter optimization and an outer loop for performance evaluation.

**Constructor Parameters:**
- `outer_folds` (int, optional): Number of outer folds for final evaluation (default: 5)
- `inner_folds` (int, optional): Number of inner folds for hyperparameter optimization (default: 3)
- `top_n` (int, optional): Number of top components to select (default: 5)
- `optimization_methods` (list, optional): List of optimization methods to compare
- `plot_results` (bool, optional): Whether to plot results (default: True)

#### WindowDataHandler

Data handler for a specific window or fold in validation.

**Constructor Parameters:**
- `train_data` (list or DataFrame): Training data for this window
- `test_data` (list or DataFrame): Testing data for this window

**Methods:**
- `get_next_train_bar()`: Get the next training bar
- `get_next_test_bar()`: Get the next testing bar
- `reset_train()`: Reset the training data pointer
- `reset_test()`: Reset the testing data pointer

#### create_train_test_windows(data, window_size, step_size, train_pct=0.7)

Create training and testing windows for walk-forward validation.

**Parameters:**
- `data` (list or DataFrame): Full dataset
- `window_size` (int): Size of each window
- `step_size` (int): Number of steps to roll forward between windows
- `train_pct` (float): Percentage of window to use for training

**Returns:**
- `list`: List of (train_window, test_window) tuples

## Advanced Usage

### Optimization Flow

Optimization flow for different strategies:

```python
from optimization import OptimizerManager, OptimizationMethod, OptimizationSequence
from regime_detection import TrendStrengthRegimeDetector

# Create optimizer manager
optimizer = OptimizerManager(data_handler)

# Register rules
optimizer.register_rule("sma_rule", SMARule, 
                       {"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]})
optimizer.register_rule("rsi_rule", RSIRule,
                       {"period": [7, 14, 21], "overbought": [70, 80], "oversold": [20, 30]})

# Register regime detector
optimizer.register_regime_detector("trend_detector", TrendStrengthRegimeDetector,
                                  {"adx_period": [10, 14, 20], "adx_threshold": [20, 25, 30]})

# Create detector
detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# Rules-first approach: First optimize rules, then optimize for regimes
strategy = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GENETIC,
    metrics='sharpe',
    sequence=OptimizationSequence.RULES_FIRST,
    regime_detector=detector,
    genetic={
        'population_size': 50,
        'num_generations': 30
    }
)
```

### Cross-Validation of Strategies

```python
from optimization.validation import CrossValidator
from optimization import OptimizationMethod

# Create cross-validator with 5 folds
validator = CrossValidator(n_folds=5, top_n=3)

# Define components to validate
configs = [
    (SMARule, {"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]}),
    (RSIRule, {"period": [7, 14, 21], "overbought": [70, 80], "oversold": [20, 30]})
]

# Run cross-validation with grid search optimization
results = validator.validate(
    component_factory=factory,
    optimization_method=OptimizationMethod.GRID_SEARCH,
    data_handler=data_handler,
    configs=configs,
    metric='sharpe',
    verbose=True
)

# Get average metrics across folds
print(f"Average Sharpe Ratio: {results['avg_sharpe']:.4f}")
print(f"Average Return: {results['avg_return']:.2f}%")
print(f"Parameter Stability: {results['parameter_stability']:.2f}")
```

### Walk-Forward Optimization

```python
from optimization.validation import WalkForwardValidator
from optimization import OptimizationMethod

# Create walk-forward validator
validator = WalkForwardValidator(
    window_size=252,  # One year of trading days
    step_size=63,     # Quarterly steps
    train_pct=0.7     # 70% for training, 30% for testing
)

# Run walk-forward validation
results = validator.validate(
    component_factory=factory,
    optimization_method=OptimizationMethod.GENETIC,
    data_handler=data_handler,
    configs=configs,
    metric='sharpe',
    verbose=True,
    genetic={
        'population_size': 30,
        'num_generations': 20,
        'mutation_rate': 0.1
    }
)

# Display window results
for i, window_result in enumerate(results['window_results']):
    print(f"Window {i+1}: Train Return: {window_result['train_return']:.2f}%, " +
          f"Test Return: {window_result['test_return']:.2f}%")

# Overall out-of-sample performance
print(f"Out-of-sample Sharpe Ratio: {results['oos_sharpe']:.4f}")
print(f"Out-of-sample Return: {results['oos_return']:.2f}%")
```

### Combining Multiple Optimization Methods

```python
from optimization import OptimizerManager, OptimizationMethod

# First use grid search for fast initial exploration
grid_results = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GRID_SEARCH,
    metrics='sharpe'
)

# Take the best parameters as starting point for genetic optimization
best_params = optimizer.best_params.copy()

# Refine parameters with genetic algorithm
refined_results = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GENETIC,
    metrics='sharpe',
    genetic={
        'initial_population': best_params,
        'population_size': 50,
        'num_generations': 50
    }
)
```

## Best Practices

1. **Avoid overfitting**: Always validate optimized parameters with out-of-sample testing

2. **Use appropriate metrics**: Choose performance metrics aligned with your trading goals

3. **Consider multiple objectives**: Balance return, risk, and other factors important to your strategy

4. **Regularize parameters**: Apply constraints to prevent extreme parameter values

5. **Start simple**: Begin with grid search to explore parameter space before using more complex methods

6. **Monitor stability**: Check if optimal parameters are stable across validation periods

7. **Balance exploration and exploitation**: Especially important in genetic algorithms

8. **Use cross-validation**: Particularly for strategies with fewer trades to ensure robustness

9. **Separate training/testing data**: Strictly maintain separation during validation

10. **Realistic simulation**: Include transaction costs, slippage, and other realistic constraints
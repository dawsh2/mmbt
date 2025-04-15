# Optimization Framework: Parameter Handling Guide

## Overview

This document explains the parameter handling mechanism in the optimization framework, addressing common issues and providing best practices for correctly passing and handling parameters throughout the optimization process.

## Core Components

The optimization framework consists of several interrelated components:

1. **OptimizerManager**: Coordinates the optimization process
2. **GridOptimizer**: Implements grid search optimization by testing different parameter combinations
3. **ComponentFactory / RuleFactory**: Creates component instances with specified parameters
4. **Rule / Component**: The actual instances that get created with different parameter sets

## Parameter Flow

Parameters flow through the system in this sequence:

1. Parameter ranges are registered with the `OptimizerManager`
2. The `OptimizerManager` passes these to the appropriate optimizer (e.g., `GridOptimizer`) 
3. The optimizer expands the parameter grid into all possible combinations
4. For each parameter set, the optimizer calls a factory to create a component instance
5. The factory passes the parameters to the component constructor
6. The component constructor applies and validates the parameters

## Known Issues and Solutions

### Issue 1: Parameters Not Correctly Passed to Component Constructors

**Problem**: The `RuleFactory.create()` method in `src/optimization/components.py` was passing parameters as a positional argument instead of a named argument.

**Solution**: Update the factory's `create()` method to pass parameters explicitly as a named argument:

```python
# In src/optimization/components.py
def create(self, rule_class, params):
    """Create a rule instance with the given parameters."""
    return rule_class(params=params)  # Pass as a named parameter
```

### Issue 2: Duplicate Factory Implementations

**Problem**: There are two `RuleFactory` implementations in the codebase:
- A simple one in `src/optimization/components.py`
- A more comprehensive one in `src/rules/rule_factory.py`

**Solution**: Either:
1. Use the comprehensive factory throughout the codebase, or
2. Ensure both factories pass parameters correctly

For consistency, it's recommended to use the more comprehensive factory from `src/rules/rule_factory.py` where possible.

### Issue 3: Invalid Parameter Combinations

**Problem**: When expanding parameter grids, some combinations may be invalid and fail validation.

**Solution**: Add try/except blocks in the `GridOptimizer.optimize()` method to catch and skip invalid parameter combinations:

```python
try:
    # Create a fresh component instance with these specific parameters
    import copy
    params_copy = copy.deepcopy(params)
    component = self.component_factory.create(component_class, params_copy)
    
    # Evaluate the component
    score = self.evaluate(component, data_handler, metric)
    component_performances[tuple(sorted(params.items()))] = score
except ValueError as e:
    # Skip invalid parameter combinations
    if verbose:
        print(f"  Skipping invalid params: {e}")
```

### Issue 4: Parameter Reference Issues

**Problem**: Shared parameter references can lead to unintended modifications.

**Solution**: Always make deep copies of parameter dictionaries before passing them:

```python
import copy
params_copy = copy.deepcopy(params)
component = self.component_factory.create(component_class, params_copy)
```

## Best Practices

### 1. Parameter Registration

When registering parameters for optimization, ensure that:
- Parameter ranges are clearly defined
- Parameter combinations make logical sense
- Default values are provided where appropriate

```python
optimizer.register_rule(
    "sma_rule",
    SMAcrossoverRule,  
    {
        'fast_window': [5, 10, 15],  # Ensure fast_window < slow_window
        'slow_window': [30, 40, 50], 
        'smooth_signals': [True, False]
    }
)
```

### 2. Parameter Validation

Implement thorough parameter validation in component constructors:

```python
def _validate_params(self):
    """Validate the parameters provided to the rule."""
    if self.params['fast_window'] >= self.params['slow_window']:
        raise ValueError("Fast window must be smaller than slow window")
```

### 3. Parameter Copying

Always make deep copies of parameters to avoid reference issues:

```python
import copy
params_copy = copy.deepcopy(params)
```

### 4. Error Handling

Use proper error handling to gracefully skip invalid parameter combinations:

```python
try:
    # Create and evaluate component with parameters
except ValueError as e:
    # Skip this combination and log the error
    if verbose:
        print(f"Skipping invalid parameter set: {e}")
```

### 5. Debug Output

Add debug output to trace parameter flow:

```python
if verbose:
    print(f"Creating {component_class.__name__} with params: {params}")
    print(f"Actual component params: {component.params}")
```

## Testing Parameter Handling

To verify correct parameter handling, create a "canary" rule with a unique fingerprint for each parameter set:

```python
class CanaryRule(Rule):
    """
    A special test rule that produces predictable results based on its parameters.
    Used to test optimizer parameter handling.
    """
    
    @classmethod
    def default_params(cls):
        return {
            'multiplier': 1.0,
            'threshold': 0.5,
            'buy_period': 5
        }
    
    def __init__(self, name="canary", params=None, description=""):
        super().__init__(name or "canary_rule", params or self.default_params(), 
                        description or "Test rule for parameter handling")
        self.params_fingerprint = f"{self.params['multiplier']}-{self.params['threshold']}-{self.params['buy_period']}"
        print(f"Created CanaryRule with params fingerprint: {self.params_fingerprint}")
```

Run optimization with this rule to verify that each parameter set creates a unique instance with the correct parameter values.

## Implementation Details

### OptimizerManager.optimize()

The `optimize()` method in `OptimizerManager` coordinates the optimization process:

```python
def optimize(self, component_type, method=OptimizationMethod.GRID_SEARCH, 
             components=None, metrics='return', verbose=True, **kwargs):
    """
    Optimize components of a specific type.
    """
    # ... existing code ...
    
    # Choose factory and evaluator based on component type
    if component_type == 'rule':
        factory = RuleFactory()  # Use the appropriate factory
        evaluator = RuleEvaluator.evaluate
    # ... other component types ...
    
    # Run optimization
    if method == OptimizationMethod.GRID_SEARCH:
        optimizer = GridOptimizer(factory, evaluator, top_n=top_n)
        optimized = optimizer.optimize(configs, self.data_handler, metrics, verbose)
    # ... other methods ...
    
    return optimized
```

### GridOptimizer.optimize()

The `optimize()` method in `GridOptimizer` handles parameter sets and component evaluation:

```python
def optimize(self, configs, data_handler, metric='return', verbose=True):
    """
    Optimize components using grid search.
    """
    # Expand parameter grids
    expanded_configs = self._expand_param_grid(configs, verbose)
    
    # ... evaluation logic with try/except blocks for parameter validation ...
    
    # Create optimized components with the best parameters
    self.best_components = {}
    for idx, data in top_components:
        # Create a fresh instance with the best parameters
        import copy
        best_params = copy.deepcopy(self.best_params[idx])
        # ... create component with best parameters ...
    
    return self.best_components
```

## Summary

Properly handling parameters in the optimization framework requires:

1. Correct parameter passing (using named parameters)
2. Deep copying parameters to avoid reference issues
3. Proper validation of parameter combinations
4. Error handling for invalid parameter sets
5. Consistent use of factory implementations

By following these guidelines, the optimization framework can correctly test different parameter combinations and identify the optimal parameters for trading rules and other components.
Optimization/components.py, and the modules that depend on it, need to be refactored to use the appropriate factory modules from the native component modules (e.g, src/rules/rules_factory). This is non-urgent, but shouldn't be forgotten. 

# Optimization Framework Update Notes

## Issue Summary

We identified critical issues in the parameter handling mechanisms of the optimization framework:

1. **Parameters not correctly passed to components** during optimization
2. **Factory implementation duplications** causing inconsistent behavior
3. **Lack of error handling** for invalid parameter combinations
4. **Documentation gaps** regarding parameter flow and validation

## Key Fixes

### 1. Parameter Passing Fix

Updated `RuleFactory.create()` in `src/optimization/components.py` to pass parameters as named arguments:

```python
def create(self, rule_class, params):
    """Create a rule instance with the given parameters."""
    return rule_class(params=params)  # Pass as a named parameter instead of positional
```

### 2. Error Handling for Invalid Parameter Combinations

Added try/except blocks in `GridOptimizer.optimize()` to gracefully handle invalid parameter sets:

```python
try:
    # Create and evaluate component with parameters
    component = self.component_factory.create(component_class, params_copy)
    score = self.evaluate(component, data_handler, metric)
    component_performances[tuple(sorted(params.items()))] = score
except ValueError as e:
    # Skip invalid parameter combinations
    if verbose:
        print(f"  Skipping invalid params: {e}")
```

### 3. Parameter Reference Issues

Implemented deep copying of parameter dictionaries to prevent reference issues:

```python
import copy
params_copy = copy.deepcopy(params)
component = self.component_factory.create(component_class, params_copy)
```

## Documentation Updates

### Parameter Flow Clarification

Added documentation of how parameters flow through the system:
1. Parameter ranges registered with `OptimizerManager`
2. Expanded into combinations by `GridOptimizer`
3. Used to create component instances via factories
4. Validated by component constructors

### Factory Implementation Notes

Documented the two different `RuleFactory` implementations:
- Simple version in `src/optimization/components.py`
- Comprehensive version in `src/rules/rule_factory.py`

### Best Practices

Added guidance on:
- Parameter validation in components
- Error handling for invalid combinations
- Debugging parameter issues
- Testing parameter handling with diagnostic rules

## Testing Approach

Implemented a "canary" rule with parameter fingerprinting to verify correct parameter application:

```python
def __init__(self, name="canary", params=None, description=""):
    super().__init__(name, params or self.default_params(), description)
    self.params_fingerprint = f"{self.params['multiplier']}-{self.params['threshold']}-{self.params['buy_period']}"
    print(f"Created CanaryRule with params fingerprint: {self.params_fingerprint}")
```

## Next Steps

1. **Refactor factories** to use the comprehensive `RuleFactory` throughout the codebase
2. **Add parameter validation tests** for all rule types
3. **Improve debug logging** during optimization to make parameter issues more visible
4. **Apply similar fixes** to other component types (RegimeDetector, Strategy, etc.)
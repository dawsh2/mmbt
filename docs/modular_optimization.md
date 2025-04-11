# Modular Optimization Architecture for Trading Systems

## Introduction

Modern trading systems require flexibility in how optimization is applied. This document outlines an architecture for creating a modular optimization framework that enables different optimization methods, sequences, and targets to be combined and interchanged easily.

## Core Architecture Components

### 1. Optimization Targets

These are the elements of the trading system that can be optimized:

- **Rule Parameters**: Parameters for individual trading rules (e.g., moving average periods)
- **Rule Weights**: How much influence each rule has in the final signal
- **Regime Detection Parameters**: How different market regimes are identified
- **Regime-Specific Weights**: Different rule weights for each market regime
- **Position Sizing Parameters**: Parameters controlling trade size based on confidence/volatility
- **Risk Management Parameters**: Stop-loss, take-profit, max drawdown constraints

### 2. Optimization Methods

Different algorithms that can be applied to any optimization target:

- **Genetic Algorithms**: Population-based evolutionary optimization
- **Grid Search**: Exhaustive search through a parameter space
- **Bayesian Optimization**: Probabilistic model-based optimization
- **Simulated Annealing**: Probabilistic technique for approximating global optimum
- **Particle Swarm**: Another population-based approach inspired by social behavior
- **Reinforcement Learning**: Learning optimal actions through reward signals

### 3. Optimization Sequences

Orders in which different targets can be optimized:

- **Sequential**: Optimize one target after another (e.g., rules first, then regimes)
- **Parallel**: Optimize multiple targets simultaneously
- **Hierarchical**: Optimize at multiple levels with nested optimization
- **Iterative**: Cycle through different targets repeatedly
- **Adaptive**: Dynamically determine what to optimize next based on results

## Interface Design

### Base Interfaces

```python
class OptimizationTarget(ABC):
    @abstractmethod
    def get_parameters(self): pass
    
    @abstractmethod
    def set_parameters(self, params): pass
    
    @abstractmethod
    def validate_parameters(self, params): pass

class OptimizationMethod(ABC):
    @abstractmethod
    def optimize(self, target, evaluation_func, constraints): pass
    
    @abstractmethod
    def get_best_result(self): pass
```

### Core Manager

```python
class OptimizationManager:
    def __init__(self):
        self.targets = {}
        self.methods = {}
        self.sequences = {}
        self.evaluators = {}
        self.results_history = {}
    
    def register_target(self, name, target):
        self.targets[name] = target
    
    def register_method(self, name, method):
        self.methods[name] = method
    
    def register_sequence(self, name, sequence_func):
        self.sequences[name] = sequence_func
    
    def register_evaluator(self, name, evaluator_func):
        self.evaluators[name] = evaluator_func
    
    def run_optimization(self, sequence_name, method_name, targets, evaluator_name, **kwargs):
        # Execute the optimization according to specified sequence
        sequence = self.sequences[sequence_name]
        return sequence(self, method_name, targets, evaluator_name, **kwargs)
```

## Joint Optimization

Joint optimization involves optimizing multiple targets simultaneously. This is more complex but can find better global solutions.

### Implementation Approaches

#### 1. Combined Parameter Space

```python
def joint_optimize(manager, method_name, targets, evaluator_name, **kwargs):
    # Create a combined parameter space across all targets
    combined_params = []
    for target_name in targets:
        combined_params.extend(manager.targets[target_name].get_parameters())
    
    # Define a wrapper evaluation function
    def evaluate(params):
        # Distribute parameters back to targets
        param_index = 0
        for target_name in targets:
            target = manager.targets[target_name]
            target_param_count = len(target.get_parameters())
            target_params = params[param_index:param_index+target_param_count]
            target.set_parameters(target_params)
            param_index += target_param_count
        
        # Evaluate the combined system
        return manager.evaluators[evaluator_name]()
    
    # Run the selected optimization method
    method = manager.methods[method_name]
    return method.optimize(combined_params, evaluate, kwargs.get('constraints', None))
```

#### 2. Nested Chromosome Encoding (for Genetic Algorithms)

For genetic algorithms, we can design specialized chromosomes that encode parameters for multiple targets:

```python
class JointChromosome:
    def __init__(self, target_params):
        self.target_params = target_params  # Dict of target_name -> params
    
    def crossover(self, other_chromosome):
        # Perform crossover for each target's parameters
        child_params = {}
        for target_name, params in self.target_params.items():
            other_params = other_chromosome.target_params[target_name]
            child_params[target_name] = self._crossover_segment(params, other_params)
        return JointChromosome(child_params)
    
    def mutate(self, mutation_rate):
        # Mutate parameters for each target
        for target_name in self.target_params:
            self.target_params[target_name] = self._mutate_segment(
                self.target_params[target_name], mutation_rate
            )
```

## Advanced Optimization Scenarios

### 1. Regime-Specific Rule Optimization

Optimize different rule parameters or weights for each identified market regime:

```python
def regime_specific_optimize(manager, **kwargs):
    # First, optimize regime detection
    manager.run_optimization('sequential', 'genetic', ['regime_detector'], 'accuracy')
    
    # Identify regimes in historical data
    regimes = manager.targets['regime_detector'].identify_regimes(kwargs['data'])
    
    # For each regime, optimize rule weights
    results = {}
    for regime_type, regime_data in regimes.items():
        results[regime_type] = manager.run_optimization(
            'sequential', 'genetic', ['rule_weights'], 'sharpe',
            data=regime_data
        )
    
    return results
```

### 2. Hierarchical Multi-Objective Optimization

Optimize for multiple objectives with different priorities:

```python
def hierarchical_multi_objective(manager, **kwargs):
    # Primary objective: Sharpe Ratio
    # Secondary objectives: Max Drawdown, Calmar Ratio
    
    # First level: Get top 20% of solutions by Sharpe
    primary_results = manager.run_optimization(
        'sequential', 'genetic', ['rule_weights'], 'sharpe',
        population_size=100, keep_population=True
    )
    
    # Filter top solutions
    top_solutions = primary_results.get_top_percent(20)
    
    # Second level: Among top Sharpe solutions, find best drawdown
    secondary_results = manager.select_best_by_metric(
        top_solutions, 'max_drawdown', minimize=True
    )
    
    return secondary_results
```

### 3. Walk-Forward Optimization

Continuously re-optimize on a rolling window to adapt to changing market conditions:

```python
def walk_forward_optimize(manager, **kwargs):
    data = kwargs['data']
    window_size = kwargs.get('window_size', 252)  # Default: 1 year of trading days
    step_size = kwargs.get('step_size', 63)      # Default: 3 months
    
    results = []
    for i in range(0, len(data) - window_size, step_size):
        train_data = data[i:i+window_size]
        test_data = data[i+window_size:i+window_size+step_size]
        
        # Optimize on training window
        optimization_result = manager.run_optimization(
            'sequential', 'genetic', ['rule_weights'], 'sharpe',
            data=train_data
        )
        
        # Test on out-of-sample data
        test_performance = manager.evaluate_performance(
            optimization_result.best_parameters, test_data
        )
        
        results.append({
            'window_start': i,
            'window_end': i + window_size,
            'parameters': optimization_result.best_parameters,
            'training_performance': optimization_result.best_fitness,
            'test_performance': test_performance
        })
    
    return results
```

## Practical Implementation Guidelines

### 1. Component Registration

Start by defining and registering the basic components:

```python
# Initialize manager
optimization_manager = OptimizationManager()

# Register targets
optimization_manager.register_target('rule_params', RuleParametersTarget(rule_objects))
optimization_manager.register_target('rule_weights', RuleWeightsTarget(rule_objects))
optimization_manager.register_target('regime_detector', RegimeDetectorTarget(detector))

# Register methods
optimization_manager.register_method('genetic', GeneticAlgorithmMethod())
optimization_manager.register_method('grid', GridSearchMethod())
optimization_manager.register_method('bayesian', BayesianOptimizationMethod())

# Register sequences
optimization_manager.register_sequence('sequential', sequential_optimization)
optimization_manager.register_sequence('joint', joint_optimization)
optimization_manager.register_sequence('iterative', iterative_optimization)

# Register evaluators
optimization_manager.register_evaluator('sharpe', lambda: calculate_sharpe(backtester))
optimization_manager.register_evaluator('total_return', lambda: calculate_return(backtester))
optimization_manager.register_evaluator('sortino', lambda: calculate_sortino(backtester))
```

### 2. Execution Examples

```python
# Basic rule parameter optimization
results = optimization_manager.run_optimization(
    sequence_name='sequential',
    method_name='genetic',
    targets=['rule_params'],
    evaluator_name='sharpe',
    population_size=50,
    num_generations=100
)

# Joint optimization of rules and regimes
results = optimization_manager.run_optimization(
    sequence_name='joint',
    method_name='genetic',
    targets=['rule_weights', 'regime_detector'],
    evaluator_name='sharpe',
    population_size=100,
    num_generations=200
)

# Walk-forward optimization
results = optimization_manager.run_optimization(
    sequence_name='walk_forward',
    method_name='genetic',
    targets=['rule_weights'],
    evaluator_name='sharpe',
    data=historical_data,
    window_size=252,
    step_size=63
)
```

## Extending the Architecture

### 1. Adding New Optimization Methods

```python
class ParticleSwarmMethod(OptimizationMethod):
    def __init__(self, num_particles=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.best_result = None
    
    def optimize(self, target, evaluation_func, constraints):
        # Implement PSO algorithm here
        pass
    
    def get_best_result(self):
        return self.best_result

# Register with manager
optimization_manager.register_method('particle_swarm', ParticleSwarmMethod())
```

### 2. Adding New Evaluation Metrics

```python
def calculate_calmar_ratio(backtester):
    # Run backtest
    results = backtester.run(use_test_data=True)
    
    # Calculate annual return
    annual_return = calculate_annual_return(results)
    
    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(results)
    
    # Calmar ratio = Annual Return / Max Drawdown
    return annual_return / max_drawdown if max_drawdown > 0 else float('inf')

# Register with manager
optimization_manager.register_evaluator('calmar', lambda: calculate_calmar_ratio(backtester))
```

### 3. Creating Custom Optimization Sequences

```python
def regime_conditional_optimization(manager, method_name, targets, evaluator_name, **kwargs):
    """First optimize regime detection, then conditionally optimize rules for each regime."""
    
    # Step 1: Optimize regime detection
    if 'regime_detector' in manager.targets:
        regime_results = manager.run_optimization(
            'sequential', method_name, ['regime_detector'], 'accuracy'
        )
    
    # Step 2: Identify regimes in historical data
    regimes = manager.targets['regime_detector'].identify_regimes(kwargs['data'])
    
    # Step 3: For each regime type, optimize rule parameters/weights
    regime_specific_results = {}
    for regime_type, regime_data in regimes.items():
        if len(regime_data) < kwargs.get('min_regime_data', 100):
            print(f"Insufficient data for regime {regime_type}, using default parameters")
            continue
            
        print(f"Optimizing for regime: {regime_type}")
        regime_specific_results[regime_type] = manager.run_optimization(
            'sequential', method_name, targets, evaluator_name,
            data=regime_data
        )
    
    return {
        'regime_detector': regime_results,
        'regime_specific': regime_specific_results
    }

# Register with manager
optimization_manager.register_sequence('regime_conditional', regime_conditional_optimization)
```

## Conclusion

A modular optimization architecture enables flexible, powerful, and maintainable trading systems. By clearly separating optimization targets, methods, sequences, and evaluation metrics, the system can evolve and adapt over time without requiring significant refactoring.

The architecture outlined here provides a foundation for implementing advanced optimization approaches like joint optimization, regime-specific optimization, and multi-objective optimization within a cohesive framework.
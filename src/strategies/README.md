# Strategy Module

This module provides a modular, extensible framework for creating and combining trading strategies. It is designed to be flexible, reusable, and maintainable.

## Architecture

The strategy module follows a clean, object-oriented design with several key components:

1. **Base Strategy**: Abstract base class that defines the common interface for all strategies
2. **Strategy Registry**: Centralized registry for strategy types that supports dynamic discovery
3. **Strategy Factory**: Factory for creating strategy instances from configurations
4. **Core Strategy Implementations**:
   - `WeightedStrategy`: Combines signals from multiple rules using weights
   - `EnsembleStrategy`: Combines signals from multiple strategies using various methods
   - `RegimeStrategy`: Adapts to different market regimes by selecting appropriate strategies
   - `TopNStrategy`: Legacy strategy migrated to the new architecture

## Class Diagram

```
┌───────────────┐
│   Strategy    │
│    (ABC)      │
└───────┬───────┘
        │
        ├───────────────┬───────────────┬───────────────┐
        │               │               │               │
┌───────▼───────┐┌──────▼────────┐┌─────▼─────────┐┌────▼────────┐
│WeightedStrategy││EnsembleStrategy││RegimeStrategy ││TopNStrategy │
└───────────────┘└───────────────┘└───────────────┘└─────────────┘
```

## Usage

### Creating a Weighted Strategy

```python
from strategies import WeightedStrategy

# Create a weighted strategy with custom weights
strategy = WeightedStrategy(
    rules=rule_objects,
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.3,
    sell_threshold=-0.3,
    name="MyWeightedStrategy"
)

# Use the strategy in a backtest
backtester = Backtester(data_handler, strategy)
results = backtester.run()
```

### Creating a Regime Strategy

```python
from strategies import WeightedStrategy, RegimeStrategy
from regime_detection import TrendStrengthRegimeDetector, RegimeType

# Create regime-specific strategies
trend_up_strategy = WeightedStrategy(
    rules=rule_objects[:3],
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.2,  # More aggressive in uptrend
    sell_threshold=-0.5
)

trend_down_strategy = WeightedStrategy(
    rules=rule_objects[2:],
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.5,  # More conservative in downtrend
    sell_threshold=-0.2
)

# Create regime detector
regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# Create regime strategy
regime_strategy = RegimeStrategy(
    regime_detector=regime_detector,
    regime_strategies={
        RegimeType.TRENDING_UP: trend_up_strategy,
        RegimeType.TRENDING_DOWN: trend_down_strategy
    },
    default_strategy=default_strategy
)
```

### Creating an Ensemble Strategy

```python
from strategies import EnsembleStrategy

# Create ensemble strategy
ensemble_strategy = EnsembleStrategy(
    strategies={
        "weighted": weighted_strategy,
        "regime": regime_strategy,
        "topn": topn_strategy
    },
    combination_method="weighted",
    weights={"weighted": 0.3, "regime": 0.5, "topn": 0.2}
)
```

### Using the Strategy Factory

```python
from strategies import StrategyFactory
from regime_detection import TrendStrengthRegimeDetector, RegimeType

# Simple factory method
weighted_strategy = StrategyFactory.create_weighted_strategy(
    rules=rule_objects,
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.3,
    sell_threshold=-0.3
)

# Using a configuration dictionary
config = {
    'type': 'RegimeStrategy',
    'params': {
        'regime_detector': regime_detector,
        'regime_strategies': {
            RegimeType.TRENDING_UP: {
                'type': 'WeightedStrategy',
                'params': {
                    'rules': rule_objects[:2],
                    'weights': [0.7, 0.3],
                    'buy_threshold': 0.2
                }
            },
            RegimeType.TRENDING_DOWN: {
                'type': 'WeightedStrategy',
                'params': {
                    'rules': rule_objects[1:],
                    'weights': [0.6, 0.4],
                    'buy_threshold': 0.5
                }
            }
        },
        'default_strategy': {
            'type': 'WeightedStrategy',
            'params': {
                'rules': rule_objects
            }
        }
    }
}

complex_strategy = StrategyFactory.create_from_config(config)
```

## Creating Custom Strategies

You can create custom strategies by subclassing the `Strategy` base class and registering them with the `StrategyRegistry`:

```python
from strategies import Strategy, StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="custom")
class MyCustomStrategy(Strategy):
    def __init__(self, parameter1, parameter2, name=None):
        super().__init__(name or "MyCustomStrategy")
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        
    def on_bar(self, event):
        bar = event.bar
        
        # Your custom logic here
        # ...
        
        # Create and return a signal
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=SignalType.BUY,  # Determined by your logic
            price=bar["Close"],
            rule_id=self.name
        )
        
    def reset(self):
        # Reset any internal state
        pass
```

## Testing Strategies

Each strategy should be tested to ensure it behaves as expected. Here's an example of testing a weighted strategy:

```python
import unittest
from strategies import WeightedStrategy
from signals import SignalType

class TestWeightedStrategy(unittest.TestCase):
    def test_equal_weights(self):
        # Create strategy with equal weights
        strategy = WeightedStrategy(
            rules=[buy_rule, sell_rule, neutral_rule]
        )
        
        # Test signal
        signal = strategy.on_bar(bar_event)
        
        # Verify signal
        self.assertEqual(signal.signal_type, SignalType.NEUTRAL)
```

## Integration with Optimization Framework

The strategy classes are designed to work seamlessly with the optimization framework:

```python
from optimization import OptimizerManager, OptimizationMethod
from strategies import WeightedStrategy

# Initialize optimizer manager
optimizer = OptimizerManager(data_handler)

# Run optimization
optimized_params = optimizer.optimize(
    component_type='strategy',
    method=OptimizationMethod.GENETIC,
    metrics='sharpe',
    strategy_class=WeightedStrategy,
    parameter_ranges={
        'buy_threshold': [0.1, 0.3, 0.5, 0.7],
        'sell_threshold': [-0.7, -0.5, -0.3, -0.1]
    }
)

# Create strategy with optimized parameters
strategy = WeightedStrategy(
    rules=rule_objects,
    **optimized_params
)
```

## Benefits of the New Architecture

1. **Modularity**: Each strategy type is in its own file with clear responsibilities
2. **Loose Coupling**: Strategies interact with rules through a well-defined interface
3. **Extensibility**: The registry pattern allows easy addition of new strategy types
4. **Reusability**: Common functionality is extracted to base classes
5. **Testability**: All components can be easily tested in isolation
6. **Configuration-Based**: Strategies can be created from configuration dictionaries
7. **Hierarchical Composition**: Strategies can be composed into more complex strategies
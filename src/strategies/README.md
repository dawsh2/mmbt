# Strategies Module Documentation

The Strategies module provides a modular, extensible framework for creating and combining trading strategies. It allows traders to create sophisticated trading algorithms by composing different strategies, enabling adaptation to various market conditions through regime detection, ensemble techniques, and weighted signal combinations.

## Core Concepts

**Strategy**: Abstract base class that defines the common interface for all strategies, processing bar data to generate trading signals.  
**WeightedStrategy**: Combines multiple components using configurable weights to generate a trading signal.  
**EnsembleStrategy**: Combines signals from multiple strategies using various methods such as voting or weighted averaging.  
**RegimeStrategy**: Adapts to different market regimes by selecting appropriate sub-strategies based on market conditions.  
**StrategyRegistry**: Central registry for strategy types that supports dynamic registration and discovery.  
**StrategyFactory**: Factory for creating strategy instances from configurations.

## Basic Usage

```python
from strategies import WeightedStrategy, StrategyFactory
from strategies import StrategyRegistry

# Create a weighted strategy with custom weights
strategy = WeightedStrategy(
    components=rule_objects,  # List of signal-generating components (rules, etc.)
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.3,
    sell_threshold=-0.3,
    name="MyWeightedStrategy"
)

# Process a bar of market data
bar_data = {
    "timestamp": "2023-06-15 10:30:00",
    "Open": 100.5,
    "High": 101.2,
    "Low": 99.8,
    "Close": 100.9,
    "Volume": 5000
}

# Create a bar event object (expected by strategies)
class BarEvent:
    def __init__(self, bar):
        self.bar = bar

event = BarEvent(bar_data)

# Get the signal from the strategy
signal = strategy.on_bar(event)

# Check the signal type
if signal.signal_type.value > 0:
    print("Buy signal generated")
elif signal.signal_type.value < 0:
    print("Sell signal generated")
else:
    print("Neutral signal generated")
```

## API Reference

### Strategy Classes

#### Strategy (Base Class)

Abstract base class for all trading strategies.

**Constructor Parameters:**
- `name` (str, optional): Name for the strategy. If not provided, uses class name.

**Methods:**
- `on_bar(event)`: Process a bar event and generate a trading signal
  - `event`: Bar event containing market data
  - Returns: Signal object with trading signal information
- `reset()`: Reset the strategy's internal state

**Example:**
```python
from strategies import Strategy
from signals import Signal, SignalType

class MyCustomStrategy(Strategy):
    def __init__(self, name=None):
        super().__init__(name or "MyCustomStrategy")
        self.last_signal = None
        
    def on_bar(self, event):
        bar = event.bar
        # Custom logic to generate signals
        signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=SignalType.BUY,  # or SELL or NEUTRAL
            price=bar["Close"],
            rule_id=self.name
        )
        self.last_signal = signal
        return signal
        
    def reset(self):
        self.last_signal = None
```

#### WeightedStrategy

Strategy that combines signals from multiple components using weights.

**Constructor Parameters:**
- `components` (list): List of signal-generating components (rules, strategies, etc.)
- `weights` (list, optional): List of weights for each component (default: equal weights)
- `buy_threshold` (float, optional): Threshold above which to generate a buy signal (default: 0.5)
- `sell_threshold` (float, optional): Threshold below which to generate a sell signal (default: -0.5)
- `name` (str, optional): Strategy name

**Example:**
```python
from strategies import WeightedStrategy
import numpy as np

# Create a weighted strategy
strategy = WeightedStrategy(
    components=signal_components,  # Any objects with on_bar() method
    weights=[0.4, 0.3, 0.3],
    buy_threshold=0.4,
    sell_threshold=-0.4,
    name="CustomWeightedStrategy"
)

# Process a bar
signal = strategy.on_bar(event)
```

#### EnsembleStrategy

Strategy that combines signals from multiple strategies using various methods.

**Constructor Parameters:**
- `strategies` (dict): Dictionary mapping strategy names to strategy objects
- `combination_method` (str, optional): Method for combining signals ('voting', 'weighted', or 'consensus') (default: 'voting')
- `weights` (dict, optional): Optional dictionary of weights for each strategy (for 'weighted' method)
- `name` (str, optional): Strategy name

**Example:**
```python
from strategies import EnsembleStrategy, WeightedStrategy

# Create individual strategies
strategy1 = WeightedStrategy(components=rule_objects[:3], weights=[0.5, 0.3, 0.2])
strategy2 = WeightedStrategy(components=rule_objects[3:], weights=[0.6, 0.4])

# Create ensemble strategy
ensemble = EnsembleStrategy(
    strategies={
        "strategy1": strategy1,
        "strategy2": strategy2
    },
    combination_method="weighted",
    weights={"strategy1": 0.6, "strategy2": 0.4},
    name="MyEnsembleStrategy"
)

# Process a bar
signal = ensemble.on_bar(event)
```

#### RegimeStrategy

Strategy that adapts to different market regimes by selecting appropriate strategies.

**Constructor Parameters:**
- `regime_detector` (object): Object that identifies market regimes
- `regime_strategies` (dict): Dictionary mapping regime types to strategies
- `default_strategy` (Strategy, optional): Strategy to use when no regime-specific strategy is available
- `name` (str, optional): Strategy name

**Example:**
```python
from strategies import RegimeStrategy, WeightedStrategy
from regime_detection import TrendStrengthRegimeDetector, RegimeType

# Create regime-specific strategies
trend_up_strategy = WeightedStrategy(
    components=rule_objects[:3],
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.2,  # More aggressive in uptrend
    sell_threshold=-0.5
)

trend_down_strategy = WeightedStrategy(
    components=rule_objects[3:],
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.5,  # More conservative in downtrend
    sell_threshold=-0.2
)

# Create default strategy
default_strategy = WeightedStrategy(components=rule_objects)

# Create regime detector
regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# Create regime strategy
regime_strategy = RegimeStrategy(
    regime_detector=regime_detector,
    regime_strategies={
        RegimeType.TRENDING_UP: trend_up_strategy,
        RegimeType.TRENDING_DOWN: trend_down_strategy
    },
    default_strategy=default_strategy,
    name="AdaptiveStrategy"
)

# Process a bar
signal = regime_strategy.on_bar(event)
```

#### TopNStrategy

Strategy that combines signals from top N rules using a consensus mechanism.

**Constructor Parameters:**
- `rule_objects` (list): List of rule objects
- `name` (str, optional): Strategy name

**Note:** This strategy uses `SignalRouter` internally to collect and aggregate signals from rules.

**Example:**
```python
from strategies import TopNStrategy

# Create TopN strategy
topn_strategy = TopNStrategy(
    rule_objects=top_performing_rules,
    name="TopRulesStrategy"
)

# Process a bar
signal = topn_strategy.on_bar(event)
```

### Strategy Factory and Registry

#### StrategyRegistry

Registry of available strategies for dynamic registration and discovery.

**Class Methods:**
- `register(category='general')`: Decorator to register a strategy class
- `get_strategy_class(name)`: Get a strategy class by name
- `list_strategies()`: List all registered strategies by category

**Example:**
```python
from strategies import Strategy, StrategyRegistry

# Register a new strategy
@StrategyRegistry.register(category="custom")
class MyCustomStrategy(Strategy):
    def on_bar(self, event):
        # Implementation...
        pass

# Get a strategy class
StrategyClass = StrategyRegistry.get_strategy_class("WeightedStrategy")

# List all registered strategies
strategy_categories = StrategyRegistry.list_strategies()
```

#### StrategyFactory

Factory for creating strategy instances with various helper methods.

**Methods:**
- `create_strategy(strategy_type, params=None)`: Create a strategy instance
- `create_weighted_strategy(components, weights=None, buy_threshold=0.5, sell_threshold=-0.5, name=None)`: Create a weighted strategy
- `create_ensemble_strategy(strategies, combination_method='voting', weights=None, name=None)`: Create an ensemble strategy
- `create_regime_strategy(regime_detector, regime_strategies, default_strategy=None, name=None)`: Create a regime strategy
- `create_topn_strategy(rule_objects, name=None)`: Create a TopN strategy
- `create_from_config(config)`: Create a strategy from a configuration dictionary

**Example:**
```python
from strategies import StrategyFactory

factory = StrategyFactory()

# Create a weighted strategy
weighted_strategy = factory.create_weighted_strategy(
    components=rule_objects,
    weights=[0.5, 0.3, 0.2],
    buy_threshold=0.4,
    sell_threshold=-0.4
)

# Create a strategy from a configuration dictionary
config = {
    'type': 'EnsembleStrategy',
    'params': {
        'strategies': {
            'strategy1': {'type': 'WeightedStrategy', 'params': {'components': rule_objects[:3]}},
            'strategy2': {'type': 'WeightedStrategy', 'params': {'components': rule_objects[3:]}}
        },
        'combination_method': 'voting'
    }
}

strategy = factory.create_from_config(config)
```

## Advanced Usage

### Creating Custom Strategies

You can create custom strategies by subclassing the Strategy base class:

```python
from strategies import Strategy, StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="custom")
class MyCustomStrategy(Strategy):
    def __init__(self, parameter1, parameter2, name=None):
        super().__init__(name or "MyCustomStrategy")
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.last_signal = None
        
    def on_bar(self, event):
        bar = event.bar
        
        # Custom logic to determine signal
        if bar["Close"] > bar["Open"] * (1 + self.parameter1):
            signal_type = SignalType.BUY
        elif bar["Close"] < bar["Open"] * (1 - self.parameter2):
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create and return signal
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name
        )
        
        return self.last_signal
        
    def reset(self):
        # Reset any internal state
        self.last_signal = None
```

### Creating a Regime Strategy

Regime strategies adapt to different market conditions by selecting appropriate sub-strategies:

```python
from strategies import WeightedStrategy, RegimeStrategy
from regime_detection import RegimeType, VolatilityRegimeDetector

# Create a regime detector
regime_detector = VolatilityRegimeDetector(
    lookback_period=20,
    volatility_threshold=0.015
)

# Create specific strategies for different volatility regimes
low_vol_strategy = WeightedStrategy(
    components=trend_following_rules,
    weights=[0.4, 0.3, 0.3],
    buy_threshold=0.3,
    sell_threshold=-0.3
)

high_vol_strategy = WeightedStrategy(
    components=mean_reversion_rules,
    weights=[0.5, 0.5],
    buy_threshold=0.6,  # Higher threshold in volatile markets
    sell_threshold=-0.6
)

# Create a regime strategy
regime_strategy = RegimeStrategy(
    regime_detector=regime_detector,
    regime_strategies={
        RegimeType.LOW_VOLATILITY: low_vol_strategy,
        RegimeType.VOLATILE: high_vol_strategy
    },
    default_strategy=low_vol_strategy  # Default to low volatility strategy
)

# Process market data with regime-adaptive strategy
signal = regime_strategy.on_bar(event)
print(f"Current regime: {regime_strategy.current_regime}")
print(f"Signal type: {signal.signal_type}")
```

### Creating an Ensemble Strategy

Ensemble strategies combine multiple strategies using various methods:

```python
from strategies import EnsembleStrategy, WeightedStrategy, TopNStrategy

# Create multiple strategies with different approaches
ma_strategy = WeightedStrategy(
    components=moving_average_rules,
    weights=[0.5, 0.5]
)

oscillator_strategy = WeightedStrategy(
    components=oscillator_rules,
    weights=[0.4, 0.3, 0.3]
)

volatility_strategy = WeightedStrategy(
    components=volatility_rules,
    weights=[0.6, 0.4]
)

# Combine strategies using ensemble with weighted voting
ensemble = EnsembleStrategy(
    strategies={
        "ma_strategy": ma_strategy,
        "oscillator_strategy": oscillator_strategy,
        "volatility_strategy": volatility_strategy
    },
    combination_method="weighted",
    weights={
        "ma_strategy": 0.4,
        "oscillator_strategy": 0.4,
        "volatility_strategy": 0.2
    }
)

# Process a bar with the ensemble
signal = ensemble.on_bar(event)

# Access detailed signal information
print(f"Consensus: {signal.metadata['consensus']}")
print(f"Agreement strength: {signal.metadata['agreement_strength']:.2f}")
print(f"Component signals: {signal.metadata['signals']}")
```

### Using Strategy Factory for Complex Configurations

The StrategyFactory provides methods to create complex strategy hierarchies from configuration:

```python
from strategies import StrategyFactory
from regime_detection import TrendStrengthRegimeDetector, RegimeType

factory = StrategyFactory()

# Complex configuration with nested strategies
config = {
    'type': 'RegimeStrategy',
    'params': {
        'regime_detector': TrendStrengthRegimeDetector(adx_period=14),
        'regime_strategies': {
            RegimeType.TRENDING_UP: {
                'type': 'EnsembleStrategy',
                'params': {
                    'strategies': {
                        'trend_follow': {'type': 'WeightedStrategy', 'params': {'components': trend_rules}},
                        'breakout': {'type': 'WeightedStrategy', 'params': {'components': breakout_rules}}
                    },
                    'combination_method': 'weighted',
                    'weights': {'trend_follow': 0.7, 'breakout': 0.3}
                }
            },
            RegimeType.TRENDING_DOWN: {
                'type': 'WeightedStrategy',
                'params': {
                    'components': defensive_rules,
                    'buy_threshold': 0.7,  # More conservative in downtrends
                    'sell_threshold': -0.3  # More eager to sell in downtrends
                }
            }
        },
        'default_strategy': {
            'type': 'WeightedStrategy',
            'params': {'components': balanced_rules}
        }
    }
}

# Create complex strategy from configuration
complex_strategy = factory.create_from_config(config)

# Process a bar with the complex strategy
signal = complex_strategy.on_bar(event)
```

### Integration with Optimization Framework

Strategies can be used with the optimization framework to find optimal parameters:

```python
from optimization import OptimizerManager, OptimizationMethod
from strategies import WeightedStrategy

# Initialize optimizer manager
optimizer = OptimizerManager(data_handler)

# Register rules for optimization
for i, rule in enumerate(rule_objects):
    optimizer.register_rule(f"rule_{i}", rule.__class__, rule_params[i], rule)

# Run optimization to find best weights and thresholds
optimized_strategy = optimizer.optimize(
    component_type='rule',
    method=OptimizationMethod.GENETIC,
    metrics='sharpe',
    genetic={
        'population_size': 50,
        'num_generations': 30,
        'mutation_rate': 0.1,
        'regularization_factor': 0.2,
        'optimize_thresholds': True
    }
)

# Use the optimized strategy in backtesting
backtester = Backtester(data_handler, optimized_strategy)
results = backtester.run()
```

### Implementing Strategy State Management

For strategies that need to maintain state across multiple bars:

```python
from strategies import Strategy
from signals import Signal, SignalType

class StateAwareStrategy(Strategy):
    def __init__(self, lookback=20, name=None):
        super().__init__(name or "StateAwareStrategy")
        self.lookback = lookback
        self.price_history = []
        self.signal_history = []
        self.current_position = 0  # 0 = flat, 1 = long, -1 = short
        
    def on_bar(self, event):
        bar = event.bar
        
        # Update state
        self.price_history.append(bar["Close"])
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        # Generate signal based on state
        signal_type = SignalType.NEUTRAL
        
        # Example: Only allow position reversal after holding for at least 5 bars
        if self.current_position == 1 and len(self.signal_history) >= 5:
            # Logic for exiting long position
            if bar["Close"] < min(self.price_history[-5:]):
                signal_type = SignalType.SELL
                self.current_position = -1
        elif self.current_position == -1 and len(self.signal_history) >= 5:
            # Logic for exiting short position
            if bar["Close"] > max(self.price_history[-5:]):
                signal_type = SignalType.BUY
                self.current_position = 1
        elif self.current_position == 0:
            # Logic for entering new position
            if bar["Close"] > max(self.price_history[:-1]) if len(self.price_history) > 1 else 0:
                signal_type = SignalType.BUY
                self.current_position = 1
            elif bar["Close"] < min(self.price_history[:-1]) if len(self.price_history) > 1 else float('inf'):
                signal_type = SignalType.SELL
                self.current_position = -1
        
        # Create signal
        signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name,
            metadata={"current_position": self.current_position}
        )
        
        # Update signal history
        self.signal_history.append(signal)
        if len(self.signal_history) > self.lookback:
            self.signal_history.pop(0)
        
        return signal
    
    def reset(self):
        self.price_history = []
        self.signal_history = []
        self.current_position = 0
```

### Creating Multi-Timeframe Strategies

To incorporate multiple timeframes in a single strategy:

```python
from strategies import Strategy
from signals import Signal, SignalType

class MultiTimeframeStrategy(Strategy):
    def __init__(self, base_timeframe="1d", higher_timeframe="1w", name=None):
        super().__init__(name or "MultiTimeframeStrategy")
        self.base_timeframe = base_timeframe
        self.higher_timeframe = higher_timeframe
        self.higher_tf_data = {}
        self.last_signal = None
        
    def on_bar(self, event):
        bar = event.bar
        
        # Check if this bar contains higher timeframe data
        if "higher_timeframe" in bar:
            self.higher_tf_data = bar["higher_timeframe"]
        
        # Generate signal based on both timeframes
        signal_type = SignalType.NEUTRAL
        
        # Example: Use higher timeframe for trend direction, lower for entry timing
        higher_tf_trend = self._determine_trend(self.higher_tf_data)
        base_tf_signal = self._analyze_base_timeframe(bar)
        
        # Only generate buy signals when higher timeframe trend is positive
        if higher_tf_trend > 0 and base_tf_signal > 0:
            signal_type = SignalType.BUY
        # Only generate sell signals when higher timeframe trend is negative
        elif higher_tf_trend < 0 and base_tf_signal < 0:
            signal_type = SignalType.SELL
        
        # Create signal
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name,
            metadata={
                "higher_tf_trend": higher_tf_trend,
                "base_tf_signal": base_tf_signal
            }
        )
        
        return self.last_signal
    
    def _determine_trend(self, higher_tf_data):
        # Logic to determine trend on higher timeframe
        if not higher_tf_data:
            return 0
            
        # Example: Simple moving average comparison
        if "SMA_50" in higher_tf_data and "SMA_200" in higher_tf_data:
            if higher_tf_data["SMA_50"] > higher_tf_data["SMA_200"]:
                return 1  # Uptrend
            elif higher_tf_data["SMA_50"] < higher_tf_data["SMA_200"]:
                return -1  # Downtrend
        
        return 0  # Neutral
    
    def _analyze_base_timeframe(self, bar):
        # Logic to generate signals on base timeframe
        # Example: RSI overbought/oversold
        if "RSI" in bar:
            if bar["RSI"] < 30:
                return 1  # Oversold - buy signal
            elif bar["RSI"] > 70:
                return -1  # Overbought - sell signal
        
        return 0  # Neutral
    
    def reset(self):
        self.higher_tf_data = {}
        self.last_signal = None
```

### Signal Router Integration

The `TopNStrategy` and some other strategies use `SignalRouter` internally to collect and aggregate signals:

```python
from strategies import Strategy
from signals import SignalRouter, Signal, SignalType

class CustomRoutingStrategy(Strategy):
    def __init__(self, rule_objects, name=None):
        super().__init__(name or "CustomRoutingStrategy")
        # Initialize SignalRouter with rule objects
        self.signal_router = SignalRouter(rule_objects)
        self.last_signal = None
        
    def on_bar(self, event):
        # Use SignalRouter to collect signals from all rules
        router_output = self.signal_router.on_bar(event)
        
        # Get the signal collection
        signal_collection = router_output["signals"]
        
        # Get weighted consensus from the collection
        consensus_signal_type = signal_collection.get_weighted_consensus()
        
        # Create a new signal with the consensus
        signal = Signal(
            timestamp=router_output["timestamp"],
            signal_type=consensus_signal_type,
            price=router_output["price"],
            rule_id=self.name,
            confidence=0.7
        )
        
        self.last_signal = signal
        return signal
        
    def reset(self):
        # Reset the router and our state
        self.signal_router.reset()
        self.last_signal = None
```

## Rule Aggregation Methods

When creating composite rules, the following aggregation methods are available:

- **majority**: Signal type with the most votes wins
- **unanimous**: Signals only when all rules agree
- **weighted**: Weight each rule's signal (requires weights parameter)
- **any**: Signal when any rule gives a non-neutral signal
- **sequence**: Signal when rules trigger in sequence

## Best Practices

1. **Start with simple strategies**: Begin with simple strategies like WeightedStrategy before moving to more complex ones.

2. **Use the StrategyFactory**: Leverage the factory pattern for creating strategies, especially when working with configurations.

3. **Maintain proper state**: Always implement the `reset()` method to properly clear internal state when needed.

4. **Apply proper thresholds**: Tune buy/sell thresholds based on the expected signal range of your components.

5. **Use appropriate weights**: Assign higher weights to more reliable components based on historical performance.

6. **Balance complexity**: More complex strategies aren't always better. Aim for the minimum complexity needed.

7. **Leverage regime detection**: For adaptive strategies, make use of market regime detection to select appropriate sub-strategies.

8. **Normalize inputs**: Ensure all components produce signals in a consistent range (usually -1 to 1).

9. **Handle edge cases**: Implement proper handling for missing data or edge-case market conditions.

10. **Document strategy logic**: Always clearly document the strategy's logic, especially for complex composite strategies.

# Regime Detection Module Documentation

The Regime Detection module identifies market regimes (trending, range-bound, volatile, etc.) to enable adaptive trading strategies. By abstracting market condition detection, it allows strategies to dynamically adjust to changing environments without modifying their core logic.

## Core Concepts

**RegimeType**: Enumeration of market regimes (TRENDING_UP, TRENDING_DOWN, RANGE_BOUND, VOLATILE, etc.).  
**DetectorBase**: Abstract base class for all regime detection algorithms.  
**DetectorRegistry**: Central registry for detector types with dynamic registration.  
**RegimeManager**: Coordinates regime detection and strategy selection.

## Basic Usage

```python
from src.regime_detection import RegimeType, DetectorRegistry, registry
from src.regime_detection.detectors.trend_detectors import TrendStrengthRegimeDetector
from src.strategies import WeightedStrategy

# Create a regime detector
detector = TrendStrengthRegimeDetector(
    config={
        "adx_period": 14,
        "adx_threshold": 25
    }
)

# Process market data
bar_data = {
    "Open": 100.5,
    "High": 101.2,
    "Low": 99.8,
    "Close": 100.9,
    "Volume": 5000
}

# Detect current regime
current_regime = detector.detect_regime(bar_data)
print(f"Current market regime: {current_regime}")  # e.g., TRENDING_UP

# Create regime-specific strategies
trending_up_strategy = WeightedStrategy(
    components=trend_following_rules,
    buy_threshold=0.3,
    sell_threshold=-0.5
)

range_bound_strategy = WeightedStrategy(
    components=mean_reversion_rules,
    buy_threshold=0.4,
    sell_threshold=-0.4
)

# Create regime manager
from src.regime_detection import RegimeManager
from src.strategy import WeightedRuleStrategyFactory

# Create strategy factory
strategy_factory = WeightedRuleStrategyFactory()

# Create regime manager
regime_manager = RegimeManager(
    regime_detector=detector,
    strategy_factory=strategy_factory,
    rule_objects=rule_objects,
    data_handler=data_handler
)

# Add regime-specific strategies
regime_manager.regime_strategies = {
    RegimeType.TRENDING_UP: trending_up_strategy,
    RegimeType.RANGE_BOUND: range_bound_strategy
}

# Use in backtest or live trading
# The regime manager automatically selects the appropriate strategy
signal = regime_manager.on_bar(event)
```

## API Reference

### RegimeType

Enumeration of different market regime types.

```python
class RegimeType(Enum):
    UNKNOWN = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGE_BOUND = auto()
    VOLATILE = auto()
    LOW_VOLATILITY = auto()
    BULL = auto()
    BEAR = auto()
    CHOPPY = auto()
```

### DetectorBase

Abstract base class for regime detection algorithms.

**Constructor Parameters:**
- `name` (str, optional): Optional name for the detector
- `config` (dict, optional): Optional configuration dictionary

**Methods:**
- `detect_regime(bar_data)`: Detect the current market regime based on bar data
- `reset()`: Reset the detector's internal state

**Example:**
```python
from src.regime_detection import DetectorBase, RegimeType

class MyCustomDetector(DetectorBase):
    def __init__(self, name=None, config=None):
        super().__init__(name, config or {"param1": 10})
        
    def detect_regime(self, bar_data):
        # Custom logic to detect regime based on bar data
        if bar_data["Close"] > bar_data["Open"]:
            return RegimeType.TRENDING_UP
        elif bar_data["Close"] < bar_data["Open"]:
            return RegimeType.TRENDING_DOWN
        else:
            return RegimeType.RANGE_BOUND
            
    def reset(self):
        # Reset internal state
        super().reset()
```

### DetectorRegistry

Registry of available regime detectors for dynamic registration and discovery.

**Methods:**
- `register(category="general")`: Decorator to register a detector class
- `get_detector_class(name)`: Get a detector class by name
- `list_detectors(category=None)`: List available detectors, optionally filtered by category

**Example:**
```python
from src.regime_detection import DetectorRegistry, registry

# Register a new detector
@registry.register(category="custom")
class MyCustomDetector(DetectorBase):
    # Implementation...
    pass

# Get a detector class
detector_class = registry.get_detector_class("TrendStrengthRegimeDetector")

# List all detectors in a category
volatility_detectors = registry.list_detectors(category="volatility")
```

### DetectorFactory

Factory for creating regime detector instances.

**Constructor Parameters:**
- `registry` (DetectorRegistry, optional): Optional DetectorRegistry instance

**Methods:**
- `create_detector(name_or_class, config=None)`: Create a detector instance by name or class
- `create_from_config(config)`: Create a detector from a configuration dictionary
- `create_composite(configs, combination_method='majority')`: Create a composite detector from multiple configurations

**Example:**
```python
from src.regime_detection import DetectorFactory

factory = DetectorFactory()

# Create detector by name
trend_detector = factory.create_detector(
    "TrendStrengthRegimeDetector", 
    config={"adx_period": 14, "adx_threshold": 25}
)

# Create detector from configuration
detector_config = {
    "type": "VolatilityRegimeDetector",
    "params": {
        "lookback_period": 20,
        "volatility_threshold": 0.015
    }
}
vol_detector = factory.create_from_config(detector_config)

# Create composite detector
composite_detector = factory.create_composite(
    [trend_detector_config, volatility_detector_config],
    combination_method='weighted'
)
```

### RegimeManager

Manages trading strategies based on detected market regimes.

**Constructor Parameters:**
- `regime_detector` (DetectorBase): RegimeDetector object for identifying regimes
- `strategy_factory` (object): Factory for creating strategies
- `rule_objects` (list, optional): List of trading rule objects (passed to the factory)
- `data_handler` (object, optional): Optional data handler for optimization

**Methods:**
- `optimize_regime_strategies(regimes_to_optimize=None, optimization_metric='sharpe', verbose=True)`: Optimize strategies for different market regimes
- `get_strategy_for_regime(regime)`: Get the optimized strategy for a specific regime
- `on_bar(event)`: Process a bar and generate trading signals using the appropriate strategy
- `reset()`: Reset the regime manager and its components

**Example:**
```python
from src.regime_detection import RegimeManager
from src.strategies import WeightedRuleStrategyFactory

# Create strategy factory
strategy_factory = WeightedRuleStrategyFactory()

# Create regime manager
regime_manager = RegimeManager(
    regime_detector=trend_detector,
    strategy_factory=strategy_factory,
    rule_objects=rule_objects,
    data_handler=data_handler
)

# Optimize regime-specific strategies
regime_manager.optimize_regime_strategies(
    optimization_metric='sharpe',
    verbose=True
)

# Process a bar
signal = regime_manager.on_bar(event)
```

## Detector Implementations

### Trend Detectors

#### TrendStrengthRegimeDetector

Detects trending and range-bound markets using the Average Directional Index (ADX).

**Configuration Parameters:**
- `adx_period` (int): Period for ADX calculation (default: 14)
- `adx_threshold` (int): Threshold for trend identification (default: 25)

**Detected Regimes:**
- `TRENDING_UP`: Strong uptrend (ADX > threshold, +DI > -DI)
- `TRENDING_DOWN`: Strong downtrend (ADX > threshold, -DI > +DI)
- `RANGE_BOUND`: No strong trend (ADX <= threshold)

**Example:**
```python
from src.regime_detection.detectors.trend_detectors import TrendStrengthRegimeDetector

detector = TrendStrengthRegimeDetector(config={
    "adx_period": 14, 
    "adx_threshold": 25
})
regime = detector.detect_regime(bar_data)
```

### Volatility Detectors

#### VolatilityRegimeDetector

Identifies volatile and low-volatility markets based on the standard deviation of returns.

**Configuration Parameters:**
- `lookback_period` (int): Period for volatility calculation (default: 20)
- `volatility_threshold` (float): Threshold for volatility regimes (default: 0.015)

**Detected Regimes:**
- `VOLATILE`: High volatility (std_dev > threshold)
- `LOW_VOLATILITY`: Low volatility (std_dev <= threshold)

**Example:**
```python
from src.regime_detection.detectors.volatility_detectors import VolatilityRegimeDetector

detector = VolatilityRegimeDetector(config={
    "lookback_period": 20,
    "volatility_threshold": 0.015
})
regime = detector.detect_regime(bar_data)
```

### Composite Detectors

#### CompositeDetector

Combines multiple regime detectors using voting or weighted methods.

**Configuration Parameters:**
- `detectors` (list): List of detector instances to combine
- `combination_method` (str): Method for combining detector outputs:
  - `majority`: Use the most common regime
  - `consensus`: Use a regime only if all detectors agree
  - `weighted`: Use weighted voting

**Example:**
```python
from src.regime_detection.detectors.composite_detectors import CompositeDetector

detector = CompositeDetector(
    detectors=[trend_detector, volatility_detector],
    combination_method='majority'
)
regime = detector.detect_regime(bar_data)
```

## Advanced Usage

### Creating Adaptive Strategies with Regime Detection

```python
from src.regime_detection import RegimeType
from src.regime_detection.detectors.volatility_detectors import VolatilityRegimeDetector
from src.strategies import WeightedStrategy, RegimeStrategy

# Create regime detector
detector = VolatilityRegimeDetector(config={
    "lookback_period": 20,
    "volatility_threshold": 0.015
})

# Define different strategies for different volatility regimes
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
    regime_detector=detector,
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

### Analyzing Regime Distribution and Performance

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.regime_detection.detectors.trend_detectors import TrendStrengthRegimeDetector

def analyze_regimes(data, detector):
    """Analyze regime distribution and performance in historical data."""
    regimes = []
    returns = []
    
    # Process each bar to detect regime
    detector.reset()
    for i in range(1, len(data)):
        bar = data.iloc[i].to_dict()
        prev_bar = data.iloc[i-1].to_dict()
        
        # Detect regime
        regime = detector.detect_regime(bar)
        regimes.append(regime)
        
        # Calculate return
        daily_return = (bar['Close'] / prev_bar['Close']) - 1
        returns.append(daily_return)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'regime': regimes,
        'return': returns
    })
    
    # Analyze returns by regime
    regime_stats = results.groupby('regime')['return'].agg(['mean', 'std', 'count'])
    regime_stats['sharpe'] = regime_stats['mean'] / regime_stats['std']
    
    # Plot regime distribution
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    regime_counts = results['regime'].value_counts()
    regime_counts.plot(kind='bar')
    plt.title('Regime Distribution')
    plt.ylabel('Number of Days')
    
    plt.subplot(2, 1, 2)
    for regime in results['regime'].unique():
        regime_returns = results[results['regime'] == regime]['return'].cumsum()
        plt.plot(regime_returns, label=regime.name)
    
    plt.title('Cumulative Returns by Regime')
    plt.legend()
    
    return regime_stats
```

### Integration with Optimization Framework

```python
from src.optimization import OptimizerManager, OptimizationMethod
from src.regime_detection.detectors.trend_detectors import TrendStrengthRegimeDetector

# Create detector with initial parameters
detector = TrendStrengthRegimeDetector(config={
    "adx_period": 14,
    "adx_threshold": 25
})

# Create optimizer for regime detector
optimizer = OptimizerManager(data_handler)

# Register detector with parameter ranges
optimizer.register_regime_detector(
    "trend_detector", 
    TrendStrengthRegimeDetector,
    {"adx_period": [10, 14, 20], 
     "adx_threshold": [20, 25, 30]}
)

# Optimize detector parameters
optimized_detectors = optimizer.optimize(
    component_type='regime_detector',
    method=OptimizationMethod.GRID_SEARCH,
    metrics='stability',  # Specialized metric for regime detectors
    verbose=True
)

# Get optimized detector
best_detector = list(optimized_detectors.values())[0]

# Use in regime strategy
regime_strategy = RegimeStrategy(
    regime_detector=best_detector,
    regime_strategies=regime_specific_strategies,
    default_strategy=default_strategy
)
```

### Building a Custom Detector

```python
from src.regime_detection import DetectorBase, RegimeType, registry

@registry.register(category="custom")
class MovingAverageRegimeDetector(DetectorBase):
    """
    Regime detector based on moving average relationships.
    """
    
    def __init__(self, name=None, config=None):
        super().__init__(name, config or {
            "fast_ma": 20,
            "medium_ma": 50,
            "slow_ma": 200
        })
        
        self.fast_ma = self.config["fast_ma"]
        self.medium_ma = self.config["medium_ma"]
        self.slow_ma = self.config["slow_ma"]
        
        # State variables
        self.fast_values = []
        self.medium_values = []
        self.slow_values = []
        self.close_history = []
    
    def detect_regime(self, bar_data):
        """Detect regime based on MA relationships."""
        # Add current price to history
        self.close_history.append(bar_data["Close"])
        
        # Need enough history to calculate all MAs
        if len(self.close_history) < self.slow_ma:
            return RegimeType.UNKNOWN
        
        # Calculate MA values
        self.fast_values.append(
            sum(self.close_history[-self.fast_ma:]) / self.fast_ma
        )
        self.medium_values.append(
            sum(self.close_history[-self.medium_ma:]) / self.medium_ma
        )
        self.slow_values.append(
            sum(self.close_history[-self.slow_ma:]) / self.slow_ma
        )
        
        # Detect regime based on MA alignment
        current_price = bar_data["Close"]
        
        if (current_price > self.fast_values[-1] > 
            self.medium_values[-1] > self.slow_values[-1]):
            self.current_regime = RegimeType.BULL
        elif (current_price < self.fast_values[-1] < 
              self.medium_values[-1] < self.slow_values[-1]):
            self.current_regime = RegimeType.BEAR
        elif (abs(self.fast_values[-1] - self.medium_values[-1]) / 
              self.medium_values[-1] < 0.02):
            self.current_regime = RegimeType.CHOPPY
        else:
            self.current_regime = RegimeType.UNKNOWN
        
        return self.current_regime
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.fast_values = []
        self.medium_values = []
        self.slow_values = []
        self.close_history = []
```

## Integration with Other Modules

### With Indicators Module

The Regime Detection module can reuse components from the Indicators module for calculations:

```python
from src.indicators.trend import average_directional_index
from src.indicators.volatility import historical_volatility
from src.regime_detection import DetectorBase, RegimeType, registry

@registry.register(category="custom")
class CustomIndicatorDetector(DetectorBase):
    """Detector that uses existing indicator functions."""
    
    def __init__(self, name=None, config=None):
        super().__init__(name, config or {"window": 14})
        self.window = self.config["window"]
        self.price_history = []
        self.high_history = []
        self.low_history = []
    
    def detect_regime(self, bar_data):
        # Update history
        self.price_history.append(bar_data["Close"])
        self.high_history.append(bar_data["High"])
        self.low_history.append(bar_data["Low"])
        
        # Ensure enough history
        if len(self.price_history) < self.window + 1:
            return RegimeType.UNKNOWN
            
        # Use indicator functions from indicators module
        adx, plus_di, minus_di = average_directional_index(
            np.array(self.high_history),
            np.array(self.low_history),
            np.array(self.price_history),
            period=self.window
        )
        
        volatility = historical_volatility(
            np.array(self.price_history),
            period=self.window
        )
        
        # Determine regime based on indicators
        if adx > 25:
            if plus_di > minus_di:
                return RegimeType.TRENDING_UP
            else:
                return RegimeType.TRENDING_DOWN
        elif volatility > 0.015:
            return RegimeType.VOLATILE
        else:
            return RegimeType.RANGE_BOUND
```

### With Strategies Module

Integrate regime detection with the strategies module:

```python
from src.strategies import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from src.regime_detection import RegimeType

@StrategyRegistry.register(category="regime_adaptive")
class AdaptiveMAStrategy(Strategy):
    """Strategy that adapts MA parameters based on market regime."""
    
    def __init__(self, regime_detector, name=None):
        super().__init__(name or "AdaptiveMAStrategy")
        self.regime_detector = regime_detector
        
        # Different MA parameters for different regimes
        self.ma_params = {
            RegimeType.VOLATILE: {"fast": 5, "slow": 15},
            RegimeType.LOW_VOLATILITY: {"fast": 10, "slow": 30},
            RegimeType.TRENDING_UP: {"fast": 10, "slow": 20},
            RegimeType.TRENDING_DOWN: {"fast": 5, "slow": 15},
            RegimeType.RANGE_BOUND: {"fast": 3, "slow": 10}
        }
        
        # Default parameters
        self.default_params = {"fast": 10, "slow": 20}
        
        # State variables
        self.current_regime = RegimeType.UNKNOWN
        self.price_history = []
        
    def on_bar(self, event):
        bar = event.bar
        
        # Update price history
        self.price_history.append(bar["Close"])
        
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(bar)
        
        # Get MA parameters for current regime
        params = self.ma_params.get(self.current_regime, self.default_params)
        fast_period = params["fast"]
        slow_period = params["slow"]
        
        # Generate signal based on adaptive MAs
        # Implementation details...
        
        # Create signal
        signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name,
            metadata={"regime": self.current_regime.name}
        )
        
        return signal
```

### With Risk Management Module

Adjust risk parameters based on detected regime:

```python
from src.risk_management import RiskManager
from src.risk_management.types import RiskParameters
from src.regime_detection import RegimeType

class RegimeAwareRiskManager(RiskManager):
    """Risk manager that adapts parameters based on market regime."""
    
    def __init__(self, regime_detector, **kwargs):
        super().__init__(**kwargs)
        self.regime_detector = regime_detector
        
        # Define different risk parameters for different regimes
        self.regime_risk_params = {
            RegimeType.VOLATILE: RiskParameters(
                stop_loss_pct=3.0,
                take_profit_pct=6.0,
                max_position_pct=0.03  # Smaller positions in volatile markets
            ),
            RegimeType.TRENDING_UP: RiskParameters(
                stop_loss_pct=2.0,
                take_profit_pct=5.0,
                max_position_pct=0.05  # Normal position size in uptrend
            ),
            RegimeType.TRENDING_DOWN: RiskParameters(
                stop_loss_pct=1.5,
                take_profit_pct=4.0,
                max_position_pct=0.04  # Slightly smaller in downtrend
            ),
            RegimeType.RANGE_BOUND: RiskParameters(
                stop_loss_pct=1.0,
                take_profit_pct=2.0,
                max_position_pct=0.05  # Normal position size
            )
        }
    
    def open_trade(self, trade_id, direction, entry_price, entry_time, **kwargs):
        """Override to use regime-specific risk parameters."""
        # Get current market regime
        current_bar = kwargs.get('current_bar', {})
        current_regime = self.regime_detector.detect_regime(current_bar)
        
        # Get risk parameters for current regime
        regime_params = self.regime_risk_params.get(current_regime)
        
        if regime_params:
            # Use regime-specific parameters
            original_params = self.risk_params
            self.risk_params = regime_params
            
            # Call parent method with new parameters
            result = super().open_trade(trade_id, direction, entry_price, entry_time, **kwargs)
            
            # Restore original parameters
            self.risk_params = original_params
            
            return result
        else:
            # Use default parameters
            return super().open_trade(trade_id, direction, entry_price, entry_time, **kwargs)
```

## Best Practices

1. **Choose appropriate detection methods**: Different markets require different detection approaches; trend detection works well for directional markets, volatility detection for mean-reverting markets.

2. **Avoid frequent regime changes**: Implement smoothing or hysteresis in regime transitions to prevent whipsaws.

3. **Validate regime detectors**: Use historical data to confirm that your regime detector identifies meaningful regimes that correlate with strategy performance.

4. **Adapt parameters, not strategies**: Often better to use the same strategy with regime-optimized parameters than completely different strategies per regime.

5. **Consider regime stability**: Check how stable detected regimes are over time to avoid overfitting.

6. **Combine multiple detection methods**: Using composite detectors often provides more robust regime identification.

7. **Handle regime transitions**: Pay special attention to transitions between regimes, which may require special handling.

8. **Monitor regime distribution**: Track the percentage of time the market spends in each regime to ensure your detection is balanced.

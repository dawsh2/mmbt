# Features Module Documentation

The Features module provides a framework for transforming raw price data and indicators into meaningful inputs for trading rules. Features represent derived values, patterns, or conditions within market data that can be used to generate trading signals.

## Core Concepts

**Feature**: A calculation that transforms raw price data into a meaningful input for trading rules.  
**FeatureSet**: A collection of related features that can be calculated together.  
**StatefulFeature**: A feature that maintains internal state between calculations.  
**CompositeFeature**: A feature composed of multiple sub-features combined using a specified function.

## Basic Usage

```python
from features import NormalizedPriceFeature, MACrossoverFeature, combine_features
from features import create_default_feature_set

# Create a single feature
norm_price = NormalizedPriceFeature(
    name="normalized_price",
    params={"method": "z-score", "window": 20}
)

# Calculate the feature value from market data
data = {
    "Close": [101.2, 102.5, 103.1, 102.8, 103.5],
    "High": [102.1, 103.0, 103.7, 103.2, 104.1],
    "Low": [100.5, 101.9, 102.4, 102.1, 102.8],
    "Volume": [5000, 6200, 5800, 5500, 6800]
}

# Get the feature value
value = norm_price.calculate(data)
print(f"Normalized price: {value}")

# Create a feature set with multiple features
feature_set = create_default_feature_set()

# Calculate all features at once
results = feature_set.calculate_all(data)
```

## API Reference

### Feature Classes

#### Feature (Base Class)

Abstract base class for all features.

**Constructor Parameters:**
- `name` (str): Unique identifier for the feature
- `params` (dict, optional): Dictionary of parameters for feature calculation
- `description` (str, optional): Human-readable description of the feature

**Methods:**
- `calculate(data)`: Calculate the feature value from the provided data
  - `data` (dict): Dictionary containing price data and indicators
  - Returns: The calculated feature value (can be scalar, array, or dictionary)

**Example:**
```python
class MyCustomFeature(Feature):
    def __init__(self, name="my_feature", params=None, description=""):
        super().__init__(name, params or {"window": 10}, description)
        
    def calculate(self, data):
        window = self.params["window"]
        if "Close" not in data or len(data["Close"]) < window:
            return 0
        
        # Custom calculation
        return sum(data["Close"][-window:]) / window
```

#### StatefulFeature

Feature that maintains internal state between calculations.

**Additional Parameters:**
- `max_history` (int, optional): Maximum history length to maintain (default: 100)

**Additional Methods:**
- `update(data)`: Update the feature state with new data and return the new value
- `reset()`: Reset the feature's internal state

**Example:**
```python
# Create a stateful feature that maintains history
volatility_feature = VolumeProfileFeature(
    name="volume_profile",
    params={"num_bins": 10, "window": 20},
    max_history=50
)

# Update with new data
new_value = volatility_feature.update(new_data)

# Reset state when starting a new analysis
volatility_feature.reset()
```

#### CompositeFeature

A feature composed of multiple sub-features.

**Constructor Parameters:**
- `name` (str): Unique identifier for the feature
- `features` (list): List of component features
- `combiner_func` (callable): Function that combines component feature values
- `params` (dict, optional): Dictionary of parameters passed to the combiner function
- `description` (str, optional): Human-readable description

**Example:**
```python
# Create component features
rsi_feature = OscillatorStateFeature(name="rsi_state", params={"oscillator": "RSI"})
macd_feature = OscillatorStateFeature(name="macd_state", params={"oscillator": "MACD"})

# Create composite feature using a combiner function
combined_feature = CompositeFeature(
    name="oscillator_consensus",
    features=[rsi_feature, macd_feature],
    combiner_func=weighted_average_combiner,
    params={"weights": [0.7, 0.3]}
)
```

#### FeatureSet

A collection of features with convenience methods for batch calculation.

**Constructor Parameters:**
- `features` (list, optional): List of Feature objects to include in the set
- `name` (str, optional): Name for the feature set

**Methods:**
- `add_feature(feature)`: Add a feature to the set
- `calculate_all(data)`: Calculate all features in the set on the provided data
- `to_dataframe(data)`: Calculate all features and return as a DataFrame

**Example:**
```python
# Create a feature set
feature_set = FeatureSet(name="technical_indicators")

# Add features
feature_set.add_feature(MACrossoverFeature(name="sma_crossover"))
feature_set.add_feature(OscillatorStateFeature(name="rsi_state"))

# Calculate all features at once
results = feature_set.calculate_all(data)

# Get results as DataFrame
df = feature_set.to_dataframe(data)
```

### Feature Categories

#### Price Features

Features derived directly from price data.

**ReturnFeature**: Calculate price returns over specified periods.
```python
# Calculate 1-day and 5-day returns
return_feature = ReturnFeature(
    name="returns",
    params={
        "periods": [1, 5],
        "price_key": "Close",
        "log_returns": True
    }
)
```

**NormalizedPriceFeature**: Normalize price relative to a reference point.
```python
# Z-score normalization with 20-day window
norm_price = NormalizedPriceFeature(
    name="norm_price",
    params={
        "method": "z-score",  # Options: 'z-score', 'min-max', 'relative'
        "window": 20,
        "price_key": "Close"
    }
)
```

**PricePatternFeature**: Detect specific price patterns in the data.
```python
# Detect double top pattern
pattern_feature = PricePatternFeature(
    name="double_top",
    params={
        "pattern": "double_top",  # Options: 'double_top', 'double_bottom', 'head_shoulders'
        "window": 20,
        "threshold": 0.03
    }
)
```

**VolumeProfileFeature**: Calculate volume profile features based on price and volume data.
```python
volume_profile = VolumeProfileFeature(
    name="vol_profile",
    params={
        "num_bins": 10,
        "window": 20
    }
)
```

**PriceDistanceFeature**: Calculate distance of price from reference levels.
```python
# Distance from 50-day moving average
price_distance = PriceDistanceFeature(
    name="ma_distance",
    params={
        "reference": "ma",  # Options: 'ma', 'level', 'high_low', 'bands'
        "period": 50,
        "as_percentage": True
    }
)
```

#### Technical Features

Features derived from technical indicators.

**MACrossoverFeature**: Detect crossovers between moving averages.
```python
ma_cross = MACrossoverFeature(
    name="sma_crossover",
    params={
        "fast_ma": "SMA_10",
        "slow_ma": "SMA_30",
        "smooth": False
    }
)
```

**OscillatorStateFeature**: Analyze oscillator indicators (RSI, Stochastic, etc.).
```python
rsi_state = OscillatorStateFeature(
    name="rsi_state",
    params={
        "oscillator": "RSI",
        "overbought": 70,
        "oversold": 30,
        "check_divergence": True
    }
)
```

**TrendStrengthFeature**: Measure the strength and direction of the current trend.
```python
trend_strength = TrendStrengthFeature(
    name="adx_trend",
    params={
        "method": "adx",  # Options: 'adx', 'aroon', 'slope'
        "threshold": 25,
        "lookback": 14
    }
)
```

**VolatilityFeature**: Analyze market volatility using various methods.
```python
volatility = VolatilityFeature(
    name="volatility",
    params={
        "method": "atr",  # Options: 'atr', 'bb_width', 'std_dev'
        "period": 14,
        "normalize": True
    }
)
```

**SupportResistanceFeature**: Identify support and resistance levels.
```python
support_resistance = SupportResistanceFeature(
    name="sr_levels",
    params={
        "method": "peaks",  # Options: 'peaks', 'pivots', 'volume'
        "lookback": 100,
        "strength_threshold": 2,
        "proximity_threshold": 3.0
    }
)
```

**SignalAgreementFeature**: Analyze agreement among multiple technical indicators.
```python
signal_agreement = SignalAgreementFeature(
    name="indicator_consensus",
    params={
        "indicators": ["MA_crossover", "RSI", "MACD", "BB"],
        "weights": {"MA_crossover": 0.3, "RSI": 0.3, "MACD": 0.2, "BB": 0.2},
        "threshold": 0.6
    }
)
```

**DivergenceFeature**: Detect divergences between price and indicators.
```python
divergence = DivergenceFeature(
    name="rsi_divergence",
    params={
        "indicator": "RSI",
        "lookback": 20,
        "peak_threshold": 3
    }
)
```

#### Time Features

Features derived from time and date information.

**TimeOfDayFeature**: Extract time of day patterns.
```python
time_of_day = TimeOfDayFeature(
    name="time_of_day",
    params={
        "format": "%Y-%m-%d %H:%M:%S",
        "trading_hours": [(9, 16)],  # 9:00 AM to 4:00 PM
        "zones": {"morning": (9, 12), "afternoon": (12, 16)}
    }
)
```

**DayOfWeekFeature**: Extract day of week patterns.
```python
day_of_week = DayOfWeekFeature(
    name="day_of_week",
    params={
        "format": "%Y-%m-%d %H:%M:%S",
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday (0-based)
        "with_cyclical": True  # Generate sin/cos encoding
    }
)
```

**MonthFeature**: Extract month and seasonal patterns.
```python
month_feature = MonthFeature(
    name="month",
    params={
        "format": "%Y-%m-%d %H:%M:%S",
        "with_cyclical": True,
        "with_quarters": True
    }
)
```

**SeasonalityFeature**: Detect seasonal patterns in price data.
```python
seasonality = SeasonalityFeature(
    name="monthly_pattern",
    params={
        "period": "month",  # Options: 'day', 'week', 'month', 'quarter'
        "lookback": 252,
        "min_pattern_strength": 0.6
    }
)
```

**EventFeature**: Identify special events and calendar dates.
```python
event_feature = EventFeature(
    name="calendar_events",
    params={
        "format": "%Y-%m-%d %H:%M:%S",
        "holidays": {"12-25": "Christmas", "01-01": "New Year"},
        "events": {"2023-04-28": "Earnings Release"},
        "event_window": 1  # Days before/after to consider
    }
)
```

### Feature Utility Functions

**combine_features**: Combine multiple features into a composite feature.
```python
# Combine features with weighted average
combined = combine_features(
    features=[feature1, feature2, feature3],
    combiner_func=weighted_average_combiner,
    name="combined_feature",
    params={"weights": [0.5, 0.3, 0.2]},
    description="Weighted combination of features"
)
```

**weighted_average_combiner**: Combine feature values using a weighted average.
```python
# Usage in combine_features:
combined = combine_features(
    features=[feature1, feature2],
    combiner_func=weighted_average_combiner,
    params={"weights": [0.7, 0.3]}
)
```

**logical_combiner**: Combine feature values using logical operations.
```python
# Usage in combine_features:
combined = combine_features(
    features=[feature1, feature2, feature3],
    combiner_func=logical_combiner,
    params={"operation": "majority", "threshold": 0.5}
)
```

**threshold_combiner**: Combine feature values using thresholds to generate a signal.
```python
# Usage in combine_features:
combined = combine_features(
    features=[feature1, feature2, feature3],
    combiner_func=threshold_combiner,
    params={
        "thresholds": [0.5, 0.7, 0.3],
        "directions": [1, 1, -1],
        "min_agreements": 2
    }
)
```

**cross_feature_indicator**: Create a crossover indicator from two features.
```python
# Usage in combine_features:
crossover = combine_features(
    features=[fast_ma_feature, slow_ma_feature],
    combiner_func=cross_feature_indicator,
    params={"direction": 1}
)
```

**create_feature_vector**: Create a feature vector from multiple features.
```python
# Create numeric vector for machine learning
feature_vector = create_feature_vector(
    features=[feature1, feature2, feature3],
    data=market_data
)
```

## Advanced Usage

### Creating Custom Features

You can create custom features by subclassing the Feature or StatefulFeature base class:

```python
from features import Feature, register_feature

@register_feature(category="custom")
class MyCustomFeature(Feature):
    def __init__(self, name="my_custom_feature", params=None, description=""):
        super().__init__(name, params or self.default_params, description)
    
    @property
    def default_params(self):
        return {
            "window": 10,
            "threshold": 0.5
        }
    
    def _validate_params(self):
        """Validate parameters (called during initialization)"""
        if self.params["window"] <= 0:
            raise ValueError("Window size must be positive")
    
    def calculate(self, data):
        """Calculate feature value from data"""
        window = self.params["window"]
        threshold = self.params["threshold"]
        
        if "Close" not in data or len(data["Close"]) < window:
            return {"signal": 0, "value": 0}
        
        # Custom calculation logic
        prices = data["Close"][-window:]
        avg_price = sum(prices) / len(prices)
        
        # Return dictionary with calculated values
        return {
            "value": avg_price,
            "signal": 1 if prices[-1] > avg_price * (1 + threshold) else 0
        }
```

### Working with FeatureSet for Machine Learning

FeatureSets are useful for preparing data for machine learning models:

```python
import pandas as pd
from features import FeatureSet, ReturnFeature, MACrossoverFeature, OscillatorStateFeature

# Create features for machine learning
ml_features = FeatureSet(name="ml_features")
ml_features.add_feature(ReturnFeature(name="returns", params={"periods": [1, 5, 10]}))
ml_features.add_feature(MACrossoverFeature(name="ma_cross"))
ml_features.add_feature(OscillatorStateFeature(name="rsi"))
ml_features.add_feature(OscillatorStateFeature(name="macd", params={"oscillator": "MACD"}))

# Convert market data to feature DataFrame
feature_df = ml_features.to_dataframe(market_data)

# Split into features and target for ML
X = feature_df.drop('returns_1', axis=1)  # Features
y = (feature_df['returns_1'] > 0).astype(int)  # Target (positive returns)

# Now use with your favorite ML library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Creating Complex Feature Hierarchies

Features can be combined in hierarchical structures to create sophisticated trading logic:

```python
from features import (
    combine_features, logical_combiner, weighted_average_combiner,
    MACrossoverFeature, TrendStrengthFeature, VolatilityFeature, OscillatorStateFeature
)

# Create base features
trend = TrendStrengthFeature(name="adx_trend")
volatility = VolatilityFeature(name="volatility")
ma_cross = MACrossoverFeature(name="sma_cross")
rsi = OscillatorStateFeature(name="rsi_state")

# First level: combine trend and volatility
market_regime = combine_features(
    features=[trend, volatility],
    combiner_func=weighted_average_combiner,
    name="market_regime",
    params={"weights": [0.7, 0.3]}
)

# Second level: combine technical signals
tech_signals = combine_features(
    features=[ma_cross, rsi],
    combiner_func=logical_combiner,
    name="tech_signals",
    params={"operation": "majority"}
)

# Top level: combine regime and signals
final_feature = combine_features(
    features=[market_regime, tech_signals],
    combiner_func=weighted_average_combiner,
    name="final_signal",
    params={"weights": [0.4, 0.6]}
)

# Calculate the final result
result = final_feature.calculate(market_data)
```

### Handling Timestamp Data

When working with time features, ensure your data has properly formatted timestamps:

```python
import datetime

# Market data with timestamp
data = {
    "timestamp": "2023-06-01 10:30:00",
    "Open": 100.5,
    "High": 101.2,
    "Low": 100.1,
    "Close": 100.8,
    "Volume": 5000
}

# Parse timestamp if it's a string
if isinstance(data["timestamp"], str):
    data["timestamp"] = datetime.datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")

# Create time features
time_feature = TimeOfDayFeature()
day_feature = DayOfWeekFeature()

# Calculate features
time_info = time_feature.calculate(data)
day_info = day_feature.calculate(data)

print(f"Trading session: {time_info['session']}")
print(f"Day of week: {day_info['day_name']} (is_trading_day: {day_info['is_trading_day']})")
```

### Using Feature Registry

The feature registry allows you to create features by name:

```python
from features import get_registry

registry = get_registry()

# Create a feature by name
rsi_feature = registry.create_feature(
    name="OscillatorStateFeature",
    params={"oscillator": "RSI", "overbought": 75, "oversold": 25},
    feature_name="custom_rsi"
)

# List available features in a category
technical_features = registry.list_features(category="technical")
print(f"Available technical features: {technical_features}")
```

## Creating Default Features Set

For convenience, the module provides a function to create a set of common features:

```python
from features import create_default_feature_set

# Create a default set of features
default_features = create_default_feature_set(name="default_features")

# Calculate all default features
results = default_features.calculate_all(market_data)

# Get the price features
normalized_price = results["norm_price_zscore"]
returns_1d = results["return_1d"]
returns_5d = results["return_5d"]

# Get the technical features
sma_crossover = results["sma_crossover"]
rsi_state = results["rsi_state"]
volatility = results["volatility"]

# Get the time features
day_of_week = results["day_of_week"]
monthly_pattern = results["monthly_pattern"]
```
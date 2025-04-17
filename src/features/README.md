# Features Module

Features Module

This module provides the feature layer of the trading system, which transforms
raw price data and technical indicators into meaningful trading signals.

## Contents

- [feature_base](#feature_base)
- [feature_registry](#feature_registry)
- [feature_utils](#feature_utils)
- [price_features](#price_features)
- [technical_features](#technical_features)
- [time_features](#time_features)

## feature_base

Base Feature Module

This module defines the base Feature class and related abstractions for the feature layer
of the trading system. Features transform raw price data and indicators into meaningful
inputs for trading rules.

### Classes

#### `Feature`

Base class for all features in the trading system.

Features transform raw price data and indicators into a format suitable for 
use by trading rules. They encapsulate the logic for calculating derived values
from market data while maintaining a consistent interface.

##### Methods

###### `__init__(name, params=None, description='')`

Initialize a feature with parameters.

Args:
    name: Unique identifier for the feature
    params: Dictionary of parameters for feature calculation
    description: Human-readable description of what the feature measures

###### `_validate_params()`

*Returns:* `None`

Validate the parameters provided to the feature.

This method should be overridden by subclasses to provide
specific parameter validation logic.

Raises:
    ValueError: If parameters are invalid

###### `calculate(data)`

*Returns:* `Any`

Calculate the feature value from the provided data.

Args:
    data: Dictionary containing price data and calculated indicators
         required by this feature
         
Returns:
    The calculated feature value (can be scalar, array, or any type)

###### `default_params()`

*Returns:* `Dict[str, Any]`

Get the default parameters for this feature.

Returns:
    Dictionary of default parameter values

###### `__str__()`

*Returns:* `str`

String representation of the feature.

###### `__repr__()`

*Returns:* `str`

Detailed representation of the feature.

#### `FeatureSet`

A collection of features with convenience methods for batch calculation.

FeatureSet provides a way to organize related features and calculate them
together efficiently on the same dataset.

##### Methods

###### `__init__(features=None, name='')`

Initialize a feature set with a list of features.

Args:
    features: List of Feature objects to include in the set
    name: Optional name for the feature set

###### `add_feature(feature)`

*Returns:* `None`

Add a feature to the set.

Args:
    feature: Feature object to add

###### `calculate_all(data)`

*Returns:* `Dict[str, Any]`

Calculate all features in the set on the provided data.

Args:
    data: Dictionary containing price data and indicators
    
Returns:
    Dictionary mapping feature names to calculated values

###### `to_dataframe(data)`

*Returns:* `pd.DataFrame`

Calculate all features and return as a DataFrame.

Args:
    data: Dictionary containing price data and indicators
    
Returns:
    DataFrame with columns for each feature

###### `__len__()`

*Returns:* `int`

Get the number of features in the set.

###### `__getitem__(index)`

*Returns:* `Feature`

Get a feature by index.

###### `__iter__()`

Iterate through features.

#### `CompositeFeature`

A feature composed of multiple sub-features.

CompositeFeature allows combining multiple features into a single feature,
making it easier to create complex feature hierarchies.

##### Methods

###### `__init__(name, features, combiner_func, params=None, description='')`

Initialize a composite feature.

Args:
    name: Unique identifier for the feature
    features: List of component features
    combiner_func: Function that combines component feature values
    params: Dictionary of parameters
    description: Human-readable description

###### `calculate(data)`

*Returns:* `Any`

Calculate the composite feature by combining sub-feature values.

Args:
    data: Dictionary containing price data and indicators
         
Returns:
    The calculated composite feature value

###### `__repr__()`

*Returns:* `str`

Detailed representation of the composite feature.

#### `StatefulFeature`

A feature that maintains internal state between calculations.

StatefulFeature is useful for features that depend on historical values
or require incremental updates, such as EMA-based features.

##### Methods

###### `__init__(name, params=None, description='', max_history=100)`

Initialize a stateful feature.

Args:
    name: Unique identifier for the feature
    params: Dictionary of parameters
    description: Human-readable description
    max_history: Maximum history length to maintain

###### `update(data)`

*Returns:* `Any`

Update the feature state with new data and return the new value.

Args:
    data: Dictionary containing price data and indicators
    
Returns:
    The newly calculated feature value

###### `reset()`

*Returns:* `None`

Reset the feature's internal state.

## feature_registry

Feature Registry Module

This module provides a registry system for features, allowing them to be
registered, discovered, and instantiated by name throughout the trading system.

### Functions

#### `get_registry()`

*Returns:* `FeatureRegistry`

Get the global feature registry instance.

Returns:
    The FeatureRegistry singleton instance

#### `register_feature(category='general')`

Decorator to register a feature class in the registry.

Args:
    category (str): Category for the feature

Returns:
    decorator function

*Returns:* decorator function

#### `register_features_in_module(module, category='general')`

*Returns:* `None`

Register all Feature classes in a module with the registry.

Args:
    module: The module object containing feature classes
    category: Category to group the features under

### Classes

#### `FeatureRegistry`

Registry for feature classes in the trading system.

The FeatureRegistry maintains a mapping of feature names to their implementing
classes, allowing features to be instantiated by name.

##### Methods

###### `__new__(cls)`

Implement singleton pattern for the registry.

###### `register(feature_class, name=None, category='general')`

*Returns:* `None`

Register a feature class with the registry.

Args:
    feature_class: The Feature class to register
    name: Optional name to register the feature under (defaults to class name)
    category: Category to group the feature under
    
Raises:
    ValueError: If a feature with the same name is already registered

###### `get_feature_class(name)`

Get a feature class by name.

Args:
    name: Name of the feature to retrieve
    
Returns:
    The feature class
    
Raises:
    KeyError: If no feature with the given name is registered

*Returns:* The feature class

###### `create_feature(name, params=None, feature_name=None)`

Create an instance of a feature by name.

Args:
    name: Name of the feature class to instantiate
    params: Parameters to pass to the feature constructor
    feature_name: Optional name for the created feature instance
                (defaults to the registered name)
    
Returns:
    An instance of the requested feature
    
Raises:
    KeyError: If no feature with the given name is registered

*Returns:* An instance of the requested feature

###### `list_features(category=None)`

*Returns:* `List[str]`

List all registered features, optionally filtered by category.

Args:
    category: Optional category to filter by
    
Returns:
    List of feature names

###### `list_categories()`

*Returns:* `List[str]`

List all feature categories.

Returns:
    List of category names

###### `clear()`

*Returns:* `None`

Clear all registered features (useful for testing).

## feature_utils

Feature Utilities Module

This module provides utility functions for combining and transforming
features in various ways to create complex feature compositions.

### Functions

#### `combine_features(features, combiner_func, name='composite_feature', params=None, description='Combined features')`

*Returns:* `CompositeFeature`

Combine multiple features into a composite feature.

Args:
    features: List of feature objects to combine
    combiner_func: Function that takes a list of feature values and returns a combined value
    name: Name for the new composite feature
    params: Parameters to pass to the combiner function
    description: Description of the new feature
    
Returns:
    CompositeFeature object

#### `weighted_average_combiner(feature_values)`

*Returns:* `float`

Combine feature values using a weighted average.

Args:
    feature_values: List of feature values to combine
    params: Must include 'weights' as a list of weights
    
Returns:
    Weighted average of feature values

#### `logical_combiner(feature_values)`

*Returns:* `bool`

Combine feature values using logical operations.

Args:
    feature_values: List of feature values to combine
    params: Must include 'operation' as one of 'and', 'or', 'majority'
    
Returns:
    Boolean result of logical operation

#### `threshold_combiner(feature_values)`

*Returns:* `int`

Combine feature values using thresholds to generate a signal.

Args:
    feature_values: List of feature values to combine
    params: Dictionary containing:
        - 'thresholds': List of thresholds for each feature
        - 'directions': List of expected directions (1 or -1) for each feature
        - 'min_agreements': Minimum number of agreements needed for a signal
    
Returns:
    Signal value: 1 (buy), -1 (sell), or 0 (neutral)

#### `cross_feature_indicator(feature_values)`

*Returns:* `Dict[str, Any]`

Create a crossover indicator from two features.

Args:
    feature_values: List of exactly two feature values
    params: Dictionary containing:
        - 'direction': Expected crossover direction (1 for first > second, -1 for first < second)
    
Returns:
    Dictionary with crossover status information

#### `z_score_normalize(feature_values)`

*Returns:* `List[float]`

Normalize feature values using Z-score normalization.

Args:
    feature_values: List of feature values to normalize
    params: Dictionary containing additional parameters (not used)
    
Returns:
    List of normalized feature values

#### `create_feature_vector(features, data)`

*Returns:* `np.ndarray`

Create a feature vector from multiple features.

Args:
    features: List of feature objects
    data: Dictionary containing the data to calculate features from
    
Returns:
    Numpy array containing feature values

#### `combine_time_series_features(features, data, lookback=10)`

*Returns:* `np.ndarray`

Create a time series of feature vectors from multiple features.

Args:
    features: List of feature objects
    data: Dictionary containing historical data (with each value as a list/array)
    lookback: Number of historical time steps to include
    
Returns:
    2D numpy array with shape (lookback, num_features)

## price_features

Price Features Module

This module provides features derived directly from price data, such as returns,
normalized prices, and price patterns.

### Classes

#### `ReturnFeature`

Calculate price returns over specified periods.

This feature computes price returns (percentage change) over one or more time periods.

##### Methods

###### `__init__(name='return', params=None, description='Price returns over specified periods')`

Initialize the return feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - periods: List of periods to calculate returns for (default: [1])
        - price_key: Key to price data in the input dictionary (default: 'Close')
        - log_returns: Whether to compute log returns instead of simple returns (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for return calculation.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[int, float]`

Calculate returns for each specified period.

Args:
    data: Dictionary containing price history with keys like 'Open', 'High', 'Low', 'Close'
         Expected format: {'Close': np.array or list of prices, ...}

Returns:
    Dictionary mapping periods to calculated returns

#### `NormalizedPriceFeature`

Normalize price data relative to a reference point.

This feature normalizes prices using various methods such as z-score normalization,
min-max scaling, or relative to a moving average.

##### Methods

###### `__init__(name='normalized_price', params=None, description='Normalized price data')`

Initialize the normalized price feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - method: Normalization method ('z-score', 'min-max', or 'relative') (default: 'z-score')
        - window: Window size for calculating statistics (default: 20)
        - price_key: Key to price data in the input dictionary (default: 'Close')
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for normalization.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `float`

Calculate normalized price.

Args:
    data: Dictionary containing price history

Returns:
    Normalized price value

#### `PricePatternFeature`

Detect specific price patterns in the data.

This feature identifies common price patterns such as double tops/bottoms,
head and shoulders, or trendline breaks.

##### Methods

###### `__init__(name='price_pattern', params=None, description='Price pattern detection')`

Initialize the price pattern feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - pattern: Pattern to detect ('double_top', 'double_bottom', 'head_shoulders', etc.)
        - window: Window size for pattern detection (default: 20)
        - threshold: Threshold for pattern confirmation (default: 0.03)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for pattern detection.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Detect price patterns in the data.

Args:
    data: Dictionary containing price history

Returns:
    Dictionary with pattern detection results, including:
    - detected: Boolean indicating if pattern was detected
    - confidence: Confidence score for the detection (0.0-1.0)
    - details: Additional pattern-specific details

###### `_detect_double_top(highs, lows, closes, threshold)`

Detect double top pattern.

###### `_detect_double_bottom(highs, lows, closes, threshold)`

Detect double bottom pattern.

###### `_detect_head_shoulders(highs, lows, closes, threshold)`

Detect head and shoulders pattern.

###### `_detect_inverse_head_shoulders(highs, lows, closes, threshold)`

Detect inverse head and shoulders pattern.

#### `VolumeProfileFeature`

Calculate volume profile features based on price and volume data.

This feature analyzes the distribution of volume across price levels.

##### Methods

###### `__init__(name='volume_profile', params=None, description='Volume profile analysis', max_history=100)`

Initialize the volume profile feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - num_bins: Number of price bins for volume distribution (default: 10)
        - window: Window size for calculation (default: 20)
    description: Feature description
    max_history: Maximum history to maintain

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for volume profile.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate volume profile features.

Args:
    data: Dictionary containing price and volume history

Returns:
    Dictionary with volume profile metrics

#### `PriceDistanceFeature`

Calculate distance of price from various reference points.

This feature measures how far the current price is from reference
levels such as moving averages, support/resistance, or previous highs/lows.

##### Methods

###### `__init__(name='price_distance', params=None, description='Distance from reference levels')`

Initialize the price distance feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - reference: Reference type ('ma', 'level', 'high_low', 'bands')
        - period: Period for moving average if reference is 'ma' (default: 20)
        - levels: List of price levels if reference is 'level'
        - lookback: Lookback period for high/low if reference is 'high_low' (default: 20)
        - as_percentage: Whether to return distance as percentage (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for price distance.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, float]`

Calculate price distance from reference levels.

Args:
    data: Dictionary containing price history and indicators

Returns:
    Dictionary with distance measurements

## technical_features

Technical Features Module

This module provides features derived from technical indicators, such as
moving average crossovers, oscillator states, and indicator divergences.

### Classes

#### `VolatilityFeature`

Volatility measurement feature.

This feature analyzes market volatility using indicators like ATR,
Bollinger Band width, or standard deviation of returns.

##### Methods

###### `__init__(name='volatility', params=None, description='Market volatility measurement')`

Initialize the volatility feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - method: Method for volatility calculation ('atr', 'bb_width', 'std_dev')
        - period: Period for calculation (default: 14)
        - normalize: Whether to normalize the volatility value (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for volatility calculation.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate volatility metrics.

Args:
    data: Dictionary containing price data and indicators

Returns:
    Dictionary with volatility information

###### `_calculate_atr(data, period, normalize)`

*Returns:* `Dict[str, Any]`

Calculate volatility using Average True Range.

###### `_calculate_bb_width(data, normalize)`

*Returns:* `Dict[str, Any]`

Calculate volatility using Bollinger Band width.

###### `_calculate_std_dev(data, period, normalize)`

*Returns:* `Dict[str, Any]`

Calculate volatility using standard deviation of returns.

#### `SupportResistanceFeature`

Support and Resistance Levels feature.

This feature identifies key support and resistance levels and measures
the distance of the current price from these levels.

##### Methods

###### `__init__(name='support_resistance', params=None, description='Support and resistance analysis')`

Initialize the support/resistance feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - method: Method for level detection ('peaks', 'pivots', 'volume')
        - lookback: Lookback period for calculation (default: 100)
        - strength_threshold: Threshold for level strength (default: 2)
        - proximity_threshold: Threshold for level proximity in % (default: 3.0)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for support/resistance calculation.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Identify support/resistance levels and calculate price distance.

Args:
    data: Dictionary containing price data and indicators

Returns:
    Dictionary with support/resistance information

###### `_find_peak_levels(highs, lows, strength_threshold)`

Find support/resistance levels based on price peaks.

###### `_find_pivot_levels(highs, lows, closes)`

Find support/resistance levels based on pivot points.

###### `_find_volume_levels(highs, lows, closes, volumes)`

Find support/resistance levels based on volume profile.

#### `SignalAgreementFeature`

Signal Agreement feature.

This feature analyzes the agreement or disagreement among multiple
technical indicators to provide a consensus signal.

##### Methods

###### `__init__(name='signal_agreement', params=None, description='Technical indicator consensus')`

Initialize the signal agreement feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - indicators: List of indicators to include in consensus
        - weights: Optional dictionary of weights for each indicator
        - threshold: Threshold for signal consensus (default: 0.6)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for signal agreement.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate signal agreement among indicators.

Args:
    data: Dictionary containing technical indicators

Returns:
    Dictionary with consensus information

###### `_get_indicator_signal(indicator, value)`

Extract signal from indicator value.

#### `DivergenceFeature`

Divergence Detection feature.

This feature detects divergences between price and indicators,
which can signal potential trend reversals.

##### Methods

###### `__init__(name='divergence', params=None, description='Price-indicator divergence detection')`

Initialize the divergence feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - indicator: Indicator to check for divergence ('RSI', 'MACD', 'CCI', etc.)
        - lookback: Lookback period for divergence detection (default: 20)
        - peak_threshold: Threshold for identifying peaks/troughs (default: 3)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for divergence detection.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Detect divergences between price and indicator.

Args:
    data: Dictionary containing price data and indicators

Returns:
    Dictionary with divergence information

###### `_find_peaks(data, threshold)`

Find peaks in data.

###### `_find_troughs(data, threshold)`

Find troughs in data.

###### `_check_bullish_divergence(prices, indicator, price_troughs, indicator_troughs)`

Check for bullish divergence: lower price lows but higher indicator lows.

###### `_check_bearish_divergence(prices, indicator, price_peaks, indicator_peaks)`

Check for bearish divergence: higher price highs but lower indicator highs.

###### `_check_hidden_bullish(prices, indicator, price_troughs, indicator_troughs)`

Check for hidden bullish divergence: higher price lows but lower indicator lows.

###### `_check_hidden_bearish(prices, indicator, price_peaks, indicator_peaks)`

Check for hidden bearish divergence: lower price highs but higher indicator highs.

#### `MACrossoverFeature`

Moving Average Crossover feature.

This feature detects crossovers between two moving averages and provides
information about the state and direction of the crossover.

##### Methods

###### `__init__(name='ma_crossover', params=None, description='Moving average crossover detection')`

Initialize the moving average crossover feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - fast_ma: Name of the fast MA indicator (default: 'SMA_10')
        - slow_ma: Name of the slow MA indicator (default: 'SMA_30')
        - smooth: Whether to smooth the crossover signal (default: False)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for moving average crossover.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate moving average crossover state and signals.

Args:
    data: Dictionary containing technical indicators

Returns:
    Dictionary with crossover information

#### `OscillatorStateFeature`

Oscillator State feature.

This feature analyzes oscillator indicators (RSI, Stochastic, etc.) and
identifies overbought/oversold conditions and divergences.

##### Methods

###### `__init__(name='oscillator_state', params=None, description='Oscillator state analysis')`

Initialize the oscillator state feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - oscillator: Name of the oscillator indicator (default: 'RSI')
        - overbought: Overbought threshold (default: 70)
        - oversold: Oversold threshold (default: 30)
        - check_divergence: Whether to check for divergence (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for oscillator state.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate oscillator state and identify conditions.

Args:
    data: Dictionary containing price data and indicators

Returns:
    Dictionary with oscillator state information

#### `TrendStrengthFeature`

Trend Strength feature.

This feature measures the strength and direction of the current trend
using indicators like ADX, Aroon, or directional movement.

##### Methods

###### `__init__(name='trend_strength', params=None, description='Trend strength measurement')`

Initialize the trend strength feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - method: Method for trend strength calculation ('adx', 'aroon', 'slope')
        - threshold: Threshold for strong trend (default: 25 for ADX)
        - lookback: Lookback period for calculation (default: 14)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for trend strength.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Calculate trend strength and direction.

Args:
    data: Dictionary containing price data and indicators

Returns:
    Dictionary with trend information

###### `_calculate_adx(data, threshold)`

*Returns:* `Dict[str, Any]`

Calculate trend strength using ADX.

###### `_calculate_aroon(data, threshold)`

*Returns:* `Dict[str, Any]`

Calculate trend strength using Aroon indicators.

###### `_calculate_slope(data, threshold)`

*Returns:* `Dict[str, Any]`

Calculate trend strength using price slope.

#### `SMA_Crossover`

Feature that detects crossovers between two SMAs.

##### Methods

###### `__init__(fast_window=10, slow_window=30, name=None)`

No docstring provided.

###### `calculate(bar_data, history=None)`

No docstring provided.

## time_features

Time Features Module

This module provides features derived from time and date information, such as
seasonality patterns, day of week effects, and other calendar-based features.

### Classes

#### `TimeOfDayFeature`

Time of Day feature.

This feature extracts time of day information and identifies patterns
based on the hour of the day.

##### Methods

###### `__init__(name='time_of_day', params=None, description='Time of day analysis')`

Initialize the time of day feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
        - trading_hours: List of trading hour ranges (default: [(9, 16)])
        - zones: Custom time zones to create (default: None)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for time of day.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Extract time of day features.

Args:
    data: Dictionary containing timestamp information

Returns:
    Dictionary with time of day features

#### `DayOfWeekFeature`

Day of Week feature.

This feature extracts day of week information and identifies patterns
based on the day of the week.

##### Methods

###### `__init__(name='day_of_week', params=None, description='Day of week analysis')`

Initialize the day of week feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
        - trading_days: List of trading days (default: [0, 1, 2, 3, 4])
        - with_cyclical: Whether to include cyclical encoding (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for day of week.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Extract day of week features.

Args:
    data: Dictionary containing timestamp information

Returns:
    Dictionary with day of week features

#### `MonthFeature`

Month feature.

This feature extracts month information and identifies patterns
based on the month of the year.

##### Methods

###### `__init__(name='month', params=None, description='Month analysis')`

Initialize the month feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
        - with_cyclical: Whether to include cyclical encoding (default: True)
        - with_quarters: Whether to include quarter information (default: True)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for month.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Extract month features.

Args:
    data: Dictionary containing timestamp information

Returns:
    Dictionary with month features

#### `SeasonalityFeature`

Seasonality feature.

This feature detects seasonal patterns in price data based on time periods
such as day of week, month of year, etc.

##### Methods

###### `__init__(name='seasonality', params=None, description='Seasonality pattern detection')`

Initialize the seasonality feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - period: Seasonality period to analyze ('day', 'week', 'month', 'quarter')
        - lookback: Lookback period for pattern analysis (default: 252)
        - min_pattern_strength: Minimum strength for pattern detection (default: 0.6)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for seasonality.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this feature.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Detect seasonality patterns in price data.

Args:
    data: Dictionary containing price history and timestamp information

Returns:
    Dictionary with seasonality information

#### `EventFeature`

Event detection feature.

This feature identifies special events like earnings releases, holidays,
economic announcements, etc. based on the calendar date.

##### Methods

###### `__init__(name='event', params=None, description='Calendar event detection')`

Initialize the event feature.

Args:
    name: Feature name
    params: Dictionary containing:
        - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
        - holidays: List or dictionary of holiday dates
        - events: Dictionary of other special events/dates
        - event_window: Number of days to consider around an event (default: 1)
    description: Feature description

###### `default_params()`

*Returns:* `Dict[str, Any]`

Default parameters for event detection.

###### `calculate(data)`

*Returns:* `Dict[str, Any]`

Detect calendar events based on date.

Args:
    data: Dictionary containing timestamp information

Returns:
    Dictionary with event information

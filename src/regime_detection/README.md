# Regime_detection Module

Market regime detection module for algorithmic trading.

This module provides tools for identifying market regimes and adapting 
trading strategies accordingly.

## Contents

- [detector_base](#detector_base)
- [detector_factory](#detector_factory)
- [detector_registry](#detector_registry)
- [composite_detectors](#composite_detectors)
- [trend_detectors](#trend_detectors)
- [volatility_detectors](#volatility_detectors)
- [regime_manager](#regime_manager)
- [regime_type](#regime_type)

## detector_base

Base class for market regime detectors.

### Classes

#### `DetectorBase`

Abstract base class for regime detection algorithms.

This class defines the interface that all regime detectors must implement.
Subclasses should override the detect_regime method to provide their specific
regime detection logic.

##### Methods

###### `__init__(name=None, config=None)`

Initialize the regime detector.

Args:
    name: Optional name for the detector
    config: Optional configuration dictionary

###### `detect_regime(bar_data)`

Detect the current market regime based on bar data.

Args:
    bar_data: Dictionary containing market data (OHLCV)
    
Returns:
    RegimeType: The detected market regime

*Returns:* RegimeType: The detected market regime

###### `reset()`

Reset the detector's internal state.

This method should be called when restarting analysis or backtesting.

###### `__str__()`

Return a string representation of the detector.

## detector_factory

Factory for creating market regime detectors.

### Classes

#### `DetectorFactory`

Factory for creating regime detector instances.

This class provides methods to instantiate detectors from their registered classes,
either by name or from configuration dictionaries.

##### Methods

###### `__init__(registry=None)`

Initialize the detector factory.

Args:
    registry: Optional DetectorRegistry instance

###### `create_detector(name_or_class, config=None)`

Create a detector instance.

Args:
    name_or_class: String name or class reference
    config: Optional configuration dictionary
    
Returns:
    DetectorBase: Instantiated detector

*Returns:* DetectorBase: Instantiated detector

###### `create_from_config(config)`

Create a detector from a configuration dictionary.

Args:
    config: Dictionary with 'type' and optional 'params'
    
Returns:
    DetectorBase: Instantiated detector

*Returns:* DetectorBase: Instantiated detector

###### `create_composite(configs, combination_method='majority')`

Create a composite detector from multiple configurations.

Args:
    configs: List of detector configurations
    combination_method: Method to combine detector outputs
    
Returns:
    CompositeDetector: Composite detector instance

*Returns:* CompositeDetector: Composite detector instance

## detector_registry

Registry for market regime detectors.

### Classes

#### `DetectorRegistry`

Registry for market regime detectors.

This class maintains a registry of available detector classes organized by
category. It provides decorator-based registration and methods to retrieve
detectors by name.

##### Methods

###### `__init__()`

Initialize an empty detector registry.

###### `register(category='general')`

Decorator to register a detector class.

Args:
    category: Category to register the detector under
    
Returns:
    function: Decorator function

*Returns:* function: Decorator function

###### `get_detector_class(name)`

Get a detector class by name.

Args:
    name: Name of the detector class
    
Returns:
    class: The detector class, or None if not found

*Returns:* class: The detector class, or None if not found

###### `list_detectors(category=None)`

List available detectors, optionally filtered by category.

Args:
    category: Optional category to filter by
    
Returns:
    list: Names of available detectors

*Returns:* list: Names of available detectors

###### `list_categories()`

List available detector categories.

Returns:
    list: Names of available categories

*Returns:* list: Names of available categories

## composite_detectors

Composite regime detectors.

### Classes

#### `CompositeDetector`

Composite detector that combines multiple regime detectors.

This detector aggregates the outputs of multiple detectors using
different combination methods (majority vote, consensus, etc.).

##### Methods

###### `__init__(name=None, config=None, detectors=None, combination_method='majority')`

Initialize the composite detector.

Args:
    name: Optional name for the detector
    config: Optional configuration dictionary
    detectors: List of detector instances to combine
    combination_method: Method to combine detector outputs:
        - majority: Use the most common regime
        - consensus: Use a regime only if all detectors agree
        - weighted: Use weighted voting (requires weights in config)

###### `add_detector(detector)`

Add a detector to the composite.

Args:
    detector: Detector instance to add

###### `detect_regime(bar_data)`

Detect the current market regime based on combined detector outputs.

Args:
    bar_data: Bar data dictionary
    
Returns:
    RegimeType: The combined market regime

*Returns:* RegimeType: The combined market regime

###### `reset()`

Reset the detector and all sub-detectors.

## trend_detectors

Trend-based regime detectors.

### Classes

#### `TrendStrengthRegimeDetector`

Regime detector based on trend strength using the ADX indicator.

This detector identifies trending and range-bound markets using the Average
Directional Index (ADX) and directional movement indicators (+DI, -DI).

##### Methods

###### `__init__(name=None, config=None)`

Initialize the trend strength detector.

Args:
    name: Optional name for the detector
    config: Optional configuration dictionary with parameters:
        - adx_period: Period for ADX calculation (default: 14)
        - adx_threshold: Threshold for trend identification (default: 25)

###### `detect_regime(bar)`

Detect the current market regime based on ADX and directional movements.

Args:
    bar: Bar data dictionary with 'High', 'Low', 'Close' keys
    
Returns:
    RegimeType: The detected market regime

*Returns:* RegimeType: The detected market regime

###### `reset()`

Reset the detector state.

## volatility_detectors

Volatility-based regime detectors.

### Classes

#### `VolatilityRegimeDetector`

Regime detector based on market volatility.

This detector identifies volatile and low-volatility markets based on
the standard deviation of returns over a specified lookback period.

##### Methods

###### `__init__(name=None, config=None)`

Initialize the volatility detector.

Args:
    name: Optional name for the detector
    config: Optional configuration dictionary with parameters:
        - lookback_period: Period for volatility calculation (default: 20)
        - volatility_threshold: Threshold for volatility regimes (default: 0.015)

###### `detect_regime(bar)`

Detect the current market regime based on volatility.

Args:
    bar: Bar data dictionary with 'Close' key
    
Returns:
    RegimeType: The detected market regime

*Returns:* RegimeType: The detected market regime

###### `reset()`

Reset the detector state.

## regime_manager

Regime manager for adapting trading strategies based on market regimes.

### Classes

#### `RegimeManager`

Manages trading strategies based on detected market regimes.

This class uses a regime detector to identify the current market regime
and selects the appropriate strategy accordingly.

##### Methods

###### `__init__(regime_detector, strategy_factory, rule_objects=None, data_handler=None)`

Initialize the regime manager.

Args:
    regime_detector: RegimeDetector object for identifying regimes
    strategy_factory: Factory for creating strategies
    rule_objects: List of trading rule objects (passed to the factory)
    data_handler: Optional data handler for optimization

###### `optimize_regime_strategies(regimes_to_optimize=None, optimization_metric='sharpe', verbose=True)`

Optimize strategies for different market regimes.

Args:
    regimes_to_optimize: List of regimes to optimize for (or None for all)
    optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
    verbose: Whether to print optimization progress
    
Returns:
    dict: Mapping from regime to optimized parameters

*Returns:* dict: Mapping from regime to optimized parameters

###### `_identify_regime_bars()`

Identify which bars belong to each regime.

Returns:
    dict: Mapping from regime to list of (index, bar) tuples

*Returns:* dict: Mapping from regime to list of (index, bar) tuples

###### `_create_regime_specific_data(regime_bars)`

Create a mock data handler with only bars from a specific regime.

Args:
    regime_bars: List of (index, bar) tuples for the regime
    
Returns:
    object: A data handler-like object for the specific regime

*Returns:* object: A data handler-like object for the specific regime

###### `get_strategy_for_regime(regime)`

Get the optimized strategy for a specific regime.

Args:
    regime: RegimeType to get strategy for
    
Returns:
    object: Strategy instance for the regime

*Returns:* object: Strategy instance for the regime

###### `on_bar(event)`

Process a bar and generate trading signals using the appropriate strategy.

Args:
    event: Bar event containing market data
    
Returns:
    object: Signal information from the strategy

*Returns:* object: Signal information from the strategy

###### `reset()`

Reset the regime manager and its components.

## regime_type

Enumeration of market regime types for regime detection.

### Classes

#### `RegimeType`

Enumeration of different market regime types.

##### Methods

###### `__str__()`

Return a readable string representation.

###### `from_string(cls, regime_name)`

Create a RegimeType from string name.

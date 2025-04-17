# Strategies Module

Strategies Module

This module provides a modular framework for creating and combining trading strategies.

## Contents

- [ensemble_strategy](#ensemble_strategy)
- [regime_strategy](#regime_strategy)
- [strategy_base](#strategy_base)
- [strategy_factory](#strategy_factory)
- [strategy_registry](#strategy_registry)
- [strategy_utils](#strategy_utils)
- [topn_strategy](#topn_strategy)
- [weighted_strategy](#weighted_strategy)

## ensemble_strategy

Ensemble Strategy Module

This module provides the EnsembleStrategy class that combines signals from multiple
strategies using configurable combination methods.

### Classes

#### `EnsembleStrategy`

Strategy that combines signals from multiple sub-strategies.

This strategy implements various methods for combining the signals from
multiple strategies, including voting, weighted averaging, and consensus.

##### Methods

###### `__init__(strategies, combination_method='voting', weights=None, name=None)`

Initialize the ensemble strategy.

Args:
    strategies: Dictionary mapping strategy names to strategy objects
    combination_method: Method to combine signals ('voting', 'weighted', or 'consensus')
    weights: Optional dictionary of weights for each strategy (for 'weighted' method)
    name: Strategy name

###### `on_bar(event)`

Process a bar and generate a signal by combining multiple strategies.

Args:
    event: Bar event containing market data
    
Returns:
    Signal: Combined signal

*Returns:* Signal: Combined signal

###### `reset()`

Reset all strategies in the ensemble.

## regime_strategy

Regime Strategy Module

This module provides the RegimeStrategy class that adapts to different market regimes
by selecting appropriate sub-strategies.

### Classes

#### `RegimeStrategy`

Strategy that adapts to different market regimes.

This strategy uses a regime detector to identify the current market regime
and then delegates to the appropriate strategy for that regime.

##### Methods

###### `__init__(regime_detector, regime_strategies, default_strategy=None, name=None)`

Initialize the regime strategy.

Args:
    regime_detector: Object that identifies market regimes
    regime_strategies: Dictionary mapping regime types to strategies
    default_strategy: Strategy to use when no regime-specific strategy is available
    name: Strategy name

###### `get_strategy_for_regime(regime)`

*Returns:* `Strategy`

Get the appropriate strategy for a specific regime.

Args:
    regime: Market regime
    
Returns:
    Strategy: The strategy for this regime, or default if none exists

###### `on_bar(event)`

Process a bar and delegate to the appropriate regime-specific strategy.

Args:
    event: Bar event containing market data
    
Returns:
    Signal: Trading signal from the regime-specific strategy

*Returns:* Signal: Trading signal from the regime-specific strategy

###### `reset()`

Reset the regime detector and all strategies.

## strategy_base

Strategy Base Module

This module provides the base class for all trading strategies in the system.
It standardizes the interface for processing signals and handling events.

### Classes

#### `Strategy`

Base class for all trading strategies.

This class provides the standard interface for strategies to receive
market data events and process signals in response. Strategies typically
evaluate signals from multiple rules or other sources to make trading decisions.

##### Methods

###### `__init__(name, event_bus=None)`

Initialize strategy.

Args:
    name: Strategy name
    event_bus: Optional event bus for emitting signals

###### `on_bar(event)`

*Returns:* `Optional[SignalEvent]`

Process a bar event and process signals.

This method is called when a new bar event is received.
It extracts bar data and delegates to process_signals.

Args:
    event: Bar event
    
Returns:
    SignalEvent if signal generated, None otherwise

###### `update_indicators(bar_data)`

*Returns:* `None`

Update indicators with new bar data.

This method should be implemented by subclasses to update
any technical indicators or state based on new bar data.

Args:
    bar_data: Dictionary containing bar data (OHLCV)

###### `process_signals(bar_event)`

*Returns:* `Optional[SignalEvent]`

Process signals based on market data.

This method must be implemented by subclasses to process
signals from rules or other sources and determine a trading decision.

Args:
    bar_event: BarEvent containing market data
    
Returns:
    SignalEvent if a signal is produced, None otherwise

###### `set_event_bus(event_bus)`

*Returns:* `None`

Set the event bus for emitting signals.

Args:
    event_bus: Event bus instance

###### `reset()`

*Returns:* `None`

Reset strategy state.

###### `get_state()`

*Returns:* `Dict[str, Any]`

Get the current strategy state.

Returns:
    Dictionary with strategy state

## strategy_factory

Strategy Factory Module

This module provides the StrategyFactory class that simplifies the creation
of strategy instances.

### Classes

#### `StrategyFactory`

Factory for creating strategy instances.

This class provides methods for creating various types of strategies
from configuration dictionaries or parameters.

##### Methods

###### `create_strategy(strategy_type, params=None)`

*Returns:* `Strategy`

Create a strategy instance.

Args:
    strategy_type: Name or class of the strategy
    params: Parameters for strategy initialization
    
Returns:
    Strategy: Initialized strategy instance

###### `create_weighted_strategy(rules, weights=None, buy_threshold=0.5, sell_threshold, name=None)`

*Returns:* `WeightedStrategy`

Create a weighted strategy.

Args:
    rules: List of rule objects
    weights: Optional weights for each rule
    buy_threshold: Threshold for buy signals
    sell_threshold: Threshold for sell signals
    name: Strategy name
    
Returns:
    WeightedStrategy: Initialized weighted strategy

###### `create_ensemble_strategy(strategies, combination_method='voting', weights=None, name=None)`

*Returns:* `EnsembleStrategy`

Create an ensemble strategy.

Args:
    strategies: Dictionary of strategies
    combination_method: Method for combining signals
    weights: Optional weights for each strategy
    name: Strategy name
    
Returns:
    EnsembleStrategy: Initialized ensemble strategy

###### `create_regime_strategy(regime_detector, regime_strategies, default_strategy=None, name=None)`

*Returns:* `RegimeStrategy`

Create a regime strategy.

Args:
    regime_detector: Regime detection object
    regime_strategies: Dictionary mapping regimes to strategies
    default_strategy: Default strategy when no regime-specific one exists
    name: Strategy name
    
Returns:
    RegimeStrategy: Initialized regime strategy

###### `create_topn_strategy(rule_objects, name=None)`

*Returns:* `TopNStrategy`

Create a TopN strategy.

Args:
    rule_objects: List of rule objects
    name: Strategy name
    
Returns:
    TopNStrategy: Initialized TopN strategy

###### `create_from_config(config)`

*Returns:* `Strategy`

Create a strategy from a configuration dictionary.

Args:
    config: Strategy configuration
    
Returns:
    Strategy: Initialized strategy
    
Example config:
{
    'type': 'WeightedStrategy',
    'params': {
        'rules': [rule1, rule2, rule3],
        'weights': [0.5, 0.3, 0.2],
        'buy_threshold': 0.4,
        'sell_threshold': -0.4
    }
}

## strategy_registry

Strategy Registry Module

This module provides a registry for strategies that allows dynamic registration
and discovery of strategy implementations.

### Classes

#### `StrategyRegistry`

Registry of available strategies.

This class provides a centralized registry where strategies can be registered
and retrieved by name. It supports decorator-based registration.

##### Methods

###### `register(cls, category='general')`

Decorator to register a strategy class.

Args:
    category: Category to place the strategy in
    
Returns:
    Decorator function

*Returns:* Decorator function

###### `get_strategy_class(cls, name)`

Get a strategy class by name.

Args:
    name: Name of the strategy class
    
Returns:
    Type[Strategy]: The strategy class
    
Raises:
    ValueError: If strategy name is not found

*Returns:* Type[Strategy]: The strategy class

###### `list_strategies(cls)`

*Returns:* `Dict[str, List[str]]`

List all registered strategies by category.

Returns:
    Dict[str, list]: Mapping of categories to strategy names

## strategy_utils

Strategy Utilities Module

This module provides utility functions for working with strategies and signals
in the trading system. It standardizes signal generation and strategy operations.

### Functions

#### `create_signal_event(signal_type, price, symbol='default', rule_id=None, confidence=1.0, metadata=None, timestamp=None)`

*Returns:* `SignalEvent`

Create a standardized SignalEvent.

Args:
    signal_type: Type of signal (BUY, SELL, NEUTRAL)
    price: Price at signal generation
    symbol: Instrument symbol
    rule_id: ID of the rule that generated the signal
    confidence: Signal confidence (0-1)
    metadata: Additional signal metadata
    timestamp: Signal timestamp
    
Returns:
    SignalEvent object

#### `extract_bar_data(event)`

*Returns:* `Dict[str, Any]`

Extract bar data from an event for strategy processing.

Args:
    event: Event object
    
Returns:
    Bar data dictionary

#### `get_indicator_value(indicators, name, default=None)`

*Returns:* `Any`

Safely get indicator value with fallback.

Args:
    indicators: Dictionary of indicators
    name: Indicator name to retrieve
    default: Default value if not found
    
Returns:
    Indicator value or default

#### `analyze_bar_pattern(bars, window=5)`

*Returns:* `Dict[str, Any]`

Analyze a pattern in a series of bars.

Args:
    bars: List of bar data dictionaries
    window: Analysis window size
    
Returns:
    Dictionary with pattern analysis results

#### `calculate_signal_confidence(indicators, trend_strength=0.5)`

*Returns:* `float`

Calculate confidence score for a signal based on indicators.

Args:
    indicators: Dictionary of indicator values
    trend_strength: Strength of the current trend (0-1)
    
Returns:
    Confidence score (0-1)

## topn_strategy

Top-N Strategy Module

This module provides the TopNStrategy class that combines signals from top N rules
using a voting mechanism. This version uses standardized SignalEvent objects.

### Classes

#### `TopNStrategy`

Strategy that combines signals from top N rules using consensus.

##### Methods

###### `__init__(rule_objects, name=None, event_bus=None)`

Initialize the TopN strategy.

Args:
    rule_objects: List of rule objects
    name: Strategy name
    event_bus: Optional event bus for emitting events

###### `process_signals(bar_event)`

*Returns:* `Optional[SignalEvent]`

Process a bar and generate a consensus signal.

Args:
    bar_event: BarEvent containing market data
    
Returns:
    SignalEvent if generated, None otherwise

###### `reset()`

Reset the strategy state.

## weighted_strategy

Weighted Strategy Module

This module provides the WeightedStrategy class that combines signals from multiple
components using configurable weights.

### Classes

#### `WeightedStrategy`

Strategy that combines signals from multiple components using weights.

This strategy takes a list of components (rules or other signal generators)
and combines their signals using configurable weights to generate a final 
trading signal.

##### Methods

###### `__init__(components, weights=None, buy_threshold=0.5, sell_threshold, name=None, event_bus=None)`

Initialize the weighted strategy.

Args:
    components: List of components that generate signals
    weights: List of weights for each component (default: equal weights)
    buy_threshold: Threshold above which to generate a buy signal
    sell_threshold: Threshold below which to generate a sell signal
    name: Strategy name
    event_bus: Optional event bus for emitting events

###### `process_signals(bar_event)`

*Returns:* `Optional[SignalEvent]`

Process signals from components and generate a weighted trading signal.

Args:
    bar_event: BarEvent containing market data

Returns:
    SignalEvent representing the weighted decision

###### `reset()`

Reset all components in the strategy.

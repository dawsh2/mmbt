# Rules Module

Rules Module

This module provides a framework for creating trading rules that generate signals
based on technical indicators.

## Contents

- [crossover_rules](#crossover_rules)
- [oscillator_rules](#oscillator_rules)
- [rule_base](#rule_base)
- [rule_factory](#rule_factory)
- [rule_registry](#rule_registry)
- [trend_rules](#trend_rules)
- [volatility_rules](#volatility_rules)

## crossover_rules

Crossover Rules Module

This module implements various crossover-based trading rules such as
moving average crossovers, price-MA crossovers, and other indicator crossovers.

### Classes

#### `SMACrossoverRule`

Simple Moving Average crossover rule.

Generates buy signals when the fast SMA crosses above the slow SMA,
and sell signals when it crosses below.

##### Methods

###### `__init__(name, params=None, description='', event_bus=None)`

Initialize SMA crossover rule.

Args:
    name: Rule name
    params: Rule parameters including:
        - fast_window: Window size for fast SMA (default: 10)
        - slow_window: Window size for slow SMA (default: 30)
    description: Rule description
    event_bus: Optional event bus for emitting signals

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Get default parameters for SMA crossover rule.

Returns:
    Dictionary of default parameter values

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

Raises:
    ValueError: If parameters are invalid

###### `generate_signal(bar_event)`

*Returns:* `Optional[SignalEvent]`

Generate a signal based on SMA crossover.

This method implements the SMA crossover strategy logic:
1. Update price history with the latest price
2. Calculate fast and slow SMAs if enough data
3. Check for crossover between SMAs
4. Generate appropriate signals on crossover

Args:
    bar_event: BarEvent containing market data
    
Returns:
    SignalEvent if crossover occurs, None otherwise

###### `_emit_signal(signal)`

Emit a signal event to the event bus if available.

Args:
    signal: SignalEvent to emit

###### `on_bar(event)`

*Returns:* `Optional[SignalEvent]`

Process a bar event and generate a trading signal.

This method overrides the base class to ensure signals are emitted.

Args:
    event: Event containing a BarEvent in its data attribute
    
Returns:
    SignalEvent if a signal is generated, None otherwise

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `ExponentialMACrossoverRule`

Exponential Moving Average (EMA) Crossover Rule.

This rule generates buy signals when a faster EMA crosses above a slower EMA,
and sell signals when the faster EMA crosses below the slower EMA.

##### Methods

###### `__init__(name='ema_crossover', params=None, description='EMA crossover rule')`

Initialize the EMA crossover rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - fast_period: Period for fast EMA (default: 12)
        - slow_period: Period for slow EMA (default: 26)
        - smooth_signals: Whether to generate signals when MAs are aligned (default: False)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on EMA crossover.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `MACDCrossoverRule`

Moving Average Convergence Divergence (MACD) Crossover Rule.

This rule generates buy signals when the MACD line crosses above the signal line,
and sell signals when the MACD line crosses below the signal line.

##### Methods

###### `__init__(name='macd_crossover', params=None, description='MACD crossover rule')`

Initialize the MACD crossover rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - fast_period: Period for fast EMA (default: 12)
        - slow_period: Period for slow EMA (default: 26)
        - signal_period: Period for signal line (default: 9)
        - use_histogram: Whether to use MACD histogram for signals (default: False)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on MACD crossover.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `PriceMACrossoverRule`

Price-Moving Average Crossover Rule.

This rule generates buy signals when the price crosses above a moving average,
and sell signals when the price crosses below.

##### Methods

###### `__init__(name='price_ma_crossover', params=None, description='Price-MA crossover rule')`

Initialize the Price-MA crossover rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - ma_period: Period for the moving average (default: 20)
        - ma_type: Type of moving average ('sma', 'ema') (default: 'sma')
        - smooth_signals: Whether to generate signals when price and MA are aligned (default: False)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on price-MA crossover.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `BollingerBandsCrossoverRule`

Bollinger Bands Crossover Rule.

This rule generates buy signals when price crosses below the lower band
and sell signals when price crosses above the upper band.

##### Methods

###### `__init__(name='bollinger_bands_crossover', params=None, description='Bollinger Bands crossover rule')`

Initialize the Bollinger Bands crossover rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - period: Period for the moving average (default: 20)
        - num_std_dev: Number of standard deviations for bands (default: 2.0)
        - use_middle_band: Whether to also generate signals on middle band crosses (default: False)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Bollinger Bands crossover.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `StochasticCrossoverRule`

Stochastic Oscillator Crossover Rule.

This rule generates signals based on %K crossing %D in the Stochastic Oscillator.

##### Methods

###### `__init__(name='stochastic_crossover', params=None, description='Stochastic crossover rule')`

Initialize the Stochastic crossover rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - k_period: Period for %K calculation (default: 14)
        - d_period: Period for %D calculation (default: 3)
        - slowing: Slowing period for %K (default: 3)
        - use_extremes: Whether to also generate signals on overbought/oversold levels (default: True)
        - overbought: Overbought level (default: 80)
        - oversold: Oversold level (default: 20)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Stochastic crossover.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

## oscillator_rules

Oscillator Rules Module

This module implements various oscillator-based trading rules such as
RSI, Stochastic, CCI, and other momentum oscillators.

### Classes

#### `RSIRule`

Relative Strength Index (RSI) Rule.

This rule generates signals based on RSI, an oscillator that measures the 
speed and change of price movements on a scale of 0 to 100.

##### Methods

###### `__init__(name='rsi_rule', params=None, description='RSI overbought/oversold rule')`

Initialize the RSI rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - rsi_period: Period for RSI calculation (default: 14)
        - overbought: Overbought level (default: 70)
        - oversold: Oversold level (default: 30)
        - signal_type: Signal generation method ('levels', 'divergence', 'midline') (default: 'levels')
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(bar_event)`

*Returns:* `Optional[SignalEvent]`

Generate a trading signal based on RSI.

Args:
    bar_event: BarEvent containing market data
    
Returns:
    SignalEvent if conditions are met, None otherwise

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `StochasticRule`

Stochastic Oscillator Rule.

This rule generates signals based on the Stochastic Oscillator, which measures
the current price relative to the price range over a period of time.

##### Methods

###### `__init__(name='stochastic_rule', params=None, description='Stochastic oscillator rule')`

Initialize the Stochastic Oscillator rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - k_period: %K period (default: 14)
        - k_slowing: %K slowing period (default: 3)
        - d_period: %D period (default: 3)
        - overbought: Overbought level (default: 80)
        - oversold: Oversold level (default: 20)
        - signal_type: Signal generation method ('levels', 'crossover', 'both') (default: 'both')
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Stochastic Oscillator.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `CCIRule`

Commodity Channel Index (CCI) Rule.

This rule uses the CCI oscillator to identify overbought and oversold 
conditions as well as trend strength and potential reversals.

##### Methods

###### `__init__(name='cci_rule', params=None, description='CCI oscillator rule')`

Initialize the CCI rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - period: CCI calculation period (default: 20)
        - overbought: Overbought level (default: 100)
        - oversold: Oversold level (default: -100)
        - extreme_overbought: Extreme overbought level (default: 200)
        - extreme_oversold: Extreme oversold level (default: -200)
        - zero_line_cross: Whether to use zero line crossovers (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on CCI.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `MACDHistogramRule`

MACD Histogram Rule.

This rule uses the MACD histogram to identify momentum shifts in a trend,
focusing on changes in the histogram rather than just MACD line crossovers.

##### Methods

###### `__init__(name='macd_histogram', params=None, description='MACD histogram rule')`

Initialize the MACD Histogram rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - fast_period: Fast EMA period (default: 12)
        - slow_period: Slow EMA period (default: 26)
        - signal_period: Signal line period (default: 9)
        - zero_line_cross: Whether to use zero line crossovers (default: True)
        - divergence: Whether to detect divergence (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on MACD histogram.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

## rule_base

Rule Base Module

This module defines the base Rule class and related abstractions for the rules layer
of the trading system. Rules analyze market data to generate trading signals.

### Classes

#### `Rule`

Base class for all trading rules in the system.

Rules transform market data into trading signals by applying decision logic.
Each rule encapsulates a specific trading strategy or signal generation logic.

##### Methods

###### `__init__(name, params=None, description='', event_bus=None)`

Initialize a rule.

Args:
    name: Unique identifier for this rule
    params: Dictionary of configuration parameters
    description: Human-readable description of the rule
    event_bus: Optional event bus for emitting signals

###### `_validate_params()`

*Returns:* `None`

Validate the parameters provided to the rule.

This method should be overridden by subclasses to provide
specific parameter validation logic.

Raises:
    ValueError: If parameters are invalid

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Get the default parameters for this rule.

Returns:
    Dictionary of default parameter values

###### `generate_signal(bar_event)`

*Returns:* `Optional[SignalEvent]`

Analyze market data and generate a trading signal.

This method is responsible for the signal generation process:
1. Analyzing the bar data 
2. Determining if a signal should be generated
3. Creating and returning the appropriate SignalEvent

Subclasses must implement this method with their specific trading logic.

Args:
    bar_event: BarEvent containing market data
         
Returns:
    SignalEvent if conditions warrant a signal, None otherwise

###### `on_bar(event)`

*Returns:* `Optional[SignalEvent]`

Process a bar event and generate a trading signal.

This method extracts the bar data from the event, passes it to
generate_signal() for signal generation, and emits any signals
to the event bus.

Args:
    event: Event containing a BarEvent in its data attribute
    
Returns:
    SignalEvent if a signal is generated, None otherwise

###### `set_event_bus(event_bus)`

Set the event bus for this rule.

Args:
    event_bus: Event bus for emitting signals

###### `update_state(key, value)`

*Returns:* `None`

Update the rule's internal state.

Args:
    key: State dictionary key
    value: Value to store

###### `get_state(key=None, default=None)`

*Returns:* `Any`

Get a value from the rule's state or the entire state dict.

Args:
    key: State dictionary key, or None to get the entire state
    default: Default value if key is not found
    
Returns:
    The value from state or default, or the entire state dict

###### `reset()`

*Returns:* `None`

Reset the rule's internal state and signal history.

This method should be called when reusing a rule instance
for a new backtest or trading session.

###### `__str__()`

*Returns:* `str`

String representation of the rule.

###### `__repr__()`

*Returns:* `str`

Detailed representation of the rule.

#### `CompositeRule`

A rule composed of multiple sub-rules.

CompositeRule combines signals from multiple rules using a specified
aggregation method to produce a final signal.

##### Methods

###### `__init__(name, rules, aggregation_method='majority', params=None, description='', event_bus=None)`

Initialize a composite rule.

Args:
    name: Unique identifier for the rule
    rules: List of component rules
    aggregation_method: Method to combine signals ('majority', 'unanimous', 'weighted')
    params: Dictionary of parameters
    description: Human-readable description
    event_bus: Optional event bus for emitting signals

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this composite rule.

###### `generate_signal(bar_event)`

*Returns:* `Optional[SignalEvent]`

Generate a composite signal by combining signals from sub-rules.

Args:
    bar_event: BarEvent containing market data
         
Returns:
    SignalEvent representing the combined decision, or None if no signal

###### `_majority_vote(signals, bar_event)`

*Returns:* `SignalEvent`

Combine signals using a majority vote.

Args:
    signals: List of signals from sub-rules
    bar_event: Original bar event
    
Returns:
    Combined signal

###### `reset()`

*Returns:* `None`

Reset this rule and all sub-rules.

#### `FeatureBasedRule`

A rule that generates signals based on features.

FeatureBasedRule uses a list of features to generate trading signals,
abstracting away the direct handling of price data and indicators.

##### Methods

###### `__init__(name, feature_names, params=None, description='', event_bus=None)`

Initialize a feature-based rule.

Args:
    name: Unique identifier for the rule
    feature_names: List of feature names this rule depends on
    params: Dictionary of parameters
    description: Human-readable description
    event_bus: Optional event bus for emitting signals

###### `generate_signal(bar_event)`

*Returns:* `Optional[SignalEvent]`

Generate a trading signal based on features.

Args:
    bar_event: BarEvent containing market data
         
Returns:
    SignalEvent representing the trading decision, or None if no signal

###### `make_decision(features, bar_event)`

*Returns:* `Optional[SignalEvent]`

Make a trading decision based on the features.

This method should be implemented by subclasses to define
the specific decision logic using features. It should directly
return a SignalEvent object when appropriate.

Args:
    features: Dictionary of feature values
    bar_event: Original bar event
         
Returns:
    SignalEvent representing the trading decision, or None if no signal

###### `_validate_param_application()`

Validate that parameters were correctly applied to this instance.
Called after initialization.

## rule_factory

Rule Factory Module

This module provides factory functions for creating and configuring rule instances
with proper parameter handling and validation.

### Functions

#### `create_rule(rule_name, params=None)`

*Returns:* `Rule`

Create a rule instance using the global registry.

Args:
    rule_name: Name of the rule class to instantiate
    params: Parameters for the rule
    **kwargs: Additional keyword arguments for the rule constructor
    
Returns:
    Instantiated rule

#### `create_composite_rule(name, rule_configs, aggregation_method='majority', params=None)`

*Returns:* `CompositeRule`

Create a composite rule from multiple rule configurations.

Args:
    name: Name for the composite rule
    rule_configs: List of rule configurations
    aggregation_method: Method to combine signals
    params: Parameters for the composite rule
    
Returns:
    CompositeRule instance

### Classes

#### `RuleFactory`

Factory for creating rule instances with proper parameter handling.

This class provides methods for creating rule instances with various
parameter combinations and validation.

##### Methods

###### `__init__(registry=None)`

Initialize the rule factory.

Args:
    registry: Optional rule registry to use (defaults to global registry)

###### `create_rule(rule_name, params=None, instance_name=None)`

*Returns:* `Rule`

Create a rule instance with the given parameters.

Args:
    rule_name: Name of the rule class to instantiate
    params: Parameters for the rule
    instance_name: Optional name for the rule instance
    **kwargs: Additional keyword arguments for the rule constructor
    
Returns:
    Instantiated rule
    
Raises:
    KeyError: If rule is not found in registry

###### `create_rule_variants(rule_name, param_grid, base_params=None, name_format='{rule}_{param}_{value}')`

*Returns:* `List[Rule]`

Create multiple rule instances with different parameter combinations.

Args:
    rule_name: Name of the rule class to instantiate
    param_grid: Dictionary mapping parameter names to lists of values
    base_params: Base parameters to use for all variants
    name_format: Format string for naming rule instances
    
Returns:
    List of rule instances with different parameter combinations

###### `create_composite_rule(name, rule_configs, aggregation_method='majority', params=None)`

*Returns:* `CompositeRule`

Create a composite rule from multiple rule configurations.

Args:
    name: Name for the composite rule
    rule_configs: List of rule configurations, which can be:
                  - Rule instance
                  - Rule name (string)
                  - Dict with 'name' and optional 'params' keys
    aggregation_method: Method to combine signals
    params: Parameters for the composite rule
    
Returns:
    CompositeRule instance
    
Raises:
    ValueError: If rule_configs is empty

###### `create_from_config(config)`

*Returns:* `Rule`

Create a rule from a configuration dictionary.

Args:
    config: Dictionary with rule configuration
    
Returns:
    Rule instance
    
Raises:
    ValueError: If config is invalid

#### `RuleOptimizer`

Optimizer for rule parameters based on performance metrics.

This class provides methods for optimizing rule parameters to
maximize performance metrics using different optimization methods.

##### Methods

###### `__init__(rule_factory, evaluation_func)`

Initialize the rule optimizer.

Args:
    rule_factory: RuleFactory instance for creating rules
    evaluation_func: Function that takes a rule and returns a performance score

###### `optimize_grid_search(rule_name, param_grid, base_params=None)`

*Returns:* `Tuple[Dict[str, Any], float]`

Optimize rule parameters using grid search.

Args:
    rule_name: Name of the rule to optimize
    param_grid: Dictionary mapping parameter names to lists of values
    base_params: Base parameters to use for all variants
    
Returns:
    Tuple of (best parameters, best score)

###### `optimize_random_search(rule_name, param_distributions, base_params=None, n_iterations=10)`

*Returns:* `Tuple[Dict[str, Any], float]`

Optimize rule parameters using random search.

Args:
    rule_name: Name of the rule to optimize
    param_distributions: Dictionary mapping parameter names to distributions
                       or lists of values
    base_params: Base parameters to use for all variants
    n_iterations: Number of random combinations to try
    
Returns:
    Tuple of (best parameters, best score)

## rule_registry

Rule Registry Module

This module provides a registry system for rules, allowing them to be
registered, discovered, and instantiated by name throughout the trading system.

### Functions

#### `register_rule(name=None, category='general')`

Decorator for registering rule classes with the registry.

Args:
    name: Optional name for the rule (defaults to class name)
    category: Category to group the rule under
    
Returns:
    Decorator function that registers the rule class

*Returns:* Decorator function that registers the rule class

#### `register_rules_in_module(module, category='general')`

*Returns:* `None`

Register all Rule classes in a module with the registry.

Args:
    module: The module object containing rule classes
    category: Category to group the rules under

#### `get_registry()`

*Returns:* `RuleRegistry`

Get the global rule registry instance.

Returns:
    The RuleRegistry singleton instance

### Classes

#### `RuleRegistry`

Registry for rule classes in the trading system.

The RuleRegistry maintains a mapping of rule names to their implementing
classes, allowing rules to be instantiated by name.

##### Methods

###### `__new__(cls)`

Implement singleton pattern for the registry.

###### `register(rule_class, name=None, category='general')`

*Returns:* `None`

Register a rule class with the registry.

Args:
    rule_class: The Rule class to register
    name: Optional name to register the rule under (defaults to class name)
    category: Category to group the rule under
    
Raises:
    ValueError: If a rule with the same name is already registered

###### `get_rule_class(name)`

*Returns:* `Type[Rule]`

Get a rule class by name.

Args:
    name: Name of the rule to retrieve
    
Returns:
    The rule class
    
Raises:
    KeyError: If no rule with the given name is registered

###### `create_rule(name, params=None, rule_name=None)`

*Returns:* `Rule`

Create an instance of a rule by name.

Args:
    name: Name of the rule class to instantiate
    params: Parameters to pass to the rule constructor
    rule_name: Optional name for the created rule instance
              (defaults to the registered name)
    **kwargs: Additional keyword arguments to pass to the constructor
    
Returns:
    An instance of the requested rule
    
Raises:
    KeyError: If no rule with the given name is registered

###### `list_rules(category=None)`

*Returns:* `List[str]`

List all registered rules, optionally filtered by category.

Args:
    category: Optional category to filter by
    
Returns:
    List of rule names

###### `list_categories()`

*Returns:* `List[str]`

List all rule categories.

Returns:
    List of category names

###### `clear()`

*Returns:* `None`

Clear all registered rules (useful for testing).

## trend_rules

Trend Rules Module

This module implements various trend-based trading rules such as
ADX rules, trend strength indicators, and directional movement.

### Classes

#### `ADXRule`

Average Directional Index (ADX) Rule.

This rule generates signals based on the ADX indicator and the directional movement indicators
(+DI and -DI) to identify strong trends and their direction.

##### Methods

###### `__init__(name='adx_rule', params=None, description='ADX trend strength rule')`

Initialize the ADX rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - adx_period: Period for ADX calculation (default: 14)
        - adx_threshold: Threshold to consider a trend strong (default: 25)
        - use_di_cross: Whether to use DI crossovers for signals (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on ADX and directional movement.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `IchimokuRule`

Ichimoku Cloud Rule.

This rule generates signals based on the Ichimoku Cloud indicator,
which provides multiple signals including trend direction, support/resistance,
and momentum.

##### Methods

###### `__init__(name='ichimoku_rule', params=None, description='Ichimoku Cloud rule')`

Initialize the Ichimoku Cloud rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - tenkan_period: Tenkan-sen period (default: 9)
        - kijun_period: Kijun-sen period (default: 26)
        - senkou_span_b_period: Senkou Span B period (default: 52)
        - signal_type: Signal generation method ('cloud', 'tk_cross', 'price_cross') (default: 'cloud')
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Ichimoku Cloud.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `VortexRule`

Vortex Indicator Rule.

This rule uses the Vortex Indicator (VI) to identify trend reversals based on
the relationship between the positive and negative VI lines.

##### Methods

###### `__init__(name='vortex_rule', params=None, description='Vortex indicator rule')`

Initialize the Vortex Indicator rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - period: Calculation period for VI+ and VI- (default: 14)
        - smooth_signals: Whether to generate signals when VIs are aligned (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on the Vortex Indicator.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

## volatility_rules

Volatility Rules Module

This module implements various volatility-based trading rules such as
Bollinger Bands, ATR strategies, and volatility breakout systems.

### Classes

#### `BollingerBandRule`

Bollinger Bands Breakout Rule.

This rule generates signals when price breaks out of Bollinger Bands,
indicating potential trend reversals or continuation based on volatility.

##### Methods

###### `__init__(name='bollinger_breakout', params=None, description='Bollinger Bands breakout rule')`

Initialize the Bollinger Bands Breakout rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - period: Period for SMA calculation (default: 20)
        - std_dev: Number of standard deviations for bands (default: 2.0)
        - breakout_type: Type of breakout to monitor ('upper', 'lower', 'both') (default: 'both')
        - use_confirmations: Whether to require confirmation (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Bollinger Bands breakout.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `ATRTrailingStopRule`

Average True Range (ATR) Trailing Stop Rule.

This rule uses ATR to set dynamic trailing stops that adapt to market volatility,
providing a strategy for managing exits.

##### Methods

###### `__init__(name='atr_trailing_stop', params=None, description='ATR trailing stop rule')`

Initialize the ATR Trailing Stop rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - atr_period: Period for ATR calculation (default: 14)
        - atr_multiplier: Multiplier for ATR to set stop distance (default: 3.0)
        - use_trend_filter: Whether to use trend filter for entries (default: True)
        - trend_ma_period: Period for trend moving average (default: 50)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on ATR trailing stop.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `VolatilityBreakoutRule`

Volatility Breakout Rule.

This rule generates signals based on price breaking out of a volatility-adjusted range,
which can identify the start of new trends after periods of consolidation.

##### Methods

###### `__init__(name='volatility_breakout', params=None, description='Volatility breakout rule')`

Initialize the Volatility Breakout rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - lookback_period: Period for calculating the range (default: 20)
        - volatility_measure: Method to measure volatility ('atr', 'stdev', 'range') (default: 'atr')
        - breakout_multiplier: Multiplier for breakout threshold (default: 1.5)
        - require_confirmation: Whether to require confirmation (default: True)
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on volatility breakout.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

#### `KeltnerChannelRule`

Keltner Channel Rule.

This rule uses Keltner Channels which are similar to Bollinger Bands but use ATR
for volatility measurement instead of standard deviation.

##### Methods

###### `__init__(name='keltner_channel', params=None, description='Keltner Channel volatility rule')`

Initialize the Keltner Channel rule.

Args:
    name: Rule name
    params: Dictionary containing:
        - ema_period: Period for EMA calculation (default: 20)
        - atr_period: Period for ATR calculation (default: 10)
        - multiplier: Multiplier for channels (default: 2.0)
        - signal_type: Signal generation method ('channel_cross', 'channel_touch') (default: 'channel_cross')
    description: Rule description

###### `default_params(cls)`

*Returns:* `Dict[str, Any]`

Default parameters for the rule.

###### `_validate_params()`

*Returns:* `None`

Validate the parameters for this rule.

###### `generate_signal(data)`

*Returns:* `Signal`

Generate a trading signal based on Keltner Channels.

Args:
    data: Dictionary containing price data
         
Returns:
    Signal object representing the trading decision

###### `reset()`

*Returns:* `None`

Reset the rule's internal state.

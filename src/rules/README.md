# Rules Module Documentation

The rules module provides a framework for creating trading rules that generate signals based on technical indicators.

## Core Concepts

### Rule
Each rule:
- Maintains internal state and history
- Processes bar data (OHLCV)
- Returns standardized Signal objects

### Signal
Each signal contains:
- Signal type (BUY, SELL, NEUTRAL)
- Confidence score (0.0-1.0)
- Price at generation
- Metadata with calculations

## Basic Usage

```python
from rules import create_rule

# Create a rule
sma_rule = create_rule('SMAcrossoverRule', {
    'fast_window': 10,
    'slow_window': 30
})

# Process a bar
bar_data = {
    'timestamp': datetime.now(),
    'Open': 100.0, 'High': 102.0, 
    'Low': 99.5, 'Close': 101.2
}

signal = sma_rule.on_bar(bar_data)

# Use the signal
if signal.signal_type == SignalType.BUY:
    print(f"Buy signal with {signal.confidence:.2f} confidence")
```

### Creating Composite Rules
```python
from rules import create_composite_rule

# Combine multiple rules into a composite rule
composite = create_composite_rule(
    name="my_composite_rule",
    rule_configs=[
        'SMAcrossoverRule',  # Use default parameters
        {'name': 'RSIRule', 'params': {'rsi_period': 14, 'overbought': 75}},
        existing_rule_instance  # Can also pass existing rule objects
    ],
    aggregation_method="majority"  # How to combine signals
)
```

## Rule Reference

The system includes the following rules with their parameters:

### Crossover Rules

#### SMAcrossoverRule
Simple Moving Average crossover rule.

**Parameters:**
- `fast_window`: Window size for fast SMA (default: 5)
- `slow_window`: Window size for slow SMA (default: 20)
- `smooth_signals`: Whether to generate signals when MAs are aligned (default: False)

```python
sma_rule = create_rule('SMAcrossoverRule', {
    'fast_window': 10,
    'slow_window': 30,
    'smooth_signals': True
})
```

#### ExponentialMACrossoverRule
Exponential Moving Average crossover rule.

**Parameters:**
- `fast_period`: Period for fast EMA (default: 12)
- `slow_period`: Period for slow EMA (default: 26)
- `smooth_signals`: Whether to generate signals when MAs are aligned (default: False)

```python
ema_rule = create_rule('ExponentialMACrossoverRule', {
    'fast_period': 12,
    'slow_period': 26,
    'smooth_signals': True
})
```

#### MACDCrossoverRule
MACD line crossing signal line rule.

**Parameters:**
- `fast_period`: Period for fast EMA (default: 12)
- `slow_period`: Period for slow EMA (default: 26)
- `signal_period`: Period for signal line (default: 9)
- `use_histogram`: Whether to use MACD histogram for signals (default: False)

```python
macd_rule = create_rule('MACDCrossoverRule', {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
    'use_histogram': True
})
```

#### PriceMACrossoverRule
Price crossing a moving average rule.

**Parameters:**
- `ma_period`: Period for the moving average (default: 20)
- `ma_type`: Type of moving average ('sma', 'ema') (default: 'sma')
- `smooth_signals`: Whether to generate signals when price and MA are aligned (default: False)

```python
price_ma_rule = create_rule('PriceMACrossoverRule', {
    'ma_period': 20,
    'ma_type': 'ema',
    'smooth_signals': True
})
```

#### BollingerBandsCrossoverRule
Price crossing Bollinger Bands rule.

**Parameters:**
- `period`: Period for the moving average (default: 20)
- `num_std_dev`: Number of standard deviations for bands (default: 2.0)
- `use_middle_band`: Whether to also generate signals on middle band crosses (default: False)

```python
bbands_rule = create_rule('BollingerBandsCrossoverRule', {
    'period': 20,
    'num_std_dev': 2.0,
    'use_middle_band': True
})
```

#### StochasticCrossoverRule
%K crossing %D in Stochastic Oscillator rule.

**Parameters:**
- `k_period`: Period for %K calculation (default: 14)
- `d_period`: Period for %D calculation (default: 3)
- `slowing`: Slowing period for %K (default: 3)
- `use_extremes`: Whether to also generate signals on overbought/oversold levels (default: True)
- `overbought`: Overbought level (default: 80)
- `oversold`: Oversold level (default: 20)

```python
stoch_cross_rule = create_rule('StochasticCrossoverRule', {
    'k_period': 14,
    'd_period': 3,
    'slowing': 3,
    'use_extremes': True,
    'overbought': 80,
    'oversold': 20
})
```

### Oscillator Rules

#### RSIRule
Relative Strength Index (RSI) rule.

**Parameters:**
- `rsi_period`: Period for RSI calculation (default: 14)
- `overbought`: Overbought level (default: 70)
- `oversold`: Oversold level (default: 30)
- `signal_type`: Signal generation method ('levels', 'divergence', 'midline') (default: 'levels')

```python
rsi_rule = create_rule('RSIRule', {
    'rsi_period': 14,
    'overbought': 70,
    'oversold': 30,
    'signal_type': 'levels'
})
```

#### StochasticRule
Stochastic Oscillator rule.

**Parameters:**
- `k_period`: %K period (default: 14)
- `k_slowing`: %K slowing period (default: 3)
- `j_period`: J-Line period (default: 3)
- `d_period`: %D period (default: 3)
- `overbought`: Overbought level (default: 80)
- `oversold`: Oversold level (default: 20)
- `signal_type`: Signal generation method ('levels', 'crossover', 'both') (default: 'both')

```python
stoch_rule = create_rule('StochasticRule', {
    'k_period': 14,
    'k_slowing': 3,
    'd_period': 3,
    'overbought': 80,
    'oversold': 20,
    'signal_type': 'both'
})
```

#### CCIRule
Commodity Channel Index (CCI) rule.

**Parameters:**
- `period`: CCI calculation period (default: 20)
- `overbought`: Overbought level (default: 100)
- `oversold`: Oversold level (default: -100)
- `extreme_overbought`: Extreme overbought level (default: 200)
- `extreme_oversold`: Extreme oversold level (default: -200)
- `zero_line_cross`: Whether to use zero line crossovers (default: True)

```python
cci_rule = create_rule('CCIRule', {
    'period': 20,
    'overbought': 100,
    'oversold': -100,
    'extreme_overbought': 200,
    'extreme_oversold': -200,
    'zero_line_cross': True
})
```

#### MACDHistogramRule
MACD histogram signal rule.

**Parameters:**
- `fast_period`: Fast EMA period (default: 12)
- `slow_period`: Slow EMA period (default: 26)
- `signal_period`: Signal line period (default: 9)
- `zero_line_cross`: Whether to use zero line crossovers (default: True)
- `divergence`: Whether to detect divergence (default: True)

```python
macd_hist_rule = create_rule('MACDHistogramRule', {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
    'zero_line_cross': True,
    'divergence': True
})
```

### Trend Rules

#### ADXRule
Average Directional Index (ADX) rule.

**Parameters:**
- `adx_period`: Period for ADX calculation (default: 14)
- `adx_threshold`: Threshold to consider a trend strong (default: 25)
- `use_di_cross`: Whether to use DI crossovers for signals (default: True)

```python
adx_rule = create_rule('ADXRule', {
    'adx_period': 14,
    'adx_threshold': 25,
    'use_di_cross': True
})
```

#### IchimokuRule
Ichimoku Cloud rule.

**Parameters:**
- `tenkan_period`: Tenkan-sen period (default: 9)
- `kijun_period`: Kijun-sen period (default: 26)
- `senkou_span_b_period`: Senkou Span B period (default: 52)
- `signal_type`: Signal generation method ('cloud', 'tk_cross', 'price_cross') (default: 'cloud')

```python
ichimoku_rule = create_rule('IchimokuRule', {
    'tenkan_period': 9,
    'kijun_period': 26,
    'senkou_span_b_period': 52,
    'signal_type': 'cloud'
})
```

#### VortexRule
Vortex Indicator rule.

**Parameters:**
- `period`: Calculation period for VI+ and VI- (default: 14)
- `smooth_signals`: Whether to generate signals when VIs are aligned (default: True)

```python
vortex_rule = create_rule('VortexRule', {
    'period': 14,
    'smooth_signals': True
})
```

### Volatility Rules

#### BollingerBandRule
Volatility-based signals using Bollinger Bands.

**Parameters:**
- `period`: Period for SMA calculation (default: 20)
- `std_dev`: Number of standard deviations for bands (default: 2.0)
- `signal_type`: Signal generation method ('band_touch', 'band_cross', 'squeeze') (default: 'band_cross')

```python
bb_rule = create_rule('BollingerBandRule', {
    'period': 20,
    'std_dev': 2.0,
    'signal_type': 'band_cross'
})
```

#### ATRTrailingStopRule
ATR-based trailing stop rule.

**Parameters:**
- `atr_period`: Period for ATR calculation (default: 14)
- `atr_multiplier`: Multiplier for ATR to set stop distance (default: 3.0)
- `use_trend_filter`: Whether to use trend filter for entries (default: True)
- `trend_ma_period`: Period for trend moving average (default: 50)

```python
atr_stop_rule = create_rule('ATRTrailingStopRule', {
    'atr_period': 14,
    'atr_multiplier': 3.0,
    'use_trend_filter': True,
    'trend_ma_period': 50
})
```

#### VolatilityBreakoutRule
Breakouts from volatility ranges rule.

**Parameters:**
- `lookback_period`: Period for calculating the range (default: 20)
- `volatility_measure`: Method to measure volatility ('atr', 'stdev', 'range') (default: 'atr')
- `breakout_multiplier`: Multiplier for breakout threshold (default: 1.5)
- `require_confirmation`: Whether to require confirmation (default: True)

```python
vol_breakout_rule = create_rule('VolatilityBreakoutRule', {
    'lookback_period': 20,
    'volatility_measure': 'atr',
    'breakout_multiplier': 1.5,
    'require_confirmation': True
})
```

#### KeltnerChannelRule
Keltner Channels for volatility rule.

**Parameters:**
- `ema_period`: Period for EMA calculation (default: 20)
- `atr_period`: Period for ATR calculation (default: 10)
- `multiplier`: Multiplier for channels (default: 2.0)
- `signal_type`: Signal generation method ('channel_cross', 'channel_touch') (default: 'channel_cross')

```python
keltner_rule = create_rule('KeltnerChannelRule', {
    'ema_period': 20,
    'atr_period': 10,
    'multiplier': 2.0,
    'signal_type': 'channel_cross'
})
```

## Rule Aggregation Methods

When creating composite rules, the following aggregation methods are available:

- **majority**: Signal type with the most votes wins
- **unanimous**: Signals only when all rules agree
- **weighted**: Weight each rule's signal (requires weights parameter)
- **any**: Signal when any rule gives a non-neutral signal
- **sequence**: Signal when rules trigger in sequence

## Rule Factory

The RuleFactory provides advanced methods for creating rules:

```python
from rules import RuleFactory

factory = RuleFactory()

# Create multiple variants of a rule with different parameters
sma_variants = factory.create_rule_variants(
    rule_name='SMAcrossoverRule',
    param_grid={
        'fast_window': [5, 10, 15],
        'slow_window': [20, 30, 50]
    }
)

# Create a rule from a configuration dictionary
rule = factory.create_from_config({
    'type': 'RSIRule',
    'params': {
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'name': 'my_rsi_rule'
})
```

## Rule Optimization

Rules can be optimized to find optimal parameters:

```python
from rules import RuleOptimizer

def evaluate_rule(rule):
    # Custom evaluation function
    # Returns a performance score
    return performance_score

optimizer = RuleOptimizer(
    rule_factory=RuleFactory(),
    evaluation_func=evaluate_rule
)

# Grid search optimization
best_params, best_score = optimizer.optimize_grid_search(
    rule_name='SMAcrossoverRule',
    param_grid={
        'fast_window': [5, 10, 15],
        'slow_window': [20, 30, 50]
    }
)

# Random search optimization
best_params, best_score = optimizer.optimize_random_search(
    rule_name='RSIRule',
    param_distributions={
        'rsi_period': [7, 14, 21],
        'overbought': lambda: np.random.randint(70, 85),
        'oversold': lambda: np.random.randint(15, 30)
    },
    n_iterations=20
)
```

## Advanced Usage

### Creating Custom Rules

You can create custom rules by subclassing the Rule base class:

```python
from rules import Rule, register_rule
from signals import Signal, SignalType

@register_rule(category="custom")
class MyCustomRule(Rule):
    @classmethod
    def default_params(cls):
        return {
            'parameter1': 10,
            'parameter2': 20
        }
    
    def _validate_params(self):
        if self.params['parameter1'] <= 0:
            raise ValueError("parameter1 must be positive")
    
    def generate_signal(self, data):
        # Custom signal generation logic
        # ...
        return Signal(
            timestamp=data.get('timestamp'),
            signal_type=SignalType.BUY,  # or SELL or NEUTRAL
            price=data.get('Close'),
            rule_id=self.name,
            confidence=0.7  # 0.0 to 1.0
        )
```

### Resetting Rule State

Rules maintain state between bars. Reset them when starting a new analysis:

```python
# Reset a rule's internal state
rule.reset()

# Reset all rules in a composite rule
composite_rule.reset()
```
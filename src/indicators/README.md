# Indicators Module Documentation

The Indicators module provides pure functions for calculating technical indicators from price data. Each function takes price data arrays as input and returns the calculated indicator values without maintaining any internal state.

## Core Concepts

**Pure Functions**: All indicators are implemented as stateless functions that produce the same output given the same input.  
**Numpy Vectorization**: Functions use numpy for efficient array operations.  
**Categories**: Indicators are organized by functional category (moving averages, oscillators, volatility, trend, volume).

## Directory Structure

```
src/indicators/
├── __init__.py               # Package exports
├── moving_averages.py        # Moving average indicator implementations
├── oscillators.py            # Oscillator indicator implementations
├── volatility.py             # Volatility indicator implementations
├── volume.py                 # Volume-based indicator implementations
├── trend.py                  # Trend indicator implementations
└── README.md                 # This documentation file
```

## Basic Usage

```python
import numpy as np
from src.indicators.moving_averages import simple_moving_average, exponential_moving_average
from src.indicators.oscillators import relative_strength_index
from src.indicators.volatility import bollinger_bands

# Sample price data
prices = np.array([100.0, 101.2, 99.8, 102.5, 103.1, 102.8, 103.5])

# Calculate Simple Moving Average
sma = simple_moving_average(prices, window=3)
print(f"SMA: {sma}")  # Output: [100.33 101.17 101.8 102.8]

# Calculate Exponential Moving Average
ema = exponential_moving_average(prices, span=3)
print(f"EMA: {ema}")

# Calculate RSI
rsi = relative_strength_index(prices, period=5)
print(f"RSI: {rsi}")

# Calculate Bollinger Bands
middle_band, upper_band, lower_band = bollinger_bands(prices, period=5, num_std_dev=2)
print(f"Upper Band: {upper_band}")
print(f"Middle Band: {middle_band}")
print(f"Lower Band: {lower_band}")
```

## Important Notes

- **Initial Values**: Most indicators will return zeros or NaN values for the initial periods (equal to the calculation period) since they require a minimum amount of data. Handle these initial values appropriately in your code.
- **Input Types**: All functions accept both numpy arrays and regular lists/tuples, but internally convert to numpy arrays for calculation.
- **Return Types**: All functions return numpy arrays (or tuples of numpy arrays).
- **Avoiding Division by Zero**: The implementations include protection against division by zero, but be aware that some edge cases might still produce NaN or infinite values.

## API Reference

### Moving Averages (src.indicators.moving_averages)

#### simple_moving_average(prices, window)

Calculate Simple Moving Average.

**Parameters:**
- `prices` (array-like): Array of price values
- `window` (int): Size of the moving window

**Returns:**
- `ndarray`: SMA values (valid only from index `window-1` onwards)

**Example:**
```python
sma = simple_moving_average(prices, window=10)
```

#### exponential_moving_average(prices, span)

Calculate Exponential Moving Average.

**Parameters:**
- `prices` (array-like): Array of price values
- `span` (int): Specified period for the EMA

**Returns:**
- `ndarray`: EMA values

**Example:**
```python
ema = exponential_moving_average(prices, span=12)
```

#### weighted_moving_average(prices, window)

Calculate Weighted Moving Average.

**Parameters:**
- `prices` (array-like): Array of price values
- `window` (int): Size of the moving window

**Returns:**
- `ndarray`: WMA values

#### double_exponential_moving_average(prices, span)

Calculate Double Exponential Moving Average.

**Parameters:**
- `prices` (array-like): Array of price values
- `span` (int): Specified period for the DEMA

**Returns:**
- `ndarray`: DEMA values

#### triple_exponential_moving_average(prices, span)

Calculate Triple Exponential Moving Average.

**Parameters:**
- `prices` (array-like): Array of price values
- `span` (int): Specified period for the TEMA

**Returns:**
- `ndarray`: TEMA values

#### hull_moving_average(prices, window)

Calculate Hull Moving Average, designed to reduce lag and improve smoothness.

**Parameters:**
- `prices` (array-like): Array of price values
- `window` (int): Size of the moving window

**Returns:**
- `ndarray`: HMA values

#### kaufman_adaptive_moving_average(prices, n=10, fast_ema=2, slow_ema=30)

Calculate Kaufman's Adaptive Moving Average (KAMA), which adjusts the smoothing based on market efficiency.

**Parameters:**
- `prices` (array-like): Array of price values
- `n` (int): Efficiency ratio lookback period
- `fast_ema` (int): Fast EMA period
- `slow_ema` (int): Slow EMA period

**Returns:**
- `ndarray`: KAMA values

#### variable_index_dynamic_average(prices, period=9, vi_period=6)

Calculate Variable Index Dynamic Average (VIDYA), an EMA that adjusts based on volatility.

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): VIDYA period
- `vi_period` (int): Volatility index period

**Returns:**
- `ndarray`: VIDYA values

### Oscillators (src.indicators.oscillators)

#### relative_strength_index(prices, period=14)

Calculate Relative Strength Index (RSI).

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): RSI calculation period

**Returns:**
- `ndarray`: RSI values (0-100)

**Example:**
```python
rsi = relative_strength_index(prices, period=14)
```

#### stochastic_oscillator(high_prices, low_prices, close_prices, k_period=14, d_period=3)

Calculate Stochastic Oscillator.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `k_period` (int): %K period
- `d_period` (int): %D period

**Returns:**
- `tuple`: (%K values, %D values)

**Example:**
```python
k_values, d_values = stochastic_oscillator(highs, lows, closes, k_period=14, d_period=3)
```

#### commodity_channel_index(high_prices, low_prices, close_prices, period=20, constant=0.015)

Calculate Commodity Channel Index (CCI).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): CCI period
- `constant` (float): CCI constant

**Returns:**
- `ndarray`: CCI values

#### williams_r(high_prices, low_prices, close_prices, period=14)

Calculate Williams %R.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): Lookback period

**Returns:**
- `ndarray`: Williams %R values

#### money_flow_index(high_prices, low_prices, close_prices, volume, period=14)

Calculate Money Flow Index (MFI).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values
- `period` (int): MFI period

**Returns:**
- `ndarray`: MFI values

#### macd(prices, fast_period=12, slow_period=26, signal_period=9)

Calculate Moving Average Convergence Divergence (MACD).

**Parameters:**
- `prices` (array-like): Array of price values
- `fast_period` (int): Fast EMA period
- `slow_period` (int): Slow EMA period
- `signal_period` (int): Signal line period

**Returns:**
- `tuple`: (MACD line, signal line, histogram)

**Example:**
```python
macd_line, signal_line, histogram = macd(prices, fast_period=12, slow_period=26, signal_period=9)
```

#### rate_of_change(prices, period=10)

Calculate Rate of Change (ROC).

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): ROC period

**Returns:**
- `ndarray`: ROC values

#### awesome_oscillator(high_prices, low_prices, fast_period=5, slow_period=34)

Calculate Awesome Oscillator.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `fast_period` (int): Fast SMA period
- `slow_period` (int): Slow SMA period

**Returns:**
- `ndarray`: Awesome Oscillator values

### Volatility Indicators (src.indicators.volatility)

#### bollinger_bands(prices, period=20, num_std_dev=2)

Calculate Bollinger Bands.

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): Moving average period
- `num_std_dev` (float): Number of standard deviations for bands

**Returns:**
- `tuple`: (middle band, upper band, lower band)

**Example:**
```python
middle, upper, lower = bollinger_bands(prices, period=20, num_std_dev=2)
```

#### average_true_range(high_prices, low_prices, close_prices, period=14)

Calculate Average True Range (ATR).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): ATR period

**Returns:**
- `ndarray`: ATR values

**Example:**
```python
atr = average_true_range(highs, lows, closes, period=14)
```

#### keltner_channels(high_prices, low_prices, close_prices, ema_period=20, atr_period=10, multiplier=2)

Calculate Keltner Channels.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `ema_period` (int): EMA period for middle line
- `atr_period` (int): ATR period
- `multiplier` (float): Multiplier for bands

**Returns:**
- `tuple`: (middle line, upper line, lower line)

#### donchian_channels(high_prices, low_prices, period=20)

Calculate Donchian Channels.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `period` (int): Lookback period

**Returns:**
- `tuple`: (upper band, middle band, lower band)

#### historical_volatility(close_prices, period=21, trading_periods=252)

Calculate Historical Volatility.

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `period` (int): Lookback period
- `trading_periods` (int): Number of trading periods in a year

**Returns:**
- `ndarray`: Historical volatility values (annualized)

#### volatility_index(close_prices, period=30)

Calculate a simple Volatility Index based on standard deviation.

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `period` (int): Lookback period

**Returns:**
- `ndarray`: Volatility index values as percentage

#### chaikin_volatility(high_prices, low_prices, ema_period=10, roc_period=10)

Calculate Chaikin Volatility.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `ema_period` (int): EMA period for smoothing
- `roc_period` (int): Rate of change period

**Returns:**
- `ndarray`: Chaikin Volatility values

#### standard_deviation(prices, period=20)

Calculate Rolling Standard Deviation.

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): Lookback period

**Returns:**
- `ndarray`: Standard deviation values

### Trend Indicators (src.indicators.trend)

#### average_directional_index(high_prices, low_prices, close_prices, period=14)

Calculate Average Directional Index (ADX).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): ADX period

**Returns:**
- `tuple`: (ADX, +DI, -DI)

**Example:**
```python
adx, plus_di, minus_di = average_directional_index(highs, lows, closes, period=14)
```

#### moving_average_convergence_divergence(prices, fast_period=12, slow_period=26, signal_period=9)

Calculate Moving Average Convergence Divergence (MACD) - alias for `macd()` in oscillators.

**Parameters:**
- `prices` (array-like): Array of price values
- `fast_period` (int): Fast EMA period
- `slow_period` (int): Slow EMA period
- `signal_period` (int): Signal line period

**Returns:**
- `tuple`: (MACD line, signal line, histogram)

#### parabolic_sar(high_prices, low_prices, af_start=0.02, af_step=0.02, af_max=0.2)

Calculate Parabolic SAR (Stop and Reverse).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `af_start` (float): Starting acceleration factor
- `af_step` (float): Acceleration factor step
- `af_max` (float): Maximum acceleration factor

**Returns:**
- `ndarray`: PSAR values

#### aroon(high_prices, low_prices, period=25)

Calculate Aroon indicators.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `period` (int): Lookback period

**Returns:**
- `tuple`: (Aroon Up, Aroon Down, Aroon Oscillator)

#### ichimoku_cloud(high_prices, low_prices, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26)

Calculate Ichimoku Cloud components.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `tenkan_period` (int): Tenkan-sen period
- `kijun_period` (int): Kijun-sen period
- `senkou_b_period` (int): Senkou Span B period
- `chikou_period` (int): Chikou Span shift

**Returns:**
- `tuple`: (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)

#### trix(prices, period=15)

Calculate the TRIX indicator (Triple Exponential Average).

**Parameters:**
- `prices` (array-like): Array of price values
- `period` (int): EMA period

**Returns:**
- `ndarray`: TRIX values

#### vortex_indicator(high_prices, low_prices, close_prices, period=14)

Calculate Vortex Indicator.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): Lookback period

**Returns:**
- `tuple`: (VI+, VI-)

#### supertrend(high_prices, low_prices, close_prices, period=10, multiplier=3.0)

Calculate SuperTrend indicator.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `period` (int): ATR period
- `multiplier` (float): ATR multiplier

**Returns:**
- `tuple`: (SuperTrend, Direction)

### Volume Indicators (src.indicators.volume)

#### on_balance_volume(close_prices, volume)

Calculate On-Balance Volume (OBV).

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values

**Returns:**
- `ndarray`: OBV values

**Example:**
```python
obv = on_balance_volume(closes, volume)
```

#### volume_price_trend(close_prices, volume, period=14)

Calculate Volume-Price Trend (VPT).

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values
- `period` (int): SMA period for signal line

**Returns:**
- `tuple`: (VPT, Signal Line)

#### accumulation_distribution(high_prices, low_prices, close_prices, volume)

Calculate Accumulation/Distribution Line.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values

**Returns:**
- `ndarray`: A/D Line values

#### chaikin_oscillator(high_prices, low_prices, close_prices, volume, fast_period=3, slow_period=10)

Calculate Chaikin Oscillator.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values
- `fast_period` (int): Fast EMA period
- `slow_period` (int): Slow EMA period

**Returns:**
- `ndarray`: Chaikin Oscillator values

#### money_flow_index(high_prices, low_prices, close_prices, volume, period=14)

Calculate Money Flow Index (MFI). Note: This is also available in the oscillators module.

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values
- `period` (int): MFI period

**Returns:**
- `ndarray`: MFI values

#### ease_of_movement(high_prices, low_prices, volume, period=14)

Calculate Ease of Movement (EOM).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `volume` (array-like): Array of volume values
- `period` (int): SMA period for smoothing

**Returns:**
- `ndarray`: EOM values

#### volume_weighted_average_price(high_prices, low_prices, close_prices, volume, period=14)

Calculate Volume Weighted Average Price (VWAP).

**Parameters:**
- `high_prices` (array-like): Array of high prices
- `low_prices` (array-like): Array of low prices
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values
- `period` (int): VWAP period

**Returns:**
- `ndarray`: VWAP values

#### negative_volume_index(close_prices, volume)

Calculate Negative Volume Index (NVI).

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values

**Returns:**
- `ndarray`: NVI values

#### positive_volume_index(close_prices, volume)

Calculate Positive Volume Index (PVI).

**Parameters:**
- `close_prices` (array-like): Array of close prices
- `volume` (array-like): Array of volume values

**Returns:**
- `ndarray`: PVI values

## Advanced Usage

### Combining Multiple Indicators

```python
import numpy as np
from src.indicators.moving_averages import simple_moving_average, exponential_moving_average
from src.indicators.oscillators import relative_strength_index
from src.indicators.volatility import bollinger_bands

def sma_rsi_bollinger_strategy(prices, highs, lows, volume):
    """Combined indicator strategy example."""
    # Calculate indicators
    sma_20 = simple_moving_average(prices, window=20)
    sma_50 = simple_moving_average(prices, window=50)
    
    rsi = relative_strength_index(prices, period=14)
    
    middle, upper, lower = bollinger_bands(prices, period=20, num_std_dev=2)
    
    # Generate signals (last valid data point only)
    # Check if price is above/below key moving averages
    trend_signal = 1 if prices[-1] > sma_50[-1] else -1 if prices[-1] < sma_50[-1] else 0
    
    # Check if RSI indicates overbought/oversold
    rsi_signal = -1 if rsi[-1] > 70 else 1 if rsi[-1] < 30 else 0
    
    # Check if price is near Bollinger Bands
    bb_signal = -1 if prices[-1] > upper[-1] else 1 if prices[-1] < lower[-1] else 0
    
    # Combine signals (simple approach for illustration)
    if trend_signal > 0 and rsi_signal >= 0:
        return "BUY"
    elif trend_signal < 0 and rsi_signal <= 0:
        return "SELL"
    else:
        return "NEUTRAL"
```

### Optimizing for Performance

For handling large datasets efficiently:

```python
import numpy as np
from src.indicators.moving_averages import simple_moving_average

# Pre-allocate arrays for better performance
prices = np.array([100.0, 101.2, 99.8, 102.5, 103.1, 102.8, 103.5] * 1000)

# Use NumPy vectorized operations when possible
returns = np.diff(prices) / prices[:-1]

# Calculate multiple indicator periods at once
sma_periods = [5, 10, 20, 50, 200]
sma_results = {period: simple_moving_average(prices, window=period) for period in sma_periods}
```

### Working with Pandas DataFrames

```python
import pandas as pd
import numpy as np
from src.indicators.oscillators import relative_strength_index
from src.indicators.volatility import bollinger_bands

# Convert DataFrame to numpy arrays for indicator functions
df = pd.read_csv('price_data.csv')
closes = df['Close'].values
highs = df['High'].values
lows = df['Low'].values
volumes = df['Volume'].values

# Calculate indicators
rsi = relative_strength_index(closes, period=14)
middle, upper, lower = bollinger_bands(closes, period=20, num_std_dev=2)

# Add results back to DataFrame
df['RSI'] = np.nan  # Initialize with NaN
df['BB_Middle'] = np.nan
df['BB_Upper'] = np.nan
df['BB_Lower'] = np.nan

# Fill in calculated values (accounting for lookback periods)
df.loc[13:, 'RSI'] = rsi[14:]  # RSI with period 14 starts giving values at index 14
df.loc[19:, 'BB_Middle'] = middle[20:]  # BB with period 20 starts giving values at index 20
df.loc[19:, 'BB_Upper'] = upper[20:]
df.loc[19:, 'BB_Lower'] = lower[20:]
```

## Best Practices

1. **Use NumPy arrays as input**: Convert data to NumPy arrays before passing to indicator functions for better performance

2. **Handle initial values**: Most indicators return fewer meaningful values than input (e.g., a 20-period SMA returns 20 fewer values); pad with NaN when adding back to time series

3. **Normalize inputs**: Some indicators expect specific input ranges; normalize if needed

4. **Prefer vectorized operations**: Use vector operations rather than loops when possible

5. **Consider window effects**: Be aware of lookback window effects when combining multiple indicators

6. **Watch for edge cases**: Handle zeros and edge cases appropriately, especially for ratio-based indicators

7. **Ensure consistent array lengths**: When using multiple indicators, ensure you're comparing values at the same index position, accounting for different lookback periods

8. **Test with small datasets first**: Verify calculation correctness with small datasets before scaling to larger ones

9. **Use np.nan for missing data**: Rather than zeros, use np.nan to represent missing or invalid data points

10. **Document indicator combinations**: When creating composite indicators or strategies, document the rationale and interpretation

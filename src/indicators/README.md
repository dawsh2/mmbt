# Indicators Module

No module overview available.

## Contents

- [moving_averages](#moving_averages)
- [oscillators](#oscillators)
- [trend](#trend)
- [volatility](#volatility)
- [volume](#volume)

## moving_averages

Moving average indicators.

This module provides functions for calculating various types of moving averages.
These functions are pure, stateless computations on price data.

### Functions

#### `simple_moving_average(prices, window)`

Calculate Simple Moving Average.

Args:
    prices: Array of price values
    window: Size of the moving window
    
Returns:
    numpy.ndarray: SMA values

*Returns:* numpy.ndarray: SMA values

#### `weighted_moving_average(prices, window)`

Calculate Weighted Moving Average.

Args:
    prices: Array of price values
    window: Size of the moving window
    
Returns:
    numpy.ndarray: WMA values

*Returns:* numpy.ndarray: WMA values

#### `exponential_moving_average(prices, span)`

Calculate Exponential Moving Average.

Args:
    prices: Array of price values
    span: Specified period for the EMA
    
Returns:
    numpy.ndarray: EMA values

*Returns:* numpy.ndarray: EMA values

#### `double_exponential_moving_average(prices, span)`

Calculate Double Exponential Moving Average.

Args:
    prices: Array of price values
    span: Specified period for the DEMA
    
Returns:
    numpy.ndarray: DEMA values

*Returns:* numpy.ndarray: DEMA values

#### `triple_exponential_moving_average(prices, span)`

Calculate Triple Exponential Moving Average.

Args:
    prices: Array of price values
    span: Specified period for the TEMA
    
Returns:
    numpy.ndarray: TEMA values

*Returns:* numpy.ndarray: TEMA values

#### `hull_moving_average(prices, window)`

Calculate Hull Moving Average.

The Hull Moving Average (HMA) is designed to reduce lag and improve smoothness.

Args:
    prices: Array of price values
    window: Size of the moving window
    
Returns:
    numpy.ndarray: HMA values

*Returns:* numpy.ndarray: HMA values

#### `kaufman_adaptive_moving_average(prices, n=10, fast_ema=2, slow_ema=30)`

Calculate Kaufman's Adaptive Moving Average (KAMA).

KAMA adjusts the smoothing based on market efficiency.

Args:
    prices: Array or Series of price values
    n: Efficiency ratio lookback period
    fast_ema: Fast EMA period (typically 2)
    slow_ema: Slow EMA period (typically 30)
    
Returns:
    numpy.ndarray: KAMA values

*Returns:* numpy.ndarray: KAMA values

#### `variable_index_dynamic_average(prices, period=9, vi_period=6)`

Calculate Variable Index Dynamic Average (VIDYA).

VIDYA is an EMA that adjusts based on volatility.

Args:
    prices: Array or Series of price values
    period: VIDYA period
    vi_period: Volatility index period
    
Returns:
    numpy.ndarray: VIDYA values

*Returns:* numpy.ndarray: VIDYA values

## oscillators

Oscillator indicators.

This module provides functions for calculating various oscillator indicators.
These functions are pure, stateless computations on price data.

### Functions

#### `relative_strength_index(prices, period=14)`

Calculate Relative Strength Index (RSI).

Args:
    prices: Array of price values
    period: RSI calculation period (default 14)
    
Returns:
    numpy.ndarray: RSI values

*Returns:* numpy.ndarray: RSI values

#### `stochastic_oscillator(high_prices, low_prices, close_prices, k_period=14, d_period=3)`

Calculate Stochastic Oscillator.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    k_period: %K period (default 14)
    d_period: %D period (default 3)
    
Returns:
    tuple: (%K values, %D values)

*Returns:* tuple: (%K values, %D values)

#### `commodity_channel_index(high_prices, low_prices, close_prices, period=20, constant=0.015)`

Calculate Commodity Channel Index (CCI).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: CCI period (default 20)
    constant: CCI constant (default 0.015)
    
Returns:
    numpy.ndarray: CCI values

*Returns:* numpy.ndarray: CCI values

#### `williams_r(high_prices, low_prices, close_prices, period=14)`

Calculate Williams %R.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: Lookback period (default 14)
    
Returns:
    numpy.ndarray: Williams %R values

*Returns:* numpy.ndarray: Williams %R values

#### `money_flow_index(high_prices, low_prices, close_prices, volume, period=14)`

Calculate Money Flow Index (MFI).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    volume: Array of volume values
    period: MFI period (default 14)
    
Returns:
    numpy.ndarray: MFI values

*Returns:* numpy.ndarray: MFI values

#### `macd(prices, fast_period=12, slow_period=26, signal_period=9)`

Calculate Moving Average Convergence Divergence (MACD).

Args:
    prices: Array of price values
    fast_period: Fast EMA period (default 12)
    slow_period: Slow EMA period (default 26)
    signal_period: Signal line period (default 9)
    
Returns:
    tuple: (MACD line, signal line, histogram)

*Returns:* tuple: (MACD line, signal line, histogram)

#### `rate_of_change(prices, period=10)`

Calculate Rate of Change (ROC).

Args:
    prices: Array of price values
    period: ROC period (default 10)
    
Returns:
    numpy.ndarray: ROC values

*Returns:* numpy.ndarray: ROC values

#### `awesome_oscillator(high_prices, low_prices, fast_period=5, slow_period=34)`

Calculate Awesome Oscillator.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    fast_period: Fast SMA period (default 5)
    slow_period: Slow SMA period (default 34)
    
Returns:
    numpy.ndarray: Awesome Oscillator values

*Returns:* numpy.ndarray: Awesome Oscillator values

## trend

Trend indicators.

This module provides functions for calculating trend-based indicators.
These functions are pure, stateless computations on price data.

### Functions

#### `average_directional_index(high_prices, low_prices, close_prices, period=14)`

Calculate Average Directional Index (ADX).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: ADX period (default 14)
    
Returns:
    tuple: (ADX, +DI, -DI)

*Returns:* tuple: (ADX, +DI, -DI)

#### `moving_average_convergence_divergence(prices, fast_period=12, slow_period=26, signal_period=9)`

Calculate Moving Average Convergence Divergence (MACD).

Args:
    prices: Array of price values
    fast_period: Fast EMA period (default 12)
    slow_period: Slow EMA period (default 26)
    signal_period: Signal line period (default 9)
    
Returns:
    tuple: (MACD line, signal line, histogram)

*Returns:* tuple: (MACD line, signal line, histogram)

#### `parabolic_sar(high_prices, low_prices, af_start=0.02, af_step=0.02, af_max=0.2)`

Calculate Parabolic SAR (Stop and Reverse).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    af_start: Starting acceleration factor (default 0.02)
    af_step: Acceleration factor step (default 0.02)
    af_max: Maximum acceleration factor (default 0.2)
    
Returns:
    numpy.ndarray: PSAR values

*Returns:* numpy.ndarray: PSAR values

#### `aroon(high_prices, low_prices, period=25)`

Calculate Aroon indicators.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    period: Lookback period (default 25)
    
Returns:
    tuple: (Aroon Up, Aroon Down, Aroon Oscillator)

*Returns:* tuple: (Aroon Up, Aroon Down, Aroon Oscillator)

#### `ichimoku_cloud(high_prices, low_prices, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26)`

Calculate Ichimoku Cloud components.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    tenkan_period: Tenkan-sen period (default 9)
    kijun_period: Kijun-sen period (default 26)
    senkou_b_period: Senkou Span B period (default 52)
    chikou_period: Chikou Span shift (default 26)
    
Returns:
    tuple: (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)

*Returns:* tuple: (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)

#### `trix(prices, period=15)`

Calculate the TRIX indicator (Triple Exponential Average).

Args:
    prices: Array of price values
    period: EMA period (default 15)
    
Returns:
    numpy.ndarray: TRIX values

*Returns:* numpy.ndarray: TRIX values

#### `vortex_indicator(high_prices, low_prices, close_prices, period=14)`

Calculate Vortex Indicator.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: Lookback period (default 14)
    
Returns:
    tuple: (VI+, VI-)

*Returns:* tuple: (VI+, VI-)

#### `supertrend(high_prices, low_prices, close_prices, period=10, multiplier=3.0)`

Calculate SuperTrend indicator.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: ATR period (default 10)
    multiplier: ATR multiplier (default 3.0)
    
Returns:
    tuple: (SuperTrend, Direction)

*Returns:* tuple: (SuperTrend, Direction)

## volatility

Volatility indicators.

This module provides functions for calculating volatility-based indicators.
These functions are pure, stateless computations on price data.

### Functions

#### `average_true_range(high_prices, low_prices, close_prices, period=14)`

Calculate Average True Range (ATR).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    period: ATR period (default 14)
    
Returns:
    numpy.ndarray: ATR values

*Returns:* numpy.ndarray: ATR values

#### `bollinger_bands(prices, period=20, num_std_dev=2)`

Calculate Bollinger Bands.

Args:
    prices: Array of price values
    period: Moving average period (default 20)
    num_std_dev: Number of standard deviations for bands (default 2)
    
Returns:
    tuple: (middle band, upper band, lower band)

*Returns:* tuple: (middle band, upper band, lower band)

#### `keltner_channels(high_prices, low_prices, close_prices, ema_period=20, atr_period=10, multiplier=2)`

Calculate Keltner Channels.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    ema_period: EMA period for middle line (default 20)
    atr_period: ATR period (default 10)
    multiplier: Multiplier for ATR (default 2)
    
Returns:
    tuple: (middle line, upper line, lower line)

*Returns:* tuple: (middle line, upper line, lower line)

#### `donchian_channels(high_prices, low_prices, period=20)`

Calculate Donchian Channels.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    period: Lookback period (default 20)
    
Returns:
    tuple: (upper band, middle band, lower band)

*Returns:* tuple: (upper band, middle band, lower band)

#### `volatility_index(close_prices, period=30)`

Calculate a simple Volatility Index based on standard deviation.

Args:
    close_prices: Array of close prices
    period: Lookback period (default 30)
    
Returns:
    numpy.ndarray: Volatility index values as percentage

*Returns:* numpy.ndarray: Volatility index values as percentage

#### `chaikin_volatility(high_prices, low_prices, ema_period=10, roc_period=10)`

Calculate Chaikin Volatility.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    ema_period: EMA period for smoothing (default 10)
    roc_period: Rate of change period (default 10)
    
Returns:
    numpy.ndarray: Chaikin Volatility values

*Returns:* numpy.ndarray: Chaikin Volatility values

#### `historical_volatility(close_prices, period=21, trading_periods=252)`

Calculate Historical Volatility.

Args:
    close_prices: Array of close prices
    period: Lookback period (default 21)
    trading_periods: Number of trading periods in a year (default 252)
    
Returns:
    numpy.ndarray: Historical volatility values (annualized)

*Returns:* numpy.ndarray: Historical volatility values (annualized)

#### `standard_deviation(prices, period=20)`

Calculate Rolling Standard Deviation.

Args:
    prices: Array of price values
    period: Lookback period (default 20)
    
Returns:
    numpy.ndarray: Standard deviation values

*Returns:* numpy.ndarray: Standard deviation values

## volume

Volume indicators.

This module provides functions for calculating volume-based indicators.
These functions are pure, stateless computations on price and volume data.

### Functions

#### `on_balance_volume(close_prices, volume)`

Calculate On-Balance Volume (OBV).

Args:
    close_prices: Array of close prices
    volume: Array of volume values
    
Returns:
    numpy.ndarray: OBV values

*Returns:* numpy.ndarray: OBV values

#### `volume_price_trend(close_prices, volume, period=14)`

Calculate Volume-Price Trend (VPT).

Args:
    close_prices: Array of close prices
    volume: Array of volume values
    period: SMA period for signal line (default 14)
    
Returns:
    tuple: (VPT, Signal Line)

*Returns:* tuple: (VPT, Signal Line)

#### `accumulation_distribution(high_prices, low_prices, close_prices, volume)`

Calculate Accumulation/Distribution Line.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    volume: Array of volume values
    
Returns:
    numpy.ndarray: A/D Line values

*Returns:* numpy.ndarray: A/D Line values

#### `chaikin_oscillator(high_prices, low_prices, close_prices, volume, fast_period=3, slow_period=10)`

Calculate Chaikin Oscillator.

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    volume: Array of volume values
    fast_period: Fast EMA period (default 3)
    slow_period: Slow EMA period (default 10)
    
Returns:
    numpy.ndarray: Chaikin Oscillator values

*Returns:* numpy.ndarray: Chaikin Oscillator values

#### `money_flow_index(high_prices, low_prices, close_prices, volume, period=14)`

Calculate Money Flow Index (MFI).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    volume: Array of volume values
    period: MFI period (default 14)
    
Returns:
    numpy.ndarray: MFI values

*Returns:* numpy.ndarray: MFI values

#### `ease_of_movement(high_prices, low_prices, volume, period=14)`

Calculate Ease of Movement (EOM).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    volume: Array of volume values
    period: SMA period for smoothing (default 14)
    
Returns:
    numpy.ndarray: EOM values

*Returns:* numpy.ndarray: EOM values

#### `volume_weighted_average_price(high_prices, low_prices, close_prices, volume, period=14)`

Calculate Volume Weighted Average Price (VWAP).

Args:
    high_prices: Array of high prices
    low_prices: Array of low prices
    close_prices: Array of close prices
    volume: Array of volume values
    period: VWAP period (default 14)
    
Returns:
    numpy.ndarray: VWAP values

*Returns:* numpy.ndarray: VWAP values

#### `negative_volume_index(close_prices, volume)`

Calculate Negative Volume Index (NVI).

Args:
    close_prices: Array of close prices
    volume: Array of volume values
    
Returns:
    numpy.ndarray: NVI values

*Returns:* numpy.ndarray: NVI values

#### `positive_volume_index(close_prices, volume)`

Calculate Positive Volume Index (PVI).

Args:
    close_prices: Array of close prices
    volume: Array of volume values
    
Returns:
    numpy.ndarray: PVI values

*Returns:* numpy.ndarray: PVI values

# src/indicators/oscillators.py
"""
Oscillator indicators.

This module provides functions for calculating various oscillator indicators.
These functions are pure, stateless computations on price data.
"""

import numpy as np
import pandas as pd

def relative_strength_index(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Array of price values
        period: RSI calculation period (default 14)
        
    Returns:
        numpy.ndarray: RSI values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Initialize arrays for gains and losses
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    # Separate gains and losses
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]  # Convert to positive values
    
    # Initialize averages with simple moving averages for first period
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Calculate first averages
    if len(gains) >= period:
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
    
    # Calculate subsequent EMA-based values
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.zeros_like(prices)
    rsi = np.zeros_like(prices)
    
    # Avoid division by zero
    valid_indices = (avg_loss > 0) & (avg_gain > 0)
    rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]
    rsi[valid_indices] = 100 - (100 / (1 + rs[valid_indices]))
    
    # Handle edge case where avg_loss is zero
    zero_loss_indices = (avg_loss == 0) & (avg_gain > 0)
    rsi[zero_loss_indices] = 100
    
    return rsi

def stochastic_oscillator(high_prices, low_prices, close_prices, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)
        
    Returns:
        tuple: (%K values, %D values)
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Initialize arrays
    k_values = np.zeros_like(close_prices)
    
    # Calculate %K
    for i in range(k_period - 1, len(close_prices)):
        highest_high = np.max(high_prices[i-k_period+1:i+1])
        lowest_low = np.min(low_prices[i-k_period+1:i+1])
        
        # Avoid division by zero
        if highest_high != lowest_low:
            k_values[i] = ((close_prices[i] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Calculate %D (simple moving average of %K)
    d_values = np.zeros_like(close_prices)
    for i in range(k_period + d_period - 2, len(close_prices)):
        d_values[i] = np.mean(k_values[i-d_period+1:i+1])
    
    return k_values, d_values

def commodity_channel_index(high_prices, low_prices, close_prices, period=20, constant=0.015):
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: CCI period (default 20)
        constant: CCI constant (default 0.015)
        
    Returns:
        numpy.ndarray: CCI values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate typical price
    tp = (high_prices + low_prices + close_prices) / 3
    
    # Initialize arrays
    sma_tp = np.zeros_like(tp)
    mad = np.zeros_like(tp)  # Mean Absolute Deviation
    cci = np.zeros_like(tp)
    
    # Calculate SMA and MAD for each point
    for i in range(period - 1, len(tp)):
        window = tp[i-period+1:i+1]
        sma_tp[i] = np.mean(window)
        mad[i] = np.mean(np.abs(window - sma_tp[i]))
    
    # Calculate CCI
    for i in range(period - 1, len(tp)):
        if mad[i] > 0:  # Avoid division by zero
            cci[i] = (tp[i] - sma_tp[i]) / (constant * mad[i])
    
    return cci

def williams_r(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Williams %R.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: Lookback period (default 14)
        
    Returns:
        numpy.ndarray: Williams %R values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Initialize Williams %R array
    williams_r_values = np.zeros_like(close_prices)
    
    # Calculate Williams %R
    for i in range(period - 1, len(close_prices)):
        highest_high = np.max(high_prices[i-period+1:i+1])
        lowest_low = np.min(low_prices[i-period+1:i+1])
        
        # Avoid division by zero
        if highest_high != lowest_low:
            williams_r_values[i] = -100 * ((highest_high - close_prices[i]) / (highest_high - lowest_low))
    
    return williams_r_values

def money_flow_index(high_prices, low_prices, close_prices, volume, period=14):
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        volume: Array of volume values
        period: MFI period (default 14)
        
    Returns:
        numpy.ndarray: MFI values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Calculate typical price
    tp = (high_prices + low_prices + close_prices) / 3
    
    # Calculate raw money flow
    money_flow = tp * volume
    
    # Calculate positive and negative money flow
    pos_flow = np.zeros_like(tp)
    neg_flow = np.zeros_like(tp)
    
    # First element has no price change
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:
            pos_flow[i] = money_flow[i]
        elif tp[i] < tp[i-1]:
            neg_flow[i] = money_flow[i]
    
    # Calculate MFI
    mfi = np.zeros_like(tp)
    
    for i in range(period, len(tp)):
        pos_sum = np.sum(pos_flow[i-period+1:i+1])
        neg_sum = np.sum(neg_flow[i-period+1:i+1])
        
        if neg_sum > 0:  # Avoid division by zero
            money_ratio = pos_sum / neg_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
        elif pos_sum > 0:  # All positive, no negative
            mfi[i] = 100
        else:  # Both sums are zero
            mfi[i] = 50  # Neutral
    
    return mfi

def macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Array of price values
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        
    Returns:
        tuple: (MACD line, signal line, histogram)
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate EMAs
    # Use pandas for this calculation for better accuracy
    prices_series = pd.Series(prices)
    fast_ema = prices_series.ewm(span=fast_period, adjust=False).mean().values
    slow_ema = prices_series.ewm(span=slow_period, adjust=False).mean().values
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD line)
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal_period, adjust=False).mean().values
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def rate_of_change(prices, period=10):
    """
    Calculate Rate of Change (ROC).
    
    Args:
        prices: Array of price values
        period: ROC period (default 10)
        
    Returns:
        numpy.ndarray: ROC values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Initialize ROC array
    roc = np.zeros_like(prices)
    
    # Calculate ROC
    for i in range(period, len(prices)):
        if prices[i-period] != 0:  # Avoid division by zero
            roc[i] = ((prices[i] - prices[i-period]) / prices[i-period]) * 100
    
    return roc

def awesome_oscillator(high_prices, low_prices, fast_period=5, slow_period=34):
    """
    Calculate Awesome Oscillator.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        fast_period: Fast SMA period (default 5)
        slow_period: Slow SMA period (default 34)
        
    Returns:
        numpy.ndarray: Awesome Oscillator values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Calculate median price
    median_price = (high_prices + low_prices) / 2
    
    # Calculate SMAs
    fast_sma = np.zeros_like(median_price)
    slow_sma = np.zeros_like(median_price)
    
    for i in range(fast_period - 1, len(median_price)):
        fast_sma[i] = np.mean(median_price[i-fast_period+1:i+1])
    
    for i in range(slow_period - 1, len(median_price)):
        slow_sma[i] = np.mean(median_price[i-slow_period+1:i+1])
    
    # Calculate Awesome Oscillator
    ao = fast_sma - slow_sma
    
    return ao

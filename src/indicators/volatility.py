# src/indicators/volatility.py
"""
Volatility indicators.

This module provides functions for calculating volatility-based indicators.
These functions are pure, stateless computations on price data.
"""

import numpy as np
import pandas as pd

def average_true_range(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: ATR period (default 14)
        
    Returns:
        numpy.ndarray: ATR values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate true range
    true_range = np.zeros_like(close_prices)
    
    # For the first bar, TR is simply High - Low
    if len(true_range) > 0:
        true_range[0] = high_prices[0] - low_prices[0]
    
    # For subsequent bars, TR is the maximum of:
    # 1. Current High - Current Low
    # 2. |Current High - Previous Close|
    # 3. |Current Low - Previous Close|
    for i in range(1, len(close_prices)):
        tr1 = high_prices[i] - low_prices[i]
        tr2 = abs(high_prices[i] - close_prices[i-1])
        tr3 = abs(low_prices[i] - close_prices[i-1])
        true_range[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using Wilder's smoothing method
    atr = np.zeros_like(close_prices)
    
    # Initialize first ATR value with simple average
    if len(true_range) >= period:
        atr[period-1] = np.mean(true_range[:period])
    
    # Calculate remaining ATR values
    for i in range(period, len(close_prices)):
        atr[i] = ((period - 1) * atr[i-1] + true_range[i]) / period
    
    return atr

def bollinger_bands(prices, period=20, num_std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Array of price values
        period: Moving average period (default 20)
        num_std_dev: Number of standard deviations for bands (default 2)
        
    Returns:
        tuple: (middle band, upper band, lower band)
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate middle band (SMA)
    middle_band = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        middle_band[i] = np.mean(prices[i-period+1:i+1])
    
    # Calculate standard deviation
    std_dev = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        std_dev[i] = np.std(prices[i-period+1:i+1], ddof=1)  # Using sample standard deviation
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def keltner_channels(high_prices, low_prices, close_prices, ema_period=20, atr_period=10, multiplier=2):
    """
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
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate middle line (EMA of close prices)
    # Use pandas for this calculation for better accuracy
    close_series = pd.Series(close_prices)
    middle_line = close_series.ewm(span=ema_period, adjust=False).mean().values
    
    # Calculate ATR
    atr = average_true_range(high_prices, low_prices, close_prices, atr_period)
    
    # Calculate upper and lower lines
    upper_line = middle_line + (multiplier * atr)
    lower_line = middle_line - (multiplier * atr)
    
    return middle_line, upper_line, lower_line

def donchian_channels(high_prices, low_prices, period=20):
    """
    Calculate Donchian Channels.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        period: Lookback period (default 20)
        
    Returns:
        tuple: (upper band, middle band, lower band)
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Initialize bands
    upper_band = np.zeros_like(high_prices)
    lower_band = np.zeros_like(low_prices)
    
    # Calculate upper and lower bands
    for i in range(period - 1, len(high_prices)):
        upper_band[i] = np.max(high_prices[i-period+1:i+1])
        lower_band[i] = np.min(low_prices[i-period+1:i+1])
    
    # Calculate middle band
    middle_band = (upper_band + lower_band) / 2
    
    return upper_band, middle_band, lower_band

def volatility_index(close_prices, period=30):
    """
    Calculate a simple Volatility Index based on standard deviation.
    
    Args:
        close_prices: Array of close prices
        period: Lookback period (default 30)
        
    Returns:
        numpy.ndarray: Volatility index values as percentage
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate returns
    returns = np.zeros_like(close_prices)
    for i in range(1, len(close_prices)):
        if close_prices[i-1] > 0:  # Avoid division by zero
            returns[i] = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
    
    # Calculate volatility as standard deviation of returns
    volatility = np.zeros_like(close_prices)
    for i in range(period, len(close_prices)):
        volatility[i] = np.std(returns[i-period+1:i+1], ddof=1) * 100  # Convert to percentage
    
    return volatility

def chaikin_volatility(high_prices, low_prices, ema_period=10, roc_period=10):
    """
    Calculate Chaikin Volatility.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        ema_period: EMA period for smoothing (default 10)
        roc_period: Rate of change period (default 10)
        
    Returns:
        numpy.ndarray: Chaikin Volatility values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Calculate high-low range
    hl_range = high_prices - low_prices
    
    # Smooth the range with EMA
    # Use pandas for this calculation for better accuracy
    hl_range_series = pd.Series(hl_range)
    smooth_range = hl_range_series.ewm(span=ema_period, adjust=False).mean().values
    
    # Calculate rate of change
    chaikin_vol = np.zeros_like(smooth_range)
    for i in range(roc_period, len(smooth_range)):
        if smooth_range[i-roc_period] > 0:  # Avoid division by zero
            chaikin_vol[i] = ((smooth_range[i] - smooth_range[i-roc_period]) / smooth_range[i-roc_period]) * 100
    
    return chaikin_vol

def historical_volatility(close_prices, period=21, trading_periods=252):
    """
    Calculate Historical Volatility.
    
    Args:
        close_prices: Array of close prices
        period: Lookback period (default 21)
        trading_periods: Number of trading periods in a year (default 252)
        
    Returns:
        numpy.ndarray: Historical volatility values (annualized)
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate log returns
    log_returns = np.zeros_like(close_prices)
    for i in range(1, len(close_prices)):
        if close_prices[i-1] > 0:  # Avoid division by zero
            log_returns[i] = np.log(close_prices[i] / close_prices[i-1])
    
    # Calculate historical volatility
    hv = np.zeros_like(close_prices)
    for i in range(period, len(close_prices)):
        # Standard deviation of log returns
        std_dev = np.std(log_returns[i-period+1:i+1], ddof=1)
        
        # Annualize by multiplying by square root of trading periods
        hv[i] = std_dev * np.sqrt(trading_periods) * 100  # Convert to percentage
    
    return hv

def standard_deviation(prices, period=20):
    """
    Calculate Rolling Standard Deviation.
    
    Args:
        prices: Array of price values
        period: Lookback period (default 20)
        
    Returns:
        numpy.ndarray: Standard deviation values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate standard deviation
    std_dev = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        std_dev[i] = np.std(prices[i-period+1:i+1], ddof=1)  # Using sample standard deviation
    
    return std_dev

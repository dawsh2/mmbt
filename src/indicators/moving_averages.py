# src/indicators/moving_averages.py
"""
Moving average indicators.

This module provides functions for calculating various types of moving averages.
These functions are pure, stateless computations on price data.
"""

import numpy as np
import pandas as pd

def simple_moving_average(prices, window):
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Array of price values
        window: Size of the moving window
        
    Returns:
        numpy.ndarray: SMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
        
    # Faster implementation using convolution
    weights = np.ones(window) / window
    return np.convolve(prices, weights, mode='valid')

def weighted_moving_average(prices, window):
    """
    Calculate Weighted Moving Average.
    
    Args:
        prices: Array of price values
        window: Size of the moving window
        
    Returns:
        numpy.ndarray: WMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
        
    # Create weights that increase linearly
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    # Calculate WMA using convolution
    wma = np.zeros(len(prices) - window + 1)
    for i in range(len(wma)):
        wma[i] = np.sum(prices[i:i+window] * weights)
    
    return wma

def exponential_moving_average(prices, span):
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Array of price values
        span: Specified period for the EMA
        
    Returns:
        numpy.ndarray: EMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    alpha = 2 / (span + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Initialize with first price
    
    # Calculate EMA
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def double_exponential_moving_average(prices, span):
    """
    Calculate Double Exponential Moving Average.
    
    Args:
        prices: Array of price values
        span: Specified period for the DEMA
        
    Returns:
        numpy.ndarray: DEMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    ema1 = exponential_moving_average(prices, span)
    ema2 = exponential_moving_average(ema1, span)
    
    # DEMA = 2 * EMA1 - EMA2
    dema = 2 * ema1 - ema2
    
    return dema

def triple_exponential_moving_average(prices, span):
    """
    Calculate Triple Exponential Moving Average.
    
    Args:
        prices: Array of price values
        span: Specified period for the TEMA
        
    Returns:
        numpy.ndarray: TEMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    ema1 = exponential_moving_average(prices, span)
    ema2 = exponential_moving_average(ema1, span)
    ema3 = exponential_moving_average(ema2, span)
    
    # TEMA = 3 * (EMA1 - EMA2) + EMA3
    tema = 3 * (ema1 - ema2) + ema3
    
    return tema

def hull_moving_average(prices, window):
    """
    Calculate Hull Moving Average.
    
    The Hull Moving Average (HMA) is designed to reduce lag and improve smoothness.
    
    Args:
        prices: Array of price values
        window: Size of the moving window
        
    Returns:
        numpy.ndarray: HMA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate WMA with half period
    half_window = window // 2
    wma_half = weighted_moving_average(prices, half_window)
    
    # Calculate WMA with full period
    wma_full = weighted_moving_average(prices, window)
    
    # Adjust for lengths - expand wma_full to match wma_half length
    wma_full_adjusted = wma_full[-(len(wma_half)):]
    
    # Calculate 2 * WMA(half period) - WMA(full period)
    diff = 2 * wma_half - wma_full_adjusted
    
    # Calculate WMA of diff with sqrt(period)
    sqrt_window = int(np.sqrt(window))
    hma = weighted_moving_average(diff, sqrt_window)
    
    return hma

def kaufman_adaptive_moving_average(prices, n=10, fast_ema=2, slow_ema=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    
    KAMA adjusts the smoothing based on market efficiency.
    
    Args:
        prices: Array or Series of price values
        n: Efficiency ratio lookback period
        fast_ema: Fast EMA period (typically 2)
        slow_ema: Slow EMA period (typically 30)
        
    Returns:
        numpy.ndarray: KAMA values
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate price change and absolute price change
    change = prices.diff(n).abs()
    volatility = prices.diff().abs().rolling(n).sum()
    
    # Calculate efficiency ratio
    er = np.zeros_like(prices)
    valid_indices = ~(change.isna() | volatility.isna() | (volatility == 0))
    er[valid_indices] = change[valid_indices] / volatility[valid_indices]
    
    # Calculate smoothing constant
    fast_alpha = 2 / (fast_ema + 1)
    slow_alpha = 2 / (slow_ema + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    # Initialize KAMA
    kama = np.zeros_like(prices)
    kama[n] = prices[n]  # Start KAMA with the nth price
    
    # Calculate KAMA
    for i in range(n + 1, len(prices)):
        kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
    
    return kama

def variable_index_dynamic_average(prices, period=9, vi_period=6):
    """
    Calculate Variable Index Dynamic Average (VIDYA).
    
    VIDYA is an EMA that adjusts based on volatility.
    
    Args:
        prices: Array or Series of price values
        period: VIDYA period
        vi_period: Volatility index period
        
    Returns:
        numpy.ndarray: VIDYA values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Calculate standard deviation over VI period
    vol_index = np.zeros_like(prices)
    for i in range(vi_period, len(prices)):
        std = np.std(prices[i-vi_period:i])
        if std != 0:
            vol_index[i] = std
    
    # Normalize volatility index
    max_vol = np.max(vol_index[vi_period:])
    if max_vol > 0:
        vol_index[vi_period:] /= max_vol
    
    # Calculate VIDYA
    vidya = np.zeros_like(prices)
    vidya[vi_period] = prices[vi_period]
    
    k = 2 / (period + 1)
    
    for i in range(vi_period + 1, len(prices)):
        alpha = k * vol_index[i]
        vidya[i] = alpha * prices[i] + (1 - alpha) * vidya[i-1]
    
    return vidya

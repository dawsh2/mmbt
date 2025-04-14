# src/indicators/trend.py
"""
Trend indicators.

This module provides functions for calculating trend-based indicators.
These functions are pure, stateless computations on price data.
"""

import numpy as np
import pandas as pd
from src.indicators.moving_averages import exponential_moving_average

def average_directional_index(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: ADX period (default 14)
        
    Returns:
        tuple: (ADX, +DI, -DI)
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Initialize arrays
    tr = np.zeros_like(close_prices)  # True Range
    plus_dm = np.zeros_like(close_prices)  # Plus Directional Movement
    minus_dm = np.zeros_like(close_prices)  # Minus Directional Movement
    
    # Calculate TR, +DM, -DM for each period
    for i in range(1, len(close_prices)):
        # True Range
        tr1 = high_prices[i] - low_prices[i]
        tr2 = abs(high_prices[i] - close_prices[i-1])
        tr3 = abs(low_prices[i] - close_prices[i-1])
        tr[i] = max(tr1, tr2, tr3)
        
        # Plus Directional Movement
        up_move = high_prices[i] - high_prices[i-1]
        down_move = low_prices[i-1] - low_prices[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        
        # Minus Directional Movement
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing method
    tr_period = np.zeros_like(close_prices)
    plus_dm_period = np.zeros_like(close_prices)
    minus_dm_period = np.zeros_like(close_prices)
    
    # Initialize first values with simple average
    if len(tr) >= period:
        tr_period[period] = np.sum(tr[1:period+1])
        plus_dm_period[period] = np.sum(plus_dm[1:period+1])
        minus_dm_period[period] = np.sum(minus_dm[1:period+1])
        
        # Calculate subsequent smoothed values
        for i in range(period + 1, len(close_prices)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
            plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
            minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
    
    # Calculate +DI, -DI
    plus_di = np.zeros_like(close_prices)
    minus_di = np.zeros_like(close_prices)
    
    for i in range(period, len(close_prices)):
        if tr_period[i] > 0:  # Avoid division by zero
            plus_di[i] = 100 * plus_dm_period[i] / tr_period[i]
            minus_di[i] = 100 * minus_dm_period[i] / tr_period[i]
    
    # Calculate DX and ADX
    dx = np.zeros_like(close_prices)
    adx = np.zeros_like(close_prices)
    
    for i in range(period, len(close_prices)):
        if (plus_di[i] + minus_di[i]) > 0:  # Avoid division by zero
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    # Initialize first ADX value
    if len(dx) >= 2*period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        
        # Calculate subsequent ADX values
        for i in range(2*period, len(close_prices)):
            adx[i] = ((period - 1) * adx[i-1] + dx[i]) / period
    
    return adx, plus_di, minus_di

def moving_average_convergence_divergence(prices, fast_period=12, slow_period=26, signal_period=9):
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

def parabolic_sar(high_prices, low_prices, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    Calculate Parabolic SAR (Stop and Reverse).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        af_start: Starting acceleration factor (default 0.02)
        af_step: Acceleration factor step (default 0.02)
        af_max: Maximum acceleration factor (default 0.2)
        
    Returns:
        numpy.ndarray: PSAR values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Initialize arrays
    psar = np.zeros_like(high_prices)
    psar_bull = np.zeros_like(high_prices)  # PSAR values for bullish trend
    psar_bear = np.zeros_like(high_prices)  # PSAR values for bearish trend
    is_bull = np.zeros_like(high_prices, dtype=bool)  # Whether trend is bullish
    af = np.zeros_like(high_prices)  # Acceleration factor
    ep = np.zeros_like(high_prices)  # Extreme point
    
    # Set initial values
    if len(high_prices) >= 2:
        # Start with a bullish trend
        is_bull[1] = True
        
        # Set the first SAR value
        psar[1] = low_prices[0]  # Start with the first low
        
        # Set the extreme point
        ep[1] = high_prices[1]
        
        # Set the acceleration factor
        af[1] = af_start
    
    # Calculate PSAR for each period
    for i in range(2, len(high_prices)):
        # Previous period's values
        prev_psar = psar[i-1]
        prev_is_bull = is_bull[i-1]
        prev_af = af[i-1]
        prev_ep = ep[i-1]
        
        # Calculate current PSAR based on previous trend
        if prev_is_bull:
            # Bullish trend
            psar[i] = prev_psar + prev_af * (prev_ep - prev_psar)
            
            # Ensure PSAR is not higher than the previous two lows
            psar[i] = min(psar[i], low_prices[i-1], low_prices[i-2] if i >= 3 else low_prices[i-1])
            
            # Check if trend reverses
            if psar[i] > low_prices[i]:
                # Trend reverses to bearish
                is_bull[i] = False
                psar[i] = prev_ep  # New PSAR is the previous extreme point
                ep[i] = low_prices[i]  # New extreme point is the current low
                af[i] = af_start  # Reset acceleration factor
            else:
                # Continue bullish trend
                is_bull[i] = True
                
                # Update extreme point and acceleration factor
                if high_prices[i] > prev_ep:
                    ep[i] = high_prices[i]
                    af[i] = min(prev_af + af_step, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            # Bearish trend
            psar[i] = prev_psar - prev_af * (prev_psar - prev_ep)
            
            # Ensure PSAR is not lower than the previous two highs
            psar[i] = max(psar[i], high_prices[i-1], high_prices[i-2] if i >= 3 else high_prices[i-1])
            
            # Check if trend reverses
            if psar[i] < high_prices[i]:
                # Trend reverses to bullish
                is_bull[i] = True
                psar[i] = prev_ep  # New PSAR is the previous extreme point
                ep[i] = high_prices[i]  # New extreme point is the current high
                af[i] = af_start  # Reset acceleration factor
            else:
                # Continue bearish trend
                is_bull[i] = False
                
                # Update extreme point and acceleration factor
                if low_prices[i] < prev_ep:
                    ep[i] = low_prices[i]
                    af[i] = min(prev_af + af_step, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
    
    return psar

def aroon(high_prices, low_prices, period=25):
    """
    Calculate Aroon indicators.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        period: Lookback period (default 25)
        
    Returns:
        tuple: (Aroon Up, Aroon Down, Aroon Oscillator)
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Initialize arrays
    aroon_up = np.zeros_like(high_prices)
    aroon_down = np.zeros_like(low_prices)
    
    # Calculate Aroon Up and Down
    for i in range(period, len(high_prices)):
        # Find days since highest high in the period
        high_period = high_prices[i-period+1:i+1]
        max_idx = np.argmax(high_period)
        days_since_high = period - 1 - max_idx
        
        # Find days since lowest low in the period
        low_period = low_prices[i-period+1:i+1]
        min_idx = np.argmin(low_period)
        days_since_low = period - 1 - min_idx
        
        # Calculate Aroon indicators
        aroon_up[i] = ((period - days_since_high) / period) * 100
        aroon_down[i] = ((period - days_since_low) / period) * 100
    
    # Calculate Aroon Oscillator (Aroon Up - Aroon Down)
    aroon_osc = aroon_up - aroon_down
    
    return aroon_up, aroon_down, aroon_osc

def ichimoku_cloud(high_prices, low_prices, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26):
    """
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
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    
    # Initialize arrays
    tenkan_sen = np.zeros_like(high_prices)
    kijun_sen = np.zeros_like(high_prices)
    senkou_span_a = np.zeros_like(high_prices)
    senkou_span_b = np.zeros_like(high_prices)
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    for i in range(tenkan_period - 1, len(high_prices)):
        highest_high = np.max(high_prices[i-tenkan_period+1:i+1])
        lowest_low = np.min(low_prices[i-tenkan_period+1:i+1])
        tenkan_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    for i in range(kijun_period - 1, len(high_prices)):
        highest_high = np.max(high_prices[i-kijun_period+1:i+1])
        lowest_low = np.min(low_prices[i-kijun_period+1:i+1])
        kijun_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted forward kijun_period periods
    for i in range(tenkan_period - 1, len(high_prices)):
        if i + kijun_period < len(high_prices):
            senkou_span_a[i + kijun_period] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, shifted forward kijun_period periods
    for i in range(senkou_b_period - 1, len(high_prices)):
        highest_high = np.max(high_prices[i-senkou_b_period+1:i+1])
        lowest_low = np.min(low_prices[i-senkou_b_period+1:i+1])
        if i + kijun_period < len(high_prices):
            senkou_span_b[i + kijun_period] = (highest_high + lowest_low) / 2
    
    # Calculate Chikou Span (Lagging Span): Current closing price shifted backwards chikou_period periods
    chikou_span = np.roll(high_prices, -chikou_period)  # Simple approximation using high prices
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def trix(prices, period=15):
    """
    Calculate the TRIX indicator (Triple Exponential Average).
    
    Args:
        prices: Array of price values
        period: EMA period (default 15)
        
    Returns:
        numpy.ndarray: TRIX values
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Use pandas for consistent EMA calculation
    prices_series = pd.Series(prices)
    
    # Calculate Triple EMA
    ema1 = prices_series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # Calculate TRIX (1-period percent change of Triple EMA)
    trix = np.zeros_like(prices)
    ema3_values = ema3.values
    
    for i in range(1, len(ema3_values)):
        if ema3_values[i-1] != 0:  # Avoid division by zero
            trix[i] = (ema3_values[i] - ema3_values[i-1]) / ema3_values[i-1] * 100
    
    return trix

def vortex_indicator(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Vortex Indicator.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: Lookback period (default 14)
        
    Returns:
        tuple: (VI+, VI-)
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate +VM and -VM
    plus_vm = np.zeros_like(high_prices)
    minus_vm = np.zeros_like(high_prices)
    
    for i in range(1, len(high_prices)):
        plus_vm[i] = abs(high_prices[i] - low_prices[i-1])
        minus_vm[i] = abs(low_prices[i] - high_prices[i-1])
    
    # Calculate True Range
    tr = np.zeros_like(close_prices)
    for i in range(1, len(close_prices)):
        tr1 = high_prices[i] - low_prices[i]
        tr2 = abs(high_prices[i] - close_prices[i-1])
        tr3 = abs(low_prices[i] - close_prices[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate VI+ and VI-
    vi_plus = np.zeros_like(high_prices)
    vi_minus = np.zeros_like(high_prices)
    
    for i in range(period, len(high_prices)):
        sum_tr = np.sum(tr[i-period+1:i+1])
        if sum_tr > 0:  # Avoid division by zero
            vi_plus[i] = np.sum(plus_vm[i-period+1:i+1]) / sum_tr
            vi_minus[i] = np.sum(minus_vm[i-period+1:i+1]) / sum_tr
    
    return vi_plus, vi_minus

def supertrend(high_prices, low_prices, close_prices, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
        
    Returns:
        tuple: (SuperTrend, Direction)
    """
    from src.indicators.volatility import average_true_range
    
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    
    # Calculate ATR
    atr = average_true_range(high_prices, low_prices, close_prices, period)
    
    # Calculate Basic Upper and Lower Bands
    basic_upper = np.zeros_like(close_prices)
    basic_lower = np.zeros_like(close_prices)
    
    for i in range(period, len(close_prices)):
        basic_upper[i] = (high_prices[i] + low_prices[i]) / 2 + multiplier * atr[i]
        basic_lower[i] = (high_prices[i] + low_prices[i]) / 2 - multiplier * atr[i]
    
    # Initialize SuperTrend and Direction arrays
    supertrend = np.zeros_like(close_prices)
    direction = np.zeros_like(close_prices)  # 1 for uptrend, -1 for downtrend
    
    # Calculate SuperTrend
    for i in range(period, len(close_prices)):
        if i == period:
            # Initialize
            supertrend[i] = basic_upper[i] if close_prices[i] <= basic_upper[i] else basic_lower[i]
            direction[i] = -1 if close_prices[i] <= basic_upper[i] else 1
        else:
            if supertrend[i-1] == basic_upper[i-1]:
                # Previous trend was down
                supertrend[i] = basic_upper[i] if close_prices[i] <= basic_upper[i] else basic_lower[i]
            else:
                # Previous trend was up
                supertrend[i] = basic_lower[i] if close_prices[i] >= basic_lower[i] else basic_upper[i]
            
            # Update direction
            direction[i] = -1 if supertrend[i] == basic_upper[i] else 1
    
    return supertrend, direction

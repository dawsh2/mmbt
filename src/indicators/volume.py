# src/indicators/volume.py
"""
Volume indicators.

This module provides functions for calculating volume-based indicators.
These functions are pure, stateless computations on price and volume data.
"""

import numpy as np
import pandas as pd

def on_balance_volume(close_prices, volume):
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close_prices: Array of close prices
        volume: Array of volume values
        
    Returns:
        numpy.ndarray: OBV values
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Initialize OBV array
    obv = np.zeros_like(close_prices)
    
    # First value is the first volume
    if len(volume) > 0:
        obv[0] = volume[0]
    
    # Calculate OBV
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i-1]:
            # Price up, add volume
            obv[i] = obv[i-1] + volume[i]
        elif close_prices[i] < close_prices[i-1]:
            # Price down, subtract volume
            obv[i] = obv[i-1] - volume[i]
        else:
            # Price unchanged, OBV unchanged
            obv[i] = obv[i-1]
    
    return obv

def volume_price_trend(close_prices, volume, period=14):
    """
    Calculate Volume-Price Trend (VPT).
    
    Args:
        close_prices: Array of close prices
        volume: Array of volume values
        period: SMA period for signal line (default 14)
        
    Returns:
        tuple: (VPT, Signal Line)
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Initialize VPT array
    vpt = np.zeros_like(close_prices)
    
    # First value is the first volume
    if len(volume) > 0:
        vpt[0] = volume[0]
    
    # Calculate VPT
    for i in range(1, len(close_prices)):
        price_change_pct = 0
        if close_prices[i-1] > 0:  # Avoid division by zero
            price_change_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
        
        vpt[i] = vpt[i-1] + volume[i] * price_change_pct
    
    # Calculate signal line (SMA of VPT)
    signal_line = np.zeros_like(vpt)
    for i in range(period - 1, len(vpt)):
        signal_line[i] = np.mean(vpt[i-period+1:i+1])
    
    return vpt, signal_line

def accumulation_distribution(high_prices, low_prices, close_prices, volume):
    """
    Calculate Accumulation/Distribution Line.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        volume: Array of volume values
        
    Returns:
        numpy.ndarray: A/D Line values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Initialize arrays
    ad_line = np.zeros_like(close_prices)
    
    # Calculate Money Flow Multiplier and Money Flow Volume
    for i in range(len(close_prices)):
        if high_prices[i] != low_prices[i]:  # Avoid division by zero
            # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
            mfm = ((close_prices[i] - low_prices[i]) - (high_prices[i] - close_prices[i])) / (high_prices[i] - low_prices[i])
            
            # Money Flow Volume = Money Flow Multiplier * Volume
            mfv = mfm * volume[i]
            
            # A/D Line = Previous A/D Line + Money Flow Volume
            if i > 0:
                ad_line[i] = ad_line[i-1] + mfv
            else:
                ad_line[i] = mfv
        elif i > 0:
            # If high equals low, use previous value
            ad_line[i] = ad_line[i-1]
    
    return ad_line

def chaikin_oscillator(high_prices, low_prices, close_prices, volume, fast_period=3, slow_period=10):
    """
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
    """
    # Calculate A/D Line
    ad_line = accumulation_distribution(high_prices, low_prices, close_prices, volume)
    
    # Use pandas for consistent EMA calculation
    ad_series = pd.Series(ad_line)
    
    # Calculate EMAs of A/D Line
    fast_ema = ad_series.ewm(span=fast_period, adjust=False).mean().values
    slow_ema = ad_series.ewm(span=slow_period, adjust=False).mean().values
    
    # Chaikin Oscillator = Fast EMA - Slow EMA
    chaikin_osc = fast_ema - slow_ema
    
    return chaikin_osc

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
    
    # Initialize arrays
    positive_flow = np.zeros_like(tp)
    negative_flow = np.zeros_like(tp)
    mfi = np.zeros_like(tp)
    
    # Identify positive and negative money flow
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:  # Prices rising
            positive_flow[i] = money_flow[i]
        elif tp[i] < tp[i-1]:  # Prices falling
            negative_flow[i] = money_flow[i]
        else:  # Prices unchanged
            # Assign to the same flow as the previous bar
            if i > 1:
                if tp[i-1] > tp[i-2]:
                    positive_flow[i] = money_flow[i]
                else:
                    negative_flow[i] = money_flow[i]
    
    # Calculate MFI using the money ratio
    for i in range(period, len(tp)):
        pos_sum = np.sum(positive_flow[i-period+1:i+1])
        neg_sum = np.sum(negative_flow[i-period+1:i+1])
        
        if neg_sum > 0:  # Avoid division by zero
            money_ratio = pos_sum / neg_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
        elif pos_sum > 0:  # All positive flow
            mfi[i] = 100
        else:  # No flow
            mfi[i] = 50
    
    return mfi

def ease_of_movement(high_prices, low_prices, volume, period=14):
    """
    Calculate Ease of Movement (EOM).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        volume: Array of volume values
        period: SMA period for smoothing (default 14)
        
    Returns:
        numpy.ndarray: EOM values
    """
    if not isinstance(high_prices, np.ndarray):
        high_prices = np.array(high_prices)
    if not isinstance(low_prices, np.ndarray):
        low_prices = np.array(low_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Calculate distance moved
    distance = ((high_prices + low_prices) / 2) - ((high_prices[:-1] + low_prices[:-1]) / 2)
    distance = np.append([0], distance)  # Prepend 0 for the first bar
    
    # Calculate box ratio
    box_ratio = volume / (high_prices - low_prices)
    box_ratio[high_prices == low_prices] = volume[high_prices == low_prices]  # Handle division by zero
    
    # Calculate raw EOM
    raw_eom = distance / box_ratio
    
    # Smooth with SMA
    eom = np.zeros_like(raw_eom)
    for i in range(period - 1, len(raw_eom)):
        eom[i] = np.mean(raw_eom[i-period+1:i+1])
    
    return eom

def volume_weighted_average_price(high_prices, low_prices, close_prices, volume, period=14):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        volume: Array of volume values
        period: VWAP period (default 14)
        
    Returns:
        numpy.ndarray: VWAP values
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
    
    # Calculate VWAP
    vwap = np.zeros_like(tp)
    
    for i in range(len(tp)):
        start_idx = max(0, i - period + 1)
        
        cum_tp_vol = np.sum(tp[start_idx:i+1] * volume[start_idx:i+1])
        cum_vol = np.sum(volume[start_idx:i+1])
        
        if cum_vol > 0:  # Avoid division by zero
            vwap[i] = cum_tp_vol / cum_vol
    
    return vwap

def negative_volume_index(close_prices, volume):
    """
    Calculate Negative Volume Index (NVI).
    
    Args:
        close_prices: Array of close prices
        volume: Array of volume values
        
    Returns:
        numpy.ndarray: NVI values
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Initialize NVI array (start at 1000)
    nvi = np.ones_like(close_prices) * 1000
    
    # Calculate NVI
    for i in range(1, len(close_prices)):
        if volume[i] < volume[i-1]:  # Volume decreased
            price_change_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1] if close_prices[i-1] > 0 else 0
            nvi[i] = nvi[i-1] * (1 + price_change_pct)
        else:  # Volume increased or unchanged
            nvi[i] = nvi[i-1]
    
    return nvi

def positive_volume_index(close_prices, volume):
    """
    Calculate Positive Volume Index (PVI).
    
    Args:
        close_prices: Array of close prices
        volume: Array of volume values
        
    Returns:
        numpy.ndarray: PVI values
    """
    if not isinstance(close_prices, np.ndarray):
        close_prices = np.array(close_prices)
    if not isinstance(volume, np.ndarray):
        volume = np.array(volume)
    
    # Initialize PVI array (start at 1000)
    pvi = np.ones_like(close_prices) * 1000
    
    # Calculate PVI
    for i in range(1, len(close_prices)):
        if volume[i] > volume[i-1]:  # Volume increased
            price_change_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1] if close_prices[i-1] > 0 else 0
            pvi[i] = pvi[i-1] * (1 + price_change_pct)
        else:  # Volume decreased or unchanged
            pvi[i] = pvi[i-1]
    
    return pvi

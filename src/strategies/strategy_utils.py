"""
Strategy Utilities Module

This module provides utility functions for working with strategies and signals
in the trading system. It standardizes signal generation and strategy operations.
"""

import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np

from src.events.signal_event import SignalEvent
from src.signals.signal_processing import SignalType
from src.events.event_bus import Event
from src.events.event_types import EventType
from src.events.event_utils import unpack_bar_event, get_event_timestamp, get_event_symbol


def create_signal_event(signal_type: SignalType, price: float, 
                       symbol: str = "default", rule_id: Optional[str] = None,
                       confidence: float = 1.0, 
                       metadata: Optional[Dict[str, Any]] = None,
                       timestamp: Optional[datetime.datetime] = None) -> SignalEvent:
    """
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
    """
    return SignalEvent(
        signal_type=signal_type,
        price=price,
        symbol=symbol,
        rule_id=rule_id,
        confidence=confidence,
        metadata=metadata,
        timestamp=timestamp
    )


def extract_bar_data(event: Event) -> Dict[str, Any]:
    """
    Extract bar data from an event for strategy processing.
    
    Args:
        event: Event object
        
    Returns:
        Bar data dictionary
    """
    try:
        return unpack_bar_event(event)
    except ValueError:
        # Fallback to empty dict if extraction fails
        return {}


def get_indicator_value(indicators: Dict[str, Any], name: str, default: Any = None) -> Any:
    """
    Safely get indicator value with fallback.
    
    Args:
        indicators: Dictionary of indicators
        name: Indicator name to retrieve
        default: Default value if not found
        
    Returns:
        Indicator value or default
    """
    if not indicators:
        return default
        
    return indicators.get(name, default)


def analyze_bar_pattern(bars: List[Dict[str, Any]], window: int = 5) -> Dict[str, Any]:
    """
    Analyze a pattern in a series of bars.
    
    Args:
        bars: List of bar data dictionaries
        window: Analysis window size
        
    Returns:
        Dictionary with pattern analysis results
    """
    if not bars or len(bars) < window:
        return {'valid': False, 'reason': 'Not enough bars'}
    
    # Extract close prices
    closes = [bar.get('Close', 0) for bar in bars[-window:]]
    
    # Calculate basic metrics
    price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0
    volatility = np.std(closes) / np.mean(closes) if np.mean(closes) else 0
    
    # Detect trend
    trend = 'up' if price_change > 0 else 'down' if price_change < 0 else 'neutral'
    trend_strength = abs(price_change)
    
    # Check for pattern
    is_higher_highs = all(closes[i] >= closes[i-1] for i in range(1, len(closes)))
    is_lower_lows = all(closes[i] <= closes[i-1] for i in range(1, len(closes)))
    
    return {
        'valid': True,
        'trend': trend,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'price_change_pct': price_change * 100,
        'is_higher_highs': is_higher_highs,
        'is_lower_lows': is_lower_lows
    }


def calculate_signal_confidence(indicators: Dict[str, Any], 
                              trend_strength: float = 0.5) -> float:
    """
    Calculate confidence score for a signal based on indicators.
    
    Args:
        indicators: Dictionary of indicator values
        trend_strength: Strength of the current trend (0-1)
        
    Returns:
        Confidence score (0-1)
    """
    # Start with base confidence
    confidence = 0.5
    factors = []
    
    # Add confidence based on RSI
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        if rsi < 30:  # Oversold
            factors.append(0.7)  # Strong buy confidence
        elif rsi > 70:  # Overbought
            factors.append(0.7)  # Strong sell confidence
        else:
            factors.append(0.5)  # Neutral confidence
    
    # Add confidence based on MACD
    if 'macd' in indicators and 'macd_signal' in indicators:
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = macd - macd_signal
        
        if abs(macd_hist) > 0.5:  # Strong MACD signal
            factors.append(0.8)
        else:
            factors.append(0.5)
    
    # Add confidence based on trend strength
    factors.append(trend_strength)
    
    # Combine factors
    if factors:
        confidence = sum(factors) / len(factors)
    
    # Ensure confidence is between 0 and 1
    return min(max(confidence, 0.0), 1.0)

"""
Strategy Utilities Module

This module provides utility functions for working with strategy components,
particularly for handling events and creating standardized signals.
"""

from typing import Tuple, Dict, Any, Optional, Union
import datetime
from src.signals import Signal, SignalType


# this (and the following function) should be imported from src/events/event_utils instead
def unpack_bar_event(event) -> Tuple[Dict[str, Any], str, float, datetime.datetime]:
    """
    Extract bar data from an event object.
    
    Args:
        event: Event object containing bar data
        
    Returns:
        tuple: (bar_dict, symbol, price, timestamp)
    """
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
        
    bar_event = event.data
    if not hasattr(bar_event, 'bar'):
        raise TypeError(f"Expected BarEvent object in event.data")
        
    bar = bar_event.bar
    symbol = bar.get('symbol', 'unknown')
    price = bar.get('Close')
    timestamp = bar.get('timestamp')
    
    return bar, symbol, price, timestamp


def create_signal(
    timestamp: datetime.datetime, 
    signal_type: SignalType, 
    price: float, 
    rule_id: Optional[str] = None, 
    confidence: float = 1.0, 
    symbol: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> Signal:
    """
    Create a standardized Signal object.
    
    Args:
        timestamp: Signal timestamp
        signal_type: Type of signal (BUY, SELL, NEUTRAL)
        price: Price at signal generation
        rule_id: Optional rule identifier
        confidence: Signal confidence (0-1)
        symbol: Instrument symbol
        metadata: Additional signal metadata
        
    Returns:
        Signal: Standardized signal object
    """
    return Signal(
        timestamp=timestamp,
        signal_type=signal_type,
        price=price,
        rule_id=rule_id,
        confidence=confidence,
        symbol=symbol,
        metadata=metadata or {}
    )


def get_indicator_value(indicators: Dict[str, Any], name: str, default: Any = None) -> Any:
    """
    Safely get an indicator value from an indicators dictionary.
    
    Args:
        indicators: Dictionary of indicator values
        name: Name of the indicator to get
        default: Default value if indicator not found
        
    Returns:
        The indicator value or default
    """
    if not indicators:
        return default
        
    return indicators.get(name, default)


def analyze_bar_pattern(
    bars: list, 
    window: int = 3
) -> Dict[str, Any]:
    """
    Analyze a pattern in recent bars.
    
    Args:
        bars: List of bar dictionaries
        window: Number of bars to consider
        
    Returns:
        Dictionary with pattern analysis
    """
    if len(bars) < window:
        return {'valid': False, 'reason': 'Not enough bars'}
        
    # Take the most recent bars
    recent_bars = bars[-window:]
    
    # Extract close prices
    closes = [bar.get('Close', 0) for bar in recent_bars]
    
    # Calculate trend
    trend = 'up' if closes[-1] > closes[0] else 'down' if closes[-1] < closes[0] else 'sideways'
    
    # Calculate volatility (simple range measure)
    high = max(closes)
    low = min(closes)
    volatility = (high - low) / low if low > 0 else 0
    
    # Detect specific patterns
    is_higher_highs = all(closes[i] > closes[i-1] for i in range(1, len(closes)))
    is_lower_lows = all(closes[i] < closes[i-1] for i in range(1, len(closes)))
    
    return {
        'valid': True,
        'trend': trend,
        'volatility': volatility,
        'higher_highs': is_higher_highs,
        'lower_lows': is_lower_lows,
        'bars_analyzed': window
    }


def calculate_signal_confidence(
    indicators: Dict[str, Any], 
    trend_strength: float = 0.5
) -> float:
    """
    Calculate a signal confidence score based on indicators.
    
    Args:
        indicators: Dictionary of indicator values
        trend_strength: Strength of the current trend (0-1)
        
    Returns:
        Confidence score between 0 and 1
    """
    # Start with a base confidence of 0.5
    confidence = 0.5
    
    # Adjust based on trend strength
    confidence += (trend_strength - 0.5) * 0.2
    
    # Adjust based on indicator agreement if available
    if 'indicator_agreement' in indicators:
        agreement = indicators['indicator_agreement']
        confidence += (agreement - 0.5) * 0.4
    
    # Adjust based on volatility if available
    if 'volatility' in indicators:
        volatility = indicators['volatility']
        # Lower confidence in high volatility
        if volatility > 0.02:  # 2% volatility threshold
            confidence -= min(volatility * 5, 0.2)  # Max reduction of 0.2
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

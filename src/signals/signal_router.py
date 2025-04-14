"""
Signal Router Module

This module provides the SignalRouter class that collects and manages signals
from multiple rules. It was created to support backward compatibility for 
migrated code from the legacy architecture.
"""

from typing import List, Dict, Any, Optional
from src.signals import Signal, SignalType


class SignalCollection:
    """A collection of signals that provides consensus methods."""
    
    def __init__(self):
        """Initialize an empty signal collection."""
        self.signals = []
        
    def add(self, signal: Signal):
        """Add a signal to the collection."""
        self.signals.append(signal)
        
    def get_weighted_consensus(self) -> SignalType:
        """Get the weighted consensus signal type from all signals."""
        if not self.signals:
            return SignalType.NEUTRAL
            
        # Calculate the weighted average of signal values
        total_weight = 0
        weighted_sum = 0
        
        for signal in self.signals:
            weight = signal.confidence if hasattr(signal, 'confidence') else 1.0
            weighted_sum += signal.signal_type.value * weight
            total_weight += weight
            
        if total_weight > 0:
            avg_value = weighted_sum / total_weight
        else:
            avg_value = 0
            
        # Convert to signal type
        if avg_value > 0.3:
            return SignalType.BUY
        elif avg_value < -0.3:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL


class SignalRouter:
    """
    Routes signals from multiple rules and provides consensus methods.
    
    This class is provided for backward compatibility with legacy strategies.
    """
    
    def __init__(self, rule_objects: List[Any]):
        """
        Initialize the signal router with rule objects.
        
        Args:
            rule_objects: List of rule objects that generate signals
        """
        self.rules = rule_objects
        
    def on_bar(self, event) -> Dict[str, Any]:
        """
        Process a bar event through all rules and collect signals.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            dict: Contains the collected signals and bar metadata
        """
        signals = SignalCollection()
        bar_data = event.bar if hasattr(event, 'bar') else event
        
        # Extract timestamp and price from bar data
        timestamp = bar_data.get('timestamp', None)
        price = bar_data.get('Close', None)
        
        # Process bar through each rule
        for rule in self.rules:
            if hasattr(rule, 'on_bar'):
                signal = rule.on_bar(event)
                if signal:
                    signals.add(signal)
        
        # Return signals and metadata
        return {
            "signals": signals,
            "timestamp": timestamp,
            "price": price
        }
        
    def reset(self):
        """Reset all rules."""
        for rule in self.rules:
            if hasattr(rule, 'reset'):
                rule.reset()

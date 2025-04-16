"""
Base Strategy Module

This module provides the Strategy abstract base class that defines the common
interface for all trading strategies in the system.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union


class Strategy(ABC):
    """Base class for all trading strategies.
    
    This abstract class defines the interface that all strategy implementations
    must follow. Strategies receive bar data events and generate trading signals.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the strategy.
        
        Args:
            name: Optional name for the strategy. If not provided, uses class name.
        """
        self.name = name or self.__class__.__name__
    
    def on_bar(self, bar_event):
        """Process a bar event and generate a trading signal.
        
        This method handles standard event validation and extraction, then
        delegates to generate_signals for actual signal logic.
        
        Args:
            bar_event: Bar event containing market data
            
        Returns:
            Signal: Trading signal object
        """
        # Enforce proper input type
        if not hasattr(bar_event, 'bar'):
            raise TypeError(f"Expected BarEvent object, got {type(bar_event).__name__}")
            
        # Extract the standard bar data dictionary
        bar = bar_event.bar
        
        # Call the implementation-specific signal generation
        return self.generate_signals(bar, bar_event)
    
    @abstractmethod
    def generate_signals(self, bar, bar_event=None):
        """Generate trading signals based on market data.
        
        Args:
            bar: Dictionary containing bar data (OHLCV, timestamp, etc.)
            bar_event: Original bar event object (optional)
            
        Returns:
            Signal: Trading signal object
        """
        pass
    
    def reset(self):
        """Reset the strategy's internal state."""
        pass
    
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.name} Strategy"


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

    def on_bar(self, event):
        """Process a bar event and generate a trading signal.

        Args:
            event: Event object containing bar data

        Returns:
            Signal: Trading signal object
        """
        # Validate the event
        if not hasattr(event, 'data'):
            raise TypeError(f"Expected Event object with data attribute, got {type(event).__name__}")

        # Pass the full event to generate_signals - all data is inside
        return self.generate_signals(event)

    @abstractmethod
    def generate_signals(self, event):
        """Generate trading signals based on market data.

        Args:
            event: Event object containing bar data

        Returns:
            Signal: Trading signal object
        """
        pass

 

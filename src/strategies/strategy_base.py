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
    
    @abstractmethod
    def on_bar(self, event):
        """Process a bar event and generate a trading signal.
        
        Args:
            event: Bar event containing market data
            
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

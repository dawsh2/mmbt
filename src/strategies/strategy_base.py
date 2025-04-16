"""
Strategy Base Module

This module provides the base class for all trading strategies in the system.
It standardizes signal generation and event handling for strategies.
"""

import datetime
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.strategies.strategy_utils import extract_bar_data

# Set up logging
logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    This class provides the standard interface for strategies to receive
    market data events and generate signal events in response.
    """
    
    def __init__(self, name: str, event_bus: Optional[Any] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            event_bus: Optional event bus for emitting signals
        """
        self.name = name
        self.event_bus = event_bus
        self.indicators = {}
        self.state = {}
        self.last_signal = None
        
    def on_bar(self, event: Event) -> Optional[SignalEvent]:
        """
        Process a bar event and generate signals.
        
        This method is called when a new bar event is received.
        It extracts bar data and delegates to generate_signals.
        
        Args:
            event: Bar event
            
        Returns:
            SignalEvent if signal generated, None otherwise
        """
        try:
            # Extract bar data from event
            if isinstance(event.data, BarEvent):
                bar_event = event.data
            elif isinstance(event.data, dict) and 'Close' in event.data:
                # Convert dictionary to BarEvent for backward compatibility
                bar_event = BarEvent(event.data, event.timestamp)
            else:
                bar_data = extract_bar_data(event)
                if not bar_data:
                    logger.warning(f"Strategy {self.name}: Failed to extract bar data from event")
                    return None
                bar_event = BarEvent(bar_data, event.timestamp)
                
            # Update indicators
            self.update_indicators(bar_event.get_data())
            
            # Generate signals
            signal_event = self.generate_signals(bar_event)
            
            # Store last signal
            self.last_signal = signal_event
            
            # Emit signal event if generated
            if signal_event is not None and self.event_bus is not None:
                self.event_bus.emit(Event(EventType.SIGNAL, signal_event))
                
            return signal_event
            
        except Exception as e:
            logger.error(f"Strategy {self.name}: Error processing bar: {str(e)}", exc_info=True)
            return None
    
    def update_indicators(self, bar_data: Dict[str, Any]) -> None:
        """
        Update indicators with new bar data.
        
        This method should be implemented by subclasses to update
        any technical indicators or state based on new bar data.
        
        Args:
            bar_data: Dictionary containing bar data (OHLCV)
        """
        pass
    
    @abstractmethod
    def generate_signals(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Generate trading signals based on market data.
        
        This method must be implemented by subclasses to generate
        signal events based on market data and strategy logic.
        
        Args:
            bar_event: BarEvent containing market data
            
        Returns:
            SignalEvent if signal generated, None otherwise
        """
        pass
    
    def set_event_bus(self, event_bus: Any) -> None:
        """
        Set the event bus for emitting signals.
        
        Args:
            event_bus: Event bus instance
        """
        self.event_bus = event_bus
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.indicators = {}
        self.state = {}
        self.last_signal = None
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current strategy state.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'name': self.name,
            'indicators': self.indicators,
            'state': self.state
        }

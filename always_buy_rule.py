#!/usr/bin/env python3
# always_buy_rule.py - Simple rule that periodically generates buy signals for debugging

from src.rules import Rule
from src.signals import Signal, SignalType
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AlwaysBuyRule(Rule):
    """
    A debugging rule that always generates buy signals at a specified frequency.
    
    This rule is intended for testing and debugging purposes only, and should
    not be used for actual trading.
    
    Parameters:
    -----------
    frequency : int
        Generate a signal every 'frequency' bars (default: 2)
    confidence : float
        Confidence level for generated signals (default: 1.0)
    """
    
    @classmethod
    def default_params(cls):
        return {
            'frequency': 2,   # Generate signals more frequently for testing
            'confidence': 1.0  # Full confidence for testing
        }
    
    def __init__(self, name="always_buy", params=None, description=""):
        super().__init__(name, params or self.default_params(), description or "Debug rule that always generates buy signals")
        self.bar_count = 0
        self.last_signal = None
        logger.info(f"Initialized AlwaysBuyRule with params: {self.params}")
    
    def _validate_params(self):
        """Validate the parameters."""
        if self.params['frequency'] <= 0:
            raise ValueError("Frequency must be positive")
        if not 0 <= self.params['confidence'] <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def reset(self):
        """Reset the rule state."""
        super().reset()
        self.bar_count = 0
        self.last_signal = None
        logger.info(f"Reset AlwaysBuyRule state")
    
    def on_bar(self, event):
        """
        Process a bar event and potentially generate a signal.
        
        Parameters:
        -----------
        event : Event
            Event object containing bar data
            
        Returns:
        --------
        Signal or None
            A buy signal at the specified frequency, or None
        """
        # Log event details for debugging
        logger.info(f"AlwaysBuyRule.on_bar called with event type: {event.event_type if hasattr(event, 'event_type') else 'Unknown'}")
        
        # Check if we have data in the event
        if not hasattr(event, 'data') or event.data is None:
            logger.warning("AlwaysBuyRule.on_bar: Event has no data")
            return None

        # Get the data from the event
        data = event.data
        
        # Log data for debugging
        logger.info(f"AlwaysBuyRule.on_bar data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
        
        # Increment bar counter
        self.bar_count += 1
        logger.info(f"AlwaysBuyRule bar_count incremented to {self.bar_count}")
        
        # Get current bar info
        timestamp = data.get("timestamp") if isinstance(data, dict) else None
        close_price = data.get("Close") if isinstance(data, dict) else None
        
        logger.info(f"AlwaysBuyRule timestamp: {timestamp}, close: {close_price}")
        
        # Log periodically for debugging
        if self.bar_count % 50 == 0:
            logger.info(f"AlwaysBuyRule processed {self.bar_count} bars")
        
        # Generate a buy signal at the specified frequency
        if self.bar_count % self.params['frequency'] == 0:
            # Create signal object
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=close_price,
                rule_id=self.name,
                confidence=self.params['confidence'],
                metadata={"bar_count": self.bar_count}
            )
            
            # Log signal creation
            logger.info(f"AlwaysBuyRule generated BUY signal at bar {self.bar_count}, price {close_price}")
            
            # Store last signal
            self.last_signal = signal
            
            return signal
        
        # No signal for this bar
        logger.info(f"AlwaysBuyRule not generating signal at bar {self.bar_count}")
        return None
    
    # Maintain the original generate_signal method for compatibility
    def generate_signal(self, data):
        """
        Legacy method to maintain compatibility.
        
        Parameters:
        -----------
        data : dict
            Market data for the current bar
            
        Returns:
        --------
        Signal or None
            A buy signal at the specified frequency, or None
        """
        logger.warning("generate_signal called directly - you should be using on_bar instead")
        return None
    
    def get_state(self):
        """Return the current state of the rule for debugging."""
        return {
            "bar_count": self.bar_count,
            "frequency": self.params['frequency'],
            "confidence": self.params['confidence'],
            "last_signal": self.last_signal.timestamp if self.last_signal else None
        }

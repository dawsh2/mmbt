#!/usr/bin/env python3
# always_buy_rule.py - A simple rule that always generates buy signals for debugging

import logging
import datetime
from src.rules.rule_base import Rule
from src.signals import Signal, SignalType

# Set up logging
logger = logging.getLogger(__name__)

class AlwaysBuyRule(Rule):
    """
    A simple rule that always generates buy signals.
    Used for debugging signal propagation through the system.
    """
    
    @classmethod
    def default_params(cls):
        return {
            'frequency': 5,  # Generate a signal every N bars
            'confidence': 0.9  # Signal confidence level
        }
    
    def __init__(self, name="always_buy", params=None, description=""):
        """
        Initialize the always buy rule.
        
        Args:
            name (str): Unique name for the rule
            params (dict): Parameters for the rule
            description (str): Description of the rule
        """
        super().__init__(name, params or self.default_params(), description or "Rule that always generates buy signals")
        
        # Initialize state
        self.bar_count = 0
        self.last_signal = None
        
    def _validate_params(self):
        """Validate rule parameters."""
        if self.params['frequency'] <= 0:
            raise ValueError("Frequency must be positive")
        
        if not 0 <= self.params['confidence'] <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def generate_signal(self, data):
        """
        Generate a buy signal based on configured frequency.
        
        Args:
            data: Market data event
            
        Returns:
            Signal or None: Trading signal if it's time to generate one, None otherwise
        """
        # Extract price and timestamp from the data
        price = self._extract_price(data)
        if price is None:
            logger.warning(f"Could not extract price from data: {data}")
            return None
            
        timestamp = self._extract_timestamp(data)
        symbol = self._extract_symbol(data)
        
        # Increment bar counter
        self.bar_count += 1
        
        # Generate signal according to frequency
        if self.bar_count % self.params['frequency'] == 0:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=price,
                rule_id=self.name,
                symbol=symbol,
                confidence=self.params['confidence'],
                metadata={
                    'bar_count': self.bar_count,
                    'rule': 'AlwaysBuyRule'
                }
            )
            
            self.last_signal = signal
            logger.info(f"Generated BUY signal at price {price} (bar {self.bar_count})")
            return signal
        
        return None
    
    def _extract_price(self, data):
        """Extract price from various data formats."""
        # Case 1: data is a dict with 'Close'
        if isinstance(data, dict) and 'Close' in data:
            return data['Close']
            
        # Case 2: data has 'bar' attribute which is a dict
        if hasattr(data, 'bar') and isinstance(data.bar, dict) and 'Close' in data.bar:
            return data.bar['Close']
            
        # Case 3: data has data attribute with Close
        if hasattr(data, 'data') and isinstance(data.data, dict) and 'Close' in data.data:
            return data.data['Close']
            
        # Failed to extract price
        logger.warning(f"Could not extract price from data type: {type(data)}")
        return None
    
    def _extract_timestamp(self, data):
        """Extract timestamp from various data formats."""
        # Case 1: data is a dict with 'timestamp'
        if isinstance(data, dict) and 'timestamp' in data:
            return data['timestamp']
            
        # Case 2: data has 'bar' attribute which is a dict
        if hasattr(data, 'bar') and isinstance(data.bar, dict) and 'timestamp' in data.bar:
            return data.bar['timestamp']
            
        # Case 3: data has timestamp attribute directly
        if hasattr(data, 'timestamp'):
            return data.timestamp
            
        # Default to current time
        return datetime.datetime.now()
    
    def _extract_symbol(self, data):
        """Extract symbol from various data formats."""
        # Case 1: data is a dict with 'symbol'
        if isinstance(data, dict) and 'symbol' in data:
            return data['symbol']
            
        # Case 2: data has 'bar' attribute which is a dict
        if hasattr(data, 'bar') and isinstance(data.bar, dict) and 'symbol' in data.bar:
            return data.bar['symbol']
            
        # Case 3: data has symbol attribute directly
        if hasattr(data, 'symbol'):
            return data.symbol
            
        # Default symbol
        return "UNKNOWN"
    
    def reset(self):
        """Reset rule state."""
        self.bar_count = 0
        self.last_signal = None
        
    def get_state(self, key=None):
        """
        Get rule state.
        
        Args:
            key (str, optional): State key to retrieve
            
        Returns:
            The value of the specified state key, or a dict of all state if key is None
        """
        state = {
            'bar_count': self.bar_count,
            'last_signal': self.last_signal,
            'params': self.params
        }
        
        if key is not None:
            return state.get(key)
        
        return state

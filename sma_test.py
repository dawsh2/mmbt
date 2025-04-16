"""
Test for SMA Crossover Rule

This module tests the SMACrossoverRule to ensure it properly
handles BarEvent objects and generates SignalEvent objects.
"""

import unittest
import datetime
import numpy as np
from collections import deque

from src.events.event_types import EventType, BarEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent
from src.signals import SignalType


# Create the SMACrossoverRule class for testing
class SMACrossoverRule:
    """
    Rule that generates signals based on moving average crossovers.
    
    This rule compares a fast moving average to a slow moving average
    and generates BUY signals when the fast MA crosses above the slow MA,
    and SELL signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, name, params=None, description=""):
        """
        Initialize the moving average crossover rule.
        
        Args:
            name: Unique identifier for this rule
            params: Parameters including fast_window and slow_window
            description: Human-readable description
        """
        # Initialize parameters
        self.name = name
        self.params = params or {}
        self.description = description or "Moving Average Crossover Rule"
        
        # Extract specific parameters
        self.fast_window = self.params.get('fast_window', 10)
        self.slow_window = self.params.get('slow_window', 30)
        
        # Initialize state
        self.state = {}
        self.signals = []
        self._validate_params()
        
        # Initialize price history
        self.prices = deque(maxlen=max(self.fast_window, self.slow_window) + 1)
        self.last_fast_ma = None
        self.last_slow_ma = None
        
    def _validate_params(self):
        """Validate rule parameters."""
        if not isinstance(self.fast_window, int) or self.fast_window <= 0:
            raise ValueError(f"Fast window must be a positive integer, got {self.fast_window}")
            
        if not isinstance(self.slow_window, int) or self.slow_window <= 0:
            raise ValueError(f"Slow window must be a positive integer, got {self.slow_window}")
            
        if self.fast_window >= self.slow_window:
            raise ValueError(f"Fast window ({self.fast_window}) must be smaller than slow window ({self.slow_window})")
            
    def get_state(self, key=None, default=None):
        """
        Get a value from the rule's state or the entire state dict.
        
        Args:
            key: State dictionary key, or None to get the entire state
            default: Default value if key is not found
            
        Returns:
            The value from state or default, or the entire state dict
        """
        if key is None:
            return self.state
        return self.state.get(key, default)
        
    def update_state(self, key, value):
        """
        Update the rule's internal state.
        
        Args:
            key: State dictionary key
            value: Value to store
        """
        self.state[key] = value
    
    def on_bar(self, event):
        """
        Process a bar event and generate a trading signal.
        
        Args:
            event: Event containing a BarEvent in its data attribute
            
        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        # Extract BarEvent properly
        if not isinstance(event, Event):
            return None

        # Case 1: Event data is a BarEvent
        if isinstance(event.data, BarEvent):
            bar_event = event.data
        # Case 2: Event data is a dict (backward compatibility)
        elif isinstance(event.data, dict) and 'Close' in event.data:
            # Convert dict to BarEvent
            bar_event = BarEvent(event.data)
        else:
            return None
        
        # Generate signal
        try:
            signal = self.generate_signals(bar_event)
            
            # Store in history
            if signal is not None:
                self.signals.append(signal)
            
            return signal
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None

    def generate_signals(self, bar_event):
        """
        Generate trading signals based on moving average crossovers.

        Args:
            bar_event: BarEvent containing market data

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        # Extract price and append to price history
        price = bar_event.get_price()
        if price is None:
            return None

        self.prices.append(price)

        # Check if we have enough prices
        if len(self.prices) < self.slow_window:
            # Not enough data yet
            return None

        # Calculate moving averages
        fast_ma = np.mean(list(self.prices)[-self.fast_window:])
        slow_ma = np.mean(list(self.prices)[-self.slow_window:])

        signal = None

        # Only check for crossovers if we have previous values
        if self.last_fast_ma is not None and self.last_slow_ma is not None:
            # Check for bullish crossover (fast crosses above slow)
            if self.last_fast_ma <= self.last_slow_ma and fast_ma > slow_ma:
                print(f"  BULLISH CROSSOVER DETECTED: {self.last_fast_ma} -> {fast_ma}, {self.last_slow_ma} -> {slow_ma}")
                signal = self._create_signal(SignalType.BUY, bar_event, fast_ma, slow_ma)
            # Check for bearish crossover (fast crosses below slow)
            elif self.last_fast_ma >= self.last_slow_ma and fast_ma < slow_ma:
                print(f"  BEARISH CROSSOVER DETECTED: {self.last_fast_ma} -> {fast_ma}, {self.last_slow_ma} -> {slow_ma}")
                signal = self._create_signal(SignalType.SELL, bar_event, fast_ma, slow_ma)

        # Store current values for the next time
        self.last_fast_ma = fast_ma
        self.last_slow_ma = slow_ma

        return signal

    def _create_signal(self, signal_type, bar_event, fast_ma, slow_ma):
        """
        Create a signal event.

        Args:
            signal_type: Type of signal to create
            bar_event: The bar event that triggered the signal
            fast_ma: Current fast moving average value
            slow_ma: Current slow moving average value

        Returns:
            SignalEvent object
        """
        # Create metadata with relevant information
        metadata = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'price': bar_event.get_price(),
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
            'symbol': bar_event.get_symbol(),
            'confidence': min(1.0, abs(fast_ma - slow_ma) / ((fast_ma + slow_ma) / 2 * 0.01))
        }

        # Create and return the signal
        return SignalEvent(
            signal_value=signal_type.value,  # Changed from signal_type to signal_value
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            metadata=metadata,  # Move confidence into metadata
            timestamp=bar_event.get_timestamp()
        )
    



        
    def reset(self):
        """Reset the rule's state."""
        self.state = {}
        self.signals = []
        self.prices.clear()
        self.last_fast_ma = None
        self.last_slow_ma = None


class TestSMACrossoverRule(unittest.TestCase):
    """Test suite for the SMACrossoverRule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule = SMACrossoverRule(
            name="test_ma_crossover",
            params={
                'fast_window': 3,
                'slow_window': 5
            }
        )
    
    def test_initialization(self):
        """Test rule initialization."""
        # Check parameters were correctly set
        self.assertEqual(self.rule.fast_window, 3)
        self.assertEqual(self.rule.slow_window, 5)
        self.assertEqual(self.rule.name, "test_ma_crossover")
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid fast window
        with self.assertRaises(ValueError):
            SMACrossoverRule(
                name="test_invalid_fast",
                params={
                    'fast_window': 0,
                    'slow_window': 5
                }
            )
            
        # Test invalid slow window
        with self.assertRaises(ValueError):
            SMACrossoverRule(
                name="test_invalid_slow",
                params={
                    'fast_window': 3,
                    'slow_window': -1
                }
            )
            
        # Test fast >= slow
        with self.assertRaises(ValueError):
            SMACrossoverRule(
                name="test_invalid_relation",
                params={
                    'fast_window': 5,
                    'slow_window': 5
                }
            )
    
    def create_bar_event(self, price, timestamp=None, symbol="TEST"):
        """Create a BarEvent with the given price."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        bar_data = {
            "timestamp": timestamp,
            "Open": price * 0.99,
            "High": price * 1.01,
            "Low": price * 0.98,
            "Close": price,
            "Volume": 1000,
            "symbol": symbol
        }
        
        return BarEvent(bar_data)
    
    def create_event(self, bar_event):
        """Create an Event containing a BarEvent."""
        return Event(EventType.BAR, bar_event)
    
    def test_not_enough_data(self):
        """Test behavior when not enough data is available."""
        # Create events for just 4 bars (not enough for slow MA of 5)
        prices = [100.0, 101.0, 102.0, 103.0]
        
        for price in prices:
            bar_event = self.create_bar_event(price)
            event = self.create_event(bar_event)
            signal = self.rule.on_bar(event)
            
            # Should not generate signal yet
            self.assertIsNone(signal)

    def test_bullish_crossover(self):
        """Test detection of bullish crossover."""
        # Start with a downtrend (slow MA above fast MA)
        # Then create an uptrend to cause a crossover
        prices = [100.0, 98.0, 96.0, 94.0, 92.0, 100.0, 120.0]

        for i, price in enumerate(prices):
            bar_event = self.create_bar_event(price)
            event = self.create_event(bar_event)
            signal = self.rule.on_bar(event)

            # Print debug info
            print(f"Bar {i+1}, Price: {price}")
            print(f"  Fast MA: {self.rule.last_fast_ma}")
            print(f"  Slow MA: {self.rule.last_slow_ma}")

            if self.rule.last_fast_ma is not None and self.rule.last_slow_ma is not None:
                print(f"  Fast MA > Slow MA: {self.rule.last_fast_ma > self.rule.last_slow_ma}")

            print(f"  Signal generated: {signal is not None}")

            # In this sequence, the signal should happen on bar 7 (index 6)
            if i == 6:
                # Should get a BUY signal on the 7th bar
                self.assertIsNotNone(signal, f"Should generate BUY signal at bar {i+1}")
                if signal is not None:
                    self.assertIsInstance(signal, SignalEvent)
                    self.assertEqual(signal.get_signal_value(), SignalEvent.BUY)  # Uses the BUY constant from SignalEvent
            else:
                # No signal on other bars
                self.assertIsNone(signal, f"Should not generate signal at bar {i+1}")
            


    
    def test_reset(self):
        """Test resetting the rule state."""
        # Add some prices
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        for price in prices:
            bar_event = self.create_bar_event(price)
            event = self.create_event(bar_event)
            self.rule.on_bar(event)
            
        # Verify internal state has data
        self.assertEqual(len(self.rule.prices), 5)
        self.assertIsNotNone(self.rule.last_fast_ma)
        self.assertIsNotNone(self.rule.last_slow_ma)
        
        # Reset the rule
        self.rule.reset()
        
        # Verify state is cleared
        self.assertEqual(len(self.rule.prices), 0)
        self.assertIsNone(self.rule.last_fast_ma)
        self.assertIsNone(self.rule.last_slow_ma)


if __name__ == "__main__":
    unittest.main()

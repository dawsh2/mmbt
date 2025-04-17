"""
Standalone SMACrossoverRule implementation that doesn't rely on rule registration.

This implementation can be included directly in a test script without requiring imports
from the rule registry system.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from src.rules.rule_registry import register_rule
from functools import wraps

# Import necessary base components
from src.events.event_types import BarEvent, EventType
from src.events.event_base import Event
from src.events.signal_event import SignalEvent
from src.rules.rule_base import Rule

# Set up logging
logger = logging.getLogger(__name__)


@register_rule(category="crossover")
class StandaloneSMACrossoverRule(Rule):
    """
    Simple Moving Average crossover rule.

    Generates buy signals when the fast SMA crosses above the slow SMA,
    and sell signals when it crosses below.
    """

    def __init__(self, name: str, params: Dict[str, Any] = None, 
                 description: str = "", event_bus=None):
        """
        Initialize SMA crossover rule.

        Args:
            name: Rule name
            params: Rule parameters including:
                - fast_window: Window size for fast SMA (default: 10)
                - slow_window: Window size for slow SMA (default: 30)
            description: Rule description
            event_bus: Optional event bus for emitting signals
        """
        # Default parameters 
        default_params = {
            'fast_window': 10, 
            'slow_window': 30
        }
        
        # Merge with provided parameters
        if params:
            default_params.update(params)
            
        # Initialize base class
        super().__init__(name, default_params, description, event_bus)
        
        # Initialize state to store price history and SMA values
        self.state = {
            'prices': [],
            'fast_sma': None,
            'slow_sma': None,
            'previous_fast_sma': None,
            'previous_slow_sma': None,
            'signals_generated': 0,
            'last_signal_time': None,
            'last_signal_price': None
        }
    
    def on_bar(self, event: Event) -> Optional[SignalEvent]:
        """
        Process a bar event and generate a trading signal directly.
        
        This method bypasses the base class's event handling to ensure direct control 
        over signal generation and event emission.
        
        Args:
            event: Event containing a BarEvent in its data attribute
            
        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        # Extract BarEvent with type checking
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event).__name__}")
            return None

        # Extract the bar event
        if isinstance(event.data, BarEvent):
            bar_event = event.data
        elif isinstance(event.data, dict) and 'Close' in event.data:
            # Convert dict to BarEvent
            bar_event = BarEvent(event.data)
            logger.warning(f"Rule {self.name}: Received dictionary instead of BarEvent, converting")
        else:
            logger.error(f"Rule {self.name}: Unable to extract BarEvent from {type(event.data).__name__}")
            return None
        
        # Generate signal directly without going through base class
        try:
            signal = self.generate_signal(bar_event)
            
            # If signal was generated, store it
            if signal is not None:
                # Store in history
                self.signals.append(signal)
                logger.info(f"Rule {self.name}: Generated {signal.get_signal_name()} signal")
                
                # Emit signal event if we have an event bus
                self._emit_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Rule {self.name}: Error generating signal: {str(e)}", exc_info=True)
            return None

    def generate_signal(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Generate a signal based on SMA crossover.

        This method implements the SMA crossover strategy logic:
        1. Update price history with the latest price
        2. Calculate fast and slow SMAs if enough data
        3. Check for crossover between SMAs
        4. Generate appropriate signals on crossover

        Args:
            bar_event: BarEvent containing market data
            
        Returns:
            SignalEvent if crossover occurs, None otherwise
        """
        # Extract data from bar event
        try:
            # Get price from specified field (default: 'Close')
            price_field = self.params.get('price_field', 'Close')
            
            # Try to get price directly from bar data
            bar_data = bar_event.get_data()
            if isinstance(bar_data, dict) and price_field in bar_data:
                close_price = bar_data[price_field]
            else:
                # Fallback to get_price() method
                close_price = bar_event.get_price()
                
            timestamp = bar_event.get_timestamp()
            symbol = bar_event.get_symbol()
            
            logger.debug(f"SMA Rule {self.name}: Processing bar for {symbol} @ {timestamp}, {price_field}: {close_price}")
        
        except Exception as e:
            logger.error(f"SMA Rule {self.name}: Error extracting data from bar event: {e}")
            return None
        
        # Get parameters
        fast_window = self.params['fast_window']
        slow_window = self.params['slow_window']
        
        # Update price history
        self.state['prices'].append(close_price)
        
        # Log price history for debugging
        logger.debug(f"Rule {self.name}: Added price {close_price}, history size: {len(self.state['prices'])}")
        
        # Keep only the necessary price history
        max_window = max(fast_window, slow_window)
        if len(self.state['prices']) > max_window + 10:  # Keep a few extra points
            self.state['prices'] = self.state['prices'][-(max_window + 10):]
        
        # Calculate SMAs if we have enough data
        if len(self.state['prices']) >= slow_window:
            # Store previous SMAs for crossover detection
            self.state['previous_fast_sma'] = self.state['fast_sma']
            self.state['previous_slow_sma'] = self.state['slow_sma']
            
            # Calculate new SMAs
            prices_array = np.array(self.state['prices'])
            self.state['fast_sma'] = np.mean(prices_array[-fast_window:])
            self.state['slow_sma'] = np.mean(prices_array[-slow_window:])
            
            fast_sma = self.state['fast_sma']
            slow_sma = self.state['slow_sma']
            prev_fast_sma = self.state['previous_fast_sma']
            prev_slow_sma = self.state['previous_slow_sma']
            
            logger.debug(f"SMA Rule {self.name}: Fast SMA = {fast_sma:.4f}, Slow SMA = {slow_sma:.4f}")
            
            # Check for crossover (and make sure we have previous values)
            if prev_fast_sma is not None and prev_slow_sma is not None:
                # Calculate differences to detect crossovers
                prev_diff = prev_fast_sma - prev_slow_sma
                curr_diff = fast_sma - slow_sma
                
                logger.debug(f"Rule {self.name}: Crossover check - Prev diff: {prev_diff:.6f}, Curr diff: {curr_diff:.6f}")
                
                # Create metadata for signal
                metadata = {
                    'rule': self.name,
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'fast_window': fast_window,
                    'slow_window': slow_window,
                    'symbol': symbol
                }
                
                # Bullish crossover (fast SMA crosses above slow SMA)
                if prev_diff <= 0 and curr_diff > 0:
                    logger.info(f"SMA Rule {self.name}: Bullish crossover for {symbol} @ {timestamp}")
                    
                    # Update state tracking
                    self.state['signals_generated'] += 1
                    self.state['last_signal_time'] = timestamp
                    self.state['last_signal_price'] = close_price
                    
                    metadata['reason'] = 'bullish_crossover'
                    
                    # Create signal
                    signal = SignalEvent(
                        signal_value=SignalEvent.BUY,
                        price=close_price,
                        symbol=symbol,
                        rule_id=self.name,
                        metadata=metadata,
                        timestamp=timestamp
                    )
                    
                    return signal
                
                # Bearish crossover (fast SMA crosses below slow SMA)
                elif prev_diff >= 0 and curr_diff < 0:
                    logger.info(f"SMA Rule {self.name}: Bearish crossover for {symbol} @ {timestamp}")
                    
                    # Update state tracking
                    self.state['signals_generated'] += 1
                    self.state['last_signal_time'] = timestamp
                    self.state['last_signal_price'] = close_price
                    
                    metadata['reason'] = 'bearish_crossover'
                    
                    # Create signal
                    signal = SignalEvent(
                        signal_value=SignalEvent.SELL,
                        price=close_price,
                        symbol=symbol,
                        rule_id=self.name,
                        metadata=metadata,
                        timestamp=timestamp
                    )
                    
                    return signal
                
                # Optional: Generate continuous signals based on current alignment
                elif self.params.get('smooth_signals', False):
                    if curr_diff > 0:
                        # Fast above slow - bullish
                        metadata['reason'] = 'bullish_alignment'
                        
                        signal = SignalEvent(
                            signal_value=SignalEvent.BUY,
                            price=close_price,
                            symbol=symbol,
                            rule_id=self.name,
                            metadata=metadata,
                            timestamp=timestamp
                        )
                        
                        return signal
                    else:
                        # Fast below slow - bearish
                        metadata['reason'] = 'bearish_alignment'
                        
                        signal = SignalEvent(
                            signal_value=SignalEvent.SELL,
                            price=close_price,
                            symbol=symbol,
                            rule_id=self.name,
                            metadata=metadata,
                            timestamp=timestamp
                        )
                        
                        return signal
        
        # No signal (not enough data or no crossover)
        return None

    def _emit_signal(self, signal):
        """
        Emit a signal event to the event bus if available.
        
        Args:
            signal: SignalEvent to emit
        """
        if hasattr(self, 'event_bus') and self.event_bus is not None:
            try:
                signal_event = Event(EventType.SIGNAL, signal)
                self.event_bus.emit(signal_event)
                logger.debug(f"Rule {self.name}: Emitted signal event: {signal_event}")
            except Exception as e:
                logger.error(f"Rule {self.name}: Error emitting signal: {str(e)}")
        
    def reset(self) -> None:
        """Reset the rule's internal state."""
        self.state = {
            'prices': [],
            'fast_sma': None,
            'slow_sma': None,
            'previous_fast_sma': None,
            'previous_slow_sma': None,
            'signals_generated': 0,
            'last_signal_time': None,
            'last_signal_price': None
        }
        # Also reset the signals list in the base class
        self.signals = []
# Previous version kept for reference 
# @register_rule(category="crossover")
# class SMAcrossoverRule(Rule):
#     """
#     Simple Moving Average (SMA) Crossover Rule.
    
#     This rule generates buy signals when a faster SMA crosses above a slower SMA,
#     and sell signals when the faster SMA crosses below the slower SMA.
#     """
    
#     def __init__(self, 
#                  name: str = "sma_crossover", 
#                  params: Optional[Dict[str, Any]] = None,
#                  description: str = "SMA crossover rule"):
#         """
#         Initialize the SMA crossover rule.
        
#         Args:
#             name: Rule name
#             params: Dictionary containing:
#                 - fast_window: Window size for fast SMA (default: 5)
#                 - slow_window: Window size for slow SMA (default: 20)
#                 - smooth_signals: Whether to generate signals when MAs are aligned (default: False)
#             description: Rule description
#         """
#         super().__init__(name, params or self.default_params(), description)
#         self.prices = deque(maxlen=max(self.params['fast_window'], self.params['slow_window']) + 10)
#         self.fast_sma_history = deque(maxlen=10)
#         self.slow_sma_history = deque(maxlen=10)
#         self.current_signal_type = SignalType.NEUTRAL
    
#     @classmethod
#     def default_params(cls) -> Dict[str, Any]:
#         """Default parameters for the rule."""
#         return {
#             'fast_window': 5,
#             'slow_window': 20,
#             'smooth_signals': False
#         }
    
#     def _validate_params(self) -> None:
#         """Validate the parameters for this rule."""
#         if self.params['fast_window'] >= self.params['slow_window']:
#             raise ValueError("Fast window must be smaller than slow window")
        
#         if self.params['fast_window'] <= 0 or self.params['slow_window'] <= 0:
#             raise ValueError("Window sizes must be positive")
    
#     def generate_signal(self, data: Dict[str, Any]) -> Signal:
#         """
#         Generate a trading signal based on SMA crossover.
        
#         Args:
#             data: Dictionary containing price data
                 
#         Returns:
#             Signal object representing the trading decision
#         """
#         # Check for required data
#         if 'Close' not in data:
#             return Signal(
#                 timestamp=data.get('timestamp', None),
#                 signal_type=SignalType.NEUTRAL,
#                 price=None,
#                 rule_id=self.name,
#                 confidence=0.0,
#                 metadata={
#                     'error': 'Missing Close price data',
#                     'symbol': data.get('symbol', 'default')
#                 }
#             )
            
#         # Get parameters
#         fast_window = self.params['fast_window']
#         slow_window = self.params['slow_window']
#         smooth_signals = self.params['smooth_signals']
        
#         # Extract price data
#         close = data['Close']
#         timestamp = data.get('timestamp', None)

#         # Extract symbol
#         symbol = data.get('symbol', 'default')
        
#         # Update price history
#         self.prices.append(close)
        
#         # Calculate SMAs
#         if len(self.prices) >= slow_window:
#             try:
#                 fast_sma = sum(list(self.prices)[-fast_window:]) / fast_window
#                 slow_sma = sum(list(self.prices)[-slow_window:]) / slow_window
                
#                 # Store in history
#                 self.fast_sma_history.append(fast_sma)
#                 self.slow_sma_history.append(slow_sma)
                
#                 # Generate signals
#                 if len(self.fast_sma_history) >= 2 and len(self.slow_sma_history) >= 2:
#                     current_fast = self.fast_sma_history[-1]
#                     current_slow = self.slow_sma_history[-1]
#                     prev_fast = self.fast_sma_history[-2]
#                     prev_slow = self.slow_sma_history[-2]
                    
#                     # Log current values for debugging
#                     if hasattr(self, 'logger'):
#                         self.logger.debug(f"Symbol: {symbol}, Fast SMA: {current_fast:.2f}, Slow SMA: {current_slow:.2f}")
#                         self.logger.debug(f"Previous Fast: {prev_fast:.2f}, Previous Slow: {prev_slow:.2f}")
                    
#                     # Check for crossover
#                     if prev_fast <= prev_slow and current_fast > current_slow:
#                         self.current_signal_type = SignalType.BUY
#                         if hasattr(self, 'logger'):
#                             self.logger.info(f"Bullish crossover detected for {symbol}")
#                     elif prev_fast >= prev_slow and current_fast < current_slow:
#                         self.current_signal_type = SignalType.SELL
#                         if hasattr(self, 'logger'):
#                             self.logger.info(f"Bearish crossover detected for {symbol}")
#                     elif smooth_signals:
#                         # If smooth signals are enabled, maintain signal based on MA relationship
#                         if current_fast > current_slow:
#                             self.current_signal_type = SignalType.BUY
#                         elif current_fast < current_slow:
#                             self.current_signal_type = SignalType.SELL
#                         else:
#                             self.current_signal_type = SignalType.NEUTRAL
#                     else:
#                         # Otherwise, revert to neutral after crossover
#                         self.current_signal_type = SignalType.NEUTRAL
                    
#                     # Calculate confidence based on distance between MAs
#                     if current_slow != 0:
#                         distance = abs(current_fast - current_slow) / current_slow
#                         confidence = min(1.0, distance * 10)  # Scale distance for confidence
#                     else:
#                         confidence = 0.5

#                     return Signal(
#                         timestamp=timestamp,
#                         signal_type=self.current_signal_type,
#                         price=close,
#                         rule_id=self.name,
#                         confidence=confidence,
#                         metadata={
#                             'fast_sma': current_fast,
#                             'slow_sma': current_slow,
#                             'distance': current_fast - current_slow,
#                             'symbol': symbol  # Add symbol to metadata
#                         }
#                     )
#             except Exception as e:
#                 # Handle any unexpected errors during calculation
#                 if hasattr(self, 'logger'):
#                     self.logger.error(f"Error calculating SMA: {e}")
#                 return Signal(
#                     timestamp=timestamp,
#                     signal_type=SignalType.NEUTRAL,
#                     price=close,
#                     rule_id=self.name,
#                     confidence=0.0,
#                     metadata={
#                         'error': f"Calculation error: {str(e)}",
#                         'symbol': symbol
#                     }
#                 )

#         # Not enough data yet, return neutral signal
#         return Signal(
#             timestamp=timestamp,
#             signal_type=SignalType.NEUTRAL,
#             price=close,
#             rule_id=self.name,
#             confidence=0.0,
#             metadata={
#                 'status': 'initializing',
#                 'symbol': symbol  # Add symbol to metadata
#             }
#         )
    
#     def reset(self) -> None:
#         """Reset the rule's internal state."""
#         super().reset()
#         self.prices = deque(maxlen=max(self.params['fast_window'], self.params['slow_window']) + 10)
#         self.fast_sma_history = deque(maxlen=10)
#         self.slow_sma_history = deque(maxlen=10)
#         self.current_signal_type = SignalType.NEUTRAL
        
#     def __str__(self) -> str:
#         """String representation of the rule."""
#         return f"{self.name} (Fast: {self.params['fast_window']}, Slow: {self.params['slow_window']})"

@register_rule(category="crossover")
class ExponentialMACrossoverRule(Rule):
    """
    Exponential Moving Average (EMA) Crossover Rule.
    
    This rule generates buy signals when a faster EMA crosses above a slower EMA,
    and sell signals when the faster EMA crosses below the slower EMA.
    """
    
    def __init__(self, 
                 name: str = "ema_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "EMA crossover rule"):
        """
        Initialize the EMA crossover rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - fast_period: Period for fast EMA (default: 12)
                - slow_period: Period for slow EMA (default: 26)
                - smooth_signals: Whether to generate signals when MAs are aligned (default: False)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=max(self.params['fast_period'], self.params['slow_period']) * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.fast_ema_history = deque(maxlen=10)
        self.slow_ema_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'smooth_signals': False
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError("Fast period must be smaller than slow period")
        
        if self.params['fast_period'] <= 0 or self.params['slow_period'] <= 0:
            raise ValueError("Periods must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on EMA crossover.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        smooth_signals = self.params['smooth_signals']
        
        # Extract price data
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.prices.append(close)
        
        # Calculate EMAs
        if self.fast_ema is None and len(self.prices) >= fast_period:
            # Initialize EMAs with SMA
            self.fast_ema = sum(list(self.prices)[-fast_period:]) / fast_period
            
        if self.slow_ema is None and len(self.prices) >= slow_period:
            # Initialize EMAs with SMA
            self.slow_ema = sum(list(self.prices)[-slow_period:]) / slow_period
            
        # Update EMAs
        if self.fast_ema is not None:
            alpha_fast = 2 / (fast_period + 1)
            self.fast_ema = (close * alpha_fast) + (self.fast_ema * (1 - alpha_fast))
            self.fast_ema_history.append(self.fast_ema)
            
        if self.slow_ema is not None:
            alpha_slow = 2 / (slow_period + 1)
            self.slow_ema = (close * alpha_slow) + (self.slow_ema * (1 - alpha_slow))
            self.slow_ema_history.append(self.slow_ema)
        
        # Generate signals
        if self.fast_ema is not None and self.slow_ema is not None:
            if len(self.fast_ema_history) >= 2 and len(self.slow_ema_history) >= 2:
                current_fast = self.fast_ema_history[-1]
                current_slow = self.slow_ema_history[-1]
                prev_fast = self.fast_ema_history[-2]
                prev_slow = self.slow_ema_history[-2]
                
                # Check for crossover
                if prev_fast <= prev_slow and current_fast > current_slow:
                    self.current_signal_type = SignalType.BUY
                elif prev_fast >= prev_slow and current_fast < current_slow:
                    self.current_signal_type = SignalType.SELL
                elif smooth_signals:
                    # If smooth signals are enabled, maintain signal based on EMA relationship
                    if current_fast > current_slow:
                        self.current_signal_type = SignalType.BUY
                    elif current_fast < current_slow:
                        self.current_signal_type = SignalType.SELL
                    else:
                        self.current_signal_type = SignalType.NEUTRAL
                else:
                    # Otherwise, revert to neutral after crossover
                    self.current_signal_type = SignalType.NEUTRAL
                
                # Calculate confidence based on distance between EMAs
                if current_slow != 0:
                    distance = abs(current_fast - current_slow) / current_slow
                    confidence = min(1.0, distance * 10)  # Scale distance for confidence
                else:
                    confidence = 0.5
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=self.current_signal_type,
                    price=close,
                    rule_id=self.name,
                    confidence=confidence,
                    metadata={
                        'fast_ema': current_fast,
                        'slow_ema': current_slow,
                        'distance': current_fast - current_slow
                    }
                )
        
        # Not enough data yet, return neutral signal
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.NEUTRAL,
            price=close,
            rule_id=self.name,
            confidence=0.0,
            metadata={'status': 'initializing'}
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=max(self.params['fast_period'], self.params['slow_period']) * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.fast_ema_history = deque(maxlen=10)
        self.slow_ema_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="crossover")
class MACDCrossoverRule(Rule):
    """
    Moving Average Convergence Divergence (MACD) Crossover Rule.
    
    This rule generates buy signals when the MACD line crosses above the signal line,
    and sell signals when the MACD line crosses below the signal line.
    """
    
    def __init__(self, 
                 name: str = "macd_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "MACD crossover rule"):
        """
        Initialize the MACD crossover rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - fast_period: Period for fast EMA (default: 12)
                - slow_period: Period for slow EMA (default: 26)
                - signal_period: Period for signal line (default: 9)
                - use_histogram: Whether to use MACD histogram for signals (default: False)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['slow_period'] * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.macd_line = deque(maxlen=self.params['signal_period'] * 3)
        self.signal_line = None
        self.signal_line_history = deque(maxlen=10)
        self.macd_history = deque(maxlen=10)
        self.histogram_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'use_histogram': False
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError("Fast period must be smaller than slow period")
        
        if self.params['fast_period'] <= 0 or self.params['slow_period'] <= 0 or self.params['signal_period'] <= 0:
            raise ValueError("Periods must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on MACD crossover.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        use_histogram = self.params['use_histogram']
        
        # Extract price data
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.prices.append(close)
        
        # Calculate EMAs for MACD
        if self.fast_ema is None and len(self.prices) >= fast_period:
            # Initialize EMAs with SMA
            self.fast_ema = sum(list(self.prices)[-fast_period:]) / fast_period
            
        if self.slow_ema is None and len(self.prices) >= slow_period:
            # Initialize EMAs with SMA
            self.slow_ema = sum(list(self.prices)[-slow_period:]) / slow_period
            
        # Update EMAs
        if self.fast_ema is not None:
            alpha_fast = 2 / (fast_period + 1)
            self.fast_ema = (close * alpha_fast) + (self.fast_ema * (1 - alpha_fast))
            
        if self.slow_ema is not None:
            alpha_slow = 2 / (slow_period + 1)
            self.slow_ema = (close * alpha_slow) + (self.slow_ema * (1 - alpha_slow))
        
        # Calculate MACD line
        if self.fast_ema is not None and self.slow_ema is not None:
            macd = self.fast_ema - self.slow_ema
            self.macd_line.append(macd)
            self.macd_history.append(macd)
            
            # Calculate signal line (EMA of MACD)
            if self.signal_line is None and len(self.macd_line) >= signal_period:
                # Initialize signal line with SMA
                self.signal_line = sum(list(self.macd_line)[-signal_period:]) / signal_period
                
            if self.signal_line is not None:
                alpha_signal = 2 / (signal_period + 1)
                self.signal_line = (macd * alpha_signal) + (self.signal_line * (1 - alpha_signal))
                self.signal_line_history.append(self.signal_line)
                
                # Calculate histogram
                histogram = macd - self.signal_line
                self.histogram_history.append(histogram)
                
                # Generate signals
                if len(self.macd_history) >= 2 and len(self.signal_line_history) >= 2:
                    if use_histogram:
                        # Use histogram for signals
                        if len(self.histogram_history) >= 2:
                            current_hist = self.histogram_history[-1]
                            prev_hist = self.histogram_history[-2]
                            
                            if prev_hist <= 0 and current_hist > 0:
                                self.current_signal_type = SignalType.BUY
                            elif prev_hist >= 0 and current_hist < 0:
                                self.current_signal_type = SignalType.SELL
                            else:
                                # Maintain current signal
                                pass
                    else:
                        # Use MACD crossover for signals
                        current_macd = self.macd_history[-1]
                        current_signal = self.signal_line_history[-1]
                        prev_macd = self.macd_history[-2]
                        prev_signal = self.signal_line_history[-2]
                        
                        if prev_macd <= prev_signal and current_macd > current_signal:
                            self.current_signal_type = SignalType.BUY
                        elif prev_macd >= prev_signal and current_macd < current_signal:
                            self.current_signal_type = SignalType.SELL
                        else:
                            # Maintain current signal
                            pass
                    
                    # Calculate confidence based on histogram size
                    if len(self.histogram_history) > 0:
                        # Use absolute histogram value for confidence
                        abs_hist = abs(self.histogram_history[-1])
                        # Scale it reasonably
                        confidence = min(1.0, abs_hist * 20)
                    else:
                        confidence = 0.5
                    
                    return Signal(
                        timestamp=timestamp,
                        signal_type=self.current_signal_type,
                        price=close,
                        rule_id=self.name,
                        confidence=confidence,
                        metadata={
                            'macd': macd,
                            'signal': self.signal_line,
                            'histogram': histogram
                        }
                    )
        
        # Not enough data yet, return neutral signal
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.NEUTRAL,
            price=close,
            rule_id=self.name,
            confidence=0.0,
            metadata={'status': 'initializing'}
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['slow_period'] * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.macd_line = deque(maxlen=self.params['signal_period'] * 3)
        self.signal_line = None
        self.signal_line_history = deque(maxlen=10)
        self.macd_history = deque(maxlen=10)
        self.histogram_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="crossover")
class PriceMACrossoverRule(Rule):
    """
    Price-Moving Average Crossover Rule.
    
    This rule generates buy signals when the price crosses above a moving average,
    and sell signals when the price crosses below.
    """
    
    def __init__(self, 
                 name: str = "price_ma_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Price-MA crossover rule"):
        """
        Initialize the Price-MA crossover rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - ma_period: Period for the moving average (default: 20)
                - ma_type: Type of moving average ('sma', 'ema') (default: 'sma')
                - smooth_signals: Whether to generate signals when price and MA are aligned (default: False)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['ma_period'] * 3)
        self.ma_value = None
        self.ma_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'ma_period': 20,
            'ma_type': 'sma',
            'smooth_signals': False
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['ma_period'] <= 0:
            raise ValueError("MA period must be positive")
            
        valid_ma_types = ['sma', 'ema']
        if self.params['ma_type'] not in valid_ma_types:
            raise ValueError(f"MA type must be one of {valid_ma_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on price-MA crossover.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        ma_period = self.params['ma_period']
        ma_type = self.params['ma_type']
        smooth_signals = self.params['smooth_signals']
        
        # Extract price data
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.prices.append(close)
        
        # Calculate MA
        if ma_type == 'sma':
            # Simple Moving Average
            if len(self.prices) >= ma_period:
                self.ma_value = sum(list(self.prices)[-ma_period:]) / ma_period
                self.ma_history.append(self.ma_value)
        elif ma_type == 'ema':
            # Exponential Moving Average
            if self.ma_value is None and len(self.prices) >= ma_period:
                # Initialize EMA with SMA
                self.ma_value = sum(list(self.prices)[-ma_period:]) / ma_period
                
            if self.ma_value is not None:
                alpha = 2 / (ma_period + 1)
                self.ma_value = (close * alpha) + (self.ma_value * (1 - alpha))
                self.ma_history.append(self.ma_value)
        
        # Generate signals
        if self.ma_value is not None and len(self.ma_history) >= 2 and len(self.prices) >= 2:
            current_ma = self.ma_history[-1]
            prev_ma = self.ma_history[-2]
            prev_price = self.prices[-2]
            
            # Check for crossover
            if prev_price <= prev_ma and close > current_ma:
                self.current_signal_type = SignalType.BUY
            elif prev_price >= prev_ma and close < current_ma:
                self.current_signal_type = SignalType.SELL
            elif smooth_signals:
                # If smooth signals are enabled, maintain signal based on price/MA relationship
                if close > current_ma:
                    self.current_signal_type = SignalType.BUY
                elif close < current_ma:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                # Otherwise, revert to neutral after crossover
                self.current_signal_type = SignalType.NEUTRAL
            
            # Calculate confidence based on distance from MA
            if current_ma != 0:
                distance = abs(close - current_ma) / current_ma
                confidence = min(1.0, distance * 10)  # Scale distance for confidence
            else:
                confidence = 0.5
            
            return Signal(
                timestamp=timestamp,
                signal_type=self.current_signal_type,
                price=close,
                rule_id=self.name,
                confidence=confidence,
                metadata={
                    'ma_value': current_ma,
                    'distance': close - current_ma
                }
            )
        
        # Not enough data yet, return neutral signal
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.NEUTRAL,
            price=close,
            rule_id=self.name,
            confidence=0.0,
            metadata={'status': 'initializing'}
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['ma_period'] * 3)
        self.ma_value = None
        self.ma_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL



        
@register_rule(category="crossover")
class BollingerBandsCrossoverRule(Rule):
    """
    Bollinger Bands Crossover Rule.
    
    This rule generates buy signals when price crosses below the lower band
    and sell signals when price crosses above the upper band.
    """
    
    def __init__(self, 
                 name: str = "bollinger_bands_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Bollinger Bands crossover rule"):
        """
        Initialize the Bollinger Bands crossover rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - period: Period for the moving average (default: 20)
                - num_std_dev: Number of standard deviations for bands (default: 2.0)
                - use_middle_band: Whether to also generate signals on middle band crosses (default: False)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['period'] * 3)
        self.upper_band_history = deque(maxlen=10)
        self.middle_band_history = deque(maxlen=10)
        self.lower_band_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'period': 20,
            'num_std_dev': 2.0,
            'use_middle_band': False
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['period'] <= 0:
            raise ValueError("Period must be positive")
            
        if self.params['num_std_dev'] <= 0:
            raise ValueError("Number of standard deviations must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Bollinger Bands crossover.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        period = self.params['period']
        num_std_dev = self.params['num_std_dev']
        use_middle_band = self.params['use_middle_band']
        
        # Extract price data
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.prices.append(close)
        
        # Calculate Bollinger Bands
        if len(self.prices) >= period:
            # Calculate SMA (middle band)
            middle_band = sum(list(self.prices)[-period:]) / period
            
            # Calculate standard deviation
            variance = sum((x - middle_band) ** 2 for x in list(self.prices)[-period:]) / period
            std_dev = np.sqrt(variance)
            
            # Calculate upper and lower bands
            upper_band = middle_band + (num_std_dev * std_dev)
            lower_band = middle_band - (num_std_dev * std_dev)
            
            # Store in history
            self.upper_band_history.append(upper_band)
            self.middle_band_history.append(middle_band)
            self.lower_band_history.append(lower_band)
            
            # Generate signals
            if len(self.prices) >= period + 1 and len(self.upper_band_history) >= 2:
                prev_price = self.prices[-2]
                prev_upper = self.upper_band_history[-2]
                prev_middle = self.middle_band_history[-2]
                prev_lower = self.lower_band_history[-2]
                
                # Check for band crossovers
                if prev_price >= prev_lower and close < lower_band:
                    # Price crossed below lower band - typically a buy signal
                    self.current_signal_type = SignalType.BUY
                elif prev_price <= prev_upper and close > upper_band:
                    # Price crossed above upper band - typically a sell signal
                    self.current_signal_type = SignalType.SELL
                elif use_middle_band:
                    # Optionally handle middle band crosses
                    if prev_price <= prev_middle and close > middle_band:
                        self.current_signal_type = SignalType.BUY
                    elif prev_price >= prev_middle and close < middle_band:
                        self.current_signal_type = SignalType.SELL
                
                # Calculate confidence based on distance from bands
                band_width = upper_band - lower_band
                if band_width > 0:
                    # Normalize based on band width
                    if close < middle_band:
                        # Distance to lower band as percentage of band width
                        distance = (middle_band - close) / (band_width / 2)
                    else:
                        # Distance to upper band as percentage of band width
                        distance = (close - middle_band) / (band_width / 2)
                    
                    confidence = min(1.0, distance)
                else:
                    confidence = 0.5
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=self.current_signal_type,
                    price=close,
                    rule_id=self.name,
                    confidence=confidence,
                    metadata={
                        'upper_band': upper_band,
                        'middle_band': middle_band,
                        'lower_band': lower_band,
                        'band_width': band_width
                    }
                )
        
        # Not enough data yet, return neutral signal
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.NEUTRAL,
            price=close,
            rule_id=self.name,
            confidence=0.0,
            metadata={'status': 'initializing'}
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['period'] * 3)
        self.upper_band_history = deque(maxlen=10)
        self.middle_band_history = deque(maxlen=10)
        self.lower_band_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="crossover")
class StochasticCrossoverRule(Rule):
    """
    Stochastic Oscillator Crossover Rule.
    
    This rule generates signals based on %K crossing %D in the Stochastic Oscillator.
    """
    
    def __init__(self, 
                 name: str = "stochastic_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Stochastic crossover rule"):
        """
        Initialize the Stochastic crossover rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - k_period: Period for %K calculation (default: 14)
                - d_period: Period for %D calculation (default: 3)
                - slowing: Slowing period for %K (default: 3)
                - use_extremes: Whether to also generate signals on overbought/oversold levels (default: True)
                - overbought: Overbought level (default: 80)
                - oversold: Oversold level (default: 20)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['k_period'] * 3)
        self.low_history = deque(maxlen=self.params['k_period'] * 3)
        self.close_history = deque(maxlen=self.params['k_period'] * 3)
        self.k_history = deque(maxlen=max(self.params['d_period'], 10))
        self.d_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'k_period': 14,
            'd_period': 3,
            'slowing': 3,
            'use_extremes': True,
            'overbought': 80,
            'oversold': 20
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['k_period'] <= 0 or self.params['d_period'] <= 0 or self.params['slowing'] <= 0:
            raise ValueError("Periods must be positive")
            
        if self.params['oversold'] >= self.params['overbought']:
            raise ValueError("Oversold level must be less than overbought level")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Stochastic crossover.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if not all(key in data for key in ['High', 'Low', 'Close']):
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=data.get('Close', None),
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing required price data'}
            )
            
        # Get parameters
        k_period = self.params['k_period']
        d_period = self.params['d_period']
        slowing = self.params['slowing']
        use_extremes = self.params['use_extremes']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        
        # Extract price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Calculate %K (Fast Stochastic)
        if len(self.close_history) >= k_period:
            # Get the highest high and lowest low over the k_period
            highest_high = max(list(self.high_history)[-k_period:])
            lowest_low = min(list(self.low_history)[-k_period:])
            
            # Calculate raw %K
            if highest_high != lowest_low:
                raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            else:
                raw_k = 50  # Default to middle if range is zero
            
            # Apply slowing (average of last 'slowing' raw %K values)
            if slowing > 1:
                # Store raw_k for slowing calculation
                if not hasattr(self, 'raw_k_history'):
                    self.raw_k_history = deque(maxlen=slowing * 2)
                
                self.raw_k_history.append(raw_k)
                
                if len(self.raw_k_history) >= slowing:
                    k_value = sum(list(self.raw_k_history)[-slowing:]) / slowing
                else:
                    k_value = raw_k
            else:
                k_value = raw_k
                
            # Store %K
            self.k_history.append(k_value)
            
            # Calculate %D (SMA of %K)
            if len(self.k_history) >= d_period:
                d_value = sum(list(self.k_history)[-d_period:]) / d_period
                self.d_history.append(d_value)
                
                # Generate signals
                if len(self.k_history) >= d_period + 1 and len(self.d_history) >= 2:
                    current_k = self.k_history[-1]
                    current_d = self.d_history[-1]
                    prev_k = self.k_history[-2]
                    prev_d = self.d_history[-2]
                    
                    # Check for %K crossing %D
                    if prev_k <= prev_d and current_k > current_d:
                        # %K crossed above %D - typically a buy signal
                        self.current_signal_type = SignalType.BUY
                    elif prev_k >= prev_d and current_k < current_d:
                        # %K crossed below %D - typically a sell signal
                        self.current_signal_type = SignalType.SELL
                    elif use_extremes:
                        # Check for overbought/oversold conditions
                        if current_k < oversold and prev_k < oversold:
                            # Oversold condition - potential buy
                            self.current_signal_type = SignalType.BUY
                        elif current_k > overbought and prev_k > overbought:
                            # Overbought condition - potential sell
                            self.current_signal_type = SignalType.SELL
                    
                    # Calculate confidence
                    # Base it on how far %K and %D are from the midpoint (50)
                    k_distance_from_mid = abs(current_k - 50) / 50
                    d_distance_from_mid = abs(current_d - 50) / 50
                    
                    # Average the distances for confidence
                    confidence = min(1.0, (k_distance_from_mid + d_distance_from_mid) / 2)
                    
                    return Signal(
                        timestamp=timestamp,
                        signal_type=self.current_signal_type,
                        price=close,
                        rule_id=self.name,
                        confidence=confidence,
                        metadata={
                            'k_value': current_k,
                            'd_value': current_d,
                            'overbought': current_k > overbought,
                            'oversold': current_k < oversold
                        }
                    )
        
        # Not enough data yet, return neutral signal
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.NEUTRAL,
            price=close,
            rule_id=self.name,
            confidence=0.0,
            metadata={'status': 'initializing'}
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['k_period'] * 3)
        self.low_history = deque(maxlen=self.params['k_period'] * 3)
        self.close_history = deque(maxlen=self.params['k_period'] * 3)
        self.k_history = deque(maxlen=max(self.params['d_period'], 10))
        self.d_history = deque(maxlen=10)
        if hasattr(self, 'raw_k_history'):
            self.raw_k_history = deque(maxlen=self.params['slowing'] * 2)
        self.current_signal_type = SignalType.NEUTRAL

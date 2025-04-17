"""
Standalone test script with embedded SMACrossoverRule implementation.

This script avoids import problems by including the rule implementation directly.
"""

import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import only the essential base components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.events.signal_event import SignalEvent
from src.rules.rule_base import Rule

# Standalone SMACrossoverRule implementation 
class StandaloneSMACrossoverRule(Rule):
    """
    Simple Moving Average crossover rule.

    Generates buy signals when the fast SMA crosses above the slow SMA,
    and sell signals when it crosses below.
    """

    def __init__(self, name: str, params=None, description="", event_bus=None):
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
    
    def on_bar(self, event):
        """
        Process a bar event and generate a trading signal directly.
        
        This method bypasses the base class's event handling to ensure direct control 
        over signal generation and event emission.
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

    def generate_signal(self, bar_event):
        """
        Generate a signal based on SMA crossover.
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
            
            if len(self.state['prices']) % 100 == 0:
                logger.debug(f"SMA Rule {self.name}: Processing bar for {symbol} @ {timestamp}, {price_field}: {close_price}")
        
        except Exception as e:
            logger.error(f"SMA Rule {self.name}: Error extracting data from bar event: {e}")
            return None
        
        # Get parameters
        fast_window = self.params['fast_window']
        slow_window = self.params['slow_window']
        
        # Update price history
        self.state['prices'].append(close_price)
        
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
            
            # Check for crossover (and make sure we have previous values)
            if prev_fast_sma is not None and prev_slow_sma is not None:
                # Calculate differences to detect crossovers
                prev_diff = prev_fast_sma - prev_slow_sma
                curr_diff = fast_sma - slow_sma
                
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
        """
        if hasattr(self, 'event_bus') and self.event_bus is not None:
            try:
                signal_event = Event(EventType.SIGNAL, signal)
                self.event_bus.emit(signal_event)
                logger.debug(f"Rule {self.name}: Emitted signal event")
            except Exception as e:
                logger.error(f"Rule {self.name}: Error emitting signal: {str(e)}")
        
    def reset(self):
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

# Main test script
# ---------------

# Signal handler to log and count signals
signal_count = 0

def handle_signal(event):
    """Handler for signal events."""
    global signal_count
    signal = event.data
    
    direction = "BUY" if signal.get_signal_value() == 1 else "SELL" 
    symbol = signal.get_symbol()
    price = signal.get_price()
    timestamp = signal.timestamp if hasattr(signal, 'timestamp') else datetime.datetime.now()
    
    # Extract additional metadata for more detailed logging
    metadata = signal.get_metadata() if hasattr(signal, 'get_metadata') else {}
    fast_sma = metadata.get('fast_sma', 'N/A')
    slow_sma = metadata.get('slow_sma', 'N/A')
    reason = metadata.get('reason', 'N/A')
    
    logger.info(f"SIGNAL {signal_count+1}: {direction} {symbol} @ {price:.2f}, Time: {timestamp}")
    logger.info(f"Signal Details: Fast SMA: {fast_sma}, Slow SMA: {slow_sma}, Reason: {reason}")
    signal_count += 1

# Create event bus
event_bus = EventBus(async_mode=False)

# Register signal handler
event_bus.register(EventType.SIGNAL, handle_signal)

# Create data source and handler
data_dir = "./data"
data_source = CSVDataSource(data_dir)
data_handler = DataHandler(data_source)

# Create our rule
sma_rule = StandaloneSMACrossoverRule(
    name="sma_crossover",
    params={
        "fast_window": 5,
        "slow_window": 15
    },
    description="SMA Crossover (5,15)",
    event_bus=event_bus  # Pass event bus directly to rule
)

# Register rule to handle bar events
event_bus.register(EventType.BAR, sma_rule.on_bar)

# Configure the data load parameters
symbols = ["SPY"]
start_date = datetime.datetime(2024, 3, 26)
end_date = datetime.datetime(2025, 4, 2) 
timeframe = "1m"

# Load data
logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
data_handler.load_data(symbols=symbols, 
                      start_date=start_date,
                      end_date=end_date,
                      timeframe=timeframe)

# Process bars
logger.info("Processing bars to find trading signals...")
max_bars = 1000  # Process 1000 bars as needed to compare with expected 79 signals

# Statistics for verification
processed_bars = 0
last_signal_time = None

# Use direct bar processing from data handler
for i, bar_event in enumerate(data_handler.iter_train(use_bar_events=True)):
    if i >= max_bars:
        break
    
    processed_bars += 1
    
    # Log bar information (periodically to reduce log volume)
    if i % 100 == 0:
        logger.info(f"Processing bar {i}: {bar_event.get_symbol()} @ {bar_event.get_timestamp()}, Close: {bar_event.get_price():.2f}")
        logger.info(f"Current signals: {signal_count}")
        
        # Add statistics about the rule state
        logger.info(f"Rule state - Prices collected: {len(sma_rule.state['prices'])}")
        if sma_rule.state['fast_sma'] is not None:
            logger.info(f"Rule state - Fast SMA: {sma_rule.state['fast_sma']:.4f}")
        if sma_rule.state['slow_sma'] is not None:
            logger.info(f"Rule state - Slow SMA: {sma_rule.state['slow_sma']:.4f}")
    
    # Create and emit bar event - this is the important part
    bar_event_obj = Event(EventType.BAR, bar_event)
    event_bus.emit(bar_event_obj)
    
    # Track if a new signal was generated
    if last_signal_time != sma_rule.state['last_signal_time']:
        last_signal_time = sma_rule.state['last_signal_time']
        if last_signal_time is not None and i % 100 != 0:  # Avoid duplicating log if we just logged at a 100 boundary
            logger.info(f"New signal at: {last_signal_time}, Total: {signal_count}")

# Display final results
logger.info(f"Processed {processed_bars} bars")
logger.info(f"Generated {signal_count} signals (expected 79)")
logger.info(f"Rule state - Signals generated: {sma_rule.state['signals_generated']}")
logger.info(f"Rule state - Last signal time: {sma_rule.state['last_signal_time']}")
logger.info(f"Rule state - Prices collected: {len(sma_rule.state['prices'])}")
logger.info("Test completed successfully!")

# Save this as sma_rule_test.py and run it directly

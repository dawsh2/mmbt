"""
Event System Demo Script

This script demonstrates the usage of the event system components
by simulating a simple trading workflow with events.
"""

import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import event components
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType
from src.events.event_handlers import (
    EventHandler, 
    FunctionEventHandler,
    LoggingHandler
)
from src.events.missing_handlers import (
    DebounceHandler,
    FilterHandler,
    EventHandlerGroup
)
from src.events.event_emitters import (
    EventEmitter,
    MarketDataEmitter,
    SignalEmitter,
    OrderEmitter,
    FillEmitter
)
from src.events.missing_emitters import (
    PortfolioEmitter,
    SystemEmitter
)

# Create logger
logger = logging.getLogger('event_demo')


# Define example components that use the event system

class Strategy(EventEmitter):
    """Simple strategy that generates signals based on moving average crossover."""
    
    def __init__(self, event_bus):
        super().__init__(event_bus)
        self.short_ma = []
        self.long_ma = []
        self.short_window = 3
        self.long_window = 5
        
    def on_bar(self, bar_data):
        # Update moving averages
        price = bar_data.get('Close', 0)
        
        self.short_ma.append(price)
        if len(self.short_ma) > self.short_window:
            self.short_ma.pop(0)
            
        self.long_ma.append(price)
        if len(self.long_ma) > self.long_window:
            self.long_ma.pop(0)
        
        # Generate signal if we have enough data
        if len(self.short_ma) == self.short_window and len(self.long_ma) == self.long_window:
            short_avg = sum(self.short_ma) / len(self.short_ma)
            long_avg = sum(self.long_ma) / len(self.long_ma)
            
            # Crossover logic
            if short_avg > long_avg:
                # Bullish signal
                logger.info(f"Strategy: Generating BUY signal for {bar_data['symbol']}")
                self.emit_signal(bar_data['symbol'], "BUY", price, 0.8)
            elif short_avg < long_avg:
                # Bearish signal
                logger.info(f"Strategy: Generating SELL signal for {bar_data['symbol']}")
                self.emit_signal(bar_data['symbol'], "SELL", price, 0.8)
    
    def emit_signal(self, symbol, signal_type, price, confidence):
        """Helper method to emit a signal event."""
        signal_emitter = SignalEmitter(self.event_bus)
        signal_emitter.emit_signal(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            rule_id="MA_Crossover"
        )


class PortfolioManager(EventEmitter):
    """Simple portfolio manager that converts signals to orders."""
    
    def __init__(self, event_bus):
        super().__init__(event_bus)
        self.positions = {}  # symbol -> position
        self.cash = 100000
        
    def on_signal(self, signal_data):
        """Process a signal and generate an order."""
        symbol = signal_data.get('symbol', '')
        signal_type = signal_data.get('signal_type', '')
        price = signal_data.get('price', 0)
        
        # Determine order direction and quantity
        direction = 1 if signal_type == "BUY" else -1
        quantity = 100  # Fixed quantity for demo
        
        # Generate an order
        logger.info(f"Portfolio: Converting {signal_type} signal to order for {quantity} shares of {symbol}")
        order_emitter = OrderEmitter(self.event_bus)
        order_emitter.emit_order(
            symbol=symbol,
            order_type="MARKET",
            quantity=quantity,
            direction=direction,
            price=price
        )
    
    def on_fill(self, fill_data):
        """Process a fill event and update portfolio."""
        symbol = fill_data.get('symbol', '')
        direction = fill_data.get('direction', 0)
        quantity = fill_data.get('quantity', 0)
        price = fill_data.get('price', 0)
        
        # Calculate cost
        cost = quantity * price * direction
        
        # Update cash
        self.cash -= cost
        
        # Update position
        if symbol not in self.positions:
            if direction > 0:
                self.positions[symbol] = quantity
            else:
                self.positions[symbol] = -quantity
        else:
            self.positions[symbol] += quantity * direction
        
        # Remove position if zero
        if abs(self.positions.get(symbol, 0)) < 0.01:
            del self.positions[symbol]
        
        logger.info(f"Portfolio: Updated position {symbol}: {self.positions.get(symbol, 0)}, Cash: {self.cash:.2f}")


class ExecutionEngine(EventEmitter):
    """Simple execution engine that processes orders and generates fills."""
    
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def on_order(self, order_data):
        """Process an order and generate a fill."""
        symbol = order_data.get('symbol', '')
        direction = order_data.get('direction', 0)
        quantity = order_data.get('quantity', 0)
        price = order_data.get('price', 0)
        order_id = order_data.get('order_id', '')
        
        # Add small slippage for realism
        execution_price = price * (1 + 0.001 * direction)
        
        logger.info(f"Execution: Executing order {order_id} for {quantity} shares of {symbol} at {execution_price:.2f}")
        
        # Generate fill event
        fill_emitter = FillEmitter(self.event_bus)
        fill_emitter.emit_fill(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
            direction=direction,
            transaction_cost=1.0  # Fixed commission for demo
        )


# Demo script
def run_event_system_demo():
    """Run a demonstration of the event system."""
    
    logger.info("Starting event system demonstration")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create components
    strategy = Strategy(event_bus)
    portfolio_manager = PortfolioManager(event_bus)
    execution_engine = ExecutionEngine(event_bus)
    
    # Create handlers
    bar_handler = FunctionEventHandler(EventType.BAR, lambda event: strategy.on_bar(event.data))
    signal_handler = FunctionEventHandler(EventType.SIGNAL, lambda event: portfolio_manager.on_signal(event.data))
    order_handler = FunctionEventHandler(EventType.ORDER, lambda event: execution_engine.on_order(event.data))
    fill_handler = FunctionEventHandler(EventType.FILL, lambda event: portfolio_manager.on_fill(event.data))
    
    # Create logging handler for all events
    logging_handler = LoggingHandler(list(EventType))
    
    # Create a high-confidence filter for signals
    def confidence_filter(event):
        return event.data.get('confidence', 0) >= 0.7
    
    # Apply filter to signal handler
    filtered_signal_handler = FilterHandler([EventType.SIGNAL], signal_handler, confidence_filter)
    
    # Group handlers for easy management
    trading_handlers = EventHandlerGroup("trading_handlers", [
        bar_handler,
        filtered_signal_handler,
        order_handler,
        fill_handler
    ])
    
    # Register handlers with event bus
    trading_handlers.register_all(event_bus)
    
    # Register logging handler
    for event_type in EventType:
        event_bus.register(event_type, logging_handler)
    
    # Create market data emitter
    market_data_emitter = MarketDataEmitter(event_bus)
    
    # Create system emitter for start/stop events
    system_emitter = SystemEmitter(event_bus)
    
    # Emit system start event
    system_emitter.emit_start("demo_script")
    
    # Generate some sample market data
    symbols = ["AAPL", "MSFT", "GOOGL"]
    prices = {
        "AAPL": [150.0, 151.0, 152.0, 153.0, 152.5, 151.8, 150.5, 149.0, 148.0, 149.5],
        "MSFT": [250.0, 252.0, 253.0, 252.5, 251.0, 250.5, 251.5, 252.5, 253.5, 255.0],
        "GOOGL": [2800.0, 2810.0, 2820.0, 2815.0, 2805.0, 2795.0, 2790.0, 2785.0, 2800.0, 2820.0]
    }
    
    logger.info("Generating market data and processing events...")
    
    # Simulate market data for each symbol
    for i in range(10):
        for symbol in symbols:
            # Create bar data
            bar_data = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "Open": prices[symbol][i] - 1.0,
                "High": prices[symbol][i] + 2.0,
                "Low": prices[symbol][i] - 2.0,
                "Close": prices[symbol][i],
                "Volume": 10000 + i * 1000
            }
            
            # Emit bar event
            market_data_emitter.emit_bar(bar_data)
            
            # Small delay for logging clarity
            time.sleep(0.1)
    
    # Emit system stop event
    system_emitter.emit_stop("demo_script", "Demonstration complete")
    
    logger.info("Event system demonstration completed")
    
    # Return the final portfolio state
    return {
        "cash": portfolio_manager.cash,
        "positions": portfolio_manager.positions
    }


if __name__ == "__main__":
    # Run the demonstration
    final_state = run_event_system_demo()
    
    # Print final portfolio state
    logger.info(f"Final portfolio state: {final_state}")

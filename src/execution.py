# execution.py
from events import EventType, FillEvent

class SimpleExecutionHandler:
    def __init__(self, event_queue, price_data=None):
        self.event_queue = event_queue
        self.price_data = price_data or {}  # Symbol -> DataFrame of prices
    
    def execute_order(self, event):
        """Execute an order and generate a fill event"""
        if event.type != EventType.ORDER:
            return
        
        symbol = event.symbol
        quantity = event.quantity
        direction = event.direction
        
        # In a real system, we'd simulate slippage, etc.
        # For now, just use the latest close price 
        if symbol in self.price_data and not self.price_data[symbol].empty:
            price = self.price_data[symbol]['Close'].iloc[-1]
        else:
            # If no price data, use a placeholder (fix this in practice)
            print(f"Warning: No price data for {symbol}")
            price = 100.0
        
        # Create fill event
        fill = FillEvent(
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            fill_price=price,
            commission=quantity * price * 0.001  # Simple commission model
        )
        
        # Add to event queue
        self.event_queue.put(fill)
        print(f"Fill generated: {fill.direction} {fill.quantity} {fill.symbol} @ {fill.fill_price}")
    
    def update_prices(self, symbol, prices):
        """Update price data for a symbol"""
        self.price_data[symbol] = prices

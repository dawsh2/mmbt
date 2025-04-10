# portfolio.py
from events import EventType, OrderEvent

class SimplePortfolio:
    def __init__(self, event_queue, initial_capital=100000.0):
        self.event_queue = event_queue
        self.initial_capital = initial_capital
        self.positions = {
            'cash': initial_capital
        }
        self.current_trades = {}
    
    def process_signal(self, event):
        """Create simple orders from signals"""
        if event.type != EventType.SIGNAL:
            return
        
        symbol = event.symbol
        signal_type = event.signal_type  # 1 for long, -1 for short
        
        # Simple fixed quantity for now (improve later with position sizing)
        quantity = 100
        
        # Generate order
        order = OrderEvent(
            symbol=symbol,
            order_type='MKT',  # Market order
            quantity=quantity,
            direction=signal_type
        )
        
        # Send to event queue
        self.event_queue.put(order)
        print(f"Order generated: {order.direction} {order.quantity} {order.symbol}")

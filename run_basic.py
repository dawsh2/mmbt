#!/usr/bin/env python
# simple_backtester.py - A simplified backtester to debug trade execution

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_backtester')

# Make sure we can import from src directory
sys.path.append(os.path.abspath('.'))

# Import signal types
from src.signals import SignalType, Signal

# Simple strategy class that generates signals based on price comparison
class SimpleStrategy:
    """A simple strategy that generates signals based on price comparison."""
    
    def __init__(self, buy_threshold=0.001, sell_threshold=-0.001):
        """Initialize the strategy."""
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_price = None
    
    def on_bar(self, bar):
        """Process a bar and generate a signal."""
        # Extract price
        if hasattr(bar, 'bar'):
            bar_data = bar.bar
        else:
            bar_data = bar
            
        if not bar_data or 'Close' not in bar_data:
            return None
            
        close = bar_data['Close']
        timestamp = bar_data.get('timestamp', datetime.now())
        symbol = bar_data.get('symbol', 'default')
        
        # Skip if we don't have a previous price
        if self.last_price is None:
            self.last_price = close
            return None
            
        # Calculate return
        price_change = (close - self.last_price) / self.last_price
        
        # Generate signal based on price change
        if price_change > self.buy_threshold:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=close,
                rule_id="SimpleStrategy",
                confidence=min(1.0, abs(price_change) * 100),
                symbol=symbol
            )
        elif price_change < self.sell_threshold:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=close,
                rule_id="SimpleStrategy",
                confidence=min(1.0, abs(price_change) * 100),
                symbol=symbol
            )
        else:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id="SimpleStrategy",
                confidence=0.0,
                symbol=symbol
            )
        
        # Update last price
        self.last_price = close
        
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        self.last_price = None

# Simple order class
class Order:
    """Represents a trading order."""
    
    def __init__(self, symbol, quantity, direction, price=None, timestamp=None):
        """Initialize an order."""
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction  # 1 for buy, -1 for sell
        self.price = price
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self):
        """String representation."""
        direction_str = "BUY" if self.direction > 0 else "SELL"
        return f"{direction_str} {self.quantity} {self.symbol} @ {self.price}"

# Simple fill class
class Fill:
    """Represents an order fill."""
    
    def __init__(self, order, price, timestamp=None):
        """Initialize a fill."""
        self.order = order
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.symbol = order.symbol
        self.quantity = order.quantity
        self.direction = order.direction
    
    def __str__(self):
        """String representation."""
        direction_str = "BOUGHT" if self.direction > 0 else "SOLD"
        return f"{direction_str} {self.quantity} {self.symbol} @ {self.price}"

# Simple portfolio class
class Portfolio:
    """Manages portfolio positions and equity."""
    
    def __init__(self, initial_capital=100000):
        """Initialize the portfolio."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.position_prices = {}  # symbol -> average price
        self.equity = initial_capital
        self.history = []
    
    def execute_order(self, order, price):
        """Execute an order."""
        # Calculate cost
        cost = order.quantity * price * order.direction
        
        # Update cash
        self.cash -= cost
        
        # Update position
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0
            self.position_prices[symbol] = 0
        
        old_quantity = self.positions[symbol]
        old_price = self.position_prices[symbol]
        
        # Calculate new average price for buys
        if order.direction > 0 and old_quantity >= 0:
            # Adding to long position
            new_quantity = old_quantity + order.quantity
            if new_quantity > 0:
                # Calculate new average price
                self.position_prices[symbol] = (old_quantity * old_price + order.quantity * price) / new_quantity
        elif order.direction < 0 and old_quantity <= 0:
            # Adding to short position
            new_quantity = old_quantity - order.quantity
            if new_quantity < 0:
                # Calculate new average price
                self.position_prices[symbol] = (old_quantity * old_price - order.quantity * price) / new_quantity
        else:
            # Reducing or closing position
            self.positions[symbol] += order.quantity * order.direction
            if self.positions[symbol] == 0:
                # Position closed
                self.position_prices[symbol] = 0
        
        # Update position quantity
        self.positions[symbol] += order.quantity * order.direction
        
        # If position is closed, remove it
        if self.positions[symbol] == 0:
            self.positions.pop(symbol)
            self.position_prices.pop(symbol)
        
        # Update equity
        self._update_equity(price, symbol)
    
    def _update_equity(self, price=None, symbol=None):
        """Update portfolio equity."""
        # Calculate position value
        position_value = 0
        for sym, qty in self.positions.items():
            # Use the provided price for the specific symbol, otherwise use the last known price
            if sym == symbol and price is not None:
                position_value += qty * price
            elif sym in self.position_prices:
                position_value += qty * self.position_prices[sym]
        
        # Update equity
        self.equity = self.cash + position_value
        
        # Add to history
        self.history.append({
            'timestamp': datetime.now(),
            'cash': self.cash,
            'equity': self.equity,
            'positions': dict(self.positions)
        })
    
    def reset(self):
        """Reset the portfolio."""
        self.cash = self.initial_capital
        self.positions = {}
        self.position_prices = {}
        self.equity = self.initial_capital
        self.history = []

# Simple backtester
class SimpleBacktester:
    """A simple backtester for testing strategies."""
    
    def __init__(self, data, strategy, initial_capital=100000):
        """Initialize the backtester."""
        self.data = data
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.signals = []
        self.orders = []
        self.fills = []
    
    def run(self, limit=None):
        """Run the backtest."""
        logger.info("Starting simple backtest...")
        
        # Reset components
        self.strategy.reset()
        self.portfolio.reset()
        self.signals = []
        self.orders = []
        self.fills = []
        
        # Stats
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        # Process each bar
        for i, row in enumerate(self.data.itertuples()):
            # Stop if we hit the limit
            if limit and i >= limit:
                break
            
            # Convert row to dictionary
            bar_data = {col: getattr(row, col) for col in self.data.columns}
            
            # Process through strategy
            signal = self.strategy.on_bar(bar_data)
            
            # Skip if no signal
            if signal is None:
                continue
            
            # Store signal
            self.signals.append(signal)
            
            # Update statistics
            if signal.signal_type == SignalType.BUY:
                buy_signals += 1
            elif signal.signal_type == SignalType.SELL:
                sell_signals += 1
            else:
                neutral_signals += 1
            
            # Skip neutral signals
            if signal.signal_type == SignalType.NEUTRAL:
                continue
            
            # Create order
            order = Order(
                symbol=signal.symbol,
                quantity=100,  # Fixed size for simplicity
                direction=1 if signal.signal_type == SignalType.BUY else -1,
                price=signal.price,
                timestamp=signal.timestamp
            )
            
            # Store order
            self.orders.append(order)
            
            # Execute order
            self.portfolio.execute_order(order, signal.price)
            
            # Create fill
            fill = Fill(
                order=order,
                price=signal.price,
                timestamp=signal.timestamp
            )
            
            # Store fill
            self.fills.append(fill)
            
            # Log progress
            if i % 10000 == 0:
                logger.info(f"Processed {i} bars...")
        
        # Log final statistics
        logger.info(f"Backtest complete. Processed {len(self.data)} bars.")
        logger.info(f"Signal distribution: BUY={buy_signals}, SELL={sell_signals}, NEUTRAL={neutral_signals}")
        logger.info(f"Trades executed: {len(self.fills)}")
        
        # Get final portfolio value
        final_equity = self.portfolio.equity
        total_return = ((final_equity / self.portfolio.initial_capital) - 1) * 100
        
        logger.info(f"Initial capital: ${self.portfolio.initial_capital:.2f}")
        logger.info(f"Final portfolio value: ${final_equity:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Return results
        return {
            'signals': self.signals,
            'orders': self.orders,
            'fills': self.fills,
            'portfolio_history': self.portfolio.history,
            'final_equity': final_equity,
            'total_return': total_return
        }

def main():
    """Run the simple backtester."""
    logger.info("Starting simple backtester...")
    
    # Load data
    csv_file = 'data_1m.csv'
    symbol = 'data'
    
    logger.info(f"Loading data from: {csv_file}")
    csv_path = os.path.join('data', csv_file)
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Format columns
    for src, dst in [('timestamp', 'timestamp'), ('open', 'Open'), ('high', 'High'), 
                    ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume')]:
        for col in df.columns:
            if col.lower() == src.lower() and col != dst:
                df.rename(columns={col: dst}, inplace=True)
                logger.info(f"Renamed column: {col} → {dst}")
    
    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info("Converted 'timestamp' to datetime")
    
    # Add symbol
    df['symbol'] = symbol
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create strategy
    strategy = SimpleStrategy(buy_threshold=0.001, sell_threshold=-0.001)
    
    # Create backtester
    backtester = SimpleBacktester(df, strategy, initial_capital=100000)
    
    # Run backtest with limit
    limit = 10000  # Limit to first 10,000 bars for testing
    results = backtester.run(limit=limit)
    
    # Print results
    logger.info("Backtest Results:")
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Number of Trades: {len(results['fills'])}")
    
    # Print some trades
    if results['fills']:
        logger.info("Sample Trades:")
        for i, fill in enumerate(results['fills'][:5]):
            logger.info(f"Trade {i+1}: {fill}")
    else:
        logger.info("No trades were executed.")
    
    logger.info("Simple backtest completed.")

if __name__ == "__main__":
    main()

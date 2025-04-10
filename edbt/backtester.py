"""
Event-Driven Backtester Framework
--------------------------------
A barebones implementation of an event-driven backtesting system for trading strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt


class Event:
    """Base Event class that all other event types will inherit from."""
    pass


class MarketEvent(Event):
    """Event signaling new market data is available."""
    def __init__(self, timestamp):
        self.type = 'MARKET'
        self.timestamp = timestamp


class SignalEvent(Event):
    """Event signaling a trading signal generated from a strategy."""
    def __init__(self, symbol, timestamp, signal_type, strength=1.0):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type  # 'LONG' or 'SHORT'
        self.strength = strength  # Strength of signal [0.0, 1.0]


class OrderEvent(Event):
    """Event for sending orders to execution system."""
    def __init__(self, symbol, order_type, quantity, direction, timestamp):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type  # 'MARKET', 'LIMIT', 'STOP', etc.
        self.quantity = quantity
        self.direction = direction  # 'BUY' or 'SELL'
        self.timestamp = timestamp


class FillEvent(Event):
    """Event signaling that an order has been filled."""
    def __init__(self, timestamp, symbol, exchange, quantity, 
                 direction, fill_price, commission=None):
        self.type = 'FILL'
        self.timestamp = timestamp
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        
        # Calculate commission if not provided
        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission
            
    def calculate_commission(self):
        """Calculate the commission."""
        # Simple commission model
        return 0.001 * self.quantity * self.fill_price


class DataHandler(ABC):
    """Abstract base class providing an interface for all data handlers."""
    
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """Returns the last N bars from the latest_symbol list."""
        raise NotImplementedError("Should implement get_latest_bars()")
    
    @abstractmethod
    def update_bars(self):
        """Pushes the latest bar to the latest_symbol_data structure."""
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """Handler for CSV files of historical OHLCV data."""
    
    def __init__(self, events_queue, csv_dir, symbol_list):
        self.events_queue = events_queue
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        
        self._open_convert_csv_files()
        
    def _open_convert_csv_files(self):
        """Opens the CSV files and converts to pandas DataFrames."""
        for symbol in self.symbol_list:
            # Load CSV file with pandas
            self.symbol_data[symbol] = pd.read_csv(
                f"{self.csv_dir}/{symbol}.csv",
                header=0, index_col=0, parse_dates=True
            )
            
            # Create empty deque to store the latest symbol data
            self.latest_symbol_data[symbol] = deque(maxlen=100)
    
    def _get_new_bar(self, symbol):
        """Returns the latest bar from the data feed as a tuple."""
        if self.bar_index < len(self.symbol_data[symbol]):
            bar_data = self.symbol_data[symbol].iloc[self.bar_index]
            bar_tuple = (
                bar_data.name,  # Timestamp
                bar_data['open'],
                bar_data['high'],
                bar_data['low'],
                bar_data['close'],
                bar_data['volume']
            )
            return bar_tuple
        return None
    
    def get_latest_bars(self, symbol, N=1):
        """Returns the last N bars from the latest_symbol list."""
        try:
            bars_list = list(self.latest_symbol_data[symbol])
        except KeyError:
            print(f"Symbol {symbol} is not available in the historical data set.")
            return []
        else:
            return bars_list[-N:]
    
    def update_bars(self):
        """Pushes the latest bar to the latest_symbol_data structure for all symbols."""
        for symbol in self.symbol_list:
            bar = self._get_new_bar(symbol)
            if bar is not None:
                self.latest_symbol_data[symbol].append(bar)
        
        # After updating all symbols, increment the bar index
        self.bar_index += 1
        
        # Create and place a MarketEvent in the queue
        if self.bar_index < len(self.symbol_data[self.symbol_list[0]]):
            timestamp = self.symbol_data[self.symbol_list[0]].index[self.bar_index-1]
            self.events_queue.append(MarketEvent(timestamp))
        else:
            self.continue_backtest = False


class Strategy(ABC):
    """Strategy abstract base class that provides an interface for all strategies."""
    
    @abstractmethod
    def calculate_signals(self, event):
        """
        The main method that generates SignalEvents based on market data.
        """
        raise NotImplementedError("Should implement calculate_signals()")


class MovingAverageCrossStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    """
    
    def __init__(self, data_handler, events_queue, short_window=50, long_window=200):
        self.data_handler = data_handler
        self.events_queue = events_queue
        self.short_window = short_window
        self.long_window = long_window
        self.bought = {symbol: False for symbol in self.data_handler.symbol_list}
    
    def calculate_signals(self, event):
        """
        Generates a new SignalEvent if a signal is triggered.
        """
        if event.type == 'MARKET':
            for symbol in self.data_handler.symbol_list:
                bars = self.data_handler.get_latest_bars(symbol, N=self.long_window)
                
                if len(bars) >= self.long_window:
                    # Extract timestamp and close prices
                    timestamps = [bar[0] for bar in bars]
                    close_prices = np.array([bar[4] for bar in bars])
                    
                    # Create the two moving averages
                    short_sma = np.mean(close_prices[-self.short_window:])
                    long_sma = np.mean(close_prices)
                    
                    # Trading signals based on moving average crossovers
                    if short_sma > long_sma and not self.bought[symbol]:
                        # Short SMA above long SMA is a BUY signal
                        self.events_queue.append(
                            SignalEvent(symbol, timestamps[-1], 'LONG')
                        )
                        self.bought[symbol] = True
                    
                    elif short_sma < long_sma and self.bought[symbol]:
                        # Short SMA below long SMA is a SELL signal
                        self.events_queue.append(
                            SignalEvent(symbol, timestamps[-1], 'SHORT')
                        )
                        self.bought[symbol] = False


class Portfolio(ABC):
    """Portfolio abstract class for tracking positions, holdings, etc."""
    
    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders.
        """
        raise NotImplementedError("Should implement update_signal()")
    
    @abstractmethod
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        raise NotImplementedError("Should implement update_fill()")


class NaivePortfolio(Portfolio):
    """
    Simple portfolio model that converts signals to orders with fixed quantity.
    """
    
    def __init__(self, data_handler, events_queue, initial_capital=100000.0):
        self.data_handler = data_handler
        self.events_queue = events_queue
        self.initial_capital = initial_capital
        self.current_positions = {symbol: 0 for symbol in self.data_handler.symbol_list}
        self.current_holdings = self._construct_initial_holdings()
        self.all_positions = []
        self.all_holdings = []
    
    def _construct_initial_holdings(self):
        """
        Constructs the initial holdings dict.
        """
        holdings = {symbol: 0.0 for symbol in self.data_handler.symbol_list}
        holdings['cash'] = self.initial_capital
        holdings['total'] = self.initial_capital
        holdings['commission'] = 0.0
        return holdings
    
    def update_timeindex(self, event):
        """
        Adds a new record to the positions and holdings matrices
        for the current market data bar.
        """
        if event.type == 'MARKET':
            # Update positions
            positions = {symbol: self.current_positions[symbol] for symbol in self.current_positions}
            positions['datetime'] = event.timestamp
            self.all_positions.append(positions)
            
            # Update holdings
            holdings = {symbol: 0.0 for symbol in self.current_positions}
            holdings['datetime'] = event.timestamp
            holdings['cash'] = self.current_holdings['cash']
            holdings['commission'] = self.current_holdings['commission']
            holdings['total'] = self.current_holdings['cash']
            
            # Add market values of all positions
            for symbol in self.current_positions:
                if self.current_positions[symbol] > 0:
                    market_bar = self.data_handler.get_latest_bars(symbol)
                    if market_bar:
                        close_price = market_bar[0][4]  # Close price
                        holdings[symbol] = self.current_positions[symbol] * close_price
                        holdings['total'] += holdings[symbol]
            
            self.all_holdings.append(holdings)
            self.current_holdings = holdings.copy()
    
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new OrderEvents.
        """
        if event.type == 'SIGNAL':
            # Fixed quantity of 100 shares
            quantity = 100
            
            # Create the corresponding OrderEvent
            order = OrderEvent(
                event.symbol,
                'MARKET',
                quantity,
                'BUY' if event.signal_type == 'LONG' else 'SELL',
                event.timestamp
            )
            
            self.events_queue.append(order)
    
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings from a FillEvent.
        """
        if event.type == 'FILL':
            # Update positions
            direction_multiplier = 1 if event.direction == 'BUY' else -1
            self.current_positions[event.symbol] += direction_multiplier * event.quantity
            
            # Update holdings
            latest_bars = self.data_handler.get_latest_bars(event.symbol)
            if latest_bars:
                fill_price = event.fill_price
                cost = direction_multiplier * event.quantity * fill_price
                
                self.current_holdings[event.symbol] += cost
                self.current_holdings['commission'] += event.commission
                self.current_holdings['cash'] -= (cost + event.commission)
                self.current_holdings['total'] -= event.commission


class ExecutionHandler(ABC):
    """Execution handler abstract class."""
    
    @abstractmethod
    def execute_order(self, event):
        """
        Takes an OrderEvent and executes it, creating a FillEvent.
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler. Simply converts all order objects
    into fill objects automatically without any latency, slippage or fill ratio issues.
    """
    
    def __init__(self, events_queue):
        self.events_queue = events_queue
    
    def execute_order(self, event):
        """
        Simply converts OrderEvents into FillEvents automatically.
        """
        if event.type == 'ORDER':
            # Obtain the latest prices from the data handler
            # In a real system, this would connect to a brokerage or exchange
            fill_event = FillEvent(
                event.timestamp,
                event.symbol,
                'SIMULATED',
                event.quantity,
                event.direction,
                # In a real system, the fill price might include slippage
                # For now, we use a simplified model
                self._get_fill_price(event),
                # Commission can be calculated here if needed
                None
            )
            
            self.events_queue.append(fill_event)
    
    def _get_fill_price(self, order_event):
        """
        Return a simplified fill price. In a real system, this would include
        market impact, slippage, etc.
        """
        # This is a simplification - in reality, you would connect to a data source
        # to get the latest prices or use the backtest data
        return 100.0  # Placeholder value, replace with actual implementation


class Backtest:
    """Encapsulates the backtesting logic."""
    
    def __init__(
        self, 
        data_handler, 
        strategy, 
        portfolio, 
        execution_handler,
        initial_capital=100000.0,
        heartbeat=0.0
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat  # Time between updates in seconds
        self.events_queue = deque()
        
        # Metrics
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_bars = 0
    
    def _run_backtest(self):
        """
        Runs the backtest.
        """
        print("Running Backtest...")
        
        while self.data_handler.continue_backtest:
            # Update the bars (creates MarketEvent)
            self.data_handler.update_bars()
            
            # Process all events
            while len(self.events_queue) > 0:
                event = self.events_queue.popleft()
                
                if event.type == 'MARKET':
                    self.portfolio.update_timeindex(event)
                    self.strategy.calculate_signals(event)
                    self.num_bars += 1
                
                elif event.type == 'SIGNAL':
                    self.signals += 1
                    self.portfolio.update_signal(event)
                
                elif event.type == 'ORDER':
                    self.orders += 1
                    self.execution_handler.execute_order(event)
                
                elif event.type == 'FILL':
                    self.fills += 1
                    self.portfolio.update_fill(event)
    
    def _output_performance(self):
        """
        Outputs the performance metrics.
        """
        print("Backtest complete.")
        print(f"Total Bars: {self.num_bars}")
        print(f"Total Signals: {self.signals}")
        print(f"Total Orders: {self.orders}")
        print(f"Total Fills: {self.fills}")
        
        # Calculate portfolio performance
        if len(self.portfolio.all_holdings) > 0:
            returns = pd.Series(
                [h['total'] for h in self.portfolio.all_holdings],
                index=[h['datetime'] for h in self.portfolio.all_holdings]
            )
            
            # Calculate metrics
            total_return = (returns.iloc[-1] - returns.iloc[0]) / returns.iloc[0] * 100
            print(f"Total Return: {total_return:.2f}%")
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(returns.index, returns.values)
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.savefig('equity_curve.png')
    
    def simulate_trading(self):
        """
        Simulates the backtest and outputs performance.
        """
        self._run_backtest()
        self._output_performance()


def main():
    # Example usage
    csv_dir = 'data'
    symbol_list = ['SPY']
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    # Initialize event queue
    events_queue = deque()
    
    # Initialize data handler
    data_handler = HistoricCSVDataHandler(events_queue, csv_dir, symbol_list)
    
    # Initialize strategy
    strategy = MovingAverageCrossStrategy(data_handler, events_queue, 50, 200)
    
    # Initialize portfolio
    portfolio = NaivePortfolio(data_handler, events_queue, initial_capital)
    
    # Initialize execution handler
    execution_handler = SimulatedExecutionHandler(events_queue)
    
    # Initialize and run backtest
    backtest = Backtest(
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        initial_capital,
        heartbeat
    )
    
    backtest.simulate_trading()


if __name__ == "__main__":
    main()


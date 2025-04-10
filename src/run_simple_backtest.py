"""
Simple backtest script for testing your event-driven system with a single symbol.
This script demonstrates how to set up and run a basic backtest using the components
from the speedrun implementation.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import your event system components
from events import EventQueue, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from data_handler import CSVDataHandler
from rule_strategy import RuleBasedStrategy  # Your adapted strategy
from portfolio import SimplePortfolio  # Your simple portfolio
from execution import SimpleExecutionHandler  # Your execution handler


def run_simple_backtest(symbol='AAPL', start_date=None, end_date=None, data_dir='data/'):
    """
    Run a simple backtest for a single symbol over a specified time period.
    
    Args:
        symbol: Symbol to backtest (default: 'AAPL')
        start_date: Start date for backtest (default: 1 year ago)
        end_date: End date for backtest (default: today)
        data_dir: Directory containing CSV data files
        
    Returns:
        Dictionary with backtest results
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)  # 1 year ago
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Running backtest for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize the event queue (central messaging system)
    event_queue = EventQueue()
    
    # Set up the data handler
    data_handler = CSVDataHandler(
        event_queue=event_queue,
        csv_dir=data_dir,
        symbol_list=[symbol],
        start_date=start_date,
        end_date=end_date
    )
    
    # Set up the strategy
    strategy = RuleBasedStrategy(
        event_queue=event_queue,
        symbols=[symbol],
        use_weights=True,  # Using weighted rules
        top_n=5  # Top 5 rules
    )
    
    # Set up portfolio and execution (simplified versions)
    portfolio = SimplePortfolio(
        event_queue=event_queue,
        initial_capital=100000.0  # $100k starting capital
    )
    
    execution_handler = SimpleExecutionHandler(
        event_queue=event_queue
    )
    
    # Containers to track events
    market_events = []
    signal_events = []
    order_events = []
    fill_events = []
    
    # Main event loop
    print("Starting backtest event loop...")
    event_counts = {
        'Market': 0,
        'Signal': 0,
        'Order': 0,
        'Fill': 0
    }
    
    while data_handler.continue_backtest:
        # Update bars (generates MarketEvents)
        data_handler.update_bars()
        
        # Process all events in the queue
        while not event_queue.empty():
            event = event_queue.get()
            
            if event.type == EventType.MARKET:
                # Handle market data event
                strategy.handle_market_event(event)
                market_events.append(event)
                event_counts['Market'] += 1
                
                # Update execution handler with latest price data
                if hasattr(execution_handler, 'update_prices'):
                    execution_handler.update_prices(
                        event.symbol,
                        data_handler.get_latest_bars(event.symbol)
                    )
            
            elif event.type == EventType.SIGNAL:
                # Process signal to generate orders
                portfolio.process_signal(event)
                signal_events.append(event)
                event_counts['Signal'] += 1
                
                # Print signal details
                print(f"Signal: {'BUY' if event.signal_type > 0 else 'SELL'} {event.symbol} @ {event.datetime}")
            
            elif event.type == EventType.ORDER:
                # Execute order
                execution_handler.execute_order(event)
                order_events.append(event)
                event_counts['Order'] += 1
            
            elif event.type == EventType.FILL:
                # Update portfolio with fill information
                if hasattr(portfolio, 'process_fill'):
                    portfolio.process_fill(event)
                fill_events.append(event)
                event_counts['Fill'] += 1
    
    # Print summary statistics
    print("\nBacktest completed!")
    print(f"Processed events: {sum(event_counts.values())}")
    print(f"  - Market events: {event_counts['Market']}")
    print(f"  - Signal events: {event_counts['Signal']}")
    print(f"  - Order events: {event_counts['Order']}")
    print(f"  - Fill events: {event_counts['Fill']}")
    
    # Calculate basic performance if portfolio tracking is implemented
    if hasattr(portfolio, 'get_performance_stats'):
        performance = portfolio.get_performance_stats()
        print("\nPerformance Summary:")
        for key, value in performance.items():
            print(f"  - {key}: {value:.4f}")
    
    # Prepare results
    results = {
        'symbol': symbol,
        'market_events': market_events,
        'signal_events': signal_events,
        'order_events': order_events,
        'fill_events': fill_events,
        'event_counts': event_counts
    }
    
    # Add performance metrics if available
    if hasattr(portfolio, 'equity_curve'):
        results['equity_curve'] = portfolio.equity_curve
    
    # Visualize signals if there are any
    if len(signal_events) > 0:
        plot_signals(symbol, signal_events, data_handler)
    
    return results


def plot_signals(symbol, signal_events, data_handler):
    """
    Create a simple visualization of trading signals.
    
    Args:
        symbol: Symbol that was traded
        signal_events: List of signal events
        data_handler: Data handler with price data
    """
    try:
        # Get price data
        if hasattr(data_handler, 'symbol_data') and symbol in data_handler.symbol_data:
            price_data = data_handler.symbol_data[symbol]
            
            # Extract signal information
            signal_dates = [event.datetime for event in signal_events]
            signal_types = [event.signal_type for event in signal_events]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot price
            plt.plot(price_data.index, price_data['Close'], label='Close Price')
            
            # Plot buy signals
            buy_dates = [date for i, date in enumerate(signal_dates) if signal_types[i] > 0]
            if buy_dates:
                buy_prices = [price_data.loc[date, 'Close'] if date in price_data.index else None for date in buy_dates]
                buy_prices = [price for price in buy_prices if price is not None]
                if buy_prices and len(buy_dates) == len(buy_prices):
                    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_dates = [date for i, date in enumerate(signal_dates) if signal_types[i] < 0]
            if sell_dates:
                sell_prices = [price_data.loc[date, 'Close'] if date in price_data.index else None for date in sell_dates]
                sell_prices = [price for price in sell_prices if price is not None]
                if sell_prices and len(sell_dates) == len(sell_prices):
                    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
            
            plt.title(f'{symbol} Price and Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save and show
            plt.savefig(f'{symbol}_signals.png')
            plt.show()
            
    except Exception as e:
        print(f"Error plotting signals: {e}")


if __name__ == "__main__":
    # Example usage:
    # Adjust these parameters as needed
    symbol = 'AAPL'  # Or any symbol you have data for
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)  # Using 1 year of data
    data_dir = 'data/'  # Path to your data directory
    
    # Run backtest
    results = run_simple_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir
    )
    
    # Access results for further analysis
    print(f"\nGenerated {len(results['signal_events'])} trading signals")

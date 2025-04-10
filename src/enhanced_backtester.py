# enhanced_backtester.py
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from events import EventQueue, EventType

class EnhancedBacktester:
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        self.event_queue = EventQueue()
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        
        # Performance tracking
        self.all_positions = []
        self.all_trades = []
    
    def run(self):
        """Run the backtest"""
        print("Starting enhanced backtest...")
        start_time = time.time()
        
        # Main event loop
        while self.data_handler.continue_backtest:
            # Update the market data (creates MarketEvents)
            self.data_handler.update_bars()
            
            # Process all events
            while not self.event_queue.empty():
                event = self.event_queue.get()
                
                if event.type == EventType.MARKET:
                    # 1. Update strategy data and calculate signals
                    self.strategy.update_data(event)
                    self.strategy.calculate_signals(event)
                    
                    # 2. Update execution handler with latest prices
                    symbol = event.symbol
                    self.execution_handler.update_prices(
                        symbol, 
                        self.data_handler.get_latest_bars(symbol)
                    )
                
                elif event.type == EventType.SIGNAL:
                    # Process signal to generate orders
                    self.portfolio.process_signal(event)
                    
                    # Track signal
                    self.all_positions.append({
                        'datetime': event.datetime,
                        'symbol': event.symbol,
                        'signal': event.signal_type
                    })
                
                elif event.type == EventType.ORDER:
                    # Execute order
                    self.execution_handler.execute_order(event)
                
                elif event.type == EventType.FILL:
                    # Process fill (in a real system, would update portfolio)
                    self.all_trades.append({
                        'datetime': event.timestamp,
                        'symbol': event.symbol,
                        'direction': event.direction,
                        'quantity': event.quantity,
                        'price': event.fill_price,
                        'commission': event.commission
                    })
        
        # Create DataFrames from collected data
        positions_df = pd.DataFrame(self.all_positions)
        trades_df = pd.DataFrame(self.all_trades)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Backtest completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(positions_df)} signals and {len(trades_df)} trades")
        
        # Simple performance metrics
        if not trades_df.empty:
            total_profit = self._calculate_profit(trades_df)
            print(f"Total profit: ${total_profit:.2f}")
        
        return {
            'positions': positions_df,
            'trades': trades_df
        }
    
    def _calculate_profit(self, trades_df):
        """Calculate simple P&L from trades"""
        if trades_df.empty:
            return 0
        
        # Calculate P&L for each trade
        trades_df['pnl'] = -trades_df['direction'] * trades_df['quantity'] * trades_df['price'] - trades_df['commission']
        
        # Sum all P&Ls
        return trades_df['pnl'].sum()

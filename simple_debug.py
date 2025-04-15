#!/usr/bin/env python3
# simple_debug.py - A simplified version of main.py to debug trading

import datetime
import os
import logging
import pandas as pd
import numpy as np

# Import essential components
from src.events import Event, EventType
from src.config import ConfigManager
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource
from always_buy_rule import AlwaysBuyRule
from src.signals import Signal, SignalType
from src.position_management.portfolio import Portfolio
from src.position_management.position_sizers import PercentOfEquitySizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None):
    """Create synthetic price data for testing."""
    # Create a synthetic dataset
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Create a price series with some randomness and a trend
    base_price = 100
    prices = [base_price]
    
    # Add a trend with noise
    for i in range(1, len(dates)):
        # Random daily change between -1% and +1% with a slight upward bias
        daily_change = np.random.normal(0.0005, 0.01) 
        
        # Add some regime changes to test rules
        if i % 60 == 0:  # Every ~2 months, change trend
            daily_change = -0.02 if prices[-1] > base_price * 1.1 else 0.02
            
        new_price = prices[-1] * (1 + daily_change)
        prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(100000, 20000)) for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} bars of data")
    
    return df

def update_portfolio_prices(portfolio, prices_dict):
    """
    Update portfolio with current prices (substitute for mark_to_market).
    
    Args:
        portfolio: Portfolio object
        prices_dict: Dictionary mapping symbols to prices
    """
    # Log portfolio before update
    logger.info(f"Portfolio before update: Cash=${portfolio.cash:.2f}, Positions={len(portfolio.positions)}")
    
    # Calculate total position value before update
    total_position_value_before = 0
    for _, position in portfolio.positions.items():
        if hasattr(position, 'last_price') and position.last_price:
            position_value = position.quantity * position.last_price
        else:
            position_value = position.quantity * position.entry_price
        total_position_value_before += position_value
    
    logger.info(f"Total position value before: ${total_position_value_before:.2f}")
    
    # Update prices for open positions
    total_unrealized_pnl = 0
    for pos_id, position in list(portfolio.positions.items()):
        if position.symbol in prices_dict:
            current_price = prices_dict[position.symbol]
            
            # Update position's last known price
            position.last_price = current_price
            
            # Calculate P&L
            if position.direction > 0:  # Long
                pnl = (current_price - position.entry_price) * position.quantity
            else:  # Short
                pnl = (position.entry_price - current_price) * position.quantity
                
            # Update position's unrealized P&L
            position.unrealized_pnl = pnl
            position.unrealized_pnl_pct = pnl / (position.entry_price * position.quantity) * 100
            
            # Add to total
            total_unrealized_pnl += pnl
            
            logger.info(f"Updated position {pos_id}: {position.symbol} at ${current_price:.2f}, P&L: ${pnl:.2f}")
    
    # Manually update portfolio's equity to include unrealized P&L
    current_equity = portfolio.cash + total_position_value_before + total_unrealized_pnl
    
    logger.info(f"Cash: ${portfolio.cash:.2f}")
    logger.info(f"Unrealized P&L: ${total_unrealized_pnl:.2f}")
    logger.info(f"Current equity: ${current_equity:.2f}")
    
    # Patch the portfolio's get_performance_metrics method to include unrealized P&L
    original_get_performance_metrics = portfolio.get_performance_metrics
    
    def patched_get_performance_metrics():
        metrics = original_get_performance_metrics()
        metrics['unrealized_pnl'] = total_unrealized_pnl
        metrics['current_equity'] = portfolio.cash + total_position_value_before + total_unrealized_pnl
        return metrics
    
    portfolio.get_performance_metrics = patched_get_performance_metrics

def main():
    """Simplified main function for debugging."""
    logger.info("Starting simplified debug script")
    
    # 1. Load configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    
    # 2. Set up market data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}.csv"
    
    # Check if we need to create synthetic data
    if not os.path.exists(filename):
        create_synthetic_data(symbol, timeframe, filename)
    
    # 3. Set up data sources and handler
    data_source = CSVDataSource(".")  # Look for CSV files in current directory
    data_handler = DataHandler(data_source)
    
    # 4. Load market data
    logger.info("Loading market data")
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    
    # Load data using the DataHandler
    data_handler.load_data(
        symbols=[symbol],  # DataHandler expects a list
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    logger.info(f"Successfully loaded {len(data_handler.train_data)} bars of data")
    
    # 5. Create portfolio
    portfolio = Portfolio(initial_capital=config.get('backtester.initial_capital'))
    logger.info(f"Portfolio initialized with ${portfolio.cash:.2f} cash")
    
    # 6. Create our AlwaysBuyRule
    always_buy_rule = AlwaysBuyRule(
        name="always_buy_debug",
        params={
            'frequency': 2,  # Generate a signal every 2 bars
            'confidence': 1.0  # Full confidence for testing
        },
        description="Debug rule that always generates buy signals"
    )
    
    # 7. Process all bars
    count = 0
    signals_generated = 0
    positions_opened = 0
    
    for bar in data_handler.iter_train():
        # Create a bar event
        bar_event = Event(EventType.BAR, bar)
        
        # Process with rule
        signal = always_buy_rule.on_bar(bar_event)
        
        # If signal was generated, process it
        if signal is not None:
            signals_generated += 1
            logger.info(f"Signal generated at bar {count}: {signal.signal_type} at price {signal.price}")
            
            # Create position directly in portfolio
            if signal.signal_type == SignalType.BUY:
                # Calculate position size (fixed for now)
                position_size = 100  # Fixed size
                
                # Check if we have the symbol in the bar data
                bar_symbol = bar.get('symbol')
                if not bar_symbol:
                    bar_symbol = symbol  # Use default symbol if not in bar
                
                # Open position in portfolio
                try:
                    # Use direction=1 for long positions
                    position = portfolio.open_position(
                        symbol=bar_symbol,
                        direction=1,  # Long position for BUY signal
                        quantity=position_size,
                        entry_price=bar['Close'],
                        entry_time=bar['timestamp']
                    )
                    positions_opened += 1
                    logger.info(f"Opened position: {position_size} shares of {bar_symbol} at ${bar['Close']:.2f}")
                except Exception as e:
                    logger.error(f"Error opening position: {e}", exc_info=True)
            
            elif signal.signal_type == SignalType.SELL:
                # We could handle closing positions here if needed
                pass
        
        # Update count
        count += 1
        
        # Log status periodically
        if count % 20 == 0:
            # Update portfolio with current prices
            update_portfolio_prices(portfolio, {symbol: bar['Close']})
            
            # Log stats
            metrics = portfolio.get_performance_metrics()
            logger.info(f"Processed {count} bars of data")
            logger.info(f"Current equity: ${metrics['current_equity']:.2f}")
            logger.info(f"Open positions: {len(portfolio.positions)}")
    
    # 8. Log final results
    logger.info("Backtest complete")
    logger.info(f"Total bars processed: {count}")
    logger.info(f"Total signals generated: {signals_generated}")
    logger.info(f"Total positions opened: {positions_opened}")
    
    # Calculate final portfolio value
    # Make sure we have the final prices
    final_bar = list(data_handler.iter_train())[-1]
    update_portfolio_prices(portfolio, {symbol: final_bar['Close']})
    
    # Get final metrics
    metrics = portfolio.get_performance_metrics()
    final_equity = metrics['current_equity']
    initial_equity = config.get('backtester.initial_capital')
    total_return = (final_equity / initial_equity - 1) * 100
    
    logger.info(f"Initial equity: ${initial_equity:.2f}")
    logger.info(f"Final equity: ${final_equity:.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    
    # Calculate total unrealized P&L
    total_unrealized_pnl = 0
    for pos_id, position in portfolio.positions.items():
        if hasattr(position, 'unrealized_pnl'):
            total_unrealized_pnl += position.unrealized_pnl
    
    logger.info(f"Total unrealized P&L: ${total_unrealized_pnl:.2f}")
    
    # Log position details
    if portfolio.positions:
        logger.info("Open positions:")
        for pos_id, position in portfolio.positions.items():
            logger.info(f"  - {position.symbol}: {position.quantity} shares at {position.entry_price:.2f}")
            if hasattr(position, 'unrealized_pnl'):
                logger.info(f"    P&L: ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2f}%)")
    
    return {
        "status": "backtest_complete",
        "bars_processed": count,
        "signals_generated": signals_generated,
        "positions_opened": positions_opened,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_unrealized_pnl": total_unrealized_pnl
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\nScript executed successfully!")
        print(f"Bars processed: {result['bars_processed']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Positions opened: {result['positions_opened']}")
        print(f"Final equity: ${result['final_equity']:.2f}")
        print(f"Total unrealized P&L: ${result['total_unrealized_pnl']:.2f}")
        print(f"Total return: {result['total_return']:.2f}%")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

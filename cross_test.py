#!/usr/bin/env python3
# sma_crossover_test.py - Test script for SMAcrossoverRule

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
from src.rules import create_rule
from src.signals import Signal, SignalType
from src.position_management.portfolio import Portfolio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None):
    """Create synthetic price data for testing with clearer trends for SMA crossover."""
    # Create a synthetic dataset
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Create a price series with a clearer trend pattern for SMA crossover
    base_price = 100
    prices = []
    
    # Generate price data with alternating trends
    for i in range(len(dates)):
        # Create a rising trend for first 60 days
        if i < 60:
            trend = 0.5  # Strong uptrend
        # Then a falling trend for next 60 days
        elif i < 120:
            trend = -0.3  # Downtrend
        # Then another rising trend
        elif i < 180:
            trend = 0.4  # Uptrend
        # Then sideways/slightly down
        elif i < 240:
            trend = -0.1  # Slight downtrend
        # Then strong uptrend to finish
        else:
            trend = 0.6  # Strong uptrend
            
        # Add some randomness
        random_component = np.random.normal(0, 0.5)
        
        # Calculate daily change
        if i == 0:
            prices.append(base_price)
        else:
            daily_change = trend + random_component
            new_price = prices[-1] * (1 + daily_change/100)  # Smaller percentage changes
            prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
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
    """Main function to test SMAcrossoverRule."""
    logger.info("Starting SMA crossover rule test")
    
    # 1. Load configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    
    # 2. Set up market data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    
    # Create filename that follows the convention
    standard_filename = f"{symbol}_{timeframe}.csv"
    
    # Always create new synthetic data optimized for SMA crossover
    create_synthetic_data(symbol, timeframe, standard_filename)
    
    # 3. Set up data sources and handler
    data_source = CSVDataSource(".")  # Look for CSV files in current directory
    data_handler = DataHandler(data_source)
    
    # 4. Load market data
    logger.info("Loading market data")
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    
    # Load data using the DataHandler with the proper method signature
    # Note: load_data does not accept a filename parameter 
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
    
    # 6. Create SMAcrossoverRule
    try:
        sma_rule = create_rule('SMAcrossoverRule', {
            'fast_window': 10,
            'slow_window': 30,
            'smooth_signals': True  # Generate signals when MAs are aligned
        })
        logger.info(f"Created SMA crossover rule with fast_window=10, slow_window=30")
    except Exception as e:
        logger.error(f"Error creating SMA rule: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
    
    # 7. Process all bars
    count = 0
    signals_generated = 0
    positions_opened = 0
    position_size = 100  # Fixed size
    
    # Track fast and slow MAs for debugging
    fast_ma_values = []
    slow_ma_values = []
    prices = []
    
    for bar in data_handler.iter_train():
        # Create a bar event
        bar_event = Event(EventType.BAR, bar)
        
        # Process with rule
        signal = sma_rule.on_bar(bar_event)
        
        # Track prices and MAs
        prices.append(bar['Close'])
        
        # Calculate MAs for debugging (simple way, not the same as the rule's internal calc)
        if len(prices) >= 10:
            fast_ma = sum(prices[-10:]) / 10
            fast_ma_values.append(fast_ma)
        else:
            fast_ma_values.append(None)
            
        if len(prices) >= 30:
            slow_ma = sum(prices[-30:]) / 30
            slow_ma_values.append(slow_ma)
        else:
            slow_ma_values.append(None)
            
        # Log MA crossovers for debugging
        if len(fast_ma_values) > 1 and len(slow_ma_values) > 1 and fast_ma_values[-2] is not None and slow_ma_values[-2] is not None:
            prev_fast, prev_slow = fast_ma_values[-2], slow_ma_values[-2]
            curr_fast, curr_slow = fast_ma_values[-1], slow_ma_values[-1]
            
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                logger.info(f"Bullish crossover at bar {count}: Fast MA={curr_fast:.2f}, Slow MA={curr_slow:.2f}")
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                logger.info(f"Bearish crossover at bar {count}: Fast MA={curr_fast:.2f}, Slow MA={curr_slow:.2f}")
        
        # If signal was generated, process it
        if signal is not None:
            signals_generated += 1
            logger.info(f"Signal generated at bar {count}: {signal.signal_type} at price {signal.price}")
            
            # Create position directly in portfolio
            if signal.signal_type == SignalType.BUY:
                # Check if we have the symbol in the bar data
                bar_symbol = bar.get('symbol')
                if not bar_symbol:
                    bar_symbol = symbol  # Use default symbol if not in bar
                
                # Calculate position size
                available_cash = portfolio.cash
                required_cash = bar['Close'] * position_size
                
                # Only open position if we have enough cash
                if available_cash >= required_cash:
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
                        logger.info(f"Opened LONG position: {position_size} shares of {bar_symbol} at ${bar['Close']:.2f}")
                    except Exception as e:
                        logger.error(f"Error opening position: {e}", exc_info=True)
                else:
                    logger.warning(f"Insufficient cash for position: Required ${required_cash:.2f}, Available ${available_cash:.2f}")
            
            elif signal.signal_type == SignalType.SELL:
                # For simplicity, we'll open short positions
                bar_symbol = bar.get('symbol', symbol)
                
                # Calculate position size
                available_cash = portfolio.cash
                required_cash = bar['Close'] * position_size
                
                # Only open position if we have enough cash
                if available_cash >= required_cash:
                    try:
                        # Use direction=-1 for short positions
                        position = portfolio.open_position(
                            symbol=bar_symbol,
                            direction=-1,  # Short position for SELL signal
                            quantity=position_size,
                            entry_price=bar['Close'],
                            entry_time=bar['timestamp']
                        )
                        positions_opened += 1
                        logger.info(f"Opened SHORT position: {position_size} shares of {bar_symbol} at ${bar['Close']:.2f}")
                    except Exception as e:
                        logger.error(f"Error opening position: {e}", exc_info=True)
                else:
                    logger.warning(f"Insufficient cash for position: Required ${required_cash:.2f}, Available ${available_cash:.2f}")
        
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
            logger.info(f"  - {position.symbol}: {position.quantity} shares at {position.entry_price:.2f}, direction: {position.direction}")
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
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Bars processed: {result.get('bars_processed', 0)}")
        print(f"Signals generated: {result.get('signals_generated', 0)}")
        print(f"Positions opened: {result.get('positions_opened', 0)}")
        print(f"Final equity: ${result.get('final_equity', 0):.2f}")
        print(f"Total unrealized P&L: ${result.get('total_unrealized_pnl', 0):.2f}")
        print(f"Total return: {result.get('total_return', 0):.2f}%")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

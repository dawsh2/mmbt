#!/usr/bin/env python3
# Final main.py - adapted to use portfolio attributes directly rather than methods

import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Import from the working event system
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType
from src.events.event_handlers import LoggingHandler
from src.events.event_emitters import MarketDataEmitter

# Import other core components
from src.config import ConfigManager
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource

# Import strategy components
from src.rules import SMAcrossoverRule, RSIRule
from src.strategies import WeightedStrategy

# Import position and execution components
from src.position_management.position_manager import PositionManager
from src.position_management.portfolio import Portfolio
from src.position_management.position_sizers import VolatilityPositionSizer, PercentOfEquitySizer
from src.position_management.allocation import EqualWeightAllocation

# Import execution components
from src.engine import ExecutionEngine, MarketSimulator

# Import risk management components
from src.risk_management.types import RiskParameters, ExitReason
from src.risk_management.risk_manager import RiskManager

# Import analytics for tracking performance
from src.analytics.metrics import calculate_max_drawdown, calculate_metrics_from_trades

# For AlwaysBuyRule
from src.signals import Signal, SignalType

# Add this to your main.py file near the beginning, after imports
import logging

# Configure root logger to show INFO level messages
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BarEvent:
    """Simple wrapper to make bar data compatible with strategy interface."""
    def __init__(self, bar):
        self.bar = bar




def debug_portfolio(portfolio, symbol=None, current_price=None):
    """Print detailed information about the portfolio state"""
    print("\nPortfolio Debug Info:")
    
    # Access portfolio attributes directly rather than using methods
    if hasattr(portfolio, 'cash'):
        print(f"Cash: ${portfolio.cash:.2f}")
    else:
        print("Portfolio has no 'cash' attribute")
    
    # Get position value by summing position values or use get_position_value if available
    if hasattr(portfolio, 'get_position_value'):
        position_value = portfolio.get_position_value()
        print(f"Position value: ${position_value:.2f}")
    else:
        position_value = 0
        if hasattr(portfolio, 'positions'):
            for pos in portfolio.positions.values():
                if hasattr(pos, 'market_value'):
                    position_value += pos.market_value
                elif hasattr(pos, 'quantity') and hasattr(pos, 'current_price'):
                    position_value += pos.quantity * pos.current_price
        print(f"Calculated position value: ${position_value:.2f}")
    
    # Calculate total value
    total_value = position_value
    if hasattr(portfolio, 'cash'):
        total_value += portfolio.cash
    print(f"Total portfolio value: ${total_value:.2f}")
    
    # Show positions
    if hasattr(portfolio, 'positions'):
        print(f"Number of positions: {len(portfolio.positions)}")
        for pos_symbol, position in portfolio.positions.items():
            print(f"  Position {pos_symbol}: {position}")
    elif hasattr(portfolio, 'get_positions'):
        positions = portfolio.get_positions()
        print(f"Number of positions: {len(positions)}")
        for position in positions:
            print(f"  Position: {position}")
    
    # Show trades if available
    if hasattr(portfolio, 'trades'):
        print(f"Number of trades: {len(portfolio.trades)}")
    
    if symbol and current_price:
        print(f"Current price of {symbol}: ${current_price:.2f}")
    
    return total_value  # Return the total portfolio value for convenience




class AlwaysBuyRule:
    def __init__(self, name="always_buy"):
        self.name = name
    
    def on_bar(self, bar):
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=SignalType.BUY,
            price=bar["Close"],
            rule_id=self.name,
            confidence=1.0
        )
    
    def reset(self):
        pass


def main():
    print("Starting the trading system setup...")
    
    # 1. Initialize the event system
    print("Initializing event system")
    event_bus = EventBus()
    
    # Add a logging handler to see events - expand to include more event types for debugging
    event_types_to_log = [
        EventType.MARKET_OPEN, EventType.MARKET_CLOSE, 
        EventType.SIGNAL, EventType.ORDER, EventType.FILL, EventType.PARTIAL_FILL
    ]
    logging_handler = LoggingHandler(event_types_to_log)
    for event_type in event_types_to_log:
        event_bus.register(event_type, logging_handler)
    
    # 2. Create and configure the config manager
    print("Loading configuration")
    config = ConfigManager()
    
    # Set default configuration values
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    config.set('backtester.market_simulation.fee_model', 'fixed')
    config.set('backtester.market_simulation.fee_bps', 10)
    
    # 3. Create synthetic data if it doesn't exist
    print("Checking for test data")
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}.csv"
    
    if not os.path.exists(filename):
        print(f"Creating synthetic data file: {filename}")
        # Create a synthetic dataset
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        # Create a price series with some randomness and a trend
        base_price = 100
        prices = [base_price]
        for i in range(1, len(dates)):
            # Random daily change between -1% and +1% with a slight upward bias
            daily_change = np.random.normal(0.0005, 0.01)  
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'Close': prices,
            'Volume': [int(np.random.normal(100000, 20000)) for _ in prices]
        })
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Created synthetic data with {len(df)} bars")
    
    # 4. Set up data sources and handler
    print("Setting up data handler")
    data_source = CSVDataSource(".")  # Look for CSV files in the current directory
    data_handler = DataHandler(data_source)
    
    # 5. Load market data
    print(f"Loading data for symbol: {symbol}, timeframe: {timeframe}")
    try:
        start_date = datetime.datetime(2022, 1, 1)
        end_date = datetime.datetime(2022, 12, 31)
        
        # Load data using the DataHandler
        data_handler.load_data(
            symbols=[symbol],  # DataHandler expects a list
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        print(f"Successfully loaded data")
        
        # 6. Create rules and strategy
        print("Creating trading rules and strategy")
        
        # # Create individual trading rules - adjust parameters to create more signals
        # sma_rule = SMAcrossoverRule(
        #     name="sma_crossover", 
        #     params={"fast_window": 3, "slow_window": 10}  # Even shorter windows for more signals
        # )

        # rsi_rule = RSIRule(
        #     name="rsi_rule", 
        #     params={
        #         "rsi_period": 7,  # Shorter period for more signals
        #         "overbought": 65,  # Lower threshold for overbought
        #         "oversold": 35,   # Higher threshold for oversold
        #         "signal_type": "levels"
        #     }
        # )

        # Create a weighted strategy that combines these rules
        # strategy = WeightedStrategy(
        #     components=[sma_rule, rsi_rule],
        #     weights=[0.6, 0.4],  # 60% weight to SMA, 40% to RSI
        #     buy_threshold=0.3,    # Buy when weighted sum exceeds 0.3
        #     sell_threshold=-0.3,  # Sell when weighted sum is below -0.3
        #     name="basic_technical_strategy"
        # )

        # Then in your main.py:
        always_buy_rule = AlwaysBuyRule()

        # Create a strategy with just this rule
        test_strategy = WeightedStrategy(
            components=[always_buy_rule],
            weights=[1.0],
            buy_threshold=0.1,  # Very low threshold to ensure signals
            sell_threshold=-0.1,
            name="test_strategy"
        )


        strategy = test_strategy
        
        # 7. Set up risk management
        print("Setting up risk management")
        risk_params = RiskParameters(
            stop_loss_pct=2.0,                  # 2% stop loss
            take_profit_pct=4.0,                # 4% take profit
            trailing_stop_activation_pct=3.0,   # Activate trailing stop after 3% gain
            trailing_stop_distance_pct=1.5      # Trail by 1.5%
        )
        
        risk_manager = RiskManager(risk_params=risk_params)
        
        # 8. Set up portfolio and position management
        print("Setting up portfolio and position management")
        # Create portfolio with initial capital
        initial_capital = config.get('backtester.initial_capital')
        portfolio = Portfolio(initial_capital=initial_capital)
        
        # Create position sizer - more aggressive sizing for more trades
        position_sizer = PercentOfEquitySizer(percent=0.1)  # 10% of equity per position
        
        # Create allocation strategy
        allocation_strategy = EqualWeightAllocation()
        
        # Create position manager
        position_manager = PositionManager(
            portfolio=portfolio,
            position_sizer=position_sizer,
            allocation_strategy=allocation_strategy,
            risk_manager=risk_manager
        )
        
        # 9. Set up market simulator and execution engine
        print("Setting up execution engine")
        # Create market simulator for handling orders
        market_simulator = MarketSimulator({
            'slippage_model': config.get('backtester.market_simulation.slippage_model'),
            'slippage_bps': config.get('backtester.market_simulation.slippage_bps'),
            'fee_model': config.get('backtester.market_simulation.fee_model'),
            'fee_bps': config.get('backtester.market_simulation.fee_bps')
        })
        
        # Create execution engine
        execution_engine = ExecutionEngine(position_manager=position_manager)

        # 10. Register components with event system
        print("Registering components with event system")

        # Import the FillHandler class
        from src.events.event_handlers import FillHandler

        # Create a custom handler to debug signals
        def debug_signal_handler(event):
            signal = event.data
            print(f"\nDEBUG - Signal received: {signal}")
            if hasattr(signal, 'signal_type'):
                signal_type = signal.signal_type
                print(f"Signal type: {signal_type}")
                print(f"Signal value: {signal_type.value if hasattr(signal_type, 'value') else None}")
                print(f"Confidence: {signal.confidence if hasattr(signal, 'confidence') else None}")
                print(f"Metadata: {signal.metadata if hasattr(signal, 'metadata') else None}")

                # Convert Signal object to dictionary format expected by position_manager
                signal_dict = {
                    'symbol': 'SYNTHETIC',  # Hardcoded for now
                    'signal_type': signal.signal_type.value,  # Convert enum to value (-1, 0, 1)
                    'price': signal.price,
                    'confidence': signal.confidence,
                    'rule_id': signal.rule_id if hasattr(signal, 'rule_id') else 'default',
                    'timestamp': signal.timestamp
                }

                # Re-emit as a format position_manager expects
                event_bus.emit(Event(EventType.SIGNAL, signal_dict))
            else:
                print("Signal has no signal_type attribute")
            return event.data

 

        # Then register it like this:
        #event_bus.register(EventType.BAR, debug_strategy_output)
        # Add this debug wrapper around the strategy.on_bar call
        def debug_strategy_output(event):
            bar_event = BarEvent(event.data)
            result = strategy.on_bar(bar_event)
            print(f"\nDEBUG - Strategy on_bar result: {result}")
            print(f"Result type: {type(result)}")

            # Check if it's a Signal object
            if hasattr(result, 'signal_type'):
                print(f"Signal type: {result.signal_type}")
                # Emit a SIGNAL event with the result
                event_bus.emit(Event(EventType.SIGNAL, result))

            return None
        
        event_bus.register(EventType.BAR, debug_strategy_output)

        #event_bus.register(EventType.BAR, lambda event: event_bus.emit(debug_strategy_output(event)))


        # Strategy handles bar events and emits signals
        # event_bus.register(EventType.BAR, lambda event: event_bus.emit(
        #     Event(EventType.SIGNAL, strategy.on_bar(BarEvent(event.data)))
        # ))

        # Uncomment to see detailed signal information
        event_bus.register(EventType.SIGNAL, debug_signal_handler)

        # Position manager handles signal events and emits orders
        event_bus.register(EventType.SIGNAL, position_manager.on_signal)

        # Execution engine handles order events
        event_bus.register(EventType.ORDER, execution_engine.on_order)

        # Create and register a FillHandler to handle fill events
        fill_handler = FillHandler(position_manager)
        event_bus.register(EventType.FILL, fill_handler)
        event_bus.register(EventType.PARTIAL_FILL, fill_handler)
        
        # 11. Run the backtest
        print("\nRunning backtest...")
        market_data_emitter = MarketDataEmitter(event_bus)
        
        # Store portfolio values for equity curve - use initial capital as starting point
        timestamps = []
        portfolio_values = [initial_capital]  # Start with initial capital
        
        # Print initial portfolio state
        print("\nInitial portfolio state:")
        initial_portfolio_value = debug_portfolio(portfolio)
        
        # Emit market open event
        market_data_emitter.emit_market_open()

        # Add this after all your setup but before the backtest loop
        print("\nTesting position manager directly:")
        test_signal = {
            'symbol': 'SYNTHETIC',
            'signal_type': 1,  # BUY
            'direction': 1,    # BUY
            'price': 100.0,
            'confidence': 1.0,
            'timestamp': datetime.datetime.now()
        }

        # Call position manager directly
        result = position_manager.on_signal(test_signal)
        print(f"Direct position manager test result: {result}")
        
        # Process all bars
        bar_count = 0
        for bar in data_handler.iter_train():
            current_timestamp = pd.to_datetime(bar['timestamp'])
            current_price = bar['Close']
            
            # Update timestamps
            timestamps.append(current_timestamp)
            
            # Update prices in the portfolio
            portfolio.update_prices({symbol: current_price}, current_timestamp)
            
            # Get total portfolio value - use direct attribute access instead of methods
            total_value = 0
            if hasattr(portfolio, 'cash'):
                cash_value = portfolio.cash
                total_value += cash_value
            else:
                cash_value = 0
                
            position_value = 0
            if hasattr(portfolio, 'get_position_value'):
                position_value = portfolio.get_position_value()
                total_value += position_value
            elif hasattr(portfolio, 'positions'):
                for pos in portfolio.positions.values():
                    if hasattr(pos, 'market_value'):
                        position_value += pos.market_value
                    elif hasattr(pos, 'quantity') and hasattr(pos, 'current_price'):
                        position_value += pos.quantity * pos.current_price
                total_value += position_value
            
            # If total_value is 0 (no cash or positions found), use previous value
            if total_value == 0 and len(portfolio_values) > 0:
                total_value = portfolio_values[-1]
                
            # Print debug info every 50 bars or if there's a large change
            if bar_count % 50 == 0 or (len(portfolio_values) > 0 and 
                                       abs(total_value / portfolio_values[-1] - 1) > 0.05):
                print(f"\nBar {bar_count} @ {current_timestamp}")
                print(f"  Current price: ${current_price:.2f}")
                print(f"  Cash: ${cash_value:.2f}")
                print(f"  Position value: ${position_value:.2f}")
                print(f"  Total value: ${total_value:.2f}")
            
            portfolio_values.append(total_value)
            
            # Emit bar event
            market_data_emitter.emit_bar(bar)
            
            # Every 20 bars, print a progress update
            bar_count += 1
            if bar_count % 20 == 0:
                print(f"Processed {bar_count} bars...")
                
        # Emit market close event
        market_data_emitter.emit_market_close()
            
        print(f"\nBacktest complete! Processed {bar_count} bars.")
        
        # Final portfolio state
        print("\nFinal portfolio state:")
        final_portfolio_value = debug_portfolio(portfolio, symbol, current_price)
        
        # 12. Calculate and display performance metrics
        print("\n==== Performance Metrics ====")
        # Use the last portfolio value for final equity
        final_equity = portfolio_values[-1]
        roi_percent = (final_equity / initial_capital - 1) * 100
        
        # Calculate returns from equity values, handling potential division by zero or NaN
        if len(portfolio_values) > 1:
            # Remove zeros to avoid division by zero
            non_zero_values = np.array([v if v != 0 else np.nan for v in portfolio_values[:-1]])
            returns = np.diff(portfolio_values) / non_zero_values
            # Replace infinities and NaNs with zeros
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            returns = np.array([])
        
        # Get trade history from the execution engine
        try:
            trade_history = execution_engine.get_trade_history()
            print(f"Found {len(trade_history)} trades in execution engine")
        except AttributeError:
            print("Warning: execution_engine does not have get_trade_history method")
            trade_history = []
        
        # Convert trade history to the format expected by calculate_metrics_from_trades
        trade_tuples = []
        for trade in trade_history:
            try:
                # Convert to the format expected by analytics functions
                # (entry_time, direction, entry_price, exit_time, exit_price, log_return)
                if hasattr(trade, 'entry_price') and hasattr(trade, 'exit_price'):
                    entry_time = getattr(trade, 'entry_time', datetime.datetime.now())
                    direction = getattr(trade, 'direction', 'long')
                    entry_price = trade.entry_price
                    exit_time = getattr(trade, 'exit_time', datetime.datetime.now())
                    exit_price = trade.exit_price
                    
                    # Calculate log return based on direction
                    if entry_price > 0:
                        if direction == 'long' or direction == 1:
                            log_return = np.log(exit_price / entry_price)
                        else:
                            log_return = np.log(entry_price / exit_price)
                    else:
                        log_return = 0.0
                        
                    trade_tuple = (
                        entry_time,
                        direction,
                        entry_price,
                        exit_time,
                        exit_price,
                        log_return
                    )
                    trade_tuples.append(trade_tuple)
            except Exception as e:
                print(f"Error processing trade: {e}")
                continue
        
        # Initialize default metrics
        metrics = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'total_trades': len(trade_tuples),
            'win_rate': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'profit_factor': 0.0,
            'avg_return': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        # Calculate metrics if we have trades
        if trade_tuples:
            try:
                metrics = calculate_metrics_from_trades(trade_tuples)
            except Exception as e:
                print(f"Error calculating trade metrics: {e}")
        
        # Calculate max drawdown using the analytics function
        try:
            if len(portfolio_values) > 1:
                max_dd = calculate_max_drawdown(portfolio_values)
            else:
                max_dd = 0.0
        except Exception as e:
            print(f"Error calculating drawdown: {e}")
            max_dd = 0.0
        
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Equity: ${final_equity:,.2f}")
        print(f"Total Return: {roi_percent:.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0.0):.4f}")
        print(f"Maximum Drawdown: {max_dd:.2f}%")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0.0):.4f}")
        
        # Print trade statistics
        if trade_history:
            print(f"\nTotal Trades: {len(trade_history)}")
            print(f"Win Rate: {metrics.get('win_rate', 0.0):.2%}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
            print(f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
            print(f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")
            
            # Calculate average win/loss
            if metrics.get('win_count', 0) > 0 and metrics.get('win_rate', 0) > 0:
                print(f"Average Win: {metrics.get('avg_win', 0.0):.4f}")
            if metrics.get('loss_count', 0) > 0 and metrics.get('win_rate', 1.0) < 1:
                print(f"Average Loss: {metrics.get('avg_loss', 0.0):.4f}")
        
        # 13. Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, portfolio_values[1:])  # Skip initial capital at index 0
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        equity_curve_file = 'equity_curve.png'
        plt.savefig(equity_curve_file)
        print(f"\nEquity curve saved to {equity_curve_file}")
        
        # Plot trades on the equity curve if we have trade history
        if trade_history and timestamps:
            # Plot trades
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, portfolio_values[1:])  # Skip initial capital at index 0
            plt.title('Portfolio Equity Curve with Trades')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the figure
            trades_plot_file = 'equity_curve_with_trades.png'
            plt.savefig(trades_plot_file)
            print(f"Equity curve with trades saved to {trades_plot_file}")
        
        # Print the event statistics
        event_counts = {}
        for event in event_bus.history:
            event_type = event.event_type
            if event_type in event_counts:
                event_counts[event_type] += 1
            else:
                event_counts[event_type] = 1
        
        print("\n==== Event Statistics ====")
        for event_type, count in event_counts.items():
            print(f"{event_type.name}: {count}")
        
        # Print specific event counts that are important for debugging
        print("\n==== Event Flow Analysis ====")
        print(f"BAR events: {event_counts.get(EventType.BAR, 0)}")
        print(f"SIGNAL events: {event_counts.get(EventType.SIGNAL, 0)}")
        print(f"ORDER events: {event_counts.get(EventType.ORDER, 0)}")
        print(f"FILL events: {event_counts.get(EventType.FILL, 0)} + " 
              f"PARTIAL_FILL events: {event_counts.get(EventType.PARTIAL_FILL, 0)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Prepare return values, handling possible errors
    final_equity_value = None
    roi_percent_value = None
    trade_count = 0
    
    if 'portfolio_values' in locals() and portfolio_values:
        final_equity_value = portfolio_values[-1]
        
    if 'initial_capital' in locals() and final_equity_value is not None:
        roi_percent_value = (final_equity_value / initial_capital - 1) * 100
        
    if 'trade_history' in locals():
        trade_count = len(trade_history)
    
    return {
        "status": "backtest_complete", 
        "data_loaded": True,
        "events_emitted": len(event_bus.history) if 'event_bus' in locals() else 0,
        "final_equity": final_equity_value,
        "roi_percent": roi_percent_value,
        "trade_count": trade_count
    }


if __name__ == "__main__":
    try:
        result = main()
        print("\nScript executed successfully!")
        print(f"Status: {result['status']}")
        print(f"Events emitted: {result['events_emitted']}")
        if result.get('final_equity'):
            print(f"Final portfolio value: ${result['final_equity']:,.2f}")
            print(f"Return on investment: {result.get('roi_percent', 0.0):.2f}%")
            print(f"Total trades executed: {result['trade_count']}")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

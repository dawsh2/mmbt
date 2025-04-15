#!/usr/bin/env python3
# opt_test.py - Script for rule testing with optimization

import datetime
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Import system components
from src.events import Event, EventType
from src.config import ConfigManager
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource
from src.rules import create_rule, Rule
from src.rules.rule_factory import RuleFactory
from src.optimization import OptimizerManager, OptimizationMethod
from src.optimization.optimizer_manager import OptimizationMethod
from src.signals import Signal, SignalType
from src.position_management.portfolio import Portfolio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None, days=365):
    """Create synthetic price data for testing with trends suitable for rules testing."""
    # Create a synthetic dataset
    dates = pd.date_range(start='2022-01-01', end=pd.Timestamp('2022-01-01') + pd.Timedelta(days=days-1), freq='D')
    
    # Create a price series with a clearer trend pattern
    base_price = 100
    prices = []
    
    # Generate price data with alternating trends
    segment_size = days // 6  # Six different trend segments
    
    for i in range(len(dates)):
        # Create different trend segments
        if i < segment_size:
            trend = 0.5  # Strong uptrend
        elif i < segment_size * 2:
            trend = -0.3  # Downtrend
        elif i < segment_size * 3:
            trend = 0.4  # Uptrend
        elif i < segment_size * 4:
            trend = -0.1  # Slight downtrend
        elif i < segment_size * 5:
            trend = 0.0  # Sideways
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
    Update portfolio with current prices.
    
    Args:
        portfolio: Portfolio object
        prices_dict: Dictionary mapping symbols to prices
    """
    # Log portfolio before update
    logger.debug(f"Portfolio before update: Cash=${portfolio.cash:.2f}, Positions={len(portfolio.positions)}")
    
    # Calculate total position value before update
    total_position_value_before = 0
    for _, position in portfolio.positions.items():
        if hasattr(position, 'last_price') and position.last_price:
            position_value = position.quantity * position.last_price
        else:
            position_value = position.quantity * position.entry_price
        total_position_value_before += position_value
    
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
    
    # Manually update portfolio's equity to include unrealized P&L
    current_equity = portfolio.cash + total_position_value_before + total_unrealized_pnl
    
    logger.debug(f"Cash: ${portfolio.cash:.2f}")
    logger.debug(f"Unrealized P&L: ${total_unrealized_pnl:.2f}")
    logger.debug(f"Current equity: ${current_equity:.2f}")
    
    # Patch the portfolio's get_performance_metrics method to include unrealized P&L
    original_get_performance_metrics = portfolio.get_performance_metrics
    
    def patched_get_performance_metrics():
        metrics = original_get_performance_metrics()
        metrics['unrealized_pnl'] = total_unrealized_pnl
        metrics['current_equity'] = portfolio.cash + total_position_value_before + total_unrealized_pnl
        return metrics
    
    portfolio.get_performance_metrics = patched_get_performance_metrics

def backtest_rule(rule, data_handler, initial_capital=100000, position_size=100):
    """
    Run a backtest for a single rule.
    
    Args:
        rule: Rule instance to test
        data_handler: DataHandler with loaded data
        initial_capital: Starting capital
        position_size: Fixed position size
    
    Returns:
        dict: Results dictionary with performance metrics
    """
    # Create portfolio
    portfolio = Portfolio(initial_capital=initial_capital)
    logger.info(f"Backtesting rule: {rule.name}")
    
    # Reset rule state to ensure clean start
    rule.reset()
    
    # Statistics tracking
    count = 0
    signals_generated = 0
    positions_opened = 0
    equity_curve = []
    
    # Process all bars
    for bar in data_handler.iter_train():
        # Create a bar event and process with rule
        bar_event = Event(EventType.BAR, bar)
        signal = rule.on_bar(bar_event)
        
        # If signal was generated, process it
        if signal is not None and signal.signal_type != SignalType.NEUTRAL:
            signals_generated += 1
            logger.debug(f"Signal generated at bar {count}: {signal.signal_type} at price {signal.price}")
            
            # Create position directly in portfolio
            if signal.signal_type == SignalType.BUY:
                # Check if we have the symbol in the bar data
                bar_symbol = bar.get('symbol', 'SYNTHETIC')
                
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
                        logger.debug(f"Opened LONG position: {position_size} shares of {bar_symbol} at ${bar['Close']:.2f}")
                    except Exception as e:
                        logger.error(f"Error opening position: {e}")
                
            elif signal.signal_type == SignalType.SELL:
                # For simplicity, we'll open short positions
                bar_symbol = bar.get('symbol', 'SYNTHETIC')
                
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
                        logger.debug(f"Opened SHORT position: {position_size} shares of {bar_symbol} at ${bar['Close']:.2f}")
                    except Exception as e:
                        logger.error(f"Error opening position: {e}")
        
        # Update portfolio with current prices
        update_portfolio_prices(portfolio, {bar.get('symbol', 'SYNTHETIC'): bar['Close']})
        
        # Record equity curve
        metrics = portfolio.get_performance_metrics()
        equity_curve.append(metrics['current_equity'])
        
        # Update count
        count += 1
        
        # Log status periodically
        if count % 50 == 0:
            logger.info(f"Processed {count} bars, current equity: ${metrics['current_equity']:.2f}")
    
    # Get final metrics
    final_metrics = portfolio.get_performance_metrics()
    final_equity = final_metrics['current_equity']
    
    # Calculate returns and other metrics
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Calculate Sharpe ratio (simplified)
    equity_returns = [0]
    for i in range(1, len(equity_curve)):
        ret = (equity_curve[i] / equity_curve[i-1]) - 1
        equity_returns.append(ret)
    
    sharpe_ratio = 0
    if len(equity_returns) > 1:
        sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)  # Annualized
    
    # Calculate drawdown
    max_drawdown = 0
    peak = equity_curve[0]
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Create and return results
    results = {
        "rule_name": rule.name,
        "rule_params": rule.params,
        "bars_processed": count,
        "signals_generated": signals_generated,
        "positions_opened": positions_opened,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve
    }
    
    logger.info(f"Backtest results for {rule.name}:")
    logger.info(f"Total return: {total_return:.2f}%")
    logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
    logger.info(f"Max drawdown: {max_drawdown:.2f}%")
    logger.info(f"Signals generated: {signals_generated}")
    logger.info(f"Positions opened: {positions_opened}")
    
    return results

def evaluate_rule_performance(rule, data_handler):
    """
    Evaluation function for the optimizer.
    
    Args:
        rule: Rule to evaluate
        data_handler: DataHandler with loaded data
        
    Returns:
        float: Performance score (Sharpe ratio)
    """
    # Run backtest
    results = backtest_rule(rule, data_handler)
    
    # Use Sharpe ratio as performance metric
    return results['sharpe_ratio']

def plot_equity_curves(results_list, title="Equity Curves Comparison"):
    """
    Plot equity curves for multiple backtest results.
    
    Args:
        results_list: List of backtest result dictionaries
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        label = f"{results['rule_name']} (Return: {results['total_return']:.2f}%, Sharpe: {results['sharpe_ratio']:.2f})"
        plt.plot(results['equity_curve'], label=label)
    
    plt.title(title)
    plt.xlabel("Trading Days")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig("equity_curves.png")
    logger.info("Saved equity curves plot to 'equity_curves.png'")
    
    return plt.gcf()

def main():
    """Main function to demonstrate rule usage and parameter optimization."""
    logger.info("Starting rule testing and optimization")
    
    # 1. Create or load synthetic data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    
    # Create filename following the convention
    standard_filename = f"{symbol}_{timeframe}.csv"
    
    # Create synthetic data if it doesn't exist
    if not os.path.exists(standard_filename):
        create_synthetic_data(symbol, timeframe, standard_filename, days=500)
    
    # 2. Create configuration
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    
    # 3. Set up data sources and handler
    data_source = CSVDataSource(".")  # Look for CSV files in current directory
    data_handler = DataHandler(data_source)
    
    # 4. Load market data
    logger.info("Loading market data")
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)  # Adjust based on your data
    
    # Load data
    data_handler.load_data(
        symbols=[symbol],  # DataHandler expects a list
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    logger.info(f"Successfully loaded {len(data_handler.train_data)} bars of data")
    
    # 5. Create rules to test
    logger.info("Creating test rules")
    
    # A. SMA Crossover rule with default parameters
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 10,
        'slow_window': 30,
        'smooth_signals': True
    })
    
    # B. RSI rule with default parameters
    rsi_rule = create_rule('RSIRule', {
    'rsi_period': 14,
        'overbought': 70,
        'oversold': 30,
        'signal_type': 'levels'  # Add this parameter
    })

    
    # 6. Run backtests for default parameters
    logger.info("Running backtests with default parameters")
    
    sma_results = backtest_rule(sma_rule, data_handler)
    rsi_results = backtest_rule(rsi_rule, data_handler)
    
    # 7. Plot equity curves for default parameters
    plot_equity_curves([sma_results, rsi_results], "Default Parameters Comparison")
    
    # 8. Set up optimizer
    logger.info("Setting up optimizer")
    optimizer = OptimizerManager(data_handler)
    
    # Define evaluation function
    def evaluate_rule(rule):
        return evaluate_rule_performance(rule, data_handler)
    
    # 9. Register SMA rule for optimization
    optimizer.register_rule(
        "sma_rule",
        sma_rule.__class__,  # Use the class of the rule
        {
            'fast_window': [5, 10, 15, 20],
            'slow_window': [20, 30, 40, 50],
            'smooth_signals': [True, False]
        }
    )
    
    # 10. Register RSI rule for optimization
    optimizer.register_rule(
        "rsi_rule",
        rsi_rule.__class__,  # Use the class of the rule
        {
            'rsi_period': [7, 14, 21],
            'overbought': [65, 70, 75, 80],
            'oversold': [20, 25, 30, 35]
        }
    )
    
    # 11. Run optimization for both rules
    logger.info("Optimizing rules")
    
    # Note: This uses grid search by default
    optimized_rules = optimizer.optimize(
        component_type='rule',
        method=OptimizationMethod.GRID_SEARCH,
        metrics='sharpe',  # Using Sharpe ratio as optimization metric
        verbose=True
    )
    
    # 12. Get the optimized rules and their parameters
    optimized_sma = None
    optimized_rsi = None
    
    # Extract the optimized rules
    for rule in optimized_rules.values():
        if isinstance(rule, sma_rule.__class__):
            optimized_sma = rule
        elif isinstance(rule, rsi_rule.__class__):
            optimized_rsi = rule
    
    if not optimized_sma or not optimized_rsi:
        logger.warning("Optimization did not return expected rules")
        return
    
    # 13. Run backtests with optimized parameters
    logger.info("Running backtests with optimized parameters")
    
    optimized_sma_results = backtest_rule(optimized_sma, data_handler)
    optimized_rsi_results = backtest_rule(optimized_rsi, data_handler)
    
    # 14. Plot final comparison
    all_results = [
        sma_results, 
        rsi_results, 
        optimized_sma_results, 
        optimized_rsi_results
    ]
    
    plot_equity_curves(all_results, "Default vs Optimized Parameters")
    
    # 15. Print summary
    logger.info("\nParameter Optimization Summary:")
    logger.info("-" * 50)
    
    logger.info("SMA Crossover Rule:")
    logger.info(f"  Default parameters: {sma_rule.params}")
    logger.info(f"  Default performance: Return={sma_results['total_return']:.2f}%, Sharpe={sma_results['sharpe_ratio']:.2f}")
    logger.info(f"  Optimized parameters: {optimized_sma.params}")
    logger.info(f"  Optimized performance: Return={optimized_sma_results['total_return']:.2f}%, Sharpe={optimized_sma_results['sharpe_ratio']:.2f}")
    logger.info(f"  Improvement: Return +{optimized_sma_results['total_return'] - sma_results['total_return']:.2f}%, Sharpe +{optimized_sma_results['sharpe_ratio'] - sma_results['sharpe_ratio']:.2f}")
    
    logger.info("\nRSI Rule:")
    logger.info(f"  Default parameters: {rsi_rule.params}")
    logger.info(f"  Default performance: Return={rsi_results['total_return']:.2f}%, Sharpe={rsi_results['sharpe_ratio']:.2f}")
    logger.info(f"  Optimized parameters: {optimized_rsi.params}")
    logger.info(f"  Optimized performance: Return={optimized_rsi_results['total_return']:.2f}%, Sharpe={optimized_rsi_results['sharpe_ratio']:.2f}")
    logger.info(f"  Improvement: Return +{optimized_rsi_results['total_return'] - rsi_results['total_return']:.2f}%, Sharpe +{optimized_rsi_results['sharpe_ratio'] - rsi_results['sharpe_ratio']:.2f}")
    
    return {
        "default_results": {
            "sma": sma_results,
            "rsi": rsi_results
        },
        "optimized_results": {
            "sma": {
                "params": optimized_sma.params,
                "results": optimized_sma_results
            },
            "rsi": {
                "params": optimized_rsi.params,
                "results": optimized_rsi_results
            }
        }
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\nScript executed successfully!")
        
        # Print best parameters
        print("\nBest SMA Parameters:")
        for param, value in results["optimized_results"]["sma"]["params"].items():
            print(f"  {param}: {value}")
        
        print("\nBest RSI Parameters:")
        for param, value in results["optimized_results"]["rsi"]["params"].items():
            print(f"  {param}: {value}")
        
        print("\nCheck equity_curves.png for a visual comparison")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

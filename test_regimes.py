#!/usr/bin/env python3
"""
Walk-Forward Analysis Tool

This script implements walk-forward analysis to validate the effectiveness of
regime-aware parameter optimization without overfitting.

The walk-forward process:
1. Divides data into sequential time windows
2. For each window:
   a. Uses the first portion for in-sample (IS) optimization
   b. Uses the remaining portion for out-of-sample (OOS) validation
3. Compares performance across all OOS periods to assess robustness
"""

import datetime
import logging
import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from your codebase
from src.events.event_bus import EventBus
from src.events.event_types import EventType
from src.events.event_base import Event
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import FixedSizeSizer, PercentOfEquitySizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.engine.backtester import Backtester
from src.engine.market_simulator import MarketSimulator

# Import the regime optimizer and strategy
from regime_optimization import RegimeOptimizer
from regime_filter_strategy import RegimeAwareSMACrossover, MarketRegimeFilter


class WalkForwardAnalyzer:
    """
    Implements walk-forward analysis for algorithmic trading strategies.
    """
    
    def __init__(self, data_handler, symbols=None, window_size=None, train_pct=0.7):
        """
        Initialize walk-forward analyzer.
        
        Args:
            data_handler: Data handler with loaded data
            symbols: Symbols to trade
            window_size: Size of each walk-forward window in bars (default: 252)
            train_pct: Percentage of window to use for training (default: 0.7)
        """
        self.data_handler = data_handler
        self.symbols = symbols or ["SPY"]
        self.window_size = window_size or 252  # Default: 1 trading year
        self.train_pct = train_pct
        self.window_results = []
        
        # Calculate window split sizes
        self.train_size = int(self.window_size * self.train_pct)
        self.test_size = self.window_size - self.train_size
        
        logger.info(f"Walk-forward analysis initialized with window size {self.window_size}")
        logger.info(f"Training size: {self.train_size} bars ({self.train_pct*100:.0f}%)")
        logger.info(f"Testing size: {self.test_size} bars ({(1-self.train_pct)*100:.0f}%)")
    
    def create_window_data_handler(self, bars, is_training=True):
        """
        Create a data handler for a specific window of data.
        
        Args:
            bars: List of bars in the window
            is_training: Whether this is a training window
            
        Returns:
            Data handler for the window
        """
        # Choose appropriate bars based on window type
        if is_training:
            window_bars = bars[:self.train_size]
        else:
            window_bars = bars[self.train_size:self.window_size]
        
        # Create a custom data handler
        class WindowDataHandler:
            def __init__(self, bars):
                self.bars = bars
                self.index = 0
            
            def get_next_train_bar(self):
                if self.index < len(self.bars):
                    bar = self.bars[self.index]
                    self.index += 1
                    return bar
                return None
            
            def get_next_test_bar(self):
                return self.get_next_train_bar()
            
            def reset_train(self):
                self.index = 0
            
            def reset_test(self):
                self.index = 0
            
            def iter_train(self, use_bar_events=True):
                self.reset_train()
                while True:
                    bar = self.get_next_train_bar()
                    if bar is None:
                        break
                    yield bar
        
        return WindowDataHandler(window_bars)
    
    def collect_window_data(self):
        """
        Collect bar data and organize into windows.
        
        Returns:
            Dictionary with window data by symbol
        """
        logger.info("Collecting data and organizing into windows")
        
        # Get all bars for each symbol
        window_data = {}
        for symbol in self.symbols:
            # Reset data handler
            self.data_handler.reset_train()
            
            # Collect all bars for this symbol
            symbol_bars = []
            while True:
                bar = self.data_handler.get_next_train_bar()
                if bar is None:
                    break
                
                # Filter by symbol
                if bar.get_symbol() == symbol:
                    symbol_bars.append(bar)
            
            # Sort bars by timestamp
            symbol_bars.sort(key=lambda x: x.get_timestamp())
            
            # Calculate number of windows
            total_bars = len(symbol_bars)
            num_windows = max(1, (total_bars - self.window_size) // self.test_size + 1)
            
            logger.info(f"Collected {total_bars} bars for {symbol}")
            logger.info(f"Creating {num_windows} windows with {self.window_size} bars each")
            
            # Create windows
            windows = []
            for i in range(num_windows):
                start_idx = i * self.test_size
                end_idx = start_idx + self.window_size
                
                # Make sure we have enough bars
                if end_idx > total_bars:
                    logger.warning(f"Window {i+1} has insufficient data, skipping")
                    continue
                
                # Extract window bars
                window_bars = symbol_bars[start_idx:end_idx]
                
                # Add to windows
                windows.append({
                    'window_id': i + 1,
                    'bars': window_bars,
                    'start_date': window_bars[0].get_timestamp(),
                    'end_date': window_bars[-1].get_timestamp()
                })
            
            # Store windows for this symbol
            window_data[symbol] = windows
        
        return window_data
    
    def optimize_window(self, window, symbol, param_grid):
        """
        Optimize strategy parameters for a specific window.
        
        Args:
            window: Window data
            symbol: Symbol to optimize for
            param_grid: Parameter grid for optimization
            
        Returns:
            Dictionary with optimization results
        """
        # Create data handler for training portion
        train_data_handler = self.create_window_data_handler(window['bars'], is_training=True)
        
        # Create optimizer with training data
        optimizer = RegimeOptimizer(train_data_handler, [symbol])
        
        # Detect regimes
        optimizer.detect_regimes()
        
        # Find optimal parameters for each regime
        optimal_params = optimizer.optimize_all_regimes(param_grid)
        
        # Return optimization results
        return {
            'window_id': window['window_id'],
            'start_date': window['start_date'],
            'end_date': window['end_date'],
            'optimal_params': optimal_params
        }
    
    def test_window(self, window, symbol, optimal_params):
        """
        Test optimized parameters on out-of-sample data.
        
        Args:
            window: Window data
            symbol: Symbol to test for
            optimal_params: Optimal parameters from training
            
        Returns:
            Dictionary with test results
        """
        # Create data handler for testing portion
        test_data_handler = self.create_window_data_handler(window['bars'], is_training=False)
        
        # Create regime-aware strategy
        regime_params = {
            'regime_filter': {
                'lookback_period': 30,
                'volatility_threshold': 1.5,
                'trend_threshold': 0.6
            },
            'fast_window': 10,  # Default values
            'slow_window': 30
        }
        
        # Create event system
        event_bus = EventBus()
        event_bus.event_counts = {}
        
        # Create portfolio
        initial_capital = 100000
        portfolio = EventPortfolio(
            initial_capital=initial_capital,
            event_bus=event_bus
        )
        
        # Create position manager
        position_sizer = PercentOfEquitySizer(percent=2.0, max_pct=20.0)
        position_manager = PositionManager(
            portfolio=portfolio,
            position_sizer=position_sizer,
            event_bus=event_bus
        )
        
        # Create market simulator
        market_sim_config = {
            'slippage_model': 'fixed',
            'slippage_bps': 1,
            'fee_model': 'fixed',
            'fee_bps': 1
        }
        market_simulator = MarketSimulator(market_sim_config)
        
        # Create regime-aware strategy
        strategy = RegimeAwareSMACrossover(
            name="regime_aware_sma",
            params=regime_params,
            event_bus=event_bus
        )
        
        # Replace the default regime filter parameter map with optimized one
        strategy.regime_filter._optimal_params_map = optimal_params
        
        # Create backtester
        backtester_config = {
            'backtester': {
                'initial_capital': initial_capital,
                'market_simulation': market_sim_config
            }
        }
        
        # Create execution engine
        execution_engine = ExecutionEngine(position_manager=position_manager)
        execution_engine.portfolio = portfolio
        execution_engine.event_bus = event_bus
        execution_engine.market_simulator = market_simulator
        
        # Create backtester
        backtester = Backtester(
            config=backtester_config,
            data_handler=test_data_handler,
            strategy=strategy,
            position_manager=position_manager
        )
        backtester.execution_engine = execution_engine
        
        # Register event handlers
        event_bus.register(EventType.SIGNAL, position_manager.on_signal)
        event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
        event_bus.register(EventType.ORDER, execution_engine.on_order)
        
        if hasattr(portfolio, 'handle_fill') and callable(portfolio.handle_fill):
            event_bus.register(EventType.FILL, portfolio.handle_fill)
        
        if hasattr(portfolio, 'handle_position_action') and callable(portfolio.handle_position_action):
            event_bus.register(EventType.POSITION_ACTION, portfolio.handle_position_action)
        
        # Run backtest
        results = backtester.run(use_test_data=False)
        
        # Extract metrics
        total_return = (portfolio.equity / portfolio.initial_capital - 1) * 100
        
        # Calculate Sharpe ratio
        sharpe_ratio = results.get('sharpe_ratio', 0)
        if sharpe_ratio == 0 and 'portfolio_history' in results:
            # Calculate manually if not provided
            history = results['portfolio_history']
            if len(history) > 1:
                try:
                    equity_values = [h.get('equity', 0) for h in history]
                    returns = np.diff(equity_values) / equity_values[:-1]
                    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                except:
                    sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = results.get('max_drawdown', 0)
        if max_drawdown == 0 and 'portfolio_history' in results:
            history = results['portfolio_history']
            if len(history) > 1:
                try:
                    equity_values = [h.get('equity', 0) for h in history]
                    peak = equity_values[0]
                    drawdowns = []
                    for equity in equity_values:
                        if equity > peak:
                            peak = equity
                        drawdown = (peak - equity) / peak if peak > 0 else 0
                        drawdowns.append(drawdown)
                    max_drawdown = max(drawdowns) if drawdowns else 0
                except:
                    max_drawdown = 0
        
        # Count trades
        trades = results.get('trades', [])
        num_trades = len(trades)
        
        # Count winning trades
        winning_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Store regime distribution if available
        regime_distribution = {}
        if hasattr(strategy.regime_filter, 'current_regime'):
            for symbol, regime in strategy.regime_filter.current_regime.items():
                regime_distribution[symbol] = regime
        
        # Return test results
        return {
            'window_id': window['window_id'],
            'start_date': window['start_date'],
            'end_date': window['end_date'],
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_equity': portfolio.equity,
            'regime_distribution': regime_distribution
        }
    
    def run_analysis(self, param_grid=None):
        """
        Run complete walk-forward analysis.
        
        Args:
            param_grid: Parameter grid for optimization
            
        Returns:
            Dictionary with analysis results
        """
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'fast_window': [5, 8, 10, 12, 15, 20],
                'slow_window': [15, 20, 25, 30, 40, 50, 60]
            }
        
        # Collect data into windows
        window_data = self.collect_window_data()
        
        # Run analysis for each symbol
        for symbol in self.symbols:
            if symbol not in window_data or not window_data[symbol]:
                logger.warning(f"No window data for {symbol}, skipping")
                continue
            
            logger.info(f"Running walk-forward analysis for {symbol} with {len(window_data[symbol])} windows")
            
            # Process each window
            for window in window_data[symbol]:
                logger.info(f"Processing window {window['window_id']} "
                           f"({window['start_date']} to {window['end_date']})")
                
                # Optimize parameters on training portion
                logger.info("Optimizing parameters on training data...")
                opt_results = self.optimize_window(window, symbol, param_grid)
                
                # Test on out-of-sample portion
                logger.info("Testing on out-of-sample data...")
                test_results = self.test_window(window, symbol, opt_results['optimal_params'])
                
                # Combine results
                window_result = {
                    'window_id': window['window_id'],
                    'symbol': symbol,
                    'start_date': window['start_date'],
                    'end_date': window['end_date'],
                    'optimal_params': opt_results['optimal_params'],
                    'performance': test_results
                }
                
                # Log window results
                logger.info(f"Window {window['window_id']} results: "
                           f"Return: {test_results['total_return']:.2f}%, "
                           f"Sharpe: {test_results['sharpe_ratio']:.2f}, "
                           f"Trades: {test_results['num_trades']}")
                
                # Add to results
                self.window_results.append(window_result)
        
        # Calculate aggregate statistics
        agg_stats = self.calculate_aggregate_stats()
        
        # Plot results
        self.plot_walk_forward_results()
        
        return {
            'window_results': self.window_results,
            'aggregate_stats': agg_stats
        }
    
    def calculate_aggregate_stats(self):
        """
        Calculate aggregate statistics across all walk-forward windows.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.window_results:
            return {}
        
        # Extract metrics from all windows
        returns = [w['performance']['total_return'] for w in self.window_results]
        sharpe_ratios = [w['performance']['sharpe_ratio'] for w in self.window_results]
        drawdowns = [w['performance']['max_drawdown'] for w in self.window_results]
        trades = [w['performance']['num_trades'] for w in self.window_results]
        win_rates = [w['performance']['win_rate'] for w in self.window_results]
        
        # Calculate statistics
        stats = {
            'num_windows': len(self.window_results),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'pct_profitable_windows': sum(1 for r in returns if r > 0) / len(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'avg_drawdown': np.mean(drawdowns),
            'avg_trades': np.mean(trades),
            'total_trades': sum(trades),
            'avg_win_rate': np.mean(win_rates)
        }
        
        # Log aggregate statistics
        logger.info("\nWalk-Forward Analysis Aggregate Statistics:")
        logger.info(f"Number of windows: {stats['num_windows']}")
        logger.info(f"Average return: {stats['avg_return']:.2f}%")
        logger.info(f"Median return: {stats['median_return']:.2f}%")
        logger.info(f"Return std dev: {stats['return_std']:.2f}%")
        logger.info(f"Min/Max return: {stats['min_return']:.2f}% / {stats['max_return']:.2f}%")
        logger.info(f"Profitable windows: {stats['pct_profitable_windows']*100:.2f}%")
        logger.info(f"Average Sharpe ratio: {stats['avg_sharpe']:.2f}")
        logger.info(f"Average max drawdown: {stats['avg_drawdown']*100:.2f}%")
        logger.info(f"Average trades per window: {stats['avg_trades']:.2f}")
        logger.info(f"Total trades across all windows: {stats['total_trades']}")
        logger.info(f"Average win rate: {stats['avg_win_rate']*100:.2f}%")
        
        return stats
    
    def plot_walk_forward_results(self, save_plot=True):
        """
        Plot walk-forward analysis results.
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            True if successful, False otherwise
        """
        if not self.window_results:
            logger.warning("No results to plot")
            return False
        
        try:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot configuration
            plt.subplot(2, 2, 1)
            window_ids = [w['window_id'] for w in self.window_results]
            returns = [w['performance']['total_return'] for w in self.window_results]
            plt.bar(window_ids, returns)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Returns by Window')
            plt.xlabel('Window ID')
            plt.ylabel('Return (%)')
            plt.grid(True)
            
            # Plot Sharpe ratios
            plt.subplot(2, 2, 2)
            sharpe_ratios = [w['performance']['sharpe_ratio'] for w in self.window_results]
            plt.bar(window_ids, sharpe_ratios)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Sharpe Ratio by Window')
            plt.xlabel('Window ID')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            
            # Plot max drawdowns
            plt.subplot(2, 2, 3)
            drawdowns = [w['performance']['max_drawdown'] * 100 for w in self.window_results]
            plt.bar(window_ids, drawdowns)
            plt.title('Maximum Drawdown by Window')
            plt.xlabel('Window ID')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            
            # Plot trade counts
            plt.subplot(2, 2, 4)
            trades = [w['performance']['num_trades'] for w in self.window_results]
            win_rates = [w['performance']['win_rate'] * 100 for w in self.window_results]
            
            # Create twin axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot trade counts on primary axis
            ax1.bar(window_ids, trades, color='b', alpha=0.5)
            ax1.set_xlabel('Window ID')
            ax1.set_ylabel('Number of Trades', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot win rates on secondary axis
            ax2.plot(window_ids, win_rates, 'r-', marker='o')
            ax2.set_ylabel('Win Rate (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add title
            plt.title('Trade Counts and Win Rates by Window')
            
            # Adjust layout
            plt.tight_layout()
            
            # Calculate aggregate stats
            stats = self.calculate_aggregate_stats()
            
            # Add suptitle with aggregate statistics
            plt.suptitle(f"Walk-Forward Analysis Results\n"
                        f"Avg Return: {stats['avg_return']:.2f}%, "
                        f"Profitable Windows: {stats['pct_profitable_windows']*100:.2f}%, "
                        f"Avg Sharpe: {stats['avg_sharpe']:.2f}", 
                        fontsize=16, y=1.05)
            
            # Save plot if requested
            if save_plot:
                plt.savefig('walk_forward_results.png', bbox_inches='tight')
                logger.info("Saved plot to walk_forward_results.png")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            return False
    
    def run_comparison_benchmark(self, benchmark_params=None):
        """
        Run benchmark comparison against fixed-parameter strategy.
        
        Args:
            benchmark_params: Parameters for benchmark strategy
            
        Returns:
            Dictionary with benchmark results
        """
        # Default benchmark parameters if not provided
        if benchmark_params is None:
            benchmark_params = {
                'fast_window': 10,
                'slow_window': 30
            }
        
        logger.info(f"Running benchmark comparison with fixed parameters: {benchmark_params}")
        
        # Collect data into windows
        window_data = self.collect_window_data()
        
        # Track results
        benchmark_results = []
        
        # Run benchmark for each symbol
        for symbol in self.symbols:
            if symbol not in window_data or not window_data[symbol]:
                logger.warning(f"No window data for {symbol}, skipping")
                continue
            
            logger.info(f"Running benchmark for {symbol} with {len(window_data[symbol])} windows")
            
            # Process each window
            for window in window_data[symbol]:
                logger.info(f"Processing window {window['window_id']} for benchmark")
                
                # Create data handler for testing portion (OOS)
                test_data_handler = self.create_window_data_handler(window['bars'], is_training=False)
                
                # Create event system
                event_bus = EventBus()
                event_bus.event_counts = {}
                
                # Create portfolio
                initial_capital = 100000
                portfolio = EventPortfolio(
                    initial_capital=initial_capital,
                    event_bus=event_bus
                )
                
                # Create position manager
                position_sizer = PercentOfEquitySizer(percent=2.0, max_pct=20.0)
                position_manager = PositionManager(
                    portfolio=portfolio,
                    position_sizer=position_sizer,
                    event_bus=event_bus
                )
                
                # Create market simulator
                market_sim_config = {
                    'slippage_model': 'fixed',
                    'slippage_bps': 1,
                    'fee_model': 'fixed',
                    'fee_bps': 1
                }
                market_simulator = MarketSimulator(market_sim_config)
                
                # Create fixed parameter strategy
                strategy = SMACrossoverRule(
                    name="benchmark_sma",
                    params=benchmark_params,
                    event_bus=event_bus
                )
                
                # Create backtester
                backtester_config = {
                    'backtester': {
                        'initial_capital': initial_capital,
                        'market_simulation': market_sim_config
                    }
                }
                
                # Create execution engine
                execution_engine = ExecutionEngine(position_manager=position_manager)
                execution_engine.portfolio = portfolio
                execution_engine.event_bus = event_bus
                execution_engine.market_simulator = market_simulator
                
                # Create backtester
                backtester = Backtester(
                    config=backtester_config,
                    data_handler=test_data_handler,
                    strategy=strategy,
                    position_manager=position_manager
                )
                backtester.execution_engine = execution_engine
                
                # Register event handlers
                event_bus.register(EventType.SIGNAL, position_manager.on_signal)
                event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
                event_bus.register(EventType.ORDER, execution_engine.on_order)
                
                if hasattr(portfolio, 'handle_fill') and callable(portfolio.handle_fill):
                    event_bus.register(EventType.FILL, portfolio.handle_fill)
                
                if hasattr(portfolio, 'handle_position_action') and callable(portfolio.handle_position_action):
                    event_bus.register(EventType.POSITION_ACTION, portfolio.handle_position_action)
                
                # Run backtest
                results = backtester.run(use_test_data=False)
                
                # Extract metrics
                total_return = (portfolio.equity / portfolio.initial_capital - 1) * 100
                
                # Calculate Sharpe ratio
                sharpe_ratio = results.get('sharpe_ratio', 0)
                if sharpe_ratio == 0 and 'portfolio_history' in results:
                    history = results['portfolio_history']
                    if len(history) > 1:
                        try:
                            equity_values = [h.get('equity', 0) for h in history]
                            returns = np.diff(equity_values) / equity_values[:-1]
                            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                        except:
                            sharpe_ratio = 0
                
                # Count trades
                trades = results.get('trades', [])
                num_trades = len(trades)
                
                # Count winning trades
                winning_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
                win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
                
                # Store benchmark result
                benchmark_result = {
                    'window_id': window['window_id'],
                    'symbol': symbol,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'final_equity': portfolio.equity
                }
                
                benchmark_results.append(benchmark_result)

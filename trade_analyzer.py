"""
Trade Analyzer Module for Algorithmic Trading Systems

This module provides comprehensive analysis and visualization tools for evaluating
trading strategy performance, including advanced metrics, trade statistics,
and visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union, Any, Optional
import math
from collections import defaultdict

class TradeAnalyzer:
    """
    Analyzer for trading strategy performance.
    
    This class provides tools to calculate various performance metrics,
    analyze trade statistics, and generate visualizations from backtest results.
    """
    
    def __init__(self, backtest_results=None):
        """
        Initialize the trade analyzer.
        
        Args:
            backtest_results: Optional dictionary containing backtest results to analyze
        """
        self.backtest_results = backtest_results
        self.trades_df = None
        self.equity_curve = None
        self.daily_returns = None
        
        if backtest_results is not None and 'trades' in backtest_results:
            self._prepare_data()
    
    def set_backtest_results(self, backtest_results):
        """
        Set backtest results to analyze.
        
        Args:
            backtest_results: Dictionary containing backtest results
        """
        self.backtest_results = backtest_results
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare data structures for analysis from backtest results.
        """
        if not self.backtest_results or 'trades' not in self.backtest_results:
            return
            
        # Convert trades to DataFrame for easier analysis
        trades = self.backtest_results['trades']
        trades_data = []
        
        for t in trades:
            # Handle different trade tuple formats by checking length
            if len(t) >= 6:  # Basic format with entry/exit info
                entry_time = t[0]
                direction = t[1]
                entry_price = t[2]
                exit_time = t[3]
                exit_price = t[4]
                log_return = t[5]
                
                # Extract entry/exit signals if available
                entry_signal = t[6] if len(t) > 6 else None
                exit_signal = t[7] if len(t) > 7 else None
                
                trades_data.append({
                    'entry_time': entry_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'log_return': log_return,
                    'percent_return': (math.exp(log_return) - 1) * 100,
                    'duration': self._calculate_duration(entry_time, exit_time),
                    'entry_signal': entry_signal,
                    'exit_signal': exit_signal,
                    'is_win': log_return > 0
                })
                
        # Create DataFrame
        self.trades_df = pd.DataFrame(trades_data)
        
        # Calculate equity curve
        self._calculate_equity_curve()
    
    def _calculate_duration(self, entry_time, exit_time):
        """
        Calculate trade duration in days.
        
        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            
        Returns:
            float: Duration in days
        """
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    entry_time = datetime.strptime(entry_time, "%Y-%m-%d")
                except ValueError:
                    return np.nan
            
        if isinstance(exit_time, str):
            try:
                exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    exit_time = datetime.strptime(exit_time, "%Y-%m-%d")
                except ValueError:
                    return np.nan
        
        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            duration = (exit_time - entry_time).total_seconds() / (24 * 3600)
            return duration
        return np.nan
    
    def _calculate_equity_curve(self):
        """
        Calculate equity curve and daily returns from trades.
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            return
            
        # Sort trades by entry time
        sorted_trades = self.trades_df.sort_values('entry_time')
        
        # Create initial equity curve with just trade points
        equity = [1.0]  # Start with $1
        timestamps = [sorted_trades.iloc[0]['entry_time']]
        
        for _, trade in sorted_trades.iterrows():
            equity.append(equity[-1] * math.exp(trade['log_return']))
            timestamps.append(trade['exit_time'])
        
        # Create equity curve DataFrame
        self.equity_curve = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity[:-1]  # Remove the last duplicate point
        })
        
        # Convert timestamps to datetime if they are strings
        if isinstance(self.equity_curve['timestamp'].iloc[0], str):
            self.equity_curve['timestamp'] = pd.to_datetime(self.equity_curve['timestamp'])
        
        # Sort by timestamp to ensure chronological order
        self.equity_curve = self.equity_curve.sort_values('timestamp')
        
        # Calculate daily returns
        try:
            # Resample to daily frequency and calculate returns
            self.equity_curve.set_index('timestamp', inplace=True)
            daily_equity = self.equity_curve.resample('D').last()
            daily_equity = daily_equity.fillna(method='ffill')
            self.daily_returns = daily_equity['equity'].pct_change().dropna()
        except Exception as e:
            print(f"Warning: Failed to calculate daily returns: {str(e)}")
            self.daily_returns = None
    
    def calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'average_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'profit_factor': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_trade_duration': 0
            }
        
        # Basic metrics
        total_trades = len(self.trades_df)
        winning_trades = self.trades_df[self.trades_df['log_return'] > 0]
        losing_trades = self.trades_df[self.trades_df['log_return'] <= 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Return metrics
        total_log_return = self.trades_df['log_return'].sum()
        total_return = (math.exp(total_log_return) - 1) * 100  # percentage
        average_log_return = self.trades_df['log_return'].mean()
        average_return = (math.exp(average_log_return) - 1) * 100  # percentage
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
        
        # Trading metrics
        profit_factor = self._calculate_profit_factor(winning_trades, losing_trades)
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_wins_losses()
        avg_trade_duration = self.trades_df['duration'].mean()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'average_return': average_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_duration': avg_trade_duration
        }
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0, annualization_factor=252):
        """
        Calculate Sharpe ratio based on trade returns.
        
        Args:
            risk_free_rate: Risk-free rate (default is 0)
            annualization_factor: Factor to annualize returns (default is 252 trading days)
            
        Returns:
            float: Sharpe ratio
        """
        if self.daily_returns is None or len(self.daily_returns) < 2:
            return 0
        
        excess_returns = self.daily_returns - (risk_free_rate / annualization_factor)
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(annualization_factor)
        return sharpe
    
    def _calculate_sortino_ratio(self, risk_free_rate=0, annualization_factor=252):
        """
        Calculate Sortino ratio based on trade returns.
        
        Args:
            risk_free_rate: Risk-free rate (default is 0)
            annualization_factor: Factor to annualize returns (default is 252 trading days)
            
        Returns:
            float: Sortino ratio
        """
        if self.daily_returns is None or len(self.daily_returns) < 2:
            return 0
        
        excess_returns = self.daily_returns - (risk_free_rate / annualization_factor)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
            
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
        return sortino
    
    def _calculate_max_drawdown(self):
        """
        Calculate maximum drawdown percentage.
        
        Returns:
            float: Maximum drawdown as a percentage
        """
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0
        
        equity = self.equity_curve['equity']
        max_dd = 0
        peak = equity.iloc[0]
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # Convert to percentage
    
    def _calculate_calmar_ratio(self, annual_return, max_drawdown):
        """
        Calculate Calmar ratio.
        
        Args:
            annual_return: Annualized return percentage
            max_drawdown: Maximum drawdown percentage
            
        Returns:
            float: Calmar ratio
        """
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0
        return annual_return / max_drawdown
    
    def _calculate_profit_factor(self, winning_trades, losing_trades):
        """
        Calculate profit factor.
        
        Args:
            winning_trades: DataFrame of winning trades
            losing_trades: DataFrame of losing trades
            
        Returns:
            float: Profit factor
        """
        total_profits = winning_trades['log_return'].sum()
        total_losses = abs(losing_trades['log_return'].sum())
        
        if total_losses == 0:
            return float('inf') if total_profits > 0 else 0
        return total_profits / total_losses
    
    def _calculate_consecutive_wins_losses(self):
        """
        Calculate maximum consecutive wins and losses.
        
        Returns:
            tuple: (max_consecutive_wins, max_consecutive_losses)
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0, 0
            
        # Get trade results in chronological order
        results = self.trades_df.sort_values('entry_time')['is_win'].astype(int).tolist()
        
        # Calculate consecutive wins
        consecutive_wins = 0
        max_consecutive_wins = 0
        
        for r in results:
            if r == 1:  # Win
                consecutive_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_wins = 0
        
        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for r in results:
            if r == 0:  # Loss
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive_wins, max_consecutive_losses
    
    def analyze_trade_durations(self):
        """
        Analyze trade durations.
        
        Returns:
            dict: Statistics on trade durations
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            return {
                'avg_duration': 0,
                'median_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'duration_bins': [],
                'duration_counts': []
            }
        
        durations = self.trades_df['duration'].dropna()
        
        if len(durations) == 0:
            return {
                'avg_duration': 0,
                'median_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'duration_bins': [],
                'duration_counts': []
            }
        
        # Calculate basic statistics
        avg_duration = durations.mean()
        median_duration = durations.median()
        min_duration = durations.min()
        max_duration = durations.max()
        
        # Create duration bins
        if max_duration > 0:
            bins = np.linspace(0, max_duration, min(10, len(durations)))
            counts, bin_edges = np.histogram(durations, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        else:
            bin_centers = []
            counts = []
        
        return {
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'duration_bins': bin_centers.tolist(),
            'duration_counts': counts.tolist()
        }
    
    def analyze_trade_distribution(self):
        """
        Analyze the distribution of trade returns.
        
        Returns:
            dict: Statistics on trade return distribution
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            return {
                'mean_return': 0,
                'median_return': 0,
                'std_return': 0,
                'skew': 0,
                'kurtosis': 0,
                'return_bins': [],
                'return_counts': []
            }
        
        returns = self.trades_df['log_return']
        
        # Calculate statistics
        mean_return = returns.mean()
        median_return = returns.median()
        std_return = returns.std()
        
        # Calculate higher moments if enough data
        if len(returns) > 3:
            from scipy import stats
            skew = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
        else:
            skew = 0
            kurtosis = 0
        
        # Create return bins
        min_return = returns.min()
        max_return = returns.max()
        margin = (max_return - min_return) * 0.1  # Add 10% margin
        bins = np.linspace(min_return - margin, max_return + margin, min(20, len(returns)))
        counts, bin_edges = np.histogram(returns, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        return {
            'mean_return': mean_return,
            'median_return': median_return,
            'std_return': std_return,
            'skew': skew,
            'kurtosis': kurtosis,
            'return_bins': bin_centers.tolist(),
            'return_counts': counts.tolist()
        }
    
    def analyze_monthly_performance(self):
        """
        Analyze performance by month.
        
        Returns:
            dict: Monthly performance metrics
        """
        if self.trades_df is None or len(self.trades_df) == 0 or self.equity_curve is None:
            return {
                'monthly_returns': {}
            }
        
        # Try to resample equity curve to monthly frequency
        try:
            # Create a copy with index that can be reset
            equity_df = self.equity_curve.copy()
            
            # Make sure we have a datetime index
            if not isinstance(equity_df.index, pd.DatetimeIndex):
                return {'monthly_returns': {}}
            
            # Resample to month-end
            monthly_equity = equity_df['equity'].resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Convert to dictionary format
            monthly_dict = {}
            for date, ret in monthly_returns.items():
                month_key = date.strftime('%Y-%m')
                monthly_dict[month_key] = ret * 100  # Convert to percentage
                
            return {
                'monthly_returns': monthly_dict
            }
            
        except Exception as e:
            print(f"Warning: Failed to calculate monthly performance: {str(e)}")
            return {
                'monthly_returns': {}
            }
    
    def analyze_drawdowns(self, top_n=5):
        """
        Analyze drawdowns.
        
        Args:
            top_n: Number of largest drawdowns to analyze
            
        Returns:
            dict: Drawdown analysis results
        """
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'drawdown_count': 0,
                'top_drawdowns': []
            }
        
        equity = self.equity_curve['equity'].values
        timestamps = self.equity_curve.index
        
        # Find drawdowns
        drawdowns = []
        peak_idx = 0
        peak_value = equity[0]
        in_drawdown = False
        current_drawdown = None
        
        for i in range(1, len(equity)):
            if equity[i] > peak_value:
                # New peak
                peak_idx = i
                peak_value = equity[i]
                
                # End any current drawdown
                if in_drawdown and current_drawdown:
                    current_drawdown['recovery_time'] = timestamps[i]
                    recovery_days = (timestamps[i] - current_drawdown['start_time']).days
                    current_drawdown['duration_days'] = recovery_days
                    drawdowns.append(current_drawdown)
                    current_drawdown = None
                    in_drawdown = False
                    
            elif equity[i] < peak_value:
                # Calculate drawdown percentage
                dd_pct = (peak_value - equity[i]) / peak_value * 100
                
                if not in_drawdown:
                    # Start new drawdown
                    in_drawdown = True
                    current_drawdown = {
                        'start_time': timestamps[peak_idx],
                        'low_time': timestamps[i],
                        'drawdown_pct': dd_pct,
                        'peak_value': peak_value,
                        'low_value': equity[i]
                    }
                elif dd_pct > current_drawdown['drawdown_pct']:
                    # Update existing drawdown if deeper
                    current_drawdown['low_time'] = timestamps[i]
                    current_drawdown['drawdown_pct'] = dd_pct
                    current_drawdown['low_value'] = equity[i]
        
        # Handle any unrecovered drawdown at the end
        if in_drawdown and current_drawdown:
            current_drawdown['recovery_time'] = None
            current_drawdown['duration_days'] = (timestamps[-1] - current_drawdown['start_time']).days
            drawdowns.append(current_drawdown)
        
        # Sort drawdowns by percentage
        drawdowns.sort(key=lambda x: x['drawdown_pct'], reverse=True)
        
        # Calculate statistics
        max_drawdown = drawdowns[0]['drawdown_pct'] if drawdowns else 0
        avg_drawdown = np.mean([dd['drawdown_pct'] for dd in drawdowns]) if drawdowns else 0
        
        # Format top drawdowns for return
        top_drawdowns = []
        for i, dd in enumerate(drawdowns[:top_n]):
            recovery_time = dd['recovery_time'].strftime('%Y-%m-%d') if dd['recovery_time'] else 'Not Recovered'
            top_drawdowns.append({
                'rank': i + 1,
                'start_date': dd['start_time'].strftime('%Y-%m-%d'),
                'low_date': dd['low_time'].strftime('%Y-%m-%d'),
                'recovery_date': recovery_time,
                'duration_days': dd['duration_days'],
                'drawdown_pct': dd['drawdown_pct']
            })
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_count': len(drawdowns),
            'top_drawdowns': top_drawdowns
        }
    
    def analyze_by_regime(self, regime_data):
        """
        Analyze strategy performance by market regime.
        
        Args:
            regime_data: Dictionary mapping timestamps to regime types
            
        Returns:
            dict: Performance metrics by regime
        """
        if self.trades_df is None or len(self.trades_df) == 0 or not regime_data:
            return {'regime_metrics': {}}
        
        # Convert regime data to DataFrame for easier alignment
        regime_df = pd.DataFrame.from_dict(regime_data, orient='index', columns=['regime'])
        regime_df.index = pd.to_datetime(regime_df.index)
        
        # Group trades by regime
        trades_by_regime = defaultdict(list)
        
        for _, trade in self.trades_df.iterrows():
            # Convert entry time to datetime if needed
            entry_time = pd.to_datetime(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
            
            # Find the regime for this trade
            try:
                # Get closest regime before trade entry
                closest_regime = regime_df.loc[:entry_time].iloc[-1]['regime']
                trades_by_regime[closest_regime].append(trade.to_dict())
            except (IndexError, KeyError):
                # If regime not found, mark as 'unknown'
                trades_by_regime['unknown'].append(trade.to_dict())
        
        # Calculate performance metrics for each regime
        regime_metrics = {}
        
        for regime, trades in trades_by_regime.items():
            if not trades:
                continue
                
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            win_rate = len(trades_df[trades_df['log_return'] > 0]) / total_trades if total_trades > 0 else 0
            total_return = (math.exp(trades_df['log_return'].sum()) - 1) * 100
            avg_return = trades_df['log_return'].mean()
            
            regime_metrics[regime] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return
            }
        
        return {'regime_metrics': regime_metrics}
    
    def analyze_strategy_attribution(self, strategy_signals):
        """
        Analyze performance attribution for multiple strategy components.
        
        Args:
            strategy_signals: Dictionary mapping strategy names to signal series
            
        Returns:
            dict: Performance attribution by strategy component
        """
        if self.trades_df is None or len(self.trades_df) == 0 or not strategy_signals:
            return {'attribution': {}}
        
        # Convert signal data to DataFrame
        signals_df = pd.DataFrame(strategy_signals)
        
        # Calculate correlations between different strategy signals
        corr_matrix = signals_df.corr()
        
        # Analyze when strategies agree vs disagree
        agreement_metrics = {}
        
        for name1 in strategy_signals.keys():
            for name2 in strategy_signals.keys():
                if name1 >= name2:  # Skip duplicates and self-correlations
                    continue
                    
                # Count agreement stats
                total = len(signals_df)
                agree = (signals_df[name1] == signals_df[name2]).sum()
                agree_pct = agree / total if total > 0 else 0
                
                # Count when both are in the same direction
                both_pos = ((signals_df[name1] > 0) & (signals_df[name2] > 0)).sum()
                both_neg = ((signals_df[name1] < 0) & (signals_df[name2] < 0)).sum()
                same_direction_pct = (both_pos + both_neg) / total if total > 0 else 0
                
                agreement_metrics[f"{name1}_vs_{name2}"] = {
                    'correlation': corr_matrix.loc[name1, name2],
                    'agreement_pct': agree_pct,
                    'same_direction_pct': same_direction_pct
                }
        
        # Analyze trade attribution - which strategies contributed to which trades
        # This would require matching strategy signals to specific trades
        # and is more complex to implement without detailed signal history
        
        return {
            'signal_correlations': corr_matrix.to_dict(),
            'agreement_metrics': agreement_metrics
        }
    
    def plot_equity_curve(self, benchmark_data=None, title="Equity Curve"):
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            benchmark_data: Optional dictionary with timestamp keys and price values
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data available.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot strategy equity curve
        ax.plot(self.equity_curve.index, self.equity_curve['equity'], label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_data:
            # Convert benchmark data to DataFrame
            benchmark_df = pd.DataFrame.from_dict(benchmark_data, orient='index', columns=['price'])
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            
            # Normalize benchmark to start at the same value as the strategy
            start_value = self.equity_curve['equity'].iloc[0]
            first_benchmark_price = benchmark_df['price'].iloc[0]
            normalized_benchmark = benchmark_df['price'] / first_benchmark_price * start_value
            
            # Plot benchmark
            ax.plot(benchmark_df.index, normalized_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def plot_drawdowns(self, top_n=3, title="Drawdown Analysis"):
        """
        Plot drawdowns over time.
        
        Args:
            top_n: Number of largest drawdowns to highlight
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data available.")
            return None
        
        # Calculate drawdowns
        equity = self.equity_curve['equity'].values
        timestamps = self.equity_curve.index
        
        # Calculate drawdown series
        running_max = np.maximum.accumulate(equity)
        drawdown_series = (equity - running_max) / running_max * 100
        
        # Create DataFrame for plotting
        drawdown_df = pd.DataFrame({'drawdown': drawdown_series}, index=timestamps)
        
        # Find major drawdown periods
        drawdown_analysis = self.analyze_drawdowns(top_n)
        top_drawdowns = drawdown_analysis['top_drawdowns']
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot drawdown series
        ax.fill_between(drawdown_df.index, 0, drawdown_df['drawdown'], color='darkred', alpha=0.3)
        ax.plot(drawdown_df.index, drawdown_df['drawdown'], color='darkred', linewidth=1)
        
        # Highlight top drawdowns
        colors = ['red', 'orange', 'blue']
        
        for i, dd in enumerate(top_drawdowns[:min(top_n, len(top_drawdowns))]):
            if i < len(colors) and dd.get('start_date') and dd.get('recovery_date') and dd['recovery_date'] != 'Not Recovered':
                start_date = pd.to_datetime(dd['start_date'])
                end_date = pd.to_datetime(dd['recovery_date'])
                ax.axvspan(start_date, end_date, alpha=0.2, color=colors[i], 
                          label=f"Drawdown {i+1}: {dd['drawdown_pct']:.2f}%")
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines at key drawdown levels
        ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-20, color='gray', linestyle='--', alpha=0.5)
        
        # Format y-axis
        ax.set_ylim(bottom=min(drawdown_df['drawdown'].min() * 1.1, -5), top=1)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns(self, title="Monthly Returns"):
        """
        Plot monthly returns as a heatmap.
        
        Args:
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data available.")
            return None
        
        try:
            # Create a copy with index that can be reset
            equity_df = self.equity_curve.copy()
            
            # Make sure we have a datetime index
            if not isinstance(equity_df.index, pd.DatetimeIndex):
                print("Equity curve doesn't have a proper datetime index.")
                return None
            
            # Resample to month-end
            monthly_equity = equity_df['equity'].resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100  # Convert to percentage
            
            # Create monthly returns table
            returns_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
            returns_table = returns_table.unstack()
            
            # If no data, return None
            if len(returns_table) == 0:
                print("Insufficient data for monthly returns plot.")
                return None
                
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define colormap with red for negative, green for positive
            cmap = plt.cm.RdYlGn
            
            # Calculate vmin/vmax for symmetric colorbar
            abs_max = max(abs(returns_table.min().min()), abs(returns_table.max().max()))
            vmin, vmax = -abs_max, abs_max
            
            # Create heatmap
            im = ax.imshow(returns_table, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Return (%)')
            
            # Add text labels
            for i in range(len(returns_table)):
                for j in range(len(returns_table.columns)):
                    try:
                        value = returns_table.iloc[i, j]
                        if not np.isnan(value):
                            text_color = 'white' if abs(value) > abs_max/2 else 'black'
                            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
                    except (IndexError, KeyError):
                        pass
            
            # Set ticks and labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(np.arange(len(month_names)))
            ax.set_xticklabels(month_names)
            
            years = returns_table.index.tolist()
            ax.set_yticks(np.arange(len(years)))
            ax.set_yticklabels(years)
            
            ax.set_title(title, fontsize=14)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating monthly returns plot: {str(e)}")
            return None
    
    def plot_trade_distribution(self, title="Trade Return Distribution"):
        """
        Plot distribution of trade returns.
        
        Args:
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available.")
            return None
        
        # Get log returns
        returns = self.trades_df['log_return']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram with KDE
        try:
            from scipy import stats
            
            # Create histogram
            n, bins, patches = ax.hist(returns, bins=30, alpha=0.7, color='skyblue', density=True)
            
            # Add KDE
            kde_x = np.linspace(min(returns), max(returns), 1000)
            kde = stats.gaussian_kde(returns)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            # Add mean and median
            mean_return = returns.mean()
            median_return = returns.median()
            ax.axvline(x=mean_return, color='green', linestyle='-', alpha=0.7, 
                      label=f'Mean: {mean_return:.4f}')
            ax.axvline(x=median_return, color='blue', linestyle='-', alpha=0.7,
                      label=f'Median: {median_return:.4f}')
            
        except ImportError:
            # Fallback to simple histogram if scipy not available
            ax.hist(returns, bins=30, alpha=0.7, color='skyblue')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            mean_return = returns.mean()
            median_return = returns.median()
            ax.axvline(x=mean_return, color='green', linestyle='-', alpha=0.7, 
                      label=f'Mean: {mean_return:.4f}')
            ax.axvline(x=median_return, color='blue', linestyle='-', alpha=0.7,
                      label=f'Median: {median_return:.4f}')
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Log Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_trade_durations(self, title="Trade Duration Distribution"):
        """
        Plot distribution of trade durations.
        
        Args:
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available.")
            return None
        
        # Filter out trades with invalid durations
        durations = self.trades_df['duration'].dropna()
        
        if len(durations) == 0:
            print("No valid trade duration data available.")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram
        ax.hist(durations, bins=20, alpha=0.7, color='darkblue')
        
        # Add vertical line at mean and median
        mean_duration = durations.mean()
        median_duration = durations.median()
        ax.axvline(x=mean_duration, color='red', linestyle='-', alpha=0.7, 
                  label=f'Mean: {mean_duration:.2f} days')
        ax.axvline(x=median_duration, color='green', linestyle='-', alpha=0.7,
                  label=f'Median: {median_duration:.2f} days')
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Duration (days)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_win_loss_analysis(self, title="Win/Loss Analysis"):
        """
        Plot win/loss analysis.
        
        Args:
            title: Plot title
            
        Returns:
            plt.Figure: Figure object with multiple subplots
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available.")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # 1. Win/Loss ratio pie chart
        ax1 = axes[0, 0]
        wins = len(self.trades_df[self.trades_df['log_return'] > 0])
        losses = len(self.trades_df[self.trades_df['log_return'] <= 0])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        ax1.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', 
              colors=['green', 'red'], startangle=90)
        ax1.set_title(f'Win/Loss Ratio: {win_rate:.2%}')
        
        # 2. Average win vs average loss
        ax2 = axes[0, 1]
        winning_trades = self.trades_df[self.trades_df['log_return'] > 0]
        losing_trades = self.trades_df[self.trades_df['log_return'] <= 0]
        
        avg_win = winning_trades['log_return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['log_return'].mean() if len(losing_trades) > 0 else 0
        
        bars = ax2.bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss], 
                     color=['green', 'red'], alpha=0.7)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Average Win vs Loss')
        ax2.set_ylabel('Log Return')
        
        # 3. Win/Loss by trade duration
        ax3 = axes[1, 0]
        
        # Create duration bins
        duration_bins = [0, 1, 3, 5, 10, 20, 30, 60, 90, float('inf')]
        self.trades_df['duration_bin'] = pd.cut(self.trades_df['duration'], bins=duration_bins)
        
        # Calculate win rate by duration bin
        win_rates = []
        labels = []
        counts = []
        
        for bin_name, group in self.trades_df.groupby('duration_bin'):
            if len(group) > 0:
                bin_win_rate = len(group[group['log_return'] > 0]) / len(group)
                win_rates.append(bin_win_rate)
                
                # Create readable labels
                lower = bin_name.left
                upper = bin_name.right
                if upper == float('inf'):
                    label = f"{int(lower)}+"
                else:
                    label = f"{int(lower)}-{int(upper)}"
                    
                labels.append(label)
                counts.append(len(group))
        
        # Plot win rates by duration
        bars = ax3.bar(labels, win_rates, alpha=0.7)
        
        # Add count labels
        for i, bar in enumerate(bars):
            ax3.text(bar.get_x() + bar.get_width()/2., 0.05,
                   f'n={counts[i]}', ha='center', va='bottom', rotation=90)
        
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
        ax3.set_ylim(0, 1)
        ax3.set_title('Win Rate by Trade Duration')
        ax3.set_xlabel('Duration (days)')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        
        # 4. Consecutive wins/losses
        ax4 = axes[1, 1]
        
        # Calculate consecutive wins/losses
        results = self.trades_df.sort_values('entry_time')['log_return'] > 0
        consecutive_win_counts = []
        consecutive_loss_counts = []
        
        current_streak = 1
        for i in range(1, len(results)):
            if results.iloc[i] == results.iloc[i-1]:
                current_streak += 1
            else:
                if results.iloc[i-1]:  # If previous was a win
                    consecutive_win_counts.append(current_streak)
                else:
                    consecutive_loss_counts.append(current_streak)
                current_streak = 1
        
        # Add the final streak
        if len(results) > 0:
            if results.iloc[-1]:
                consecutive_win_counts.append(current_streak)
            else:
                consecutive_loss_counts.append(current_streak)
        
        # Plot histogram of streaks
        if consecutive_win_counts:
            ax4.hist(consecutive_win_counts, alpha=0.7, color='green', label='Win Streaks')
        if consecutive_loss_counts:
            ax4.hist(consecutive_loss_counts, alpha=0.7, color='red', label='Loss Streaks')
        
        ax4.set_title('Consecutive Win/Loss Streaks')
        ax4.set_xlabel('Streak Length')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        return fig
    
    def plot_regime_performance(self, regime_data, title="Performance by Market Regime"):
        """
        Plot performance by market regime.
        
        Args:
            regime_data: Dictionary mapping timestamps to regime types
            title: Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        if self.trades_df is None or len(self.trades_df) == 0 or not regime_data:
            print("Insufficient data for regime performance plot.")
            return None
        
        # Get regime metrics
        regime_analysis = self.analyze_by_regime(regime_data)
        regime_metrics = regime_analysis['regime_metrics']
        
        if not regime_metrics:
            print("No regime metrics available.")
            return None
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        regimes = list(regime_metrics.keys())
        total_returns = [metrics['total_return'] for metrics in regime_metrics.values()]
        win_rates = [metrics['win_rate'] * 100 for metrics in regime_metrics.values()]
        trade_counts = [metrics['total_trades'] for metrics in regime_metrics.values()]
        
        # Plot total returns by regime
        bars1 = ax1.bar(regimes, total_returns, alpha=0.7)
        ax1.set_title('Total Return by Regime')
        ax1.set_ylabel('Return (%)')
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            label_height = height + 1 if height >= 0 else height - 5
            ax1.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Color bars based on positive/negative return
        for i, bar in enumerate(bars1):
            if total_returns[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Plot win rates by regime
        bars2 = ax2.bar(regimes, win_rates, alpha=0.7, color='blue')
        
        # Add secondary axis for trade counts
        ax2_2 = ax2.twinx()
        ax2_2.plot(regimes, trade_counts, 'o-', color='darkred', alpha=0.7)
        ax2_2.set_ylabel('Number of Trades', color='darkred')
        ax2_2.tick_params(axis='y', labelcolor='darkred')
        
        # Add data labels for win rates
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax2.set_title('Win Rate by Regime')
        ax2.set_ylabel('Win Rate (%)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust for suptitle
        return fig
    
    def create_performance_report(self, strategy_name="Trading Strategy", benchmark_data=None, regime_data=None):
        """
        Create a comprehensive performance report.
        
        Args:
            strategy_name: Name of the strategy
            benchmark_data: Optional benchmark data
            regime_data: Optional regime data
            
        Returns:
            dict: Dictionary with plot figures and metrics
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available for performance report.")
            return None
        
        # Calculate all metrics
        performance_metrics = self.calculate_performance_metrics()
        trade_distribution = self.analyze_trade_distribution()
        trade_durations = self.analyze_trade_durations()
        drawdown_analysis = self.analyze_drawdowns()
        monthly_performance = self.analyze_monthly_performance()
        
        # Create plots
        equity_plot = self.plot_equity_curve(benchmark_data, title=f"{strategy_name} - Equity Curve")
        drawdown_plot = self.plot_drawdowns(title=f"{strategy_name} - Drawdowns")
        monthly_plot = self.plot_monthly_returns(title=f"{strategy_name} - Monthly Returns")
        distribution_plot = self.plot_trade_distribution(title=f"{strategy_name} - Return Distribution")
        duration_plot = self.plot_trade_durations(title=f"{strategy_name} - Trade Durations")
        win_loss_plot = self.plot_win_loss_analysis(title=f"{strategy_name} - Win/Loss Analysis")
        
        # Add regime analysis if provided
        regime_metrics = None
        regime_plot = None
        if regime_data:
            regime_metrics = self.analyze_by_regime(regime_data)
            regime_plot = self.plot_regime_performance(regime_data, title=f"{strategy_name} - Regime Analysis")
        
        # Compile report
        report = {
            'strategy_name': strategy_name,
            'performance_metrics': performance_metrics,
            'trade_distribution': trade_distribution,
            'trade_durations': trade_durations,
            'drawdown_analysis': drawdown_analysis,
            'monthly_performance': monthly_performance,
            'regime_metrics': regime_metrics,
            'plots': {
                'equity_curve': equity_plot,
                'drawdowns': drawdown_plot,
                'monthly_returns': monthly_plot,
                'return_distribution': distribution_plot,
                'trade_durations': duration_plot,
                'win_loss_analysis': win_loss_plot,
                'regime_performance': regime_plot
            }
        }
        
        return report
    
    def print_performance_summary(self):
        """
        Print a summary of performance metrics.
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available.")
            return
            
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Print summary
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.4f}")
        print(f"Average Trade Return: {metrics['average_return']:.2f}%")
        print(f"Avg Trade Duration: {metrics['avg_trade_duration']:.2f} days")
        print(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        # Print monthly performance if available
        monthly_perf = self.analyze_monthly_performance()
        if monthly_perf and monthly_perf['monthly_returns']:
            print("\nTOP 5 MONTHS:")
            returns = pd.Series(monthly_perf['monthly_returns'])
            top_months = returns.sort_values(ascending=False).head(5)
            for month, ret in top_months.items():
                print(f"{month}: {ret:.2f}%")
                
            print("\nBOTTOM 5 MONTHS:")
            bottom_months = returns.sort_values().head(5)
            for month, ret in bottom_months.items():
                print(f"{month}: {ret:.2f}%")
        
        # Print drawdown info
        dd_analysis = self.analyze_drawdowns(top_n=3)
        if dd_analysis:
            print("\nTOP 3 DRAWDOWNS:")
            for dd in dd_analysis['top_drawdowns']:
                recovery_date = dd['recovery_date']
                print(f"  {dd['start_date']} to {recovery_date}: {dd['drawdown_pct']:.2f}% ({dd['duration_days']} days)")
                
        print("\n" + "="*50)

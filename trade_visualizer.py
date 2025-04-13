"""
Trade Visualizer Module for Algorithmic Trading Systems

This module provides advanced visualization tools for trading system analysis,
including interactive plots, regime-based visualizations, and comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import math
from collections import defaultdict

class TradeVisualizer:
    """
    Specialized visualization tools for trading systems.
    
    This class provides advanced chart creation capabilities beyond
    the basic visualizations in TradeAnalyzer.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid'):
        """
        Initialize the trade visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        self.figsize = figsize
        
        # Set plotting style if available
        try:
            plt.style.use(style)
        except:
            # Fallback to a basic style if the specified one isn't available
            try:
                plt.style.use('ggplot')
            except:
                pass  # Use default style if none of the preferred styles are available
    
    def create_regime_chart(self, price_data, regime_data, trades=None, title="Price and Regime Chart"):
        """
        Create a chart showing price with regime backgrounds and trades.
        
        Args:
            price_data: DataFrame with timestamp and price columns
            regime_data: Dict or DataFrame mapping timestamps to regime types
            trades: Optional list of trade tuples (entry_time, direction, entry_price, exit_time, exit_price, log_return)
            title: Chart title
            
        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Ensure price_data is a DataFrame
        if not isinstance(price_data, pd.DataFrame):
            if isinstance(price_data, dict):
                price_data = pd.DataFrame.from_dict(price_data, orient='index', columns=['price'])
                price_data.index.name = 'timestamp'
                price_data = price_data.reset_index()
            else:
                raise ValueError("price_data must be a DataFrame or dictionary")
        
        # Ensure timestamp column exists
        timestamp_col = 'timestamp' if 'timestamp' in price_data.columns else price_data.index.name
        if timestamp_col is None:
            raise ValueError("price_data must have a timestamp column or index")
        
        # Ensure price column exists
        price_col = 'price' if 'price' in price_data.columns else price_data.columns[0]
        
        # Convert timestamps to datetime if needed
        price_data = price_data.copy()
        if timestamp_col != price_data.index.name:
            if not pd.api.types.is_datetime64_any_dtype(price_data[timestamp_col]):
                price_data[timestamp_col] = pd.to_datetime(price_data[timestamp_col])
            price_timestamps = price_data[timestamp_col]
        else:
            if not pd.api.types.is_datetime64_any_dtype(price_data.index):
                price_data.index = pd.to_datetime(price_data.index)
            price_timestamps = price_data.index
        
        # Convert regime_data to DataFrame if needed
        if isinstance(regime_data, dict):
            regime_df = pd.DataFrame.from_dict(regime_data, orient='index', columns=['regime'])
            regime_df.index = pd.to_datetime(regime_df.index)
        else:
            regime_df = regime_data.copy()
            if 'timestamp' in regime_df.columns:
                regime_df = regime_df.set_index('timestamp')
            regime_df.index = pd.to_datetime(regime_df.index)
        
        # Plot price data
        if timestamp_col != price_data.index.name:
            ax.plot(price_data[timestamp_col], price_data[price_col], linewidth=1.5)
        else:
            ax.plot(price_data.index, price_data[price_col], linewidth=1.5)
        
        # Create a dictionary of unique regimes and colors
        unique_regimes = regime_df['regime'].unique()
        regime_colors = {
            'TRENDING_UP': 'lightgreen',
            'TRENDING_DOWN': 'lightpink',
            'RANGE_BOUND': 'lightyellow',
            'VOLATILE': 'lightcoral',
            'LOW_VOLATILITY': 'lightblue',
            'unknown': 'lightgray'
        }
        
        # Assign default colors to any regimes not in the default dict
        for regime in unique_regimes:
            if regime not in regime_colors:
                # Generate a random color for this regime
                import random
                r, g, b = random.random(), random.random(), random.random()
                regime_colors[regime] = (r, g, b, 0.3)  # Add alpha for transparency
        
        # Get min and max values for y-axis limits with some padding
        if timestamp_col != price_data.index.name:
            y_min = price_data[price_col].min() * 0.95
            y_max = price_data[price_col].max() * 1.05
        else:
            y_min = price_data[price_col].min() * 0.95
            y_max = price_data[price_col].max() * 1.05
        
        # Plot regime backgrounds
        last_timestamp = None
        last_regime = None
        
        # Sort regime_df by index to ensure chronological order
        regime_df = regime_df.sort_index()
        
        for timestamp, row in regime_df.iterrows():
            regime = row['regime']
            
            if last_timestamp is not None and last_regime is not None:
                # Add colored background for the regime
                color = regime_colors.get(last_regime, 'lightgray')
                ax.axvspan(last_timestamp, timestamp, alpha=0.3, color=color)
                
                # Potentially add text label for longer regime periods
                duration = (timestamp - last_timestamp).total_seconds() / (24 * 3600)  # days
                if duration > 30:  # Only label longer regimes
                    midpoint = last_timestamp + (timestamp - last_timestamp) / 2
                    ax.text(midpoint, y_max * 0.99, str(last_regime), 
                           ha='center', va='top', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.3', alpha=0.2, color=color))
            
            last_timestamp = timestamp
            last_regime = regime
        
        # Handle last regime (if any)
        if last_timestamp is not None and last_regime is not None:
            # Use the last price timestamp as the end of the last regime
            if timestamp_col != price_data.index.name:
                end_timestamp = price_data[timestamp_col].iloc[-1]
            else:
                end_timestamp = price_data.index[-1]
                
            color = regime_colors.get(last_regime, 'lightgray')
            ax.axvspan(last_timestamp, end_timestamp, alpha=0.3, color=color)
            
            # Label if long enough
            duration = (end_timestamp - last_timestamp).total_seconds() / (24 * 3600)  # days
            if duration > 30:
                midpoint = last_timestamp + (end_timestamp - last_timestamp) / 2
                ax.text(midpoint, y_max * 0.99, str(last_regime), 
                       ha='center', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', alpha=0.2, color=color))
        
        # Add trades if provided
        if trades:
            self._add_trades_to_chart(ax, trades, y_min, y_max)
        
        # Add legend for regimes
        handles = []
        for regime, color in regime_colors.items():
            if regime in unique_regimes:
                patch = patches.Patch(color=color, alpha=0.3, label=regime)
                handles.append(patch)
        
        ax.legend(handles=handles, loc='upper left')
        
        # Format the chart
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_ylim(y_min, y_max)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def _add_trades_to_chart(self, ax, trades, y_min, y_max):
        """
        Add trade markers and annotations to a chart.
        
        Args:
            ax: Matplotlib axis to add trades to
            trades: List of trade tuples
            y_min: Minimum y-axis value for plotting
            y_max: Maximum y-axis value for plotting
        """
        # Determine the vertical positioning of trade labels
        label_spacing = (y_max - y_min) * 0.05
        
        for i, trade in enumerate(trades):
            # Handle different trade formats
            if isinstance(trade, tuple) or isinstance(trade, list):
                if len(trade) >= 6:
                    entry_time = trade[0]
                    direction = trade[1]
                    entry_price = trade[2]
                    exit_time = trade[3]
                    exit_price = trade[4]
                    log_return = trade[5]
                else:
                    continue  # Skip if not enough elements
            elif isinstance(trade, dict):
                entry_time = trade.get('entry_time')
                direction = trade.get('direction')
                entry_price = trade.get('entry_price')
                exit_time = trade.get('exit_time')
                exit_price = trade.get('exit_price')
                log_return = trade.get('log_return')
                
                if None in (entry_time, exit_time, entry_price, exit_price):
                    continue  # Skip if missing key data
            else:
                continue  # Skip unknown trade format
            
            # Convert timestamps to datetime if they're strings
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            
            # Determine color based on trade result
            is_win = log_return > 0
            color = 'green' if is_win else 'red'
            
            # Add entry and exit points
            ax.scatter(entry_time, entry_price, color=color, marker='^' if direction.lower() == 'long' else 'v', s=100)
            ax.scatter(exit_time, exit_price, color=color, marker='o', s=80)
            
            # Connect entry and exit with a line
            ax.plot([entry_time, exit_time], [entry_price, exit_price], color=color, linestyle='--', alpha=0.6)
            
            # Add label with trade number and return
            trade_number = i + 1
            percent_return = (math.exp(log_return) - 1) * 100
            
            # Alternate label positions above/below to avoid overlap
            position = 'top' if trade_number % 2 == 0 else 'bottom'
            y_pos = exit_price + label_spacing if position == 'top' else exit_price - label_spacing
            
            ax.annotate(f"#{trade_number}: {percent_return:.1f}%", 
                       xy=(exit_time, exit_price),
                       xytext=(exit_time, y_pos),
                       ha='center',
                       va='center',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', alpha=0.7, color=color))
    
    def create_comparison_chart(self, strategies_results, title="Strategy Comparison", initial_capital=10000):
        """
        Create a comparison chart for multiple strategies.
        
        Args:
            strategies_results: Dict mapping strategy names to backtest results
            title: Chart title
            initial_capital: Initial capital for equity calculation
            
        Returns:
            plt.Figure: Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Process each strategy
        equity_curves = {}
        drawdown_curves = {}
        performance_metrics = {}
        
        for strategy_name, results in strategies_results.items():
            if 'trades' not in results or not results['trades']:
                print(f"No trades found for strategy: {strategy_name}")
                continue
                
            # Calculate equity curve
            trades = results['trades']
            equity = [initial_capital]
            
            # Create a list of timestamps for this equity curve
            timestamps = []
            
            for trade in trades:
                # Extract trade info based on format
                if isinstance(trade, tuple) or isinstance(trade, list):
                    if len(trade) >= 6:
                        entry_time = trade[0]
                        exit_time = trade[3]
                        log_return = trade[5]
                    else:
                        continue
                elif isinstance(trade, dict):
                    entry_time = trade.get('entry_time')
                    exit_time = trade.get('exit_time')
                    log_return = trade.get('log_return')
                    if None in (entry_time, exit_time, log_return):
                        continue
                else:
                    continue
                
                # Convert timestamps to datetime if needed
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                if isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                
                # Add entry timestamp and equity value at entry
                timestamps.append(entry_time)
                equity.append(equity[-1])  # Duplicate last value for step effect
                
                # Add exit timestamp and equity value after trade
                timestamps.append(exit_time)
                equity.append(equity[-2] * math.exp(log_return))  # Calculate from previous value
            
            # Remove the initial duplicate (entry of first trade)
            if len(equity) > 1:
                equity = equity[1:]
                
            # Create DataFrame for this strategy's equity curve
            df_equity = pd.DataFrame({
                'timestamp': timestamps,
                'equity': equity
            })
            
            # Sort by timestamp to ensure chronological order
            df_equity = df_equity.sort_values('timestamp')
            
            # Calculate drawdown
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
            
            # Store curves
            equity_curves[strategy_name] = df_equity
            
            # Calculate performance metrics
            if 'total_return' in results:
                total_return = results['total_return']
            elif 'total_percent_return' in results:
                total_return = results['total_percent_return']
            else:
                total_return = (equity[-1] / initial_capital - 1) * 100
                
            sharpe = results.get('sharpe', 0)
            num_trades = results.get('num_trades', len(trades))
            
            performance_metrics[strategy_name] = {
                'total_return': total_return,
                'sharpe': sharpe,
                'num_trades': num_trades
            }
        
        # Plot equity curves
        max_equity = 0  # Track maximum equity for setting y-axis limit
        
        for strategy_name, df in equity_curves.items():
            color = next(ax1._get_lines.prop_cycler)['color']
            ax1.plot(df['timestamp'], df['equity'], label=f"{strategy_name} ({performance_metrics[strategy_name]['total_return']:.1f}%)", 
                    color=color)
            max_equity = max(max_equity, df['equity'].max())
            
            # Plot drawdown on second axis
            ax2.plot(df['timestamp'], df['drawdown'], color=color)
            
            # Optionally shade the drawdown
            ax2.fill_between(df['timestamp'], 0, df['drawdown'], alpha=0.2, color=color)
        
        # Format equity curve chart
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, max_equity * 1.05)
        
        # Format drawdown chart
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal lines at key drawdown levels
        ax2.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=-20, color='gray', linestyle='--', alpha=0.5)
        
        # Find the lowest drawdown to set y-axis limit
        min_drawdown = 0
        for df in equity_curves.values():
            min_drawdown = min(min_drawdown, df['drawdown'].min())
        
        ax2.set_ylim(min(min_drawdown * 1.1, -5), 2)  # Set bottom limit to min drawdown or -5%, whichever is lower
        
        # Format x-axis dates (only on bottom subplot)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_multi_metric_chart(self, strategy_results, metrics=None, title="Strategy Performance Metrics"):
        """
        Create a chart with multiple performance metrics.
        
        Args:
            strategy_results: Dict with strategy results (can be from TradeAnalyzer.calculate_performance_metrics)
            metrics: List of metrics to include (if None, uses standard set)
            title: Chart title
            
        Returns:
            plt.Figure: Figure object
        """
        # Define default metrics if not provided
        if metrics is None:
            metrics = [
                'total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                'calmar_ratio', 'win_rate', 'profit_factor'
            ]
        
        # Filter for metrics that exist in the results
        available_metrics = [m for m in metrics if m in strategy_results]
        
        if not available_metrics:
            print("No valid metrics found in strategy_results")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create more readable labels
        metric_labels = {
            'total_return': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio',
            'max_drawdown': 'Max Drawdown (%)',
            'calmar_ratio': 'Calmar Ratio',
            'win_rate': 'Win Rate',
            'profit_factor': 'Profit Factor',
            'avg_trade_duration': 'Avg Trade Duration (days)',
            'max_consecutive_wins': 'Max Consecutive Wins',
            'max_consecutive_losses': 'Max Consecutive Losses'
        }
        
        # Extract values, handling win_rate which is often in 0-1 scale
        values = []
        for metric in available_metrics:
            value = strategy_results[metric]
            
            # Convert win_rate from 0-1 to percentage if needed
            if metric == 'win_rate' and value <= 1:
                value = value * 100
                
            values.append(value)
        
        # Get labels
        labels = [metric_labels.get(m, m) for m in available_metrics]
        
        # Create bar chart
        bars = ax.bar(labels, values, color='skyblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            if abs(height) < 0.01:
                value_text = f"{height:.4f}"
            elif abs(height) < 10:
                value_text = f"{height:.2f}"
            else:
                value_text = f"{height:.1f}"
                
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 * max(values)),
                   value_text, ha='center', va='bottom')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format chart
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_regime_performance_chart(self, regime_metrics, title="Performance by Market Regime"):
        """
        Create a comprehensive chart showing performance across different market regimes.
        
        Args:
            regime_metrics: Dict of performance metrics by regime (from TradeAnalyzer.analyze_by_regime)
            title: Chart title
            
        Returns:
            plt.Figure: Figure object
        """
        if not regime_metrics or 'regime_metrics' not in regime_metrics:
            print("No regime metrics provided")
            return None
            
        regime_data = regime_metrics['regime_metrics']
        
        if not regime_data:
            print("Empty regime metrics")
            return None
            
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        regimes = list(regime_data.keys())
        total_returns = [metrics['total_return'] for metrics in regime_data.values()]
        win_rates = [metrics['win_rate'] * 100 for metrics in regime_data.values()]
        avg_returns = [metrics['avg_return'] for metrics in regime_data.values()]
        trade_counts = [metrics['total_trades'] for metrics in regime_data.values()]
        
        # 1. Total Returns by Regime
        ax1 = axes[0, 0]
        bars1 = ax1.bar(regimes, total_returns)
        
        # Color bars based on positive/negative returns
        for i, bar in enumerate(bars1):
            bar.set_color('green' if total_returns[i] >= 0 else 'red')
            
        ax1.set_title('Total Return by Regime')
        ax1.set_ylabel('Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            y_pos = height + 0.5 if height >= 0 else height - 2.5
            ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Win Rate by Regime
        ax2 = axes[0, 1]
        bars2 = ax2.bar(regimes, win_rates, color='blue', alpha=0.7)
        
        ax2.set_title('Win Rate by Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        
        # Add data labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Average Return per Trade by Regime
        ax3 = axes[1, 0]
        bars3 = ax3.bar(regimes, avg_returns)
        
        # Color bars based on positive/negative average returns
        for i, bar in enumerate(bars3):
            bar.set_color('green' if avg_returns[i] >= 0 else 'red')
            
        ax3.set_title('Average Return per Trade')
        ax3.set_ylabel('Log Return')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add data labels
        for bar in bars3:
            height = bar.get_height()
            y_pos = height + 0.0005 if height >= 0 else height - 0.002
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # 4. Trade Count by Regime
        ax4 = axes[1, 1]
        bars4 = ax4.bar(regimes, trade_counts, color='purple', alpha=0.7)
        
        ax4.set_title('Number of Trades')
        ax4.set_ylabel('Trade Count')
        
        # Add data labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height}', ha='center', va='bottom')
        
        # Rotate regime labels for better readability on all subplots
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        return fig
    
    def create_trades_heatmap(self, trades, price_data=None, title="Trades Heatmap by Time of Day and Day of Week"):
        """
        Create a heatmap of trades by time of day and day of week.
        
        Args:
            trades: List of trade tuples or dictionaries
            price_data: Optional price data for background coloring
            title: Chart title
            
        Returns:
            plt.Figure: Figure object
        """
        # Process trades to extract timestamps and returns
        trade_times = []
        trade_returns = []
        
        for trade in trades:
            # Extract info based on format
            if isinstance(trade, tuple) or isinstance(trade, list):
                if len(trade) >= 6:
                    exit_time = trade[3]
                    log_return = trade[5]
                else:
                    continue
            elif isinstance(trade, dict):
                exit_time = trade.get('exit_time')
                log_return = trade.get('log_return')
                if None in (exit_time, log_return):
                    continue
            else:
                continue
            
            # Convert timestamp to datetime if needed
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
                
            trade_times.append(exit_time)
            trade_returns.append(log_return)
        
        if not trade_times:
            print("No valid trades for heatmap")
            return None
            
        # Create DataFrame with day of week and hour of day
        trades_df = pd.DataFrame({
            'exit_time': trade_times,
            'return': trade_returns
        })
        
        trades_df['day_of_week'] = trades_df['exit_time'].dt.day_name()
        trades_df['hour_of_day'] = trades_df['exit_time'].dt.hour
        
        # Define days of week in order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Filter out days that don't have trades
        days_present = trades_df['day_of_week'].unique()
        days_to_use = [day for day in days_order if day in days_present]
        
        if not days_to_use:
            print("No day of week data available for heatmap")
            return None
            
        # Create pivot table: rows=days, columns=hours, values=average return
        pivot = trades_df.pivot_table(
            index='day_of_week',
            columns='hour_of_day',
            values='return',
            aggfunc='mean'
        )
        
        # Reorder days
        pivot = pivot.reindex(days_to_use)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create custom colormap: red for negative, white for zero, green for positive
        colors = ['red', 'white', 'green']
        custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
        
        # Determine vmin/vmax for symmetric colorbar
        abs_max = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmin, vmax = -abs_max, abs_max
        
        # Create heatmap
        im = ax.imshow(pivot, cmap=custom_cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Average Log Return')
        
        # Set ticks and labels
        ax.set_yticks(np.arange(len(days_to_use)))
        ax.set_yticklabels(days_to_use)
        
        hours_present = sorted(trades_df['hour_of_day'].unique())
        ax.set_xticks(np.arange(len(hours_present)))
        ax.set_xticklabels([f"{h:02d}:00" for h in hours_present])
        
        # Add trade count annotations
        count_pivot = trades_df.pivot_table(
            index='day_of_week',
            columns='hour_of_day',
            values='return',
            aggfunc='count'
        ).reindex(days_to_use)
        
        # Add text annotations with trade counts and returns
        for i in range(len(days_to_use)):
            for j in range(len(hours_present)):
                hour = hours_present[j]
                day = days_to_use[i]
                
                # Check if data exists for this day/hour combination
                if hour in count_pivot.columns and day in count_pivot.index:
                    count = count_pivot.at[day, hour]
                    avg_return = pivot.at[day, hour]
                    
                    # Skip if no data
                    if pd.isna(count) or pd.isna(avg_return):
                        continue
                        
                    # Determine text color based on background
                    text_color = 'white' if abs(avg_return) > abs_max/3 else 'black'
                    
                    # Add text with count and return
                    ax.text(j, i, f"n={int(count)}\n{avg_return:.4f}", 
                          ha="center", va="center", color=text_color, fontsize=9)
        
        # Format chart
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_monte_carlo_simulation(self, trades, num_simulations=1000, confidence_level=0.95, title="Monte Carlo Simulation"):
        """
        Create a Monte Carlo simulation of trading performance.
        
        Args:
            trades: List of trade tuples or dictionaries
            num_simulations: Number of simulations to run
            confidence_level: Confidence level for intervals
            title: Chart title
            
        Returns:
            plt.Figure: Figure object
        """
        # Extract returns from trades
        returns = []
        
        for trade in trades:
            # Extract info based on format
            if isinstance(trade, tuple) or isinstance(trade, list):
                if len(trade) >= 6:
                    log_return = trade[5]
                else:
                    continue
            elif isinstance(trade, dict):
                log_return = trade.get('log_return')
                if log_return is None:
                    continue
            else:
                continue
                
            returns.append(log_return)
        
        if not returns:
            print("No valid returns for Monte Carlo simulation")
            return None
            
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run simulations
        num_trades = len(returns)
        simulated_equity_curves = []
        
        for _ in range(num_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=num_trades, replace=True)
            
            # Calculate equity curve
            equity = [1.0]
            for ret in sampled_returns:
                equity.append(equity[-1] * math.exp(ret))
                
            simulated_equity_curves.append(equity)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all simulations with low alpha
        for equity in simulated_equity_curves:
            ax.plot(equity, color='blue', alpha=0.02)
            
        # Calculate and plot percentiles
        equity_array = np.array(simulated_equity_curves)
        
        # Calculate median
        median_curve = np.median(equity_array, axis=0)
        ax.plot(median_curve, color='blue', linewidth=2, label='Median')
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_curve = np.percentile(equity_array, lower_percentile * 100, axis=0)
        upper_curve = np.percentile(equity_array, upper_percentile * 100, axis=0)
        
        ax.plot(lower_curve, color='red', linewidth=2, 
               label=f"{lower_percentile*100:.1f}th Percentile")
        ax.plot(upper_curve, color='green', linewidth=2, 
               label=f"{upper_percentile*100:.1f}th Percentile")
        
        # Add shading between confidence interval bounds
        ax.fill_between(range(len(lower_curve)), lower_curve, upper_curve, 
                      color='blue', alpha=0.1)
        
        # Calculate additional percentiles for context
        p10_curve = np.percentile(equity_array, 10, axis=0)
        p90_curve = np.percentile(equity_array, 90, axis=0)
        
        ax.plot(p10_curve, color='orange', linewidth=1, linestyle='--', 
               label='10th Percentile')
        ax.plot(p90_curve, color='purple', linewidth=1, linestyle='--', 
               label='90th Percentile')
        
        # Format chart
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Number of Trades', fontsize=12)
        ax.set_ylabel('Equity (Starting = $1)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics in text box
        final_values = equity_array[:, -1]
        stats_text = (
            f"Simulations: {num_simulations}\n"
            f"Starting Value: $1.00\n"
            f"Median Final Value: ${median_curve[-1]:.2f}\n"
            f"Mean Final Value: ${np.mean(final_values):.2f}\n"
            f"95% CI: ${lower_curve[-1]:.2f} to ${upper_curve[-1]:.2f}\n"
            f"Probability of Profit: {(final_values > 1.0).mean():.1%}"
        )
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

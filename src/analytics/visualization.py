"""
Visualization Module for Trading System

This module provides tools for visualizing trading results and performance metrics,
including equity curves, drawdowns, trade distributions, and regime-based analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

from src.analytics.metrics import calculate_max_drawdown, calculate_drawdown_periods, calculate_regime_performance


class TradeVisualizer:
    """
    Class for creating trading system visualizations.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid'):
        """
        Initialize with default figure size and style.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        
        # Set plotting style if available
        try:
            plt.style.use(style)
        except:
            # Fallback styles for different matplotlib versions
            for style_name in ['seaborn-darkgrid', 'ggplot', 'default']:
                try:
                    plt.style.use(style_name)
                    break
                except:
                    continue
    
    def plot_equity_curve(self, trades, title="Equity Curve", benchmark_data=None, initial_capital=10000):
        """
        Plot the equity curve from trades.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            benchmark_data: Optional benchmark data for comparison
            initial_capital: Initial capital for equity calculation
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades:
            print("No trades to plot.")
            return None
        
        # Calculate equity curve from trades
        equity = [initial_capital]
        timestamps = []
        
        for trade in trades:
            equity.append(equity[-1] * np.exp(trade[5]))  # Apply log return
            timestamps.append(trade[3])  # Use exit time for timestamp
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot equity curve
        if all(isinstance(ts, (datetime, np.datetime64, pd.Timestamp)) or isinstance(ts, str) for ts in timestamps):
            # Convert string timestamps to datetime
            if all(isinstance(ts, str) for ts in timestamps):
                try:
                    timestamps = [pd.to_datetime(ts) for ts in timestamps]
                except:
                    # If conversion fails, use trade indices
                    timestamps = range(len(equity))
            
            # Plot time-based equity curve
            ax.plot(timestamps, equity[1:], label='Strategy')  # Skip initial capital point
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        else:
            # Plot trade-based equity curve
            ax.plot(range(len(equity)), equity, label='Strategy')
            ax.set_xlabel('Trade Number')
        
        # Add benchmark if provided
        if benchmark_data is not None:
            # Normalize benchmark to starting equity
            benchmark_start = benchmark_data.iloc[0]
            benchmark_normalized = benchmark_data / benchmark_start * initial_capital
            ax.plot(benchmark_normalized.index, benchmark_normalized.values, 
                   label='Benchmark', alpha=0.7, linestyle='--')
        
        # Format plot
        ax.set_title(title)
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_drawdowns(self, trades, title="Drawdown Analysis", threshold=5.0, initial_capital=10000):
        """
        Plot drawdowns over time.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            threshold: Minimum drawdown percentage to highlight
            initial_capital: Initial capital for equity calculation
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades:
            print("No trades to plot.")
            return None
        
        # Calculate equity curve from trades
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] * np.exp(trade[5]))  # Apply log return
        
        # Calculate drawdown series
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # Convert to percentage
        
        # Get drawdown periods
        drawdown_periods = calculate_drawdown_periods(equity, threshold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot drawdown series
        ax.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax.plot(range(len(drawdown)), drawdown, color='red', alpha=0.5)
        
        # Highlight significant drawdowns
        colors = ['darkred', 'firebrick', 'orangered', 'coral']
        highlighted = 0
        
        for i, dd in enumerate(drawdown_periods):
            start_idx = dd['start_idx']
            max_idx = dd['max_dd_idx']
            end_idx = dd['end_idx']
            max_dd = dd['max_dd']
            
            if highlighted < len(colors):
                # Highlight drawdown period
                ax.axvspan(start_idx, end_idx, alpha=0.2, color=colors[highlighted % len(colors)])
                
                # Add text annotation
                mid_point = (start_idx + end_idx) // 2
                ax.annotate(f"{max_dd:.1f}%", 
                           xy=(max_idx, drawdown[max_idx]),
                           xytext=(mid_point, drawdown[max_idx] - 3),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                
                highlighted += 1
        
        # Add horizontal grid lines at key levels
        for level in [5, 10, 15, 20, 25, 30]:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 and extend past max drawdown
        max_dd_value = max(drawdown) if len(drawdown) > 0 else 0
        ax.set_ylim(0, max(30, max_dd_value * 1.1))
        
        return fig
    
    def plot_returns_distribution(self, trades, title="Trade Returns Distribution"):
        """
        Plot the distribution of trade returns.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades:
            print("No trades to plot.")
            return None
        
        returns = [trade[5] for trade in trades]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram
        n, bins, patches = ax.hist(returns, bins=30, alpha=0.7, color='skyblue', density=True)
        
        # Try to add a KDE (kernel density estimate) if scipy available
        try:
            from scipy import stats
            
            # Calculate KDE
            kde_x = np.linspace(min(returns), max(returns), 1000)
            kde = stats.gaussian_kde(returns)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2)
            
        except ImportError:
            pass  # Skip KDE if scipy not available
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add mean and median
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        ax.axvline(x=mean_return, color='green', linestyle='-', alpha=0.7, 
                  label=f'Mean: {mean_return:.4f}')
        ax.axvline(x=median_return, color='blue', linestyle='-', alpha=0.7,
                  label=f'Median: {median_return:.4f}')
        
        # Add win percentage
        win_pct = sum(1 for r in returns if r > 0) / len(returns)
        ax.text(0.05, 0.95, f"Win Rate: {win_pct:.1%}", transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_monthly_returns(self, trades, title="Monthly Returns"):
        """
        Plot monthly returns as a heatmap.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades:
            print("No trades to plot.")
            return None
        
        # Group trades by month
        monthly_trades = defaultdict(list)
        
        for trade in trades:
            exit_time = trade[3]  # Exit time
            log_return = trade[5]  # Log return
            
            # Convert to datetime if string
            if isinstance(exit_time, str):
                try:
                    exit_time = pd.to_datetime(exit_time)
                except:
                    continue  # Skip if conversion fails
            
            # Extract year-month and add to dictionary
            if isinstance(exit_time, (datetime, pd.Timestamp, np.datetime64)):
                month_key = f"{exit_time.year}-{exit_time.month:02d}"
                monthly_trades[month_key].append(log_return)
        
        if not monthly_trades:
            print("No valid monthly data to plot.")
            return None
        
        # Calculate returns for each month
        monthly_returns = {}
        for month, returns in monthly_trades.items():
            # Use percentage return for readability
            monthly_returns[month] = (np.exp(sum(returns)) - 1) * 100
        
        # Parse month strings to get year and month components
        df_data = []
        for month_str, ret in monthly_returns.items():
            year, month = month_str.split('-')
            df_data.append({
                'year': int(year),
                'month': int(month),
                'return': ret
            })
        
        # Create DataFrame
        returns_df = pd.DataFrame(df_data)
        
        # Create pivot table
        pivot_df = returns_df.pivot(index='year', columns='month', values='return')
        
        # Fill missing months with NaN
        months = range(1, 13)
        for month in months:
            if month not in pivot_df.columns:
                pivot_df[month] = np.nan
        
        # Sort columns
        pivot_df = pivot_df[sorted(pivot_df.columns)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Define colors for positive and negative returns
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        
        # Create heatmap
        pcm = ax.pcolormesh(pivot_df.columns, pivot_df.index, pivot_df.values, 
                         cmap=cmap, vmin=-10, vmax=10)
        
        # Add colorbar
        cbar = fig.colorbar(pcm)
        cbar.set_label('Return (%)')
        
        # Add text annotations with return values
        for y in range(len(pivot_df.index)):
            for x in range(len(pivot_df.columns)):
                if x < pivot_df.shape[1] and y < pivot_df.shape[0]:
                    value = pivot_df.iloc[y, x]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 5 else 'black'
                        ax.text(x + 0.5, y + 0.5, f"{value:.1f}%",
                               ha='center', va='center', color=text_color)
        
        # Format plot
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.arange(len(month_names)) + 0.5)
        ax.set_xticklabels(month_names)
        
        ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
        ax.set_yticklabels(pivot_df.index)
        
        ax.set_title(title)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_regime_performance(self, trades, regime_data, title="Performance by Market Regime"):
        """
        Plot performance broken down by market regime.
        
        Args:
            trades: List of trade tuples
            regime_data: Dict mapping timestamps to regime types
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades or not regime_data:
            print("No trades or regime data to plot.")
            return None
        
        # Calculate regime-specific metrics
        regime_metrics = calculate_regime_performance(trades, regime_data)
        
        if not regime_metrics:
            print("No regime-specific metrics to plot.")
            return None
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Get regimes sorted by number of trades
        regimes = sorted(regime_metrics.keys(), 
                        key=lambda r: regime_metrics[r]['num_trades'],
                        reverse=True)
        
        # 1. Plot total returns by regime
        ax1 = axes[0, 0]
        returns = [regime_metrics[r]['total_return'] for r in regimes]
        bars1 = ax1.bar(regimes, returns, color=['green' if r > 0 else 'red' for r in returns])
        
        ax1.set_title('Total Return by Regime')
        ax1.set_ylabel('Return (%)')
        ax1.set_xticklabels(regimes, rotation=45, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Plot win rates by regime
        ax2 = axes[0, 1]
        win_rates = [regime_metrics[r]['win_rate'] * 100 for r in regimes]
        bars2 = ax2.bar(regimes, win_rates, color='blue', alpha=0.7)
        
        ax2.set_title('Win Rate by Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_xticklabels(regimes, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add data labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Plot average return per trade
        ax3 = axes[1, 0]
        avg_returns = [regime_metrics[r]['avg_return'] for r in regimes]
        bars3 = ax3.bar(regimes, avg_returns, color=['green' if r > 0 else 'red' for r in avg_returns])
        
        ax3.set_title('Average Return Per Trade')
        ax3.set_ylabel('Log Return')
        ax3.set_xticklabels(regimes, rotation=45, ha='right')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 4. Plot trade count by regime
        ax4 = axes[1, 1]
        trade_counts = [regime_metrics[r]['num_trades'] for r in regimes]
        bars4 = ax4.bar(regimes, trade_counts, color='purple', alpha=0.7)
        
        ax4.set_title('Number of Trades')
        ax4.set_ylabel('Count')
        ax4.set_xticklabels(regimes, rotation=45, ha='right')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add data labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        return fig
    
    def plot_trade_analysis(self, trades, title="Trade Analysis"):
        """
        Create a comprehensive trade analysis with multiple subplots.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not trades:
            print("No trades to analyze.")
            return None
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract trade data
        returns = [trade[5] for trade in trades]
        
        # 1. Plot cumulative return
        ax1 = axes[0, 0]
        cumulative = np.cumsum(returns)
        ax1.plot(cumulative, color='blue')
        ax1.set_title("Cumulative Log Return")
        ax1.set_xlabel("Trade #")
        ax1.set_ylabel("Log Return")
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot return distribution
        ax2 = axes[0, 1]
        ax2.hist(returns, bins=20, color='skyblue', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_title("Return Distribution")
        ax2.set_xlabel("Log Return")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot win/loss ratio by trade direction
        ax3 = axes[1, 0]
        
        # Separate long and short trades
        long_returns = [t[5] for t in trades if t[1].lower() == 'long']
        short_returns = [t[5] for t in trades if t[1].lower() == 'short']
        
        long_win = sum(1 for r in long_returns if r > 0)
        long_loss = len(long_returns) - long_win
        short_win = sum(1 for r in short_returns if r > 0)
        short_loss = len(short_returns) - short_win
        
        ax3.bar(['Long Win', 'Long Loss', 'Short Win', 'Short Loss'], 
               [long_win, long_loss, short_win, short_loss], 
               color=['green', 'red', 'green', 'red'], alpha=0.7)
        
        ax3.set_title("Win/Loss by Direction")
        ax3.set_ylabel("Count")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Plot consecutive wins/losses
        ax4 = axes[1, 1]
        
        wins_losses = [1 if r > 0 else -1 for r in returns]
        streak_lengths = []
        streak_types = []
        
        current_streak = 0
        current_type = None
        
        for wl in wins_losses:
            if current_type is None or wl == current_type:
                current_streak += 1
                current_type = wl
            else:
                streak_lengths.append(current_streak)
                streak_types.append("Win Streak" if current_type == 1 else "Loss Streak")
                current_streak = 1
                current_type = wl
        
        # Add the last streak
        if current_streak > 0:
            streak_lengths.append(current_streak)
            streak_types.append("Win Streak" if current_type == 1 else "Loss Streak")
        
        # Create a custom histogram
        win_streaks = [length for length, type_str in zip(streak_lengths, streak_types) if type_str == "Win Streak"]
        loss_streaks = [length for length, type_str in zip(streak_lengths, streak_types) if type_str == "Loss Streak"]
        
        # Find the max streak length for bin edge
        max_streak = max(streak_lengths) if streak_lengths else 0
        bins = range(1, max_streak + 2)  # +2 because the right edge is exclusive
        
        ax4.hist([win_streaks, loss_streaks], bins=bins, label=['Win Streaks', 'Loss Streaks'], 
                alpha=0.7, color=['green', 'red'])
        
        ax4.set_title("Streak Distribution")
        ax4.set_xlabel("Streak Length")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        return fig
    
    def plot_trade_durations(self, trades, title="Trade Duration Analysis"):
        """
        Analyze and plot trade durations.
        
        Args:
            trades: List of trade tuples
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Calculate durations
        durations = []
        returns = []
        
        for trade in trades:
            if len(trade) >= 6:
                entry_time = trade[0]
                exit_time = trade[3]
                log_return = trade[5]
                
                # Convert to datetime if they're strings
                if isinstance(entry_time, str):
                    try:
                        entry_time = pd.to_datetime(entry_time)
                    except:
                        continue  # Skip if conversion fails
                
                if isinstance(exit_time, str):
                    try:
                        exit_time = pd.to_datetime(exit_time)
                    except:
                        continue  # Skip if conversion fails
                
                if isinstance(entry_time, (datetime, pd.Timestamp)) and isinstance(exit_time, (datetime, pd.Timestamp)):
                    duration = (exit_time - entry_time).total_seconds() / (24 * 3600)  # in days
                    durations.append(duration)
                    returns.append(log_return)
        
        if not durations:
            print("No valid duration data to plot.")
            return None
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Plot duration histogram
        ax1.hist(durations, bins=20, color='skyblue', alpha=0.7)
        ax1.set_title("Trade Duration Distribution")
        ax1.set_xlabel("Duration (days)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = (
            f"Mean: {np.mean(durations):.2f} days\n"
            f"Median: {np.median(durations):.2f} days\n"
            f"Min: {min(durations):.2f} days\n"
            f"Max: {max(durations):.2f} days"
        )
        
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Plot scatter of duration vs return
        ax2.scatter(durations, returns, alpha=0.6)
        ax2.set_title("Return vs Duration")
        ax2.set_xlabel("Duration (days)")
        ax2.set_ylabel("Log Return")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Fit a trend line
        try:
            from scipy import stats
            if len(durations) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(durations, returns)
                x_line = np.array([min(durations), max(durations)])
                y_line = slope * x_line + intercept
                ax2.plot(x_line, y_line, 'r-', alpha=0.7)
                
                # Add correlation annotation
                corr_text = f"Correlation: {r_value:.3f}\nP-value: {p_value:.3f}"
                ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except ImportError:
            pass  # Skip trend line if scipy not available
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        return fig
    
    def create_performance_dashboard(self, trades, regime_data=None, title="Trading Performance Dashboard", initial_capital=10000):
        """
        Create a comprehensive performance dashboard with multiple visualizations.
        
        Args:
            trades: List of trade tuples
            regime_data: Optional dict mapping timestamps to regime types
            title: Dashboard title
            initial_capital: Initial capital for equity calculations
            
        Returns:
            List[matplotlib.figure.Figure]: List of figure objects
        """
        if not trades:
            print("No trades to analyze.")
            return None
        
        figures = []
        
        # 1. Equity and Drawdown (combined)
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        fig1.suptitle(f"{title} - Equity & Drawdown", fontsize=16)
        
        # Calculate equity curve
        equity = [initial_capital]
        timestamps = []
        for trade in trades:
            equity.append(equity[-1] * np.exp(trade[5]))
            if isinstance(trade[3], (datetime, pd.Timestamp, np.datetime64)) or isinstance(trade[3], str):
                timestamps.append(pd.to_datetime(trade[3]) if isinstance(trade[3], str) else trade[3])
            else:
                timestamps.append(len(equity) - 1)
        
        # Plot equity curve
        if all(isinstance(ts, (datetime, np.datetime64, pd.Timestamp)) for ts in timestamps):
            ax1.plot(timestamps, equity[1:])
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig1.autofmt_xdate()
        else:
            ax1.plot(range(len(equity)), equity)
            ax1.set_xlabel('Trade Number')
        
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # Add key statistics annotation
        total_return = (equity[-1] / equity[0] - 1) * 100
        returns = [trade[5] for trade in trades]
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        stats_text = (
            f"Total Return: {total_return:.2f}%\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Trades: {len(trades)}\n"
            f"Sharpe Ratio: {sharpe:.2f}"
        )
        
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Calculate drawdown series
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        # Plot drawdown
        if all(isinstance(ts, (datetime, np.datetime64, pd.Timestamp)) for ts in timestamps):
            ax2.fill_between(timestamps, 0, drawdown[1:], color='red', alpha=0.3)
            ax2.plot(timestamps, drawdown[1:], color='red', alpha=0.5)
        else:
            ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
            ax2.plot(range(len(drawdown)), drawdown, color='red', alpha=0.5)
        
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add maximum drawdown annotation
        max_dd = max(drawdown)
        ax2.text(0.02, 0.05, f"Max Drawdown: {max_dd:.2f}%", transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        figures.append(fig1)
        
        # 2. Returns Distribution and Monthly Returns
        fig2 = self.plot_returns_distribution(trades, title=f"{title} - Returns Distribution")
        figures.append(fig2)
        
        # 3. Monthly Returns
        fig3 = self.plot_monthly_returns(trades, title=f"{title} - Monthly Returns")
        if fig3:  # Only add if valid data
            figures.append(fig3)
        
        # 4. Trade Analysis
        fig4 = self.plot_trade_analysis(trades, title=f"{title} - Trade Analysis")
        figures.append(fig4)
        
        # 5. Trade Durations
        fig5 = self.plot_trade_durations(trades, title=f"{title} - Trade Durations")
        if fig5:  # Only add if valid data
            figures.append(fig5)
        
        # 6. Regime Performance (if regime data provided)
        if regime_data:
            fig6 = self.plot_regime_performance(trades, regime_data, title=f"{title} - Regime Analysis")
            if fig6:  # Only add if valid data
                figures.append(fig6)
        
        return figures
    
    def save_dashboard(self, figures, output_dir="./analysis_output", base_filename="trading_analysis"):
        """
        Save all dashboard figures to files.
        
        Args:
            figures: List of figure objects
            output_dir: Directory to save figures
            base_filename: Base filename for saved figures
            
        Returns:
            list: List of saved file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Save each figure
        for i, fig in enumerate(figures):
            filename = f"{base_filename}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            
            # Close figure to free memory
            plt.close(fig)
        
        return saved_files

"""
Example usage of the backtesting engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import Config
from data import DataHandler
from strategy import StrategyFactory
from backtester import Backtester
import ga

def generate_sample_data(n_days=1000, filename='data.csv'):
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    
    # Start date
    start_date = datetime(2020, 1, 1)
    
    # Generate dates
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data with a trend and some noise
    base_price = 100
    trend = np.linspace(0, 20, n_days)  # Upward trend
    noise = np.random.normal(0, 1, n_days)  # Random noise
    seasonal = 5 * np.sin(np.linspace(0, 10 * np.pi, n_days))  # Seasonal component
    
    # Calculate close prices
    close = base_price + trend + noise + seasonal
    
    # Calculate daily volatility
    volatility = 0.5 + 0.5 * np.abs(np.sin(np.linspace(0, 5 * np.pi, n_days)))
    
    # Generate OHLC data
    high = close + volatility * np.random.random(n_days)
    low = close - volatility * np.random.random(n_days)
    open_price = low + (high - low) * np.random.random(n_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Generated sample data saved to {filename}")
    
    return df

def plot_strategy_performance(signals_df, title="Strategy Performance"):
    """Plot strategy performance charts."""
    # Calculate cumulative returns
    signals_df['StrategyReturn'] = signals_df['Signal'].shift(1) * signals_df['LogReturn']
    signals_df['CumReturn'] = np.exp(signals_df['LogReturn'].cumsum()) - 1
    signals_df['CumStrategyReturn'] = np.exp(signals_df['StrategyReturn'].cumsum()) - 1
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and signals
    axs[0].plot(signals_df.index, signals_df['Close'], label='Close Price')
    
    # Plot buy and sell signals
    buy_signals = signals_df[signals_df['Signal'] == 1].index
    sell_signals = signals_df[signals_df['Signal'] == -1].index
    
    axs[0].scatter(buy_signals, signals_df.loc[buy_signals, 'Close'], 
                 marker='^', color='green', s=100, label='Buy Signal')
    axs[0].scatter(sell_signals, signals_df.loc[sell_signals, 'Close'], 
                 marker='v', color='red', s=100, label='Sell Signal')
    
    axs[0].set_title(f'{title} - Price and Signals')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot cumulative returns
    axs[1].plot(signals_df.index, signals_df['CumReturn'], label='Buy & Hold')
    axs[1].plot(signals_df.index, signals_df['CumStrategyReturn'], label='Strategy')
    axs[1].set_title('Cumulative Returns')
    axs[1].set_ylabel('Return %')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot drawdowns
    cum_returns = 1 + signals_df['StrategyReturn']
    cum_returns[cum_returns < 0] = 0  # Ensure no negative values
    cum_returns = cum_returns.cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    
    axs[2].fill_between(signals_df.index, drawdowns, 0, color='red', alpha=0.3)
    axs[2].set_title('Strategy Drawdowns')
    axs[2].set_ylabel('Drawdown %')
    axs[2].set_ylim(-1, 0.1)
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def run_example():
    """Run an example backtest."""
    # Generate sample data if not exists
    try:
        data = pd.read_csv('sample_data.csv')
        print("Using existing sample data.")
    except FileNotFoundError:
        data = generate_sample_data()
    
    # Create configuration
    config = Config()
    config.data_file = 'sample_data.csv'
    config.train_size = 0.6
    config.top_n = 5
    config.use_weights = True
    config.ga_pop_size = 8
    config.ga_generations = 50  # Reduced for faster execution
    config.ga_parents = 4
    config.filter_regime = False
    
    # Create backtester
    backtester = Backtester(config)
    
    # Run backtest
    backtester.run(ga_module=ga)
    
    # Get results
    if 'signals' in backtester.results:
        # Add Close price to signals for plotting
        signals_df = backtester.results['signals']
        signals_df['Close'] = data.iloc[-(len(signals_df)):]['Close'].values
        
        # Plot results
        plot_strategy_performance(signals_df, f"Example - {backtester.strategy}")
    
    # Compare with unweighted strategy
    print("\nComparing with unweighted strategy:")
    
    unweighted_config = Config()
    unweighted_config.data_file = 'sample_data.csv'
    unweighted_config.train_size = 0.6
    unweighted_config.top_n = 5
    unweighted_config.use_weights = False
    
    strategies = [config, unweighted_config]
    backtester.compare_strategies(strategies, ga_module=ga)

if __name__ == "__main__":
    run_example()

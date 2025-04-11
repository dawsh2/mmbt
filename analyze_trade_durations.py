import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def analyze_trade_durations(trades):
    """
    Given a list of trades from backtester, analyze and plot duration.
    Each trade: (entry_time, type, entry_price, exit_time, exit_price, log_return, entry_signal, exit_signal)
    """
    durations = []

    for trade in trades:
        entry_time = pd.to_datetime(trade[0])
        exit_time = pd.to_datetime(trade[3])
        duration = (exit_time - entry_time).total_seconds() / 60  # duration in minutes
        durations.append(duration)

    if not durations:
        print("No trades found.")
        return

    durations_series = pd.Series(durations)
    avg_duration = durations_series.mean()
    median_duration = durations_series.median()

    print(f"Average Trade Duration: {avg_duration:.2f} min")
    print(f"Median Trade Duration: {median_duration:.2f} min")
    print(f"Min: {durations_series.min()} min, Max: {durations_series.max()} min")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Trade Durations")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from data_handler import DataHandler
from strategy import Strategy
from backtester import Backtester

if __name__ == "__main__":
    filepath = "~/mmbt/data/data.csv"  # Replace with your real CSV path

    data_handler = DataHandler(filepath)
    strategy = Strategy(short_window=5, long_window=20)
    backtester = Backtester(data_handler, strategy)

    signals = backtester.run()

    print("\nSignals:")
    for signal in signals:
        print(f"{signal['timestamp']} | Signal: {signal['signal']} | Price: {signal['price']:.2f}")

    results = backtester.calculate_returns()

    print("\nTrades:")
    for t in results["trades"]:
        print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[3]:.2f} | Log Return: {t[4]:.4f}")

    print(f"\nTotal Log Return: {results['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results['average_log_return']:.4f}")
    print(f"Number of Trades: {results['num_trades']}")

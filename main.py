from data_handler import DataHandler
from strategy import Rule0, TopNStrategy, Strategy
from backtester import Backtester
import os  # Import the os module to handle path expansion

if __name__ == "__main__":
    filepath = os.path.expanduser("~/mmbt/data/data.csv")  # Expand the user's home directory
    sma_window = 20
    slow_window = 50  # Example slow window for crossover

    data_handler = DataHandler(filepath)
    # Instantiate Rule0 with the expected 'params' argument (a list or tuple)
    rule = Rule0(params=[sma_window, slow_window])
    strategy = TopNStrategy(rule_objects=[rule])
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

    # Example: use a single instance of Rule0 with SMA windows = 20 and 50

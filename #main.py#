import os
from data_handler import DataHandler
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
from rule_system import EventDrivenRuleSystem
from backtester import Backtester

if __name__ == "__main__":
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = DataHandler(filepath)

    # Define the rules and their parameter grids for grid search
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]}),
        (Rule4, {'dema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule5, {'dema1_period': [10, 20], 'dema2_period': [30, 50]}),
        (Rule6, {'tema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule7, {'stoch1_period': [10, 14], 'stochma2_period': [3, 5]}),
        (Rule8, {'vortex1_period': [10, 14], 'vortex2_period': [10, 14]}),
        (Rule9, {'p1': [9, 12], 'p2': [26, 30]}),
        (Rule10, {'rsi1_period': [10, 14], 'c2_threshold': [30, 40]}),
        (Rule11, {'cci1_period': [14, 20], 'c2_threshold': [100, 150]}),
        (Rule12, {'rsi_period': [10, 14], 'hl_threshold': [70, 75], 'll_threshold': [30, 25]}),
        (Rule13, {'stoch_period': [10, 14], 'cci1_period': [14, 20], 'hl_threshold': [80, 90], 'll_threshold': [20, 10]}),
        (Rule14, {'atr_period': [14, 20]}),
        (Rule15, {'bb_period': [20, 25]}),
        # Add more rules and their parameter grids here
    ]

    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=3) # Example top_n

    # Train the rules to find the best parameters
    rule_system.train_rules(data_handler)

    # Get the top N performing rules combined into a strategy
    top_n_strategy = rule_system.get_top_n_strategy()


    # Backtest the combined strategy
    backtester = Backtester(data_handler, top_n_strategy)
    results = backtester.run()  # Assign the dictionary of results to 'results'

    # print("\nSignals from the Top N Strategy:")
    # # Access the list of signals from the 'results' dictionary
    # if 'signals' in results and isinstance(results['signals'], list):
    #     for signal in results['signals']:
    #         print(f"{signal['timestamp']} | Signal: {signal['signal']} | Price: {signal['price']:.2f}")
    # else:
    #     print("No signals found or signals are not in the expected format.")

    print("\nTrades from the Top N Strategy:")
    for t in results["trades"]:
        print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[4]:.2f} | Log Return: {t[5]:.4f}")

    print(f"\nTotal Log Return: {results['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results['average_log_return']:.4f}")
    print(f"Number of Trades: {results['num_trades']}")

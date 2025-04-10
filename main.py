import os
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15


if __name__ == "__main__":
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)

    # --- In-Sample Training and Strategy Building ---
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
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=3)
    rule_system.train_rules(data_handler) # 'data_handler' now contains the split data
    top_n_strategy = rule_system.get_top_n_strategy()

    print("\nIndividual Rule Trade Counts (during training of best parameters):")
    for index, rule_object in rule_system.trained_rule_objects.items():
        if hasattr(rule_object, 'get_trade_count'):
            print(f"Rule {rule_object.__class__.__name__} (Index {index}): {rule_object.get_trade_count()} trades")


    print("\n--- Backtesting on Out-of-Sample Data ---")
    out_of_sample_backtester = Backtester(data_handler, top_n_strategy) # Remove use_test_data here
    results_oos = out_of_sample_backtester.run(use_test_data=True) # Keep it in the run method

    print("\nOut-of-Sample Backtest Results:")
    print(f"Total Log Return: {results_oos['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results_oos['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results_oos['average_log_return']:.4f}")
    print(f"Number of Trades: {results_oos['num_trades']}")
    sharpe_oos = out_of_sample_backtester.calculate_sharpe()
    print(f"Out-of-Sample Sharpe Ratio: {sharpe_oos:.4f}")

    print("\nOut-of-Sample Trades:")
    for t in results_oos["trades"]:
        print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[4]:.2f} | Log Return: {t[5]:.4f}")

    # Optional: Backtest on In-Sample Data
    print("\n--- Backtesting on In-Sample Data ---")
    in_sample_backtester = Backtester(data_handler, top_n_strategy, use_test_data=False)
    results_is = in_sample_backtester.run(use_test_data=False)
    print("\nIn-Sample Backtest Results:")
    print(f"Total Log Return: {results_is['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results_is['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results_is['average_log_return']:.4f}")
    print(f"Number of Trades: {results_is['num_trades']}")
    sharpe_is = in_sample_backtester.calculate_sharpe()
    print(f"In-Sample Sharpe Ratio: {sharpe_is:.4f}")
    print("\nIn-Sample Trades (First 10):")
    for i, t in enumerate(results_is["trades"]):
        if i < 10:
            print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[4]:.2f} | Log Return: {t[5]:.4f}")
        else:
            break


import os
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15


if __name__ == "__main__":
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)

    # Expanded list of periods
    periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
    
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
    
    # rules_config = [
    #     # Rule0: Simple Moving Average Crossover
    #     (Rule0, {'fast_window': [5, 10, 15], 'slow_window': [50, 100, 150]}),
        
    #     # Rule1: Simple Moving Average Crossover with MA1 and MA2
    #     (Rule1, {'ma1': [5, 10, 15, 19, 23], 'ma2': [27, 35, 41, 50, 61]}),
        
    #     # Rule2: EMA and MA Crossover
    #     (Rule2, {'ema1_period': [5, 10, 15, 19, 23], 'ma2_period': [27, 35, 41, 50, 61]}),
        
    #     # Rule3: EMA and EMA Crossover
    #     (Rule3, {'ema1_period': [5, 10, 15, 19], 'ema2_period': [23, 27, 35, 41, 50, 61]}),
        
    #     # Rule4: DEMA and MA Crossover
    #     (Rule4, {'dema1_period': [5, 10, 15, 19, 23], 'ma2_period': [27, 35, 41, 50, 61]}),
        
    #     # Rule5: DEMA and DEMA Crossover
    #     (Rule5, {'dema1_period': [5, 10, 15, 19], 'dema2_period': [23, 27, 35, 41, 50, 61]}),
        
    #     # Rule6: TEMA and MA Crossover
    #     (Rule6, {'tema1_period': [5, 10, 15, 19, 23], 'ma2_period': [27, 35, 41, 50, 61]}),
        
    #     # Rule7: Stochastic Oscillator
    #     (Rule7, {'stoch1_period': [5, 10, 14, 19, 23], 'stochma2_period': [3, 5, 7, 11]}),
        
    #     # Rule8: Vortex Indicator
    #     (Rule8, {'vortex1_period': [5, 10, 14, 19, 23], 'vortex2_period': [5, 10, 14, 19, 23]}),
        
    #     # Rule9: Ichimoku Cloud
    #     (Rule9, {'p1': [7, 9, 11, 15], 'p2': [23, 26, 35, 50, 61]}),
        
    #     # Rule10: RSI Overbought/Oversold
    #     (Rule10, {'rsi1_period': [7, 11, 14, 19, 23], 'c2_threshold': [30, 40, 50, 60, 70]}),
        
    #     # Rule11: CCI Overbought/Oversold
    #     (Rule11, {'cci1_period': [7, 11, 14, 19, 23], 'c2_threshold': [80, 100, 120, 150]}),
        
    #     # Rule12: RSI-based strategy
    #     (Rule12, {'rsi_period': [7, 11, 14, 19, 23], 'overbought': [65, 70, 75, 80], 'oversold': [20, 25, 30, 35]}),
        
    #     # Rule13: Stochastic Oscillator strategy
    #     (Rule13, {'stoch_period': [7, 11, 14, 19, 23], 'stoch_d_period': [3, 5, 7], 'overbought': [70, 75, 80], 'oversold': [20, 25, 30]}),
        
    #     # Rule14: ATR Trailing Stop
    #     (Rule14, {'atr_period': [7, 11, 14, 19, 23], 'atr_multiplier': [1.5, 2.0, 2.5, 3.0]}),
        
    #     # Rule15: Bollinger Bands strategy
    #     (Rule15, {'bb_period': [10, 15, 20, 25, 30], 'bb_std_dev': [1.5, 2.0, 2.5, 3.0]}),
    # ]
    
    print("\n--- Training Individual Rules on In-Sample Data ---")
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=15) # Adjust signal threshold in TopNStrategy Class in strategy.py (perhaps run optmization on this as well)
    rule_system.train_rules(data_handler)
    top_n_strategy = rule_system.get_top_n_strategy()

    print("\nIndividual Rule Trade Counts (during training of best parameters):")
    for index, rule_object in rule_system.trained_rule_objects.items():
        if hasattr(rule_object, 'get_trade_count'):
            print(f"Rule {rule_object.__class__.__name__} (Index {index}): {rule_object.get_trade_count()} trades")

    print("\n--- Backtesting on Out-of-Sample Data ---")
    out_of_sample_backtester = Backtester(data_handler, top_n_strategy)
    results_oos = out_of_sample_backtester.run(use_test_data=True)

    print("\nOut-of-Sample Backtest Results:")
    print(f"Total Log Return: {results_oos['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results_oos['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results_oos['average_log_return']:.4f}")
    print(f"Number of Trades: {results_oos['num_trades']}")
    sharpe_oos = out_of_sample_backtester.calculate_sharpe()
    print(f"Out-of-Sample Sharpe Ratio: {sharpe_oos:.4f}")

    print("\nOut-of-Sample Trades:")
    for i, t in enumerate(results_oos["trades"]):
        if i < 10:  # Only show first 10 trades
            print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[4]:.2f} | Log Return: {t[5]:.4f}")
        else:
            break
    
    if len(results_oos["trades"]) > 10:
        print(f"... and {len(results_oos['trades']) - 10} more trades")



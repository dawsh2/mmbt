import pandas as pd
import numpy as np
from data import DataHandler
from strategy import StrategyFactory
from metrics import calculate_returns, calculate_metrics
from config import Config

def validate_backtest_pipeline():
    # 🧱 Fully mocked Args object with everything config.py expects
    class Args:
        data = "data/data.csv"  # ✅ <-- Update this path
        train_size = 0.7
        top_n = 5
        no_weights = False
        output = "results.json"
        save_params = "params.json"
        debug = False
        verbose = False
        optimize = False
        seed = 42
        train = True
        test = True
        backtest = True
        strategy_name = "TopNStrategy"

        # GA-related
        ga_pop_size = 8
        ga_generations = 100
        ga_parents = 4

        # Regime filtering
        filter_regime = False

    config = Config(Args())

    # Load and prepare data
    data_handler = DataHandler(config)
    data_handler.load_data()
    data_handler.preprocess()
    data_handler.split_data()

    # Build and train strategy
    strategy = StrategyFactory.create_strategy(config)
    strategy.train(data_handler.train_data)

    # Generate final trading signals
#    print("DEBUG - Returning type:", type(signals))
#    print("DEBUG - Returning shape:", getattr(signals, 'shape', 'N/A'))
    signals = strategy.generate_signals(data_handler.test_data)

    print("✅ Signal type:", type(signals))
    print("✅ Signal shape:", signals.shape)
    print("✅ Non-zero signal count:", (signals != 0).sum())

    # Assign to test data for visibility
    data_handler.test_data['signal'] = signals

    # Compute log returns
    close_prices = data_handler.test_data['Close']
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)

    # Apply signal to returns
    print("DEBUG - First few signal/price pairs:")
    print(pd.DataFrame({
        "Close": close_prices.head(10),
        "Signal": signals.head(10)
    }))
    strat_returns = signals.shift(1).fillna(0) * log_returns
    print("DEBUG - Non-zero strategy return count:", (strat_returns != 0).sum())
    print("DEBUG - strat_returns describe():\n", strat_returns.describe())

    # Evaluate metrics
    calculate_metrics(strat_returns, signals=signals)

    print("\n[Validation Report]")
    print("-" * 40)
    print("Sample Signals:")
    print(signals.head(10))
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k:25}: {v}")
    print("-" * 40)

if __name__ == "__main__":
    validate_backtest_pipeline()


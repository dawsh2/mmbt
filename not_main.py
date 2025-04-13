import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15


if __name__ == "__main__":
    # Setup output directory for saving analysis charts
    output_dir = "analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
        (Rule10, {'rsi1_period': [10, 14]}),
        (Rule11, {'cci1_period': [14, 20]}),
        (Rule12, {'rsi_period': [10, 14]}),
        (Rule13, {'stoch_period': [10, 14], 'stoch_d_period': [3, 5]}),
        (Rule14, {'atr_period': [14, 20]}),
        (Rule15, {'bb_period': [20, 25], 'bb_std_dev_multiplier': [1.5, 2.0, 2.5]})
    ]
    
    print("\n--- Training Individual Rules on In-Sample Data ---")
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=15)
    rule_system.train_rules(data_handler)
    top_n_strategy = rule_system.get_top_n_strategy()

    print("\nIndividual Rule Trade Counts (during training of best parameters):")
    for index, rule_object in rule_system.trained_rule_objects.items():
        if hasattr(rule_object, 'get_trade_count'):
            print(f"Rule {rule_object.__class__.__name__} (Index {index}): {rule_object.get_trade_count()} trades")

    # --- Backtest on Out-of-Sample Data and in-sample data ---
    print("\n--- Backtesting on Out-of-Sample Data ---")
    oos_backtester = Backtester(data_handler, top_n_strategy)
    results_oos = oos_backtester.run(use_test_data=True)

    print("\n--- Backtesting on In-Sample Data ---")
    is_backtester = Backtester(data_handler, top_n_strategy)
    results_is = is_backtester.run(use_test_data=False)

    # Print basic out-of-sample results
    print("\nOut-of-Sample Backtest Results:")
    print(f"Total Log Return: {results_oos['total_log_return']:.4f}")
    print(f"Total Return (compounded): {results_oos['total_percent_return']:.2f}%")
    print(f"Average Log Return per Trade: {results_oos['average_log_return']:.4f}")
    print(f"Number of Trades: {results_oos['num_trades']}")
    sharpe_oos = oos_backtester.calculate_sharpe()
    print(f"Out-of-Sample Sharpe Ratio: {sharpe_oos:.4f}")

    # Print first few trades
    print("\nOut-of-Sample Trades:")
    for i, t in enumerate(results_oos["trades"]):
        if i < 10:  # Only show first 10 trades
            print(f"{t[0]} | {t[1].upper()} | Entry: {t[2]:.2f} → Exit: {t[4]:.2f} | Log Return: {t[5]:.4f}")
        else:
            break
    
    if len(results_oos["trades"]) > 10:
        print(f"... and {len(results_oos['trades']) - 10} more trades")

    # --- Enhanced Analysis with Simple Metrics ---
    print("\n--- Enhanced Analysis with Simple Metrics ---")
    
    # Calculate basic win rate
    if results_oos['trades']:
        win_count = sum(1 for trade in results_oos['trades'] if trade[5] > 0)
        win_rate = win_count / len(results_oos['trades'])
        print(f"Win Rate: {win_rate:.2%}")
        
        # Calculate average trade duration
        trade_durations = []
        for trade in results_oos['trades']:
            entry_time = trade[0]
            exit_time = trade[3]
            
            # Convert to datetime if they're strings
            if isinstance(entry_time, str):
                from datetime import datetime
                entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            if isinstance(exit_time, str):
                from datetime import datetime
                exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                
            duration = (exit_time - entry_time).total_seconds() / (24 * 3600)  # Convert to days
            trade_durations.append(duration)
        
        avg_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
        print(f"Average Trade Duration: {avg_duration:.2f} days")
        
        # Calculate max consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in results_oos['trades']:
            log_return = trade[5]
            
            if log_return > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        print(f"Max Consecutive Wins: {max_consecutive_wins}")
        print(f"Max Consecutive Losses: {max_consecutive_losses}")
    
    # Calculate drawdowns
    if results_oos['trades']:
        # Create equity curve
        equity = [10000]  # Start with initial capital
        timestamps = []
        
        for trade in results_oos['trades']:
            equity.append(equity[-1] * np.exp(trade[5]))
            timestamps.append(trade[3])  # Use exit time
            
        # Calculate drawdowns
        peak = equity[0]
        drawdowns = []
        current_drawdown = None
        
        for i, eq in enumerate(equity):
            if eq > peak:
                peak = eq
                if current_drawdown:
                    drawdowns.append(current_drawdown)
                    current_drawdown = None
            else:
                dd_pct = (peak - eq) / peak * 100
                if not current_drawdown or dd_pct > current_drawdown['max_dd']:
                    if not current_drawdown:
                        current_drawdown = {
                            'start_value': peak,
                            'low_value': eq,
                            'max_dd': dd_pct,
                            'start_idx': i
                        }
                    else:
                        current_drawdown['low_value'] = eq
                        current_drawdown['max_dd'] = dd_pct
        
        # Add any final drawdown
        if current_drawdown:
            drawdowns.append(current_drawdown)
        
        # Sort drawdowns by depth
        drawdowns.sort(key=lambda x: x['max_dd'], reverse=True)
        
        # Print top drawdowns
        print("\nTop 3 Drawdowns:")
        for i, dd in enumerate(drawdowns[:3]):
            print(f"  Drawdown #{i+1}: {dd['max_dd']:.2f}% (from {dd['start_value']:.2f} to {dd['low_value']:.2f})")
    
    # Visual analysis with Matplotlib
    print("\n--- Creating Basic Visualizations ---")
    
    # 1. Plot equity curve
    if results_oos['trades']:
        equity = [10000]  # Start with initial capital
        for trade in results_oos['trades']:
            equity.append(equity[-1] * np.exp(trade[5]))
            
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title("Out-of-Sample Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "equity_curve.png"))
        plt.close()
        
        print(f"Equity curve saved to {output_dir}/equity_curve.png")
    
    # 2. Plot return distribution
    if results_oos['trades']:
        returns = [trade[5] for trade in results_oos['trades']]
        
        plt.figure(figsize=(12, 6))
        plt.hist(returns, bins=20, alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.axvline(x=np.mean(returns), color='green', label=f'Mean: {np.mean(returns):.4f}')
        plt.title("Trade Return Distribution")
        plt.xlabel("Log Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "return_distribution.png"))
        plt.close()
        
        print(f"Return distribution saved to {output_dir}/return_distribution.png")
    
    # 3. Compare in-sample and out-of-sample performance
    if results_is['trades'] and results_oos['trades']:
        # Create equity curves
        is_equity = [10000]
        for trade in results_is['trades']:
            is_equity.append(is_equity[-1] * np.exp(trade[5]))
            
        oos_equity = [10000]
        for trade in results_oos['trades']:
            oos_equity.append(oos_equity[-1] * np.exp(trade[5]))
        
        # Normalize to percentage returns
        is_equity_pct = [eq / is_equity[0] * 100 - 100 for eq in is_equity]
        oos_equity_pct = [eq / oos_equity[0] * 100 - 100 for eq in oos_equity]
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(is_equity_pct, label=f"In-Sample ({results_is['total_percent_return']:.1f}%)")
        plt.plot(oos_equity_pct, label=f"Out-of-Sample ({results_oos['total_percent_return']:.1f}%)")
        plt.title("In-Sample vs Out-of-Sample Performance")
        plt.xlabel("Trade Number")
        plt.ylabel("Return (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "is_vs_oos_comparison.png"))
        plt.close()
        
        print(f"Performance comparison saved to {output_dir}/is_vs_oos_comparison.png")
    
    # 4. Monthly returns if timestamps are available
    if results_oos['trades'] and isinstance(results_oos['trades'][0][0], str):
        try:
            # Convert trades to monthly returns
            from datetime import datetime
            from collections import defaultdict
            
            monthly_returns = defaultdict(list)
            
            for trade in results_oos['trades']:
                exit_time = trade[3]
                if isinstance(exit_time, str):
                    exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                month_key = f"{exit_time.year}-{exit_time.month:02d}"
                monthly_returns[month_key].append(trade[5])
            
            # Calculate average monthly return
            avg_monthly = {month: np.mean(returns) for month, returns in monthly_returns.items()}
            
            # Plot monthly returns
            months = sorted(avg_monthly.keys())
            returns = [avg_monthly[m] for m in months]
            
            plt.figure(figsize=(14, 6))
            plt.bar(months, returns, color=['green' if r > 0 else 'red' for r in returns])
            plt.title("Average Monthly Returns")
            plt.xlabel("Month")
            plt.ylabel("Average Log Return")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "monthly_returns.png"))
            plt.close()
            
            print(f"Monthly returns chart saved to {output_dir}/monthly_returns.png")
        except Exception as e:
            print(f"Error creating monthly returns chart: {e}")
    
    # 5. Create custom trade visualizations
    if results_oos['trades']:
        # Plot trades on a separate chart marking entry and exit points
        if hasattr(data_handler, 'get_full_data'):
            try:
                # Get price data
                price_data = data_handler.get_full_data()
                
                # Filter only test data
                test_start_idx = int(len(price_data) * 0.8)  # Based on train_fraction=0.8
                test_price_data = price_data.iloc[test_start_idx:].copy()
                
                # Plot price chart with trade markers
                plt.figure(figsize=(14, 7))
                plt.plot(test_price_data.index, test_price_data['Close'], color='black', alpha=0.6)
                
                # Add trade markers
                for i, trade in enumerate(results_oos['trades']):
                    # Extract trade details
                    entry_time = trade[0]
                    direction = trade[1]
                    entry_price = trade[2]
                    exit_time = trade[3]
                    exit_price = trade[4]
                    log_return = trade[5]
                    
                    # Convert timestamps to datetime if they're strings
                    if isinstance(entry_time, str):
                        entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                    if isinstance(exit_time, str):
                        exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                    
                    # Determine color based on trade result
                    color = 'green' if log_return > 0 else 'red'
                    
                    # Plot entry and exit points
                    plt.scatter(entry_time, entry_price, marker='^' if direction == 'long' else 'v', 
                               s=100, color=color, zorder=5)
                    plt.scatter(exit_time, exit_price, marker='o', s=80, color=color, zorder=5)
                    
                    # Connect entry and exit with a line
                    plt.plot([entry_time, exit_time], [entry_price, exit_price], 
                           color=color, linestyle='--', alpha=0.7)
                
                plt.title("Out-of-Sample Trades")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, "trade_markers.png"))
                plt.close()
                
                print(f"Trade visualization saved to {output_dir}/trade_markers.png")
            except Exception as e:
                print(f"Error creating trade markers chart: {e}")
    
    print(f"\nAnalysis completed. Visualizations saved to {output_dir}/")

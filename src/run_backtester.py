# run_backtest.py - Tie everything together
from events import EventQueue
from data_handler import CSVDataHandler
from strategy import SimpleRuleStrategy
from rule_strategy import RuleBasedStrategy
from portfolio import SimplePortfolio
from execution import SimpleExecutionHandler
from enhanced_backtester import EnhancedBacktester

def main():
    # Initialize event queue (shared across components)
    event_queue = EventQueue()
    
    # Set up data handler
    symbols = ['AAPL']  # Start with a single symbol
    data_handler = CSVDataHandler(event_queue, 'data/', symbols)
    
    # Set up strategy (use your existing rule system)
    strategy = RuleBasedStrategy(event_queue, symbols, use_weights=True, top_n=5)
    
    # Set up portfolio and execution
    portfolio = SimplePortfolio(event_queue)
    execution = SimpleExecutionHandler(event_queue)
    
    # Create and run backtester
    backtester = EnhancedBacktester(
        data_handler,
        strategy,
        portfolio,
        execution
    )
    
    # Run the backtest
    results = backtester.run()
    
    print(f"Backtest complete with {len(results['trades'])} trades")
    
    # You'd normally save results, create reports, etc.
    # For now, just print trade summary
    if not results['trades'].empty:
        print("\nTrade Summary:")
        print(results['trades'].head())

if __name__ == "__main__":
    main()

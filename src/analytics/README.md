def process_backtester_results(results):
    """
    Convert backtester results to standard format
    
    Args:
        results: Dictionary of results from Backtester.run()
        
    Returns:
        tuple: (trades list, equity curve)
    """
    trades = results.get('trades', [])
    equity_curve = results.get('equity_curve', [])
    
    # If equity curve not provided, calculate it
    if not equity_curve and trades:
        initial_capital = results.get('initial_capital', 10000)
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] * (1 + trade[5]))
        equity_curve = equity
        
    return trades, equity_curve

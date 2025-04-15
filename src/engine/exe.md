# Execution Engine

The Execution Engine serves as the core component responsible for order execution, position tracking, and portfolio management within the trading system. It processes orders generated from trading signals, simulates realistic market conditions, and maintains the state of the trading portfolio.

## Core Components

### ExecutionEngine

The main orchestrator that:
- Processes order events from the event bus
- Executes pending orders with market simulation
- Tracks trade history and portfolio state
- Updates positions based on current market data

```python
# Create execution engine with a position manager
execution_engine = ExecutionEngine(position_manager)

# Set a market simulator for realistic execution
execution_engine.market_simulator = MarketSimulator(config)

# Handle an order event
execution_engine.on_order(order_event)
```

### Order and Fill Classes

- **Order**: Represents a trading order with symbol, quantity, direction, etc.
- **Fill**: Represents an executed order with fill price and commission

```python
# Example Order
order = Order(
    symbol="AAPL",
    order_type="MARKET",
    quantity=100,
    direction=1,  # Buy
    timestamp=datetime.now()
)

# Example Fill
fill = Fill(
    order=order,
    fill_price=150.75,
    timestamp=datetime.now(),
    commission=1.5
)
```

### Position and Portfolio Tracking

The execution engine maintains:
- Currently open positions
- Trade execution history
- Portfolio equity history
- Signal history

```python
# Get trade history
trades = execution_engine.get_trade_history()

# Get portfolio history
portfolio_history = execution_engine.get_portfolio_history()

# Get signal history
signals = execution_engine.get_signal_history()
```

## Integration with Event System

The execution engine integrates with the event system through:

1. **Order Event Handler**: Processes incoming order events
   ```python
   event_bus.register(EventType.ORDER, execution_engine.on_order)
   ```

2. **Fill Event Emitter**: Emits fill events when orders are executed
   ```python
   # Inside execution_engine._execute_order
   self._emit_fill_event(fill)
   ```

3. **Portfolio Update**: Updates portfolio with current market data
   ```python
   execution_engine.update(bar_data)
   ```

## Market Simulation

The ExecutionEngine works with a MarketSimulator to provide realistic order execution, including:

- Slippage modeling based on order size and market conditions
- Transaction cost calculation
- Liquidity constraints simulation

```python
# Create a market simulator with configuration
market_simulator = MarketSimulator({
    'slippage_model': 'fixed',
    'slippage_bps': 5,
    'price_impact': 0.1,
    'fee_model': 'fixed',
    'fee_bps': 10
})

# Set simulator in execution engine
execution_engine.market_simulator = market_simulator
```

## Key Methods

### on_order(event)

Handles incoming order events by adding them to pending orders and potentially executing market orders immediately.

```python
execution_engine.on_order(order_event)
```

### execute_pending_orders(bar, market_simulator=None)

Executes any pending orders based on current bar data, applying realistic market simulation.

```python
fills = execution_engine.execute_pending_orders(current_bar, market_simulator)
```

### update(bar)

Updates the portfolio with latest market data and records portfolio state.

```python
execution_engine.update(current_bar)
```

### _execute_order(order, price, timestamp, commission=0.0)

Internal method to execute a single order and update the portfolio.

### _emit_fill_event(fill)

Emits a fill event to the event bus when an order is executed.

## Best Practices

1. **Reset State Between Tests**: Always call `reset()` before starting a new backtest or test scenario
   ```python
   execution_engine.reset()
   ```

2. **Use Market Simulator**: Always provide a market simulator for realistic backtesting
   ```python
   execution_engine.market_simulator = MarketSimulator(config)
   ```

3. **Handle Order Data Formats**: The on_order method handles both dict-based event data and Order objects

4. **Track Last Prices**: The execution engine maintains a cache of last known prices for immediate market order execution
   ```python
   # Last prices are updated in execute_pending_orders
   execution_engine.last_known_prices[symbol] = close_price
   ```

5. **Event Bus Integration**: Always provide the event bus to enable fill event emission
   ```python
   execution_engine.event_bus = event_bus
   ```

## Troubleshooting

- **Orders Not Executing**: Ensure the market simulator is properly configured and last known prices are available
- **Fill Events Not Processing**: Check that the appropriate fill event handler is registered with the event bus
- **Portfolio Not Updating**: Verify that the `update()` method is being called with each new bar of data
- **Unexpected Execution Prices**: Review the market simulator configuration, especially slippage and fee models
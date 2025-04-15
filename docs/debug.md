# Position Manager Debugging Summary

## Current Status

Tonight we've made significant progress in debugging the trading system. We've confirmed that:

1. ✅ The strategy is correctly generating BUY signals
2. ✅ The signals are being emitted to the event bus
3. ✅ The position manager works correctly when called directly
4. ❌ The position manager is not processing signals during the backtest

## Key Insights

The most revealing test was the direct test of the position manager:

```python
test_signal = {
    'symbol': 'SYNTHETIC',
    'signal_type': 1,  # BUY
    'direction': 1,    # BUY
    'price': 100.0,
    'confidence': 1.0,
    'timestamp': datetime.datetime.now()
}

# Call position manager directly
result = position_manager.on_signal(test_signal)
```

This test successfully generated an action:

```
Direct position manager test result: [{'action': 'entry', 'symbol': 'SYNTHETIC', 'direction': 1, 'size': 100.0, 'price': 100.0}]
```

But during the backtest, the position manager isn't processing signals through the event system.

## Next Steps for Tomorrow

### 1. Check the Event Registration

Review how the position manager is registered with the event bus:

```python
# Position manager handles signal events and emits orders
event_bus.register(EventType.SIGNAL, position_manager.on_signal)
```

Things to check:
- Is the method binding correct?
- Does the event system pass the event object or event.data to handlers?

You could run:
```
grep -r "event_bus.register" src/
```
to see how other components are registered.

### 2. Try Different Registration Approaches

Test these alternatives:

```python
# Option 1: Wrap in a lambda to extract data
event_bus.register(EventType.SIGNAL, lambda event: position_manager.on_signal(event.data))

# Option 2: Wrap in a function to add debug info
def signal_handler(event):
    print(f"Signal received by handler: {event}")
    print(f"Event data: {event.data}")
    result = position_manager.on_signal(event.data)
    print(f"Position manager result: {result}")
    return result

event_bus.register(EventType.SIGNAL, signal_handler)
```

### 3. Inspect the Position Manager Class

Look at the position manager's class definition to understand how it's intended to be used:

```
grep -r "def on_signal" src/position_management/
```

Check if there are any special decorators or handling for the method.

### 4. Check Event Handling in the Event Bus

Examine how the event bus processes events and calls handlers:

```
grep -r "_dispatch_event" src/events/
```

### 5. Verify Signal Format

Ensure your signal dictionary includes all required fields:

```python
signal_dict = {
    'symbol': 'SYNTHETIC',
    'signal_type': 1,
    'direction': 1,  # Make sure this is included
    'price': signal.price,
    'confidence': signal.confidence,
    'timestamp': signal.timestamp
}
```

### 6. Try Direct Order Creation

If needed, try bypassing the position manager entirely to test if the execution engine works:

```python
test_order = {
    'order_id': str(uuid.uuid4()),
    'symbol': 'SYNTHETIC',
    'direction': 1,
    'quantity': 100,
    'order_type': 'MARKET',
    'price': 100.0,
    'timestamp': datetime.datetime.now()
}

event_bus.emit(Event(EventType.ORDER, test_order))
```

## Promising Direction

The fact that the position manager works when called directly is very encouraging. The issue is isolated to how the event system interacts with the position manager, not with the position manager itself or any of the trading logic.

This issue is likely a simple mismatch in parameter passing or event handling, not a fundamental problem with the trading system.

For tomorrow, I recommend focusing on understanding exactly how handlers are registered and called in the event system, and ensuring the position manager's `on_signal` method is being called with the right parameter structure.
# Algorithmic Trading System Debugging Guide

## Overview

This guide documents the issues encountered while implementing and testing the algorithmic trading system, along with solutions and architectural recommendations for future development. The system is an event-driven framework designed for backtesting and implementing algorithmic trading strategies, with components for data handling, strategy execution, and risk management.

## 1. Data Loading Issues

### Issue 1.1: Timezone Handling in CSV Data

**Problem:** The data loading process failed due to timestamp columns containing timezone information (-04:00, +00:00), causing difficulties when filtering by date ranges.

**Error Message:**
```
Can only use .dt accessor with datetimelike values
```

**Root Cause:** Inconsistent handling of timezone-aware and naive datetime objects during date filtering.

**Solution:**
```python
# Proper timezone handling in data loading
if 'timestamp' in df.columns:
    # Use utc=True to standardize timezone handling
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Convert to naive datetime by removing timezone information
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    
    # Ensure input dates are also naive datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
```

### Issue 1.2: Date Range Configuration Mismatch

**Problem:** The configured date range (2020-01-01 to 2022-12-31) didn't match the actual data dates (2024-03-26 to 2025-04-02).

**Solution:** Adjust configuration to match the available data:

```python
config_dict = {
    'backtest': {
        'initial_capital': 100000,
        'start_date': '2024-03-01',  # Match actual data range
        'end_date': '2025-04-02',    # Match actual data range
        'symbols': ['SPY'],
        'timeframe': '1m'
    }
}
```

## 2. Strategy Implementation Issues

### Issue 2.1: Rule Base Class Parameter Mismatch

**Problem:** The `SMACrossoverRule` implementation called the `Rule` base class constructor with 4 parameters, but the base class only accepted 3 parameters.

**Error Message:**
```
TypeError: Rule.__init__() takes from 2 to 4 positional arguments but 5 were given
```

**Root Cause:** The Rule base class defined:
```python
def __init__(self, name: str, params: Optional[Dict[str, Any]] = None, description: str = ""):
```

But SMACrossoverRule called:
```python
super().__init__(name, default_params, description, enabled)
```

**Solution:** Modify the rule implementation to match the base class:

```python
class SMACrossoverRule(Rule):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None, description: str = ""):
        # Process parameters
        default_params = {
            'fast_window': 10,
            'slow_window': 30
        }
        if params:
            default_params.update(params)
            
        # Call base constructor with supported parameters
        super().__init__(name, default_params, description)
        
        # Initialize rule-specific state
        self.state = {
            'prices': [],
            'fast_sma': None,
            'slow_sma': None,
            'previous_fast_sma': None,
            'previous_slow_sma': None
        }
```

### Issue 2.2: Missing Required Parameters in RSI Rule

**Problem:** The RSI rule implementation required a 'signal_type' parameter which was not provided when creating the rule.

**Error Message:**
```
KeyError: 'signal_type'
```

**Root Cause:** The RSI rule had validation code that checked for this parameter:
```python
if self.params['signal_type'] not in valid_signal_types:
    raise ValueError(f"signal_type must be one of {valid_signal_types}")
```

**Solution:** When creating RSI rule, provide all required parameters:

```python
rsi_rule = RSIRule(
    name="rsi_rule",
    params={
        "rsi_period": 14, 
        "overbought": 70, 
        "oversold": 30,
        "signal_type": "levels"  # Add the missing parameter
    },
    description="RSI Overbought/Oversold"
)
```

## 3. Event System Issues

### Issue 3.1: EventBus Missing Validation Parameter

**Problem:** The EventBus constructor referenced an undefined variable 'validate_events'.

**Error Message:**
```
NameError: name 'validate_events' is not defined
```

**Root Cause:** EventBus initialization expected a validation parameter that wasn't defined:
```python
self.validate_events = validate_events
```

**Solution:** Fix the EventBus initialization logic:

```python
# Option 1: Initialize with validation disabled by default
def __init__(self, async_mode=False, validate_events=False):
    self.handlers = {}
    self.history = []
    self.metrics = {}
    self.async_mode = async_mode
    self.dispatch_queue = None
    self.running = False
    self.validate_events = validate_events

# Option 2: Create a custom EventBus wrapper
class SimpleEventBus:
    def __init__(self):
        self.handlers = {}
        
    def register(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def emit(self, event):
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in handler: {e}")
```

### Issue 3.2: EventBus Constructor Incompatibility

**Problem:** Attempted to pass `validate_events=False` to EventBus constructor, but this parameter wasn't recognized.

**Error Message:**
```
TypeError: EventBus.__init__() got an unexpected keyword argument 'validate_events'
```

**Root Cause:** EventBus implementation doesn't accept a 'validate_events' parameter.

**Solution:** Create a wrapper around the EventBus that provides consistent behavior:

```python
def create_event_bus():
    """Create an EventBus with consistent behavior across implementations."""
    try:
        # Import the original implementation
        from src.events.event_bus import EventBus as OriginalEventBus
        
        # Create a compatible wrapper
        class CompatibleEventBus(OriginalEventBus):
            def __init__(self, async_mode=False):
                # Initialize with basic attributes
                self.handlers = {}
                self.history = []
                self.metrics = {}
                self.async_mode = async_mode
                self.dispatch_queue = None
                self.running = False
                self.validate_events = False  # Default to no validation
        
        return CompatibleEventBus()
    except Exception as e:
        # Fallback to minimal implementation
        logger.error(f"Error creating EventBus: {e}")
        
        class MinimalEventBus:
            def __init__(self):
                self.handlers = {}
                
            def register(self, event_type, handler):
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                self.handlers[event_type].append(handler)
                
            def emit(self, event):
                if event.event_type in self.handlers:
                    for handler in self.handlers[event.event_type]:
                        try:
                            handler(event)
                        except Exception as e:
                            logger.error(f"Error in handler: {e}")
        
        return MinimalEventBus()
```

## 4. Signal Generation and Event Handling

### Issue 4.1: Signal Creation Method Mismatch

**Problem:** Some rules attempted to use a `create_signal` method that doesn't exist in the updated codebase.

**Root Cause:** The codebase moved to using SignalEvent objects directly, but some rules still used the older approach.

**Solution:** Update rules to use SignalEvent directly:

```python
# OLD: Using create_signal method
return self.create_signal(
    signal_type=SignalType.BUY,
    price=close_price,
    symbol=symbol,
    metadata=metadata,
    timestamp=timestamp
)

# NEW: Using SignalEvent directly
return SignalEvent(
    signal_value=SignalEvent.BUY,  # Using class constants
    price=close_price,
    symbol=symbol,
    rule_id=self.name,
    metadata=metadata,
    timestamp=timestamp
)
```

## 5. Architectural Recommendations

### 5.1. Consistent Interfaces and Base Classes

**Problem:** Inconsistent interfaces between base classes and implementations lead to runtime errors.

**Recommendation:**

1. **Document Interface Contracts Clearly:** For each base class, clearly document the expected parameters, methods, and behavior.

2. **Use Type Annotations and Validation:** Leverage Python's type hinting to catch incompatibilities at development time.

3. **Create Factory Methods:** Centralize object creation in factory methods that ensure correct initialization.

```python
def create_rule(rule_type, name, params=None, description=""):
    """Factory method for creating rules with standardized parameters."""
    if rule_type == "sma_crossover":
        # Standardize parameters for this rule type
        default_params = {
            'fast_window': 10,
            'slow_window': 30
        }
        if params:
            default_params.update(params)
        return SMACrossoverRule(name, default_params, description)
    elif rule_type == "rsi":
        # Ensure RSI required parameters
        default_params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'signal_type': 'levels'
        }
        if params:
            default_params.update(params)
        return RSIRule(name, default_params, description)
    # Add more rule types as needed
```

### 5.2. Robust Error Handling and Graceful Degradation

**Problem:** Component failures cascade through the system, causing complete failures.

**Recommendation:**

1. **Defensive Programming:** Always check inputs and handle edge cases.

2. **Fallback Components:** Implement simpler versions of components that can operate with minimal functionality.

3. **Graceful Degradation Pattern:** Allow systems to function with reduced capabilities rather than complete failure.

```python
# Example of graceful degradation in rule creation
try:
    rule = create_rule("sma_crossover", "my_crossover", params)
    logger.info("Created SMA Crossover rule")
except Exception as e:
    logger.warning(f"Failed to create optimal rule, using fallback: {e}")
    # Create a minimal rule that always returns neutral
    class FallbackRule(Rule):
        def generate_signal(self, bar_event):
            return None
    rule = FallbackRule("fallback_rule")
```

### 5.3. Testing Infrastructure

**Problem:** Issues are discovered only at runtime in the full system.

**Recommendation:**

1. **Unit Tests for Components:** Test each component in isolation.

2. **Integration Tests:** Test interactions between specific components.

3. **End-to-End Tests:** Verify system behavior with simple test cases.

4. **Test Data Fixtures:** Create standardized test data sets.

```python
# Example test for rule implementation
def test_sma_crossover_rule():
    # Create test data
    prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
    dates = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(len(prices))]
    
    # Create test bars
    bars = []
    for i, price in enumerate(prices):
        bars.append({
            'timestamp': dates[i],
            'symbol': 'TEST',
            'Open': price,
            'High': price + 1,
            'Low': price - 1,
            'Close': price,
            'Volume': 1000
        })
    
    # Create rule
    rule = SMACrossoverRule(
        name="test_sma", 
        params={"fast_window": 3, "slow_window": 5}
    )
    
    # Process bars and collect signals
    signals = []
    for bar in bars:
        bar_event = BarEvent(bar)
        signal = rule.generate_signal(bar_event)
        if signal:
            signals.append(signal)
    
    # Verify expected behavior
    assert len(signals) > 0, "Rule should generate signals"
    # Add more assertions...
```

### 5.4. Module Structure Improvements

**Problem:** Tight coupling and circular dependencies between modules make changes difficult.

**Recommendation:**

1. **Dependency Inversion:** Depend on abstractions, not concrete implementations.

2. **Clean Layering:** Establish clear module layers with well-defined interfaces.

3. **Event-Driven Communication:** Fully embrace the event system for loose coupling.

**Proposed Module Structure:**
```
src/
├── core/
│   ├── base_classes.py      # Abstract base classes
│   ├── interfaces.py        # Interface definitions
│   └── exceptions.py        # Custom exceptions
├── data/                    # Data handling (no dependencies on strategy)
├── events/                  # Event system (depends only on core)
├── rules/                   # Rule implementations (depends on core and events)
├── strategies/              # Strategy implementations (depends on rules and events)
├── risk_management/         # Risk management (depends on core and events)
├── execution/               # Order execution (depends on events)
└── analytics/               # Analysis tools (depends on events)
```

## 6. Risk Management Implementation

### 6.1. Simplified Risk Management Integration

To begin integrating risk management functionality:

```python
# 1. Create core risk parameters
risk_params = {
    'stop_loss_pct': 2.0,              # % distance for stop loss
    'take_profit_pct': 4.0,            # % distance for take profit
    'trailing_stop_activation_pct': 2.0, # % move before activating trailing stop
    'trailing_stop_distance_pct': 1.0,   # % distance for trailing stop
    'max_position_pct': 10.0,          # Max % of portfolio per position
    'risk_per_trade_pct': 1.0          # % of portfolio to risk per trade
}

# 2. Create metrics collector to track MAE/MFE
risk_metrics_collector = RiskMetricsCollector()

# 3. Create simplified risk manager
class SimpleRiskManager:
    def __init__(self, risk_params, metrics_collector=None):
        self.risk_params = risk_params
        self.metrics_collector = metrics_collector
        self.active_trades = {}
    
    def calculate_position_size(self, signal, portfolio_value):
        """Calculate position size based on risk parameters."""
        risk_amount = portfolio_value * (self.risk_params['risk_per_trade_pct'] / 100)
        stop_distance_pct = self.risk_params['stop_loss_pct'] / 100
        position_size = risk_amount / (signal.get_price() * stop_distance_pct)
        return position_size
    
    def open_trade(self, trade_id, symbol, direction, entry_price, entry_time):
        """Open and track a new trade."""
        # Calculate stop and target levels
        stop_pct = self.risk_params['stop_loss_pct'] / 100
        target_pct = self.risk_params['take_profit_pct'] / 100
        
        stop_price = entry_price * (1 - stop_pct) if direction == 'long' else entry_price * (1 + stop_pct)
        target_price = entry_price * (1 + target_pct) if direction == 'long' else entry_price * (1 - target_pct)
        
        # Start metrics tracking
        if self.metrics_collector:
            self.metrics_collector.start_trade(
                trade_id=trade_id,
                entry_time=entry_time,
                entry_price=entry_price,
                direction=direction
            )
        
        # Store trade details
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'stop_price': stop_price,
            'target_price': target_price,
            'trailing_stop_active': False,
            'trailing_stop_level': None
        }
        
        return {
            'trade_id': trade_id,
            'stop_price': stop_price,
            'target_price': target_price
        }
    
    def update_price(self, trade_id, current_price, current_time, bar_data=None):
        """Update a trade with new price information."""
        if trade_id not in self.active_trades:
            return {'status': 'unknown_trade'}
            
        trade = self.active_trades[trade_id]
        direction = trade['direction']
        
        # Update metrics collector
        if self.metrics_collector and bar_data:
            self.metrics_collector.update_price_path(trade_id, bar_data)
        
        # Check for trailing stop activation
        if not trade['trailing_stop_active']:
            # Calculate price move percentage
            price_move_pct = (current_price / trade['entry_price'] - 1) * 100
            if direction == 'short':
                price_move_pct = -price_move_pct
                
            # Activate trailing stop if price moved enough
            if price_move_pct >= self.risk_params['trailing_stop_activation_pct']:
                trade['trailing_stop_active'] = True
                
                # Set initial trailing stop level
                trailing_distance = current_price * (self.risk_params['trailing_stop_distance_pct'] / 100)
                if direction == 'long':
                    trade['trailing_stop_level'] = current_price - trailing_distance
                else:
                    trade['trailing_stop_level'] = current_price + trailing_distance
                    
        # Update trailing stop if active
        elif trade['trailing_stop_active']:
            trailing_distance = current_price * (self.risk_params['trailing_stop_distance_pct'] / 100)
            
            if direction == 'long':
                new_stop = current_price - trailing_distance
                if new_stop > trade['trailing_stop_level']:
                    trade['trailing_stop_level'] = new_stop
            else:
                new_stop = current_price + trailing_distance
                if new_stop < trade['trailing_stop_level']:
                    trade['trailing_stop_level'] = new_stop
        
        # Check exit conditions
        if direction == 'long':
            # Check stop loss
            if current_price <= trade['stop_price']:
                return {'status': 'exit', 'exit_info': {'reason': 'stop_loss', 'price': current_price}}
                
            # Check trailing stop
            if trade['trailing_stop_active'] and current_price <= trade['trailing_stop_level']:
                return {'status': 'exit', 'exit_info': {'reason': 'trailing_stop', 'price': current_price}}
                
            # Check take profit
            if current_price >= trade['target_price']:
                return {'status': 'exit', 'exit_info': {'reason': 'take_profit', 'price': current_price}}
        else:  # short
            # Check stop loss
            if current_price >= trade['stop_price']:
                return {'status': 'exit', 'exit_info': {'reason': 'stop_loss', 'price': current_price}}
                
            # Check trailing stop
            if trade['trailing_stop_active'] and current_price >= trade['trailing_stop_level']:
                return {'status': 'exit', 'exit_info': {'reason': 'trailing_stop', 'price': current_price}}
                
            # Check take profit
            if current_price <= trade['target_price']:
                return {'status': 'exit', 'exit_info': {'reason': 'take_profit', 'price': current_price}}
        
        # No exit triggered
        return {'status': 'active'}
    
    def close_trade(self, trade_id, exit_price, exit_time, exit_reason):
        """Close a trade and record final metrics."""
        if trade_id not in self.active_trades:
            return {'status': 'error', 'message': 'Trade not found'}
            
        trade = self.active_trades[trade_id]
        
        # End metrics tracking
        if self.metrics_collector:
            self.metrics_collector.end_trade(
                trade_id=trade_id,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason
            )
        
        # Remove from active trades
        result = {**trade, 'exit_price': exit_price, 'exit_time': exit_time, 'exit_reason': exit_reason}
        del self.active_trades[trade_id]
        
        return {'status': 'closed', 'trade': result}
```

## 7. Simplified Event System Implementation

If you encounter further issues with the event system, consider this minimal implementation:

```python
class SimpleEvent:
    """Minimal event implementation."""
    def __init__(self, event_type, data=None, timestamp=None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.datetime.now()
    
    def get(self, key, default=None):
        """Get a value from the event data."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return getattr(self.data, key, default)

class SimpleSignalEvent(SimpleEvent):
    """Minimal signal event implementation."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    def __init__(self, signal_value, price, symbol="default", rule_id=None, 
                 metadata=None, timestamp=None):
        data = {
            'signal_value': signal_value,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'metadata': metadata or {}
        }
        super().__init__("SIGNAL", data, timestamp)
    
    def get_signal_value(self):
        """Get the signal value."""
        return self.get('signal_value', self.NEUTRAL)
    
    def get_price(self):
        """Get the price at signal generation."""
        return self.get('price', 0)
    
    def get_symbol(self):
        """Get the instrument symbol."""
        return self.get('symbol', 'default')

class SimpleEventBus:
    """Minimal event bus implementation."""
    def __init__(self):
        self.handlers = {}
    
    def register(self, event_type, handler):
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def emit(self, event):
        """Emit an event to registered handlers."""
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in handler: {e}")
```

## 8. Minimal Working Example

Here's a complete minimal example integrating the key components:

```python
# 1. Configuration
config = {
    'backtest': {
        'initial_capital': 100000,
        'start_date': '2024-03-01',
        'end_date': '2025-04-02',
        'symbols': ['SPY'],
        'timeframe': '1m'
    },
    'risk_management': {
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0,
        'trailing_stop_activation_pct': 2.0,
        'trailing_stop_distance_pct': 1.0,
        'max_position_pct': 10.0,
        'risk_per_trade_pct': 1.0
    }
}

# 2. Create essential components
event_bus = SimpleEventBus()
data_handler = DataHandler(SimpleCSVDataSource('./data/SPY_1m.csv'))
risk_metrics_collector = RiskMetricsCollector()
risk_manager = SimpleRiskManager(config['risk_management'], risk_metrics_collector)

# 3. Create simplified trading rules
class SimpleTrendRule(Rule):
    def __init__(self, name, params=None, description=""):
        super().__init__(name, params, description)
        self.last_price = None
        
    def generate_signal(self, bar_event):
        # Extract data
        close_price = bar_event.get_price()
        timestamp = bar_event.get_timestamp()
        symbol = bar_event.get_symbol()
        
        # Simple trend following - buy when price increases, sell when it decreases
        if self.last_price is not None:
            if close_price > self.last_price * 1.005:  # 0.5% increase
                self.last_price = close_price
                return SimpleSignalEvent(
                    signal_value=SimpleSignalEvent.BUY,
                    price=close_price,
                    symbol=symbol,
                    rule_id=self.name,
                    timestamp=timestamp
                )
            elif close_price < self.last_price * 0.995:  # 0.5% decrease
                self.last_price = close_price
                return SimpleSignalEvent(
                    signal_value=SimpleSignalEvent.SELL,
                    price=close_price,
                    symbol=symbol,
                    rule_id=self.name,
                    timestamp=timestamp
                )
        
        self.last_price = close_price
        return None
        
    def reset(self):
        self.last_price = None

# 4. Load data
data_handler.load_data(
    symbols=config['backtest']['symbols'],
    start_date=config['backtest']['start_date'],
    end_date=config['backtest']['end_date'],
    timeframe=config['backtest']['timeframe']
)

# 5. Initialize portfolio
portfolio = {
    'equity': config['backtest']['initial_capital'],
    'cash': config['backtest']['initial_capital'],
    'positions': {},
    'trades': []
}

# 6. Create rule
trend_rule = SimpleTrendRule("simple_trend")

# 7. Run backtest
next_trade_id = 1
for bar in data_handler.iter_train():
    bar_event = BarEvent(bar)
    
    # Generate signal
    signal = trend_rule.generate_signal(bar_event)
    
    # Process signal
    if signal:
        symbol = signal.get_symbol()
        price = signal.get_price()
        signal_value = signal.get_signal_value()
        
        # Check for open position
        has_position = symbol in portfolio['positions']
        
        # Process entry signals
        if not has_position and signal_value == SimpleSignalEvent.BUY:
            # Calculate position size
            position_size = risk_manager.calculate_position_size(signal, portfolio['equity'])
            
            # Open trade
            trade_id = f"trade_{next_trade_id}"
            next_trade_id += 1
            
            trade_details = risk_manager.open_trade(
                trade_id=trade_id,
                symbol=symbol,
                direction='long',
                entry_price=price,
                entry_time=bar_event.get_timestamp()
            )
            
            # Update portfolio
            portfolio['positions'][symbol] = {
                'trade_id': trade_id,
                'direction': 'long',
                'quantity': position_size,
                'entry_price': price,
                'entry_time': bar_event.get_timestamp(),
                'stop_price': trade_details['stop_price'],
                'target_price': trade_details['target_price']
            }
            
            portfolio['cash'] -= position_size * price
            print(f"Opened LONG position: {position_size} shares at ${price:.2f}")
            
        elif not has_position and signal_value == SimpleSignalEvent.SELL:
            # Calculate position size
            position_size = risk_manager.calculate_position_size(signal, portfolio['equity'])
            
            # Open trade
            trade_id = f"trade_{next_trade_id}"
            next_trade_id += 1
            
            trade_details = risk_manager.open_trade(
                trade_id=trade_id,
                symbol=symbol,
                direction='short',
                entry_price=price,
                entry_time=bar_event.get_timestamp()
            )
            
            # Update portfolio
            portfolio['positions'][symbol] = {
                'trade_id': trade_id,
                'direction': 'short',
                'quantity': position_size,
                'entry_price': price,
                'entry_time': bar_event.get_timestamp(),
                'stop_price': trade_details['stop_price'],
                'target_price': trade_details['target_price']
            }
            
            portfolio['cash'] += position_size * price  # Credit for short sale
            print(f"Opened SHORT position: {position_size} shares at ${price:.2f}")
            
        # Process exit signals (opposite direction of current position)
        elif has_position:
            position = portfolio['positions'][symbol]
            
            # Exit long on sell signal
            if position['direction'] == 'long' and signal_value == SimpleSignalEvent.SELL:
                # Close the trade
                close_result = risk_manager.close_trade(
                    trade_id=position['trade_id'],
                    exit_price=price,
                    exit_time=bar_event.get_timestamp(),
                    exit_reason='reversal'
                )
                
                # Update portfolio
                portfolio['cash'] += position['quantity'] * price
                profit = (price - position['entry_price']) * position['quantity']
                portfolio['equity'] += profit
                
                # Record trade
                portfolio['trades'].append({
                    'trade_id': position['trade_id'],
                    'symbol': symbol,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'quantity': position['quantity'],
                    'profit': profit,
                    'exit_reason': 'reversal'
                })
                
                # Remove position
                del portfolio['positions'][symbol]
                print(f"Closed LONG position: ${price:.2f}, Profit: ${profit:.2f}")
                
            # Exit short on buy signal
            elif position['direction'] == 'short' and signal_value == SimpleSignalEvent.BUY:
                # Close the trade
                close_result = risk_manager.close_trade(
                    trade_id=position['trade_id'],
                    exit_price=price,
                    exit_time=bar_event.get_timestamp(),
                    exit_reason='reversal'
                )
                
                # Update portfolio
                portfolio['cash'] -= position['quantity'] * price  # Repurchase to close short
                profit = (position['entry_price'] - price) * position['quantity']
                portfolio['equity'] += profit
                
                # Record trade
                portfolio['trades'].append({
                    'trade_id': position['trade_id'],
                    'symbol': symbol,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'quantity': position['quantity'],
                    'profit': profit,
                    'exit_reason': 'reversal'
                })
                
                # Remove position
                del portfolio['positions'][symbol]
                print(f"Closed SHORT position: ${price:.2f}, Profit: ${profit:.2f}")
    
    # Update risk management for open positions
    positions_to_close = []
    for symbol, position in portfolio['positions'].items():
        # Get current price from bar data
        if bar['symbol'] == symbol:
            current_price = bar['Close']
            
            # Update risk management
            update_result = risk_manager.update_price(
                trade_id=position['trade_id'],
                current_price=current_price,
                current_time=bar_event.get_timestamp(),
                bar_data=bar
            )
            
            # Check for exit conditions
            if update_result['status'] == 'exit':
                positions_to_close.append((symbol, position, update_result['exit_info']))
    
    # Close positions that hit stop loss or take profit
    for symbol, position, exit_info in positions_to_close:
        exit_price = exit_info['price']
        exit_reason = exit_info['reason']
        
        # Close the trade in risk manager
        close_result = risk_manager.close_trade(
            trade_id=position['trade_id'],
            exit_price=exit_price,
            exit_time=bar_event.get_timestamp(),
            exit_reason=exit_reason
        )
        
        # Update portfolio
        if position['direction'] == 'long':
            portfolio['cash'] += position['quantity'] * exit_price
            profit = (exit_price - position['entry_price']) * position['quantity']
        else:  # short
            portfolio['cash'] -= position['quantity'] * exit_price
            profit = (position['entry_price'] - exit_price) * position['quantity']
            
        portfolio['equity'] += profit
        
        # Record trade
        portfolio['trades'].append({
            'trade_id': position['trade_id'],
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'profit': profit,
            'exit_reason': exit_reason
        })
        
        # Remove position
        del portfolio['positions'][symbol]
        print(f"Closed {position['direction']} position: ${exit_price:.2f}, Profit: ${profit:.2f}, Reason: {exit_reason}")

# 8. Display results
print("\nBacktest Results:")
print(f"Initial Capital: ${config['backtest']['initial_capital']:,.2f}")
print(f"Final Equity: ${portfolio['equity']:,.2f}")
print(f"Total Return: {(portfolio['equity'] / config['backtest']['initial_capital'] - 1):.2%}")
print(f"Total Trades: {len(portfolio['trades'])}")

if len(portfolio['trades']) > 0:
    winning_trades = [t for t in portfolio['trades'] if t['profit'] > 0]
    win_rate = len(winning_trades) / len(portfolio['trades'])
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Profit: ${sum(t['profit'] for t in portfolio['trades']) / len(portfolio['trades']):,.2f}")

# 9. Analyze risk metrics
if risk_metrics_collector:
    metrics_df = risk_metrics_collector.get_metrics_dataframe()
    if not metrics_df.empty:
        print("\nRisk Metrics:")
        print(f"Average MAE: {metrics_df['mae_pct'].mean():.2f}%")
        print(f"Average MFE: {metrics_df['mfe_pct'].mean():.2f}%")
        print(f"MAE/MFE Ratio: {metrics_df['mae_pct'].mean() / metrics_df['mfe_pct'].mean():.2f}")
```

This minimal example provides a working foundation that:
1. Loads data correctly 
2. Implements a simple trading rule
3. Integrates basic risk management
4. Properly handles trade entry and exit
5. Collects essential risk metrics

From this foundation, you can gradually add more complex components while ensuring stability at each step.



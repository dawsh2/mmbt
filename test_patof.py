#!/usr/bin/env python3
"""
Event Flow Debugging Script

This script runs a simplified test to debug the event flow chain,
focusing specifically on the POSITION_ACTION to ORDER transition.
"""

import datetime
import logging
import sys
import os
from typing import Dict, Any, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('event_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import system components
from src.events.event_bus import EventBus
from src.events.event_types import EventType, OrderEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator

def test_position_action_to_order_flow():
    """
    Test the flow from POSITION_ACTION to ORDER events specifically.
    """
    logger.info("===== Starting Position Action to Order Flow Test =====")
    
    # 1. Create clean event bus with explicit counting
    event_bus = EventBus()
    event_bus.event_counts = {}
    
    # 2. Create minimal components
    portfolio = EventPortfolio(initial_capital=100000, event_bus=event_bus)
    position_manager = PositionManager(portfolio=portfolio, event_bus=event_bus)
    market_simulator = MarketSimulator()
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio
    execution_engine.event_bus = event_bus
    execution_engine.market_simulator = market_simulator
    
    # 3. Add detailed event tracing
    def trace_event(event):
        event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
        logger.info(f"EVENT TRACE: {event_type} event received")
        
        # Log detailed event information
        if event.event_type == EventType.POSITION_ACTION:
            action = event.data
            if isinstance(action, dict):
                logger.info(f"POSITION_ACTION DETAILS: action_type={action.get('action_type')}, "
                           f"symbol={action.get('symbol')}, direction={action.get('direction')}, "
                           f"size={action.get('size')}")
        
        elif event.event_type == EventType.ORDER:
            order = event.data
            if hasattr(order, 'get_symbol'):
                logger.info(f"ORDER DETAILS: symbol={order.get_symbol()}, "
                           f"direction={order.get_direction()}, quantity={order.get_quantity()}, "
                           f"price={order.get_price()}")
    
    # 4. Register handler for all event types
    for event_type in EventType:
        event_bus.register(event_type, trace_event)
    
    # 5. CRITICAL: Register execution engine for position actions
    event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
    logger.info("Registered execution_engine.on_position_action for POSITION_ACTION events")
    
    # Verify the method exists and is callable
    if not hasattr(execution_engine, 'on_position_action'):
        logger.error("CRITICAL ERROR: execution_engine has no on_position_action method!")
    elif not callable(execution_engine.on_position_action):
        logger.error("CRITICAL ERROR: execution_engine.on_position_action is not callable!")
    else:
        logger.info("execution_engine.on_position_action is present and callable")
        
        # Inspect the method implementation
        import inspect
        method_code = inspect.getsource(execution_engine.on_position_action)
        logger.info(f"Method implementation:\n{method_code}")
    
    # 6. Create and emit a test position action
    test_position_action = {
        'action_type': 'entry',
        'symbol': 'SPY',
        'direction': 1,  # BUY
        'size': 10,
        'price': 450.0,
        'timestamp': datetime.datetime.now()
    }
    
    logger.info("Creating test POSITION_ACTION event")
    position_action_event = Event(EventType.POSITION_ACTION, test_position_action)
    
    # 7. Emit the event and check if an ORDER event is generated
    logger.info("Emitting POSITION_ACTION event")
    event_bus.emit(position_action_event)
    
    # 8. Check event counts
    logger.info("Event counts after test:")
    for event_type, count in event_bus.event_counts.items():
        event_type_name = event_type.name if hasattr(event_type, 'name') else str(event_type)
        logger.info(f"  {event_type_name}: {count}")
    
    # 9. Check if orders were created
    logger.info(f"Pending orders in execution engine: {len(execution_engine.pending_orders)}")
    for order in execution_engine.pending_orders:
        logger.info(f"Order: {order}")
    
    # 10. Try executing orders manually
    logger.info("Attempting to execute orders manually")
    test_bar = {
        'symbol': 'SPY',
        'Open': 450.0,
        'High': 451.0, 
        'Low': 449.0,
        'Close': 450.5,
        'Volume': 1000,
        'timestamp': datetime.datetime.now()
    }
    
    fills = execution_engine.execute_pending_orders(test_bar)
    logger.info(f"Fills generated: {len(fills)}")
    for fill in fills:
        logger.info(f"Fill: {fill}")
    
    # 11. Check final portfolio state
    logger.info(f"Final portfolio state - Equity: {portfolio.equity}, Cash: {portfolio.cash}")
    position_snapshot = portfolio.get_position_snapshot()
    for symbol, positions in position_snapshot.items():
        for pos in positions:
            logger.info(f"Position: {symbol} - {pos}")
    
    logger.info("===== Position Action to Order Flow Test Complete =====")

def test_manual_order_creation():
    """
    Test manually creating an order and processing it.
    """
    logger.info("===== Starting Manual Order Creation Test =====")
    
    # 1. Create clean event bus with explicit counting
    event_bus = EventBus()
    event_bus.event_counts = {}
    
    # 2. Create minimal components
    portfolio = EventPortfolio(initial_capital=100000, event_bus=event_bus)
    execution_engine = ExecutionEngine(position_manager=None)
    execution_engine.portfolio = portfolio
    execution_engine.event_bus = event_bus
    
    # 3. Add detailed event tracing
    def trace_event(event):
        event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
        logger.info(f"EVENT TRACE: {event_type} event received")
    
    # 4. Register handler for all event types
    for event_type in EventType:
        event_bus.register(event_type, trace_event)
    
    # 5. Register execution engine for orders
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    logger.info("Registered execution_engine.on_order for ORDER events")
    
    # 6. Create and emit a test order
    test_order = OrderEvent(
        symbol='SPY',
        direction=1,  # BUY
        quantity=10,
        price=450.0,
        order_type='MARKET'
    )
    
    logger.info("Creating test ORDER event")
    order_event = Event(EventType.ORDER, test_order)
    
    # 7. Emit the order event
    logger.info("Emitting ORDER event")
    event_bus.emit(order_event)
    
    # 8. Check event counts
    logger.info("Event counts after test:")
    for event_type, count in event_bus.event_counts.items():
        event_type_name = event_type.name if hasattr(event_type, 'name') else str(event_type)
        logger.info(f"  {event_type_name}: {count}")
    
    # 9. Check if orders were added to pending_orders
    logger.info(f"Pending orders in execution engine: {len(execution_engine.pending_orders)}")
    for order in execution_engine.pending_orders:
        logger.info(f"Order: {order}")
    
    # 10. Try executing orders manually
    logger.info("Attempting to execute orders manually")
    test_bar = {
        'symbol': 'SPY',
        'Open': 450.0, 
        'High': 451.0,
        'Low': 449.0,
        'Close': 450.5,
        'Volume': 1000,
        'timestamp': datetime.datetime.now()
    }
    
    fills = execution_engine.execute_pending_orders(test_bar)
    logger.info(f"Fills generated: {len(fills)}")
    for fill in fills:
        logger.info(f"Fill: {fill}")
    
    # 11. Check final portfolio state
    logger.info(f"Final portfolio state - Equity: {portfolio.equity}, Cash: {portfolio.cash}")
    position_snapshot = portfolio.get_position_snapshot()
    for symbol, positions in position_snapshot.items():
        for pos in positions:
            logger.info(f"Position: {symbol} - {pos}")
    
    logger.info("===== Manual Order Creation Test Complete =====")

if __name__ == "__main__":
    logger.info("Starting event flow debug tests")
    
    # Test POSITION_ACTION to ORDER flow
    test_position_action_to_order_flow()
    
    # Test manual order creation
    test_manual_order_creation()
    
    logger.info("Event flow debug tests completed")

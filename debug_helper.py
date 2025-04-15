#!/usr/bin/env python3
# debug_helper.py - Helper utilities for debugging the trading system

import logging
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

def debug_event_handler(func):
    """
    Decorator to add debugging to event handlers.
    
    Parameters:
    -----------
    func : function
        The event handler function to decorate
    """
    @wraps(func)
    def wrapper(self, event):
        event_name = event.__class__.__name__
        logger.info(f"ENTERING {self.__class__.__name__}.{func.__name__} with {event_name}")
        
        if hasattr(event, 'event_type'):
            logger.info(f"Event type: {event.event_type}")
        
        if hasattr(event, 'data'):
            # For signal events, log more details
            if hasattr(event, 'event_type') and str(event.event_type) == 'SIGNAL':
                if hasattr(event.data, 'signal_type'):
                    logger.info(f"Signal type: {event.data.signal_type}")
                if hasattr(event.data, 'price'):
                    logger.info(f"Signal price: {event.data.price}")
            
            # For order events, log more details
            elif hasattr(event, 'event_type') and str(event.event_type) == 'ORDER':
                logger.info(f"Order details: {event.data}")
        
        # Call the original function
        result = func(self, event)
        
        # Log the result
        if result:
            logger.info(f"RESULT of {self.__class__.__name__}.{func.__name__}: {result}")
        else:
            logger.info(f"EXITING {self.__class__.__name__}.{func.__name__} (no result)")
        
        return result
    
    return wrapper

def patch_position_manager(position_manager):
    """
    Patch the position manager's methods with debugging.
    
    Parameters:
    -----------
    position_manager : PositionManager
        The position manager to patch
    """
    # Store original methods
    original_on_signal = position_manager.on_signal
    
    # Replace with decorated versions
    position_manager.on_signal = debug_event_handler(original_on_signal)
    
    logger.info("Position manager patched with debugging")
    
    return position_manager

def patch_execution_engine(execution_engine):
    """
    Patch the execution engine's methods with debugging.
    
    Parameters:
    -----------
    execution_engine : ExecutionEngine
        The execution engine to patch
    """
    # Store original methods
    original_on_order = execution_engine.on_order
    
    # Replace with decorated versions
    execution_engine.on_order = debug_event_handler(original_on_order)
    
    logger.info("Execution engine patched with debugging")
    
    return execution_engine

def patch_strategy(strategy):
    """
    Patch the strategy's methods with debugging.
    
    Parameters:
    -----------
    strategy : Strategy
        The strategy to patch
    """
    # Store original methods
    original_on_bar = strategy.on_bar
    
    # Replace with decorated versions
    strategy.on_bar = debug_event_handler(original_on_bar)
    
    logger.info("Strategy patched with debugging")
    
    return strategy

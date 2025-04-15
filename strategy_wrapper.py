#!/usr/bin/env python3
# strategy_wrapper.py - Wrapper for WeightedStrategy to handle proper event signatures

import logging
from src.strategies import WeightedStrategy

# Set up logging
logger = logging.getLogger(__name__)

class StrategyWrapper:
    """
    Wrapper for the WeightedStrategy class to ensure it works with the event system.
    
    This wrapper ensures that the on_bar method takes an event parameter and
    then extracts the data to pass to the underlying strategy's on_bar method.
    """
    
    def __init__(self, strategy):
        """
        Initialize the wrapper with a strategy.
        
        Parameters:
        -----------
        strategy : WeightedStrategy
            The strategy to wrap
        """
        self.strategy = strategy
        logger.info(f"Created StrategyWrapper for {strategy.__class__.__name__}")
    
    def on_bar(self, event):
        """
        Process a bar event and pass the data to the strategy.
        
        Parameters:
        -----------
        event : Event
            Event object containing bar data
            
        Returns:
        --------
        Signal or None
            Signal from the strategy or None
        """
        # Log event reception
        logger.info(f"StrategyWrapper received bar event: {event.event_type}")
        logger.info(f"Event data keys: {event.data.keys() if hasattr(event, 'data') and event.data else 'None'}")
        
        try:
            # Log components (before)
            for i, component in enumerate(self.strategy.components):
                component_info = f"Component {i}: class={component.__class__.__name__}, methods={dir(component)}"
                logger.info(component_info)
                
                # Direct call to component (bypass strategy)
                if hasattr(component, 'on_bar'):
                    logger.info(f"Calling component.on_bar directly for {component.__class__.__name__}")
                    component_result = component.on_bar(event)
                    if component_result is not None:
                        logger.info(f"Component {component.__class__.__name__} returned signal: {component_result.signal_type}")
                    else:
                        logger.info(f"Component {component.__class__.__name__} returned None")
                else:
                    logger.warning(f"Component {component.__class__.__name__} does not have on_bar method")
                    # Check if it has generate_signal
                    if hasattr(component, 'generate_signal'):
                        logger.info(f"Component {component.__class__.__name__} has generate_signal method")
                        # Add a simple on_bar method if missing
                        component.on_bar = lambda e: component.generate_signal(e.data)
                        logger.info(f"Added on_bar method to {component.__class__.__name__}")
                    else:
                        logger.warning(f"Component {component.__class__.__name__} does not have generate_signal method either")
            
            # Try calling strategy on_bar
            logger.info("Calling strategy.on_bar")
            result = self.strategy.on_bar(event)
            
            # Log result
            if result is not None:
                logger.info(f"StrategyWrapper: strategy returned signal: {result.signal_type}")
            else:
                logger.info("StrategyWrapper: strategy returned None")
            
            return result
            
        except TypeError as e:
            # If there's a signature mismatch, try to adapt
            if "missing 1 required positional argument" in str(e):
                logger.warning(f"Strategy on_bar has incorrect signature, attempting to adapt: {e}")
                
                try:
                    # Try calling with the data instead of the event
                    result = self.strategy.on_bar(event.data)
                    
                    # Log success
                    if result is not None:
                        logger.info(f"StrategyWrapper: strategy returned signal: {result.signal_type}")
                    else:
                        logger.info("StrategyWrapper: strategy returned None")
                        
                    return result
                    
                except Exception as inner_e:
                    logger.error(f"Error calling strategy with data: {inner_e}")
                    
            logger.error(f"Error in StrategyWrapper.on_bar: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in StrategyWrapper.on_bar: {e}")
            
        # Return None in case of any error
        return None

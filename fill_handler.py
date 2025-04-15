#!/usr/bin/env python3
# fill_handler.py - Handler for fill events in the trading system

import logging
from src.events.event_handlers import EventHandler
from src.events import EventType

# Set up logging
logger = logging.getLogger(__name__)

class FillHandler(EventHandler):
    """
    Handler for fill events.
    
    This handler processes fill events from the execution engine
    and updates portfolio positions accordingly.
    """
    
    def __init__(self, position_manager):
        """
        Initialize fill handler.
        
        Args:
            position_manager: Position manager to update
        """
        super().__init__([EventType.FILL, EventType.PARTIAL_FILL])
        self.position_manager = position_manager
    
    def _process_event(self, event):
        """
        Process a fill event.
        
        Args:
            event: Event to process
        """
        logger.info(f"Processing fill event: {event.data}")
        if hasattr(self.position_manager, 'on_fill'):
            try:
                self.position_manager.on_fill(event)
                logger.info("Fill processed successfully")
            except Exception as e:
                logger.error(f"Error processing fill: {str(e)}", exc_info=True)
        else:
            logger.warning("Position manager does not have on_fill method")

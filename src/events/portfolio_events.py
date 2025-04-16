"""
Portfolio Event Classes

This module defines event classes related to portfolio management.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from src.events.event_bus import Event
from src.events.event_types import EventType


class PositionActionEvent(Event):
    """Event for position actions (entry, exit, modify)."""
    
    def __init__(self, action_type: str, **kwargs):
        """
        Initialize position action event.
        
        Args:
            action_type: Type of action ('entry', 'exit', 'modify')
            **kwargs: Additional action parameters
        """
        data = {'action_type': action_type, **kwargs}
        super().__init__(EventType.POSITION_ACTION, data)
    
    def get_action_type(self) -> str:
        """Get the action type."""
        return self.get('action_type')
    
    def get_symbol(self) -> str:
        """Get the symbol."""
        return self.get('symbol', '')
    
    def get_direction(self) -> int:
        """Get the direction."""
        return self.get('direction', 0)


class PortfolioUpdateEvent(Event):
    """Event for portfolio state updates."""
    
    def __init__(self, portfolio_state: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Initialize portfolio update event.
        
        Args:
            portfolio_state: Dictionary containing portfolio state
            timestamp: Event timestamp
        """
        super().__init__(EventType.PORTFOLIO_UPDATE, portfolio_state, timestamp)
    
    def get_equity(self) -> float:
        """Get the portfolio equity."""
        return self.get('equity', 0.0)
    
    def get_cash(self) -> float:
        """Get the portfolio cash."""
        return self.get('cash', 0.0)
    
    def get_positions_count(self) -> int:
        """Get the number of open positions."""
        return self.get('positions_count', 0)


class PositionOpenedEvent(Event):
    """Event emitted when a position is opened."""
    
    def __init__(self, position_data: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Initialize position opened event.
        
        Args:
            position_data: Dictionary containing position data
            timestamp: Event timestamp
        """
        super().__init__(EventType.POSITION_OPENED, position_data, timestamp)
    
    def get_position_id(self) -> str:
        """Get the position ID."""
        return self.get('position_id', '')
    
    def get_symbol(self) -> str:
        """Get the symbol."""
        return self.get('symbol', '')
    
    def get_direction(self) -> int:
        """Get the direction."""
        return self.get('direction', 0)
    
    def get_quantity(self) -> float:
        """Get the position quantity."""
        return self.get('quantity', 0.0)


class PositionClosedEvent(Event):
    """Event emitted when a position is closed."""
    
    def __init__(self, position_data: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Initialize position closed event.
        
        Args:
            position_data: Dictionary containing position data
            timestamp: Event timestamp
        """
        super().__init__(EventType.POSITION_CLOSED, position_data, timestamp)
    
    def get_position_id(self) -> str:
        """Get the position ID."""
        return self.get('position_id', '')
    
    def get_symbol(self) -> str:
        """Get the symbol."""
        return self.get('symbol', '')
    
    def get_realized_pnl(self) -> float:
        """Get the realized P&L."""
        return self.get('realized_pnl', 0.0)


"""
Position Management Module

This module provides a comprehensive framework for managing positions,
portfolio allocation, and risk management within the trading system.
"""

# Import core position components
from .position import (
    Position, PositionStatus, PositionFactory, 
    EntryType, ExitType
)

# Import portfolio components
from .portfolio import EventPortfolio
from src.events.portfolio_events import (
    PositionActionEvent,
    PortfolioUpdateEvent,
    PositionOpenedEvent,
    PositionClosedEvent
)

# Import position manager components
from .position_manager import PositionManager

# Import position sizing strategies
from .position_sizers import (
    PositionSizer, FixedSizeSizer, PercentOfEquitySizer,
    VolatilityPositionSizer, KellyCriterionSizer, RiskParityPositionSizer,
    PSARPositionSizer, AdaptivePositionSizer, PositionSizerFactory
)

# Import allocation strategies
from .allocation import (
    AllocationStrategy, EqualWeightAllocation, MarketCapAllocation,
    SignalStrengthAllocation, VolatilityParityAllocation,
    MaximumSharpeAllocation, ConstrainedAllocation, AllocationStrategyFactory
)

# Define version
__version__ = '0.1.0'

# Utility functions for Signal standardization
def convert_dict_to_signal(signal_dict):
    """
    Convert a signal dictionary to a Signal object for backward compatibility.
    
    Args:
        signal_dict: Dictionary containing signal data
        
    Returns:
        Signal object
    """
    from src.signals import Signal, SignalType
    
    # Extract signal type
    signal_type_val = signal_dict.get('signal_type', 'NEUTRAL')
    
    # Convert string representation to SignalType enum
    if isinstance(signal_type_val, str):
        if signal_type_val.upper() in ['BUY', 'LONG']:
            signal_type = SignalType.BUY
        elif signal_type_val.upper() in ['SELL', 'SHORT']:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
    # Handle numeric representation (1, -1, 0)
    elif isinstance(signal_type_val, (int, float)):
        if signal_type_val > 0:
            signal_type = SignalType.BUY
        elif signal_type_val < 0:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
    else:
        # Assume it's already a SignalType enum
        signal_type = signal_type_val
    
    return Signal(
        timestamp=signal_dict.get('timestamp'),
        signal_type=signal_type,
        price=signal_dict.get('price'),
        rule_id=signal_dict.get('rule_id'),
        confidence=signal_dict.get('confidence', 1.0),
        metadata=signal_dict.get('metadata', {}),
        symbol=signal_dict.get('symbol', 'default')
    )

# Create an empty Portfolio for testing when none is provided
def create_test_portfolio():
    """
    Create a minimal Portfolio instance for testing purposes.
    
    Returns:
        Portfolio: A portfolio with minimal initialization
    """
    return Portfolio(initial_capital=100000, name="TestPortfolio")

# Modify PositionManager to support simplified initialization for testing
original_init = PositionManager.__init__

def position_manager_init_with_defaults(self, portfolio=None, position_sizer=None, 
                                       allocation_strategy=None, risk_manager=None, 
                                       max_positions=0):
    """
    Initialize position manager with optional parameters for easier testing.
    
    Args:
        portfolio: Portfolio to manage (creates test portfolio if None)
        position_sizer: Strategy for determining position sizes
        allocation_strategy: Strategy for allocating capital across instruments
        risk_manager: Risk management component
        max_positions: Maximum number of positions (0 for unlimited)
    """
    # Create a test portfolio if none is provided
    if portfolio is None:
        import warnings
        warnings.warn(
            "Creating a test Portfolio since none was provided. "
            "This should only be used for testing, not in production.",
            RuntimeWarning,
            stacklevel=2
        )
        portfolio = create_test_portfolio()
        
    # Call the original init
    original_init(self, portfolio, position_sizer, allocation_strategy, risk_manager, max_positions)
    
    # Add properties to help with testing
    if not hasattr(self, 'signal_history'):
        self.signal_history = []
    if not hasattr(self, 'orders_generated'):
        self.orders_generated = []

# Replace the __init__ method
PositionManager.__init__ = position_manager_init_with_defaults

# Modify on_signal method to handle Signal objects consistently and maintain history
original_on_signal = PositionManager.on_signal

def on_signal_with_standardization(self, signal_event):
    """
    Process a signal event with standardization and history tracking.
    
    Args:
        signal_event: Signal event data (Signal object or dictionary)
        
    Returns:
        List of position action dictionaries
    """
    import warnings
    
    # Extract the signal from event or use directly
    if hasattr(signal_event, 'data'):
        signal = signal_event.data
    else:
        signal = signal_event
    
    # Convert dictionary to Signal object if needed
    if isinstance(signal, dict):
        warnings.warn(
            "Using dictionaries for signals is deprecated and will be removed in a future version. "
            "Please use Signal objects from the signals module instead.",
            DeprecationWarning,
            stacklevel=2
        )
        signal = convert_dict_to_signal(signal)
        
    # Track signal history for testing
    if hasattr(self, 'signal_history'):
        self.signal_history.append(signal)
    
    # Call original method
    actions = original_on_signal(self, signal)
    
    # Track generated orders for testing
    if hasattr(self, 'orders_generated') and actions:
        self.orders_generated.extend(actions)
    
    return actions

# Replace the on_signal method
PositionManager.on_signal = on_signal_with_standardization

# Export these classes at the module level
__all__ = [
    # Position classes
    'Position', 'PositionStatus', 'PositionFactory', 'EntryType', 'ExitType',
    
    # Portfolio classes
    'Portfolio',
    
    # Position Manager
    'PositionManager',
    
    # Position Sizers
    'PositionSizer', 'FixedSizeSizer', 'PercentOfEquitySizer',
    'VolatilityPositionSizer', 'KellyCriterionSizer', 'RiskParityPositionSizer',
    'PSARPositionSizer', 'AdaptivePositionSizer', 'PositionSizerFactory',
    
    # Allocation Strategies
    'AllocationStrategy', 'EqualWeightAllocation', 'MarketCapAllocation',
    'SignalStrengthAllocation', 'VolatilityParityAllocation',
    'MaximumSharpeAllocation', 'ConstrainedAllocation', 'AllocationStrategyFactory',
    
    # Utility functions
    'convert_dict_to_signal', 'create_test_portfolio'
]



# Add to __all__ list
__all__ += [
    'PositionActionEvent',
    'PortfolioUpdateEvent', 
    'PositionOpenedEvent',
    'PositionClosedEvent'
]

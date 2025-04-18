"""
Market Simulator for Trading System

This module simulates realistic market conditions including slippage, transaction
costs, and other market effects that impact trade execution in the backtesting system.
"""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

class SlippageModel(ABC):
    """Base class for slippage models."""
    
    @abstractmethod
    def apply_slippage(self, price: float, quantity: float, direction: int, bar: Any) -> float:
        """
        Apply slippage to a base price.
        
        Args:
            price: Base price
            quantity: Order quantity
            direction: Order direction (1 for buy, -1 for sell)
            bar: Current bar data
            
        Returns:
            float: Price after slippage
        """
        pass


class NoSlippageModel(SlippageModel):
    """No slippage model - returns the base price unchanged."""
    
    def apply_slippage(self, price: float, quantity: float, direction: int, bar: Any) -> float:
        return price


class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model - applies a fixed basis point slippage to the price.
    
    A basis point (BPS) is 1/100th of a percent. So 5 BPS = 0.05%.
    """
    
    def __init__(self, slippage_bps: float = 5):
        """
        Initialize with slippage in basis points.
        
        Args:
            slippage_bps: Slippage in basis points (5 = 0.05%)
        """
        self.slippage_bps = slippage_bps
    
    def apply_slippage(self, price: float, quantity: float, direction: int, bar: Any) -> float:
        # Convert BPS to factor (5 BPS = 0.0005)
        slippage_factor = self.slippage_bps / 10000
        
        # Apply slippage in the adverse direction
        if direction > 0:  # Buy - price goes up
            return price * (1 + slippage_factor)
        else:  # Sell - price goes down
            return price * (1 - slippage_factor)


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model that scales with order size relative to volume.
    
    Uses a price impact formula based on the square root of the ratio of
    order quantity to bar volume, scaled by the price impact parameter.
    """
    
    def __init__(self, price_impact: float = 0.1):
        """
        Initialize with price impact parameter.
        
        Args:
            price_impact: Price impact parameter (higher = more slippage)
        """
        self.price_impact = price_impact
    
    def apply_slippage(self, price: float, quantity: float, direction: int, bar: Any) -> float:
        """Apply slippage based on volume."""
        # Get volume from bar data - handle both dict and BarEvent
        volume = None
        if hasattr(bar, 'get_volume'):
            # BarEvent object
            volume = bar.get_volume()
        elif isinstance(bar, dict) and 'Volume' in bar:
            # Dictionary with Volume key
            volume = bar.get('Volume')
        
        # If no volume available, use fixed slippage
        if volume is None or volume <= 0:
            logger.debug("No valid volume data, using fixed slippage model")
            fixed_model = FixedSlippageModel()
            return fixed_model.apply_slippage(price, quantity, direction, bar)
        
        # Calculate the volume ratio and square root
        volume_ratio = abs(quantity) / volume
        impact = self.price_impact * (volume_ratio ** 0.5)
        
        # Apply impact in the adverse direction
        if direction > 0:  # Buy - price goes up
            return price * (1 + impact)
        else:  # Sell - price goes down
            return price * (1 - impact)


class FeeModel(ABC):
    """Base class for fee models."""
    
    @abstractmethod
    def calculate_fee(self, quantity: float, price: float) -> float:
        """
        Calculate transaction fee.
        
        Args:
            quantity: Order quantity
            price: Execution price
            
        Returns:
            float: Fee amount
        """
        pass


class NoFeeModel(FeeModel):
    """No fee model - returns zero fees."""
    
    def calculate_fee(self, quantity: float, price: float) -> float:
        return 0.0


class FixedFeeModel(FeeModel):
    """
    Fixed fee model - applies a fixed basis point fee to the transaction value.
    """
    
    def __init__(self, fee_bps: float = 10):
        """
        Initialize with fee in basis points.
        
        Args:
            fee_bps: Fee in basis points (10 = 0.1%)
        """
        self.fee_bps = fee_bps
    
    def calculate_fee(self, quantity: float, price: float) -> float:
        # Convert BPS to factor (10 BPS = 0.001)
        fee_factor = self.fee_bps / 10000
        
        # Calculate transaction value and apply fee
        transaction_value = abs(quantity) * price
        fee = transaction_value * fee_factor
        
        return fee


class TieredFeeModel(FeeModel):
    """
    Tiered fee model with different fees based on transaction value.
    
    For example:
    - Transactions under $10,000: 0.1%
    - Transactions $10,000-$100,000: 0.05%
    - Transactions over $100,000: 0.03%
    """
    
    def __init__(self, tier_thresholds: Optional[list] = None, tier_fees_bps: Optional[list] = None):
        """
        Initialize with tier thresholds and fees.
        
        Args:
            tier_thresholds: List of tier thresholds in ascending order
            tier_fees_bps: List of fees in basis points for each tier
        """
        self.tier_thresholds = tier_thresholds or [10000, 100000]
        self.tier_fees_bps = tier_fees_bps or [10, 5, 3]  # Default 0.1%, 0.05%, 0.03%
        
        # Validate inputs
        if len(self.tier_fees_bps) != len(self.tier_thresholds) + 1:
            raise ValueError("Number of fee tiers must be one more than number of thresholds")
    
    def calculate_fee(self, quantity: float, price: float) -> float:
        # Calculate transaction value
        transaction_value = abs(quantity) * price
        
        # Determine tier
        tier = 0
        while tier < len(self.tier_thresholds) and transaction_value >= self.tier_thresholds[tier]:
            tier += 1
        
        # Apply fee for the determined tier
        fee_bps = self.tier_fees_bps[tier]
        fee_factor = fee_bps / 10000
        fee = transaction_value * fee_factor
        
        return fee


class MarketSimulator:
    """
    Simulates market effects like slippage, delays, and transaction costs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market simulator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.slippage_model = self._get_slippage_model()
        self.fee_model = self._get_fee_model()
        
    def _get_slippage_model(self) -> SlippageModel:
        """Create the slippage model based on configuration."""
        slippage_type = self.config.get('slippage_model', 'fixed')
        
        if slippage_type == 'fixed':
            return FixedSlippageModel(self.config.get('slippage_bps', 5))
        elif slippage_type == 'volume':
            return VolumeBasedSlippageModel(self.config.get('price_impact', 0.1))
        elif slippage_type == 'none':
            return NoSlippageModel()
        else:
            # Default to fixed slippage
            return FixedSlippageModel()
            
    def _get_fee_model(self) -> FeeModel:
        """Create the fee model based on configuration."""
        fee_type = self.config.get('fee_model', 'fixed')
        
        if fee_type == 'fixed':
            return FixedFeeModel(self.config.get('fee_bps', 10))
        elif fee_type == 'tiered':
            return TieredFeeModel(
                self.config.get('tier_thresholds'), 
                self.config.get('tier_fees_bps')
            )
        elif fee_type == 'none':
            return NoFeeModel()
        else:
            # Default to fixed fee
            return FixedFeeModel()
    
    def calculate_execution_price(self, order, bar: Any) -> float:
        """
        Calculate the execution price including slippage.
        
        Args:
            order: Order object with quantity and direction
            bar: Current bar data (either BarEvent or dict)
            
        Returns:
            float: Execution price with slippage
        """
        # Determine base price from bar data
        price_field = self.config.get('price_field', 'Close')
        base_price = None
        
        # Handle BarEvent objects
        if hasattr(bar, 'get_price'):
            # BarEvent provides get_price() that returns Close by default
            base_price = bar.get_price()
        elif hasattr(bar, 'get_data'):
            # If it has get_data method, try to get price from the data dict
            data = bar.get_data()
            if isinstance(data, dict):
                base_price = data.get(price_field, data.get('Close'))
        elif isinstance(bar, dict):
            # Handle dictionary bar data
            base_price = bar.get(price_field, bar.get('Close'))
        
        # If we couldn't get a price, log warning and use fallback
        if base_price is None:
            logger.warning(f"Could not extract price from bar data, using order price")
            if hasattr(order, 'get_price'):
                base_price = order.get_price()
            elif hasattr(order, 'price'):
                base_price = order.price
            else:
                base_price = 0
                logger.error(f"No price available in bar or order")
        
        # Apply slippage
        quantity = order.quantity if hasattr(order, 'quantity') else order.get_quantity()
        direction = order.direction if hasattr(order, 'direction') else order.get_direction()
        
        execution_price = self.slippage_model.apply_slippage(
            base_price, quantity, direction, bar
        )
        
        return execution_price
    
    def calculate_fees(self, order, execution_price: float) -> float:
        """
        Calculate transaction fees.
        
        Args:
            order: Order object with quantity
            execution_price: Price after slippage
            
        Returns:
            float: Fee amount
        """
        # Handle both object properties and getter methods
        if hasattr(order, 'quantity'):
            quantity = order.quantity
        elif hasattr(order, 'get_quantity'):
            quantity = order.get_quantity()
        else:
            logger.error(f"Order has no quantity attribute or method")
            return 0.0
            
        return self.fee_model.calculate_fee(quantity, execution_price)

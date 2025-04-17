"""
Position Sizers Module

This module provides different position sizing strategies to determine the appropriate
position size based on risk parameters and market conditions.
"""

import numpy as np
import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable

from src.events.signal_event import SignalEvent

# Set up logging
logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    
    Position sizers determine the appropriate position size based on
    risk parameters, portfolio state, and market conditions.
    """
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size (number of units)
        """
        pass


class FixedSizeSizer(PositionSizer):
    """
    Position sizer that always returns a fixed size.
    
    This sizer always returns the same position size regardless
    of portfolio value or market conditions.
    """
    
    def __init__(self, fixed_size: float = 100):
        """
        Initialize fixed size position sizer.
        
        Args:
            fixed_size: Number of units to trade
        """
        self.fixed_size = fixed_size
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Fixed position size
        """
        # Apply direction
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        
        return self.fixed_size * direction


class PercentOfEquitySizer(PositionSizer):
    """
    Position sizer that sizes based on a percentage of portfolio equity.
    
    This sizer calculates position size to be a specified percentage of
    the current portfolio equity.
    """
    
    def __init__(self, percent: float = 2.0, max_pct: float = 25.0):
        """
        Initialize percent of equity position sizer.
        
        Args:
            percent: Percentage of equity to allocate (2.0 = 2%)
            max_pct: Maximum percentage of equity for any one position
        """
        self.percent = percent
        self.max_pct = max_pct

    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size based on a percentage of equity.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size (positive for buy, negative for sell)
        """
        # Get equity from portfolio
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
        
        # Get direction from signal
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        
        # Skip neutral signals
        if direction == 0:
            return 0
        
        # Use price from signal if not provided
        if current_price is None:
            current_price = signal.price
        
        if current_price is None or current_price <= 0:
            return 0
        
        # Calculate position size as percentage of equity
        size = equity * (self.percent / 100) / current_price
        
        # Apply maximum percentage constraint
        max_size = equity * (self.max_pct / 100) / current_price
        size = min(size, max_size)
        
        # Apply direction
        return size * direction


class VolatilityPositionSizer(PositionSizer):
    """
    Position sizer that sizes based on asset volatility.
    
    This sizer calculates position size to risk a specified percentage of
    portfolio equity per unit of volatility.
    """
    
    def __init__(self, risk_pct: float = 1.0, atr_multiplier: float = 2.0, 
                lookback_period: int = 20, max_pct: float = 25.0):
        """
        Initialize volatility-based position sizer.
        
        Args:
            risk_pct: Percentage of equity to risk (1.0 = 1%)
            atr_multiplier: Multiplier for ATR to set stop distance
            lookback_period: Period for ATR calculation
            max_pct: Maximum percentage of equity for any one position
        """
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
        self.lookback_period = lookback_period
        self.max_pct = max_pct
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size based on volatility
        """
        # Get portfolio equity
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            logger.warning("Portfolio equity is zero or negative")
            return 0
            
        # Use provided price or get from signal
        if current_price is None:
            current_price = signal.price
            
        if current_price is None or current_price <= 0:
            logger.warning("Current price is invalid")
            return 0
            
        # Get ATR from signal metadata or use default volatility estimate
        atr = None
        if hasattr(signal, 'metadata') and signal.metadata:
            atr = signal.metadata.get('atr')
            
        if atr is None:
            # Try other common names for volatility
            if hasattr(signal, 'metadata') and signal.metadata:
                atr = signal.metadata.get('volatility')
                if atr is None:
                    atr = signal.metadata.get('atr_value')
            
        if atr is None:
            # Default to estimated volatility based on price
            atr = current_price * 0.015  # Assume 1.5% daily volatility
            logger.debug(f"No ATR provided in signal, using estimate: {atr:.4f}")
            
        # Calculate stop distance
        stop_distance = atr * self.atr_multiplier
        
        # Calculate dollar amount to risk
        risk_amount = equity * (self.risk_pct / 100)
        
        # Calculate units based on risk amount and stop distance
        if stop_distance <= 0:
            logger.warning("Stop distance is zero or negative")
            return 0
            
        units = risk_amount / stop_distance
        
        # Limit by maximum percentage of equity
        max_units = (equity * (self.max_pct / 100)) / current_price
        units = min(units, max_units)
        
        # Apply direction
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
            
        return units * direction


class KellyCriterionSizer(PositionSizer):
    """
    Position sizer based on the Kelly Criterion formula.
    
    This sizer calculates the optimal position size based on the estimated
    win rate and reward-to-risk ratio.
    """
    
    def __init__(self, win_rate: float = 0.5, reward_risk_ratio: float = 1.0, 
                fraction: float = 0.5, max_pct: float = 25.0):
        """
        Initialize Kelly Criterion position sizer.
        
        Args:
            win_rate: Expected win rate (0.5 = 50%)
            reward_risk_ratio: Expected reward-to-risk ratio
            fraction: Fraction of Kelly to use (0.5 = "Half Kelly")
            max_pct: Maximum percentage of equity for any one position
        """
        self.win_rate = win_rate
        self.reward_risk_ratio = reward_risk_ratio
        self.fraction = fraction
        self.max_pct = max_pct
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size based on Kelly Criterion
        """
        # Get portfolio equity
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
            
        # Use provided price or get from signal
        if current_price is None:
            current_price = signal.price
            
        if current_price is None or current_price <= 0:
            return 0
        
        # Get win rate and reward-risk ratio from signal or use defaults
        win_rate = self.win_rate
        reward_risk_ratio = self.reward_risk_ratio
        
        if hasattr(signal, 'metadata') and signal.metadata:
            if 'win_rate' in signal.metadata:
                win_rate = signal.metadata['win_rate']
            if 'reward_risk_ratio' in signal.metadata:
                reward_risk_ratio = signal.metadata['reward_risk_ratio']
        
        # Calculate Kelly percentage
        # Kelly formula: f* = (p×b - q) ÷ b
        # where: f* = Kelly percentage, p = win rate, q = loss rate (1-p), b = odds received on a bet
        loss_rate = 1 - win_rate
        kelly_pct = (win_rate * reward_risk_ratio - loss_rate) / reward_risk_ratio
        
        # Apply fraction and limit
        allocation_pct = min(kelly_pct * self.fraction, self.max_pct / 100)
        
        # Ensure non-negative (Kelly can be negative if edge is negative)
        allocation_pct = max(0, allocation_pct)
        
        # Calculate dollar amount to allocate
        allocation = equity * allocation_pct
        
        # Calculate units based on price
        if current_price <= 0:
            return 0
            
        units = allocation / current_price
        
        # Apply direction
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
            
        return units * direction


class RiskParityPositionSizer(PositionSizer):
    """
    Position sizer based on risk parity principles.
    
    This sizer allocates capital to maintain equal risk contribution 
    across different assets based on their volatility.
    """
    
    def __init__(self, target_portfolio_volatility: float = 0.10, max_pct: float = 25.0):
        """
        Initialize risk parity position sizer.
        
        Args:
            target_portfolio_volatility: Target annualized portfolio volatility
            max_pct: Maximum percentage of equity for any one position
        """
        self.target_portfolio_volatility = target_portfolio_volatility
        self.max_pct = max_pct
        self.asset_volatilities = {}  # Cache for asset volatilities
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size based on risk parity
        """
        # Get portfolio equity
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
            
        # Use provided price or get from signal
        if current_price is None:
            current_price = signal.price
            
        if current_price is None or current_price <= 0:
            return 0
            
        # Get symbol
        symbol = signal.symbol
        
        # Get asset volatility from signal or use cached/default
        volatility = None
        if hasattr(signal, 'metadata') and signal.metadata:
            volatility = signal.metadata.get('volatility')
            if volatility is None:
                volatility = signal.metadata.get('atr', None)
                if volatility is not None:
                    # Convert ATR to percentage volatility
                    volatility = volatility / current_price
        
        if volatility is None:
            volatility = self.asset_volatilities.get(symbol, 0.01)  # Default to 1% daily volatility
            
        # Cache the volatility for future use
        self.asset_volatilities[symbol] = volatility
            
        # Get number of assets in portfolio
        num_assets = 1
        if hasattr(portfolio, 'positions'):
            positions = portfolio.positions
            if isinstance(positions, dict):
                unique_symbols = set([symbol] + list(positions.keys()))
                num_assets = len(unique_symbols)
        elif hasattr(portfolio, 'get_position_count'):
            num_assets = max(1, portfolio.get_position_count() + 1) # +1 for new position
        
        # Calculate target risk per asset assuming equal risk allocation
        risk_per_asset = self.target_portfolio_volatility / math.sqrt(num_assets)
        
        # Calculate position size to achieve target risk
        # Units × Price × Volatility = Risk Amount
        # Units = Risk Amount / (Price × Volatility)
        risk_amount = equity * risk_per_asset
        units = risk_amount / (current_price * volatility)
        
        # Limit by maximum percentage of equity
        max_units = (equity * (self.max_pct / 100)) / current_price
        units = min(units, max_units)
        
        # Apply direction
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
            
        return units * direction


class PSARPositionSizer(PositionSizer):
    """
    Position sizer based on Parabolic SAR stops.
    
    This sizer calculates position size using Parabolic SAR for stop placement
    and a fixed risk percentage of equity.
    """
    
    def __init__(self, risk_pct: float = 1.0, psar_factor: float = 0.02, 
                psar_max: float = 0.2, max_pct: float = 25.0):
        """
        Initialize PSAR-based position sizer.
        
        Args:
            risk_pct: Percentage of equity to risk (1.0 = 1%)
            psar_factor: PSAR acceleration factor
            psar_max: Maximum PSAR acceleration
            max_pct: Maximum percentage of equity for any one position
        """
        self.risk_pct = risk_pct
        self.psar_factor = psar_factor
        self.psar_max = psar_max
        self.max_pct = max_pct
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size based on PSAR stops
        """
        # Get portfolio equity
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
            
        # Use provided price or get from signal
        if current_price is None:
            current_price = signal.price
            
        if current_price is None or current_price <= 0:
            return 0
            
        # Get PSAR value from signal metadata
        psar = None
        if hasattr(signal, 'metadata') and signal.metadata:
            psar = signal.metadata.get('psar')
            
        if psar is None:
            logger.warning(f"No PSAR value provided in signal, using estimated stop distance")
            # Default to estimated stop distance based on price and direction
            direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
            psar = current_price * 0.98 if direction > 0 else current_price * 1.02
            
        # Calculate stop distance
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        if direction == 0:
            return 0
            
        if direction > 0:  # Long position
            stop_distance = current_price - psar
        else:  # Short position
            stop_distance = psar - current_price
            
        # Ensure positive stop distance
        stop_distance = max(0.0001, stop_distance)
        
        # Calculate dollar amount to risk
        risk_amount = equity * (self.risk_pct / 100)
        
        # Calculate units based on risk amount and stop distance
        units = risk_amount / stop_distance
        
        # Limit by maximum percentage of equity
        max_units = (equity * (self.max_pct / 100)) / current_price
        units = min(units, max_units)
            
        return units * direction


class AdaptivePositionSizer(PositionSizer):
    """
    Position sizer that adapts sizing based on market conditions.
    
    This sizer dynamically adjusts position sizing based on market volatility,
    trend strength, and other market conditions.
    """
    
    def __init__(self, base_risk_pct: float = 1.0, 
                volatility_factor: float = 0.5,
                trend_factor: float = 0.5,
                max_pct: float = 25.0):
        """
        Initialize adaptive position sizer.
        
        Args:
            base_risk_pct: Base percentage of equity to risk (1.0 = 1%)
            volatility_factor: How much to adjust for volatility (0-1)
            trend_factor: How much to adjust for trend strength (0-1)
            max_pct: Maximum percentage of equity for any one position
        """
        self.base_risk_pct = base_risk_pct
        self.volatility_factor = volatility_factor
        self.trend_factor = trend_factor
        self.max_pct = max_pct
        
    def calculate_position_size(self, signal: SignalEvent, 
                               portfolio: Any, 
                               current_price: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Position size based on adaptive factors
        """
        # Get portfolio equity
        equity = 0
        if hasattr(portfolio, 'equity'):
            equity = portfolio.equity
        elif hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
            
        # Use provided price or get from signal
        if current_price is None:
            current_price = signal.price
            
        if current_price is None or current_price <= 0:
            return 0
            
        # Get relevant metrics from signal metadata
        volatility = 0.01  # Default 1% volatility
        trend_strength = 0.5  # Default neutral trend strength
        confidence = 0.5  # Default moderate confidence
        
        if hasattr(signal, 'metadata') and signal.metadata:
            metadata = signal.metadata
            volatility = metadata.get('volatility', volatility)
            trend_strength = metadata.get('trend_strength', trend_strength)
            confidence = metadata.get('confidence', confidence) if hasattr(signal, 'confidence') else confidence
            
            # Look for alternative volatility measure
            if volatility == 0.01 and 'atr' in metadata:
                volatility = metadata['atr'] / current_price
        
        # 1. Decrease risk when volatility is high
        volatility_adjustment = 1.0 - (self.volatility_factor * (volatility / 0.01 - 1.0))
        volatility_adjustment = max(0.25, min(2.0, volatility_adjustment))  # Limit adjustment range
        
        # 2. Increase risk when trend is strong and aligned with position
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        if direction == 0:
            return 0
            
        trend_direction = 0
        if hasattr(signal, 'metadata') and signal.metadata:
            trend_direction = signal.metadata.get('trend_direction', 0)
            
        trend_alignment = direction * trend_direction  # Positive if aligned, negative if counter-trend
        
        trend_adjustment = 1.0 + (self.trend_factor * trend_alignment * trend_strength)
        trend_adjustment = max(0.5, min(1.5, trend_adjustment))  # Limit adjustment range
        
        # 3. Adjust based on signal confidence
        confidence_adjustment = confidence / 0.5  # 1.0 for moderate confidence
        confidence_adjustment = max(0.5, min(2.0, confidence_adjustment))  # Limit adjustment range
        
        # Combine adjustments
        risk_pct = self.base_risk_pct * volatility_adjustment * trend_adjustment * confidence_adjustment
        
        # Get ATR or use volatility for stop distance
        atr = None
        if hasattr(signal, 'metadata') and signal.metadata:
            atr = signal.metadata.get('atr')
        
        if atr is None:
            atr = volatility * current_price
            
        stop_distance = atr * 2.0  # Use 2 ATRs for stop distance
        
        # Calculate dollar amount to risk
        risk_amount = equity * (risk_pct / 100)
        
        # Calculate units based on risk amount and stop distance
        if stop_distance <= 0:
            return 0
            
        units = risk_amount / stop_distance
        
        # Limit by maximum percentage of equity
        max_units = (equity * (self.max_pct / 100)) / current_price
        units = min(units, max_units)
        
        return units * direction


class PositionSizerFactory:
    """
    Factory for creating position sizers.
    
    This factory creates different types of position sizers based on
    configuration parameters.
    """
    
    @staticmethod
    def create_sizer(sizer_type: str, **kwargs) -> PositionSizer:
        """
        Create a position sizer of the specified type.
        
        Args:
            sizer_type: Type of position sizer to create
            **kwargs: Parameters for the position sizer
            
        Returns:
            PositionSizer instance
            
        Raises:
            ValueError: If sizer type is not recognized
        """
        sizer_type = sizer_type.lower()
        
        if sizer_type == 'fixed':
            return FixedSizeSizer(
                fixed_size=kwargs.get('fixed_size', 100)
            )
        elif sizer_type == 'percent_equity':
            return PercentOfEquitySizer(
                percent=kwargs.get('percent', 2.0),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        elif sizer_type == 'volatility':
            return VolatilityPositionSizer(
                risk_pct=kwargs.get('risk_pct', 1.0),
                atr_multiplier=kwargs.get('atr_multiplier', 2.0),
                lookback_period=kwargs.get('lookback_period', 20),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        elif sizer_type == 'kelly':
            return KellyCriterionSizer(
                win_rate=kwargs.get('win_rate', 0.5),
                reward_risk_ratio=kwargs.get('reward_risk_ratio', 1.0),
                fraction=kwargs.get('fraction', 0.5),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        elif sizer_type == 'risk_parity':
            return RiskParityPositionSizer(
                target_portfolio_volatility=kwargs.get('target_portfolio_volatility', 0.10),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        elif sizer_type == 'psar':
            return PSARPositionSizer(
                risk_pct=kwargs.get('risk_pct', 1.0),
                psar_factor=kwargs.get('psar_factor', 0.02),
                psar_max=kwargs.get('psar_max', 0.2),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        elif sizer_type == 'adaptive':
            return AdaptivePositionSizer(
                base_risk_pct=kwargs.get('base_risk_pct', 1.0),
                volatility_factor=kwargs.get('volatility_factor', 0.5),
                trend_factor=kwargs.get('trend_factor', 0.5),
                max_pct=kwargs.get('max_pct', 25.0)
            )
        else:
            raise ValueError(f"Unknown position sizer type: {sizer_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> PositionSizer:
        """
        Create a position sizer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            PositionSizer instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        sizer_type = config.get('type')
        if not sizer_type:
            raise ValueError("Configuration must include 'type' key")
            
        params = config.get('params', {})
        
        return PositionSizerFactory.create_sizer(sizer_type, **params)

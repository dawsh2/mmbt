"""
Allocation Module

This module provides portfolio allocation strategies for distributing
capital across multiple trading instruments.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class AllocationStrategy(ABC):
    """
    Abstract base class for portfolio allocation strategies.
    
    Allocation strategies determine how to distribute capital
    across multiple instruments in a portfolio.
    """
    
    @abstractmethod
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital across instruments.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of allocation weights by symbol (0-1)
        """
        pass


class EqualWeightAllocation(AllocationStrategy):
    """
    Equal weight allocation strategy.
    
    This strategy allocates equal capital to each instrument 
    in the portfolio.
    """
    
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital equally across instruments.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of equal allocation weights by symbol
        """
        # Get list of symbols with signals
        symbols = list(signals.keys())
        
        if not symbols:
            return {}
            
        # Equal weight allocation
        weight = 1.0 / len(symbols)
        
        return {symbol: weight for symbol in symbols}


class MarketCapAllocation(AllocationStrategy):
    """
    Market capitalization weighted allocation.
    
    This strategy allocates capital based on relative market 
    capitalization of each instrument.
    """
    
    def __init__(self, market_caps: Dict[str, float]):
        """
        Initialize market cap allocation strategy.
        
        Args:
            market_caps: Dictionary of market capitalizations by symbol
        """
        self.market_caps = market_caps
        
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital based on market capitalization.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of market cap-weighted allocation weights by symbol
        """
        # Get list of symbols with signals and market caps
        symbols = [s for s in signals.keys() if s in self.market_caps]
        
        if not symbols:
            return {}
            
        # Calculate total market cap for normalization
        total_market_cap = sum(self.market_caps[s] for s in symbols)
        
        if total_market_cap <= 0:
            # Fallback to equal weight if total market cap is zero
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}
            
        # Market cap weighted allocation
        weights = {s: self.market_caps[s] / total_market_cap for s in symbols}
        
        return weights


class SignalStrengthAllocation(AllocationStrategy):
    """
    Signal strength weighted allocation.
    
    This strategy allocates capital based on the relative strength
    or confidence of trading signals.
    """
    
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital based on signal strength.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of signal strength-weighted allocation weights by symbol
        """
        if not signals:
            return {}
            
        # Get signal confidence/strength for each symbol
        strengths = {}
        
        for symbol, signal in signals.items():
            # Try different signal strength indicators
            strength = signal.get('confidence', None)
            if strength is None:
                strength = signal.get('strength', None)
            if strength is None:
                strength = signal.get('probability', None)
            if strength is None:
                strength = 0.5  # Default to moderate confidence
                
            strengths[symbol] = max(0.0, min(1.0, strength))  # Ensure in 0-1 range
        
        # Calculate total strength for normalization
        total_strength = sum(strengths.values())
        
        if total_strength <= 0:
            # Fallback to equal weight if total strength is zero
            weight = 1.0 / len(signals)
            return {symbol: weight for symbol in signals}
            
        # Signal strength weighted allocation
        weights = {symbol: strength / total_strength for symbol, strength in strengths.items()}
        
        return weights


class VolatilityParityAllocation(AllocationStrategy):
    """
    Volatility parity (risk parity) allocation.
    
    This strategy allocates capital to achieve equal risk contribution
    from each instrument based on historical volatility.
    """
    
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital based on volatility parity.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of volatility-parity allocation weights by symbol
        """
        if not signals:
            return {}
            
        # Get volatility for each symbol
        volatilities = {}
        
        for symbol, signal in signals.items():
            # Get volatility from signal or use default
            volatility = signal.get('volatility', None)
            if volatility is None:
                volatility = signal.get('atr', 0.01) / prices.get(symbol, 100.0)
            if volatility <= 0:
                volatility = 0.01  # Default to 1% volatility
                
            volatilities[symbol] = volatility
        
        # Calculate inverse volatility for each symbol (higher vol = lower weight)
        inv_volatilities = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        
        # Calculate total inverse volatility for normalization
        total_inv_vol = sum(inv_volatilities.values())
        
        if total_inv_vol <= 0:
            # Fallback to equal weight
            weight = 1.0 / len(signals)
            return {symbol: weight for symbol in signals}
            
        # Risk parity weighted allocation
        weights = {symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_volatilities.items()}
        
        return weights


class MaximumSharpeAllocation(AllocationStrategy):
    """
    Maximum Sharpe ratio allocation.
    
    This strategy allocates capital to maximize the portfolio's expected
    Sharpe ratio based on expected returns and covariance matrix.
    """
    
    def __init__(self, expected_returns: Dict[str, float] = None, 
                covariance_matrix: Dict[str, Dict[str, float]] = None,
                risk_free_rate: float = 0.0):
        """
        Initialize maximum Sharpe ratio allocation strategy.
        
        Args:
            expected_returns: Dictionary of expected returns by symbol
            covariance_matrix: Dictionary of covariance values between symbols
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.expected_returns = expected_returns or {}
        self.covariance_matrix = covariance_matrix or {}
        self.risk_free_rate = risk_free_rate
        
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital to maximize Sharpe ratio.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of allocation weights by symbol
        """
        if not signals:
            return {}
            
        # Get list of symbols with signals
        symbols = list(signals.keys())
        
        # Get expected returns for each symbol (from signals or predefined)
        returns = {}
        for symbol in symbols:
            if symbol in self.expected_returns:
                returns[symbol] = self.expected_returns[symbol]
            else:
                # Try to get expected return from signal
                signal = signals[symbol]
                expected_return = signal.get('expected_return', None)
                if expected_return is None:
                    # Use signal confidence as a proxy
                    confidence = signal.get('confidence', 0.5)
                    direction = 1 if signal.get('signal_type', '').upper() in ['BUY', 'LONG'] else -1
                    expected_return = direction * confidence * 0.05  # Scale to reasonable range
                    
                returns[symbol] = expected_return
        
        # Convert to numpy arrays
        symbols_array = np.array(symbols)
        returns_array = np.array([returns[s] for s in symbols])
        
        # Construct covariance matrix
        n = len(symbols)
        cov_matrix = np.zeros((n, n))
        
        # Fill covariance matrix from provided data or use default
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if (symbol_i in self.covariance_matrix and 
                    symbol_j in self.covariance_matrix[symbol_i]):
                    cov_matrix[i, j] = self.covariance_matrix[symbol_i][symbol_j]
                elif i == j:
                    # Default variance on diagonal (from signals or default)
                    vol = signals[symbol_i].get('volatility', 0.01)
                    cov_matrix[i, j] = vol * vol
                else:
                    # Default small covariance for different symbols
                    cov_matrix[i, j] = 0.0001
        
        # Ensure covariance matrix is positive definite
        # Add small values to diagonal if necessary
        if n > 0:
            min_eig = np.min(np.linalg.eigvals(cov_matrix))
            if min_eig < 0:
                cov_matrix += np.eye(n) * (abs(min_eig) + 1e-6)
        
        try:
            # Calculate inverse of covariance matrix
            cov_inv = np.linalg.inv(cov_matrix)
            
            # Calculate weights for maximum Sharpe ratio
            excess_returns = returns_array - self.risk_free_rate
            weights = np.dot(cov_inv, excess_returns)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Create weights dictionary
            return {s: max(0, w) for s, w in zip(symbols, weights)}
        except (np.linalg.LinAlgError, ZeroDivisionError):
            logger.warning("Error calculating max Sharpe allocation. Falling back to equal weight.")
            # Fallback to equal weights
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}


class ConstrainedAllocation(AllocationStrategy):
    """
    Constrained allocation strategy.
    
    This strategy applies constraints (min/max weights, sector limits)
    to an underlying allocation strategy.
    """
    
    def __init__(self, base_strategy: AllocationStrategy, 
                min_weight: float = 0.0, max_weight: float = 0.25,
                sector_limits: Dict[str, float] = None,
                sector_mapping: Dict[str, str] = None):
        """
        Initialize constrained allocation strategy.
        
        Args:
            base_strategy: Underlying allocation strategy
            min_weight: Minimum weight for any symbol
            max_weight: Maximum weight for any symbol
            sector_limits: Maximum allocation per sector
            sector_mapping: Mapping of symbols to sectors
        """
        self.base_strategy = base_strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sector_limits = sector_limits or {}
        self.sector_mapping = sector_mapping or {}
        
    def allocate(self, portfolio: Any, signals: Dict[str, Dict[str, Any]], 
                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate portfolio capital with constraints.
        
        Args:
            portfolio: Current portfolio state
            signals: Dictionary of signals by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of constrained allocation weights by symbol
        """
        # Get base allocation from underlying strategy
        base_weights = self.base_strategy.allocate(portfolio, signals, prices)
        
        if not base_weights:
            return {}
            
        # Apply min/max constraints
        constrained_weights = {s: min(self.max_weight, max(self.min_weight, w)) 
                              for s, w in base_weights.items()}
        
        # Normalize weights after applying min/max constraints
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {s: w / total_weight for s, w in constrained_weights.items()}
        
        # Apply sector constraints if provided
        if self.sector_limits and self.sector_mapping:
            # Calculate current sector allocations
            sector_allocations = {}
            for symbol, weight in constrained_weights.items():
                sector = self.sector_mapping.get(symbol)
                if sector:
                    sector_allocations[sector] = sector_allocations.get(sector, 0) + weight
            
            # Check if any sector exceeds its limit
            excess_allocation = {}
            for sector, allocation in sector_allocations.items():
                limit = self.sector_limits.get(sector, 1.0)
                if allocation > limit:
                    excess_allocation[sector] = allocation - limit
            
            # If any sector exceeds its limit, redistribute excess allocation
            if excess_allocation:
                # Calculate total excess
                total_excess = sum(excess_allocation.values())
                
                # Reduce weights for symbols in sectors with excess allocation
                for symbol, weight in list(constrained_weights.items()):
                    sector = self.sector_mapping.get(symbol)
                    if sector in excess_allocation:
                        excess_ratio = excess_allocation[sector] / sector_allocations[sector]
                        constrained_weights[symbol] = weight * (1 - excess_ratio)
                
                # Normalize weights again
                total_weight = sum(constrained_weights.values())
                if total_weight > 0:
                    constrained_weights = {s: w / total_weight for s, w in constrained_weights.items()}
        
        return constrained_weights


class AllocationStrategyFactory:
    """
    Factory for creating allocation strategies.
    
    This factory creates different types of allocation strategies
    based on configuration parameters.
    """
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> AllocationStrategy:
        """
        Create an allocation strategy of the specified type.
        
        Args:
            strategy_type: Type of allocation strategy to create
            **kwargs: Parameters for the allocation strategy
            
        Returns:
            AllocationStrategy instance
            
        Raises:
            ValueError: If strategy type is not recognized
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type == 'equal_weight':
            return EqualWeightAllocation()
        elif strategy_type == 'market_cap':
            return MarketCapAllocation(
                market_caps=kwargs.get('market_caps', {})
            )
        elif strategy_type == 'signal_strength':
            return SignalStrengthAllocation()
        elif strategy_type == 'volatility_parity':
            return VolatilityParityAllocation()
        elif strategy_type == 'max_sharpe':
            return MaximumSharpeAllocation(
                expected_returns=kwargs.get('expected_returns'),
                covariance_matrix=kwargs.get('covariance_matrix'),
                risk_free_rate=kwargs.get('risk_free_rate', 0.0)
            )
        elif strategy_type == 'constrained':
            base_strategy = kwargs.get('base_strategy')
            if not base_strategy:
                raise ValueError("Constrained allocation requires 'base_strategy' parameter")
                
            return ConstrainedAllocation(
                base_strategy=base_strategy,
                min_weight=kwargs.get('min_weight', 0.0),
                max_weight=kwargs.get('max_weight', 0.25),
                sector_limits=kwargs.get('sector_limits'),
                sector_mapping=kwargs.get('sector_mapping')
            )
        else:
            raise ValueError(f"Unknown allocation strategy type: {strategy_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> AllocationStrategy:
        """
        Create an allocation strategy from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            AllocationStrategy instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        strategy_type = config.get('type')
        if not strategy_type:
            raise ValueError("Configuration must include 'type' key")
            
        params = config.get('params', {})
        
        # Handle nested base strategy for constrained allocation
        if strategy_type.lower() == 'constrained' and 'base_strategy' in params:
            base_config = params['base_strategy']
            base_strategy = AllocationStrategyFactory.create_from_config(base_config)
            params['base_strategy'] = base_strategy
        
        return AllocationStrategyFactory.create_strategy(strategy_type, **params)


# Example usage
if __name__ == "__main__":
    # Sample portfolio
    class SamplePortfolio:
        def __init__(self, equity):
            self.equity = equity
            self.positions = {}
    
    portfolio = SamplePortfolio(100000)
    
    # Sample signals
    signals = {
        'AAPL': {'signal_type': 'BUY', 'confidence': 0.8, 'volatility': 0.012},
        'MSFT': {'signal_type': 'BUY', 'confidence': 0.6, 'volatility': 0.010},
        'GOOGL': {'signal_type': 'BUY', 'confidence': 0.7, 'volatility': 0.015},
        'AMZN': {'signal_type': 'SELL', 'confidence': 0.5, 'volatility': 0.020}
    }
    
    # Sample prices
    prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 2000.0,
        'AMZN': 3000.0
    }
    
    # Create different allocation strategies
    equal_weight = AllocationStrategyFactory.create_strategy('equal_weight')
    signal_strength = AllocationStrategyFactory.create_strategy('signal_strength')
    volatility_parity = AllocationStrategyFactory.create_strategy('volatility_parity')
    
    # Constrained allocation based on equal weight
    constrained = AllocationStrategyFactory.create_strategy(
        'constrained',
        base_strategy=equal_weight,
        max_weight=0.4
    )
    
    # Calculate allocations
    equal_allocation = equal_weight.allocate(portfolio, signals, prices)
    signal_allocation = signal_strength.allocate(portfolio, signals, prices)
    vol_allocation = volatility_parity.allocate(portfolio, signals, prices)
    constrained_allocation = constrained.allocate(portfolio, signals, prices)
    
    print("Equal Weight Allocation:")
    for symbol, weight in equal_allocation.items():
        print(f"  {symbol}: {weight:.2%}")
        
    print("\nSignal Strength Allocation:")
    for symbol, weight in signal_allocation.items():
        print(f"  {symbol}: {weight:.2%}")
        
    print("\nVolatility Parity Allocation:")
    for symbol, weight in vol_allocation.items():
        print(f"  {symbol}: {weight:.2%}")
        
    print("\nConstrained Allocation:")
    for symbol, weight in constrained_allocation.items():
        print(f"  {symbol}: {weight:.2%}")

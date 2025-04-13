"""
Position Management Module for Trading System

This module provides position sizing, risk management, and allocation strategies
for controlling how signals are converted into actual trading positions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class SizingStrategy(ABC):
    """Base class for position sizing strategies."""
    
    @abstractmethod
    def calculate_size(self, signal, portfolio):
        """
        Calculate position size based on signal and portfolio.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            float: Position size (positive for buy, negative for sell)
        """
        pass


class FixedSizingStrategy(SizingStrategy):
    """Position sizing using a fixed number of units."""
    
    def __init__(self, fixed_size=100):
        """
        Initialize with fixed size.
        
        Args:
            fixed_size: Number of units to trade
        """
        self.fixed_size = fixed_size
        
    def calculate_size(self, signal, portfolio):
        """Calculate position size."""
        # Direction from signal
        if hasattr(signal, 'signal_type'):
            direction = signal.signal_type.value
        elif hasattr(signal, 'signal'):
            direction = signal.signal
        else:
            direction = 1  # Default to buy
            
        return self.fixed_size * direction


class PercentOfEquitySizing(SizingStrategy):
    """Size position as a percentage of portfolio equity."""
    
    def __init__(self, percent=0.02):
        """
        Initialize with percent of equity to risk.
        
        Args:
            percent: Percentage of equity to allocate (0.02 = 2%)
        """
        self.percent = percent
        
    def calculate_size(self, signal, portfolio):
        """Calculate position size."""
        # Base size on portfolio equity
        equity_amount = portfolio.equity * self.percent
        
        # Convert to number of shares based on current price
        price = signal.price if hasattr(signal, 'price') else 100  # Default price if not available
        num_shares = equity_amount / price
        
        # Adjust direction based on signal
        if hasattr(signal, 'signal_type'):
            direction = signal.signal_type.value
        elif hasattr(signal, 'signal'):
            direction = signal.signal
        else:
            direction = 1  # Default to buy
            
        if direction < 0:
            num_shares = -num_shares
            
        return num_shares


class VolatilityBasedSizing(SizingStrategy):
    """Size positions based on asset volatility."""
    
    def __init__(self, risk_pct=0.01, lookback_period=20):
        """
        Initialize with risk percentage and lookback.
        
        Args:
            risk_pct: Percentage of equity to risk per unit of volatility
            lookback_period: Period for calculating volatility
        """
        self.risk_pct = risk_pct
        self.lookback_period = lookback_period
        self.price_history = {}  # symbol -> deque of prices
        
    def calculate_size(self, signal, portfolio):
        """Calculate position size based on volatility."""
        # Get symbol from signal
        symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_period)
        
        price = signal.price if hasattr(signal, 'price') else 100  # Default price if not available
        self.price_history[symbol].append(price)
        
        # Calculate volatility (standard deviation of returns)
        prices = list(self.price_history[symbol])
        if len(prices) < 2:
            return 0
            
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        volatility = np.std(returns) if returns else 0.01  # Default if not enough data
        
        # Don't divide by zero
        if volatility < 0.0001:
            volatility = 0.0001
            
        # Calculate position size based on volatility
        risk_amount = portfolio.equity * self.risk_pct
        dollar_volatility = price * volatility
        
        # Size to risk a fixed percentage of portfolio per unit of volatility
        num_shares = risk_amount / dollar_volatility
        
        # Adjust direction based on signal
        if hasattr(signal, 'signal_type'):
            direction = signal.signal_type.value
        elif hasattr(signal, 'signal'):
            direction = signal.signal
        else:
            direction = 1  # Default to buy
            
        if direction < 0:
            num_shares = -num_shares
            
        return num_shares


class KellySizingStrategy(SizingStrategy):
    """Position sizing based on the Kelly Criterion."""
    
    def __init__(self, win_rate=0.5, win_loss_ratio=1.0, fraction=0.5):
        """
        Initialize with Kelly parameters.
        
        Args:
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Ratio of average win to average loss
            fraction: Fraction of Kelly to use (0-1, lower is more conservative)
        """
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.fraction = fraction
        self.trade_history = []
        
    def update_parameters(self, trade_history):
        """
        Update win rate and win/loss ratio based on trade history.
        
        Args:
            trade_history: List of trade results
        """
        if not trade_history:
            return
            
        # Calculate win rate
        wins = sum(1 for trade in trade_history if trade > 0)
        total_trades = len(trade_history)
        self.win_rate = wins / total_trades if total_trades > 0 else 0.5
        
        # Calculate win/loss ratio
        avg_win = np.mean([trade for trade in trade_history if trade > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(trade) for trade in trade_history if trade < 0])
        
        if avg_loss > 0:
            self.win_loss_ratio = avg_win / avg_loss
        else:
            self.win_loss_ratio = 1.0
        
    def calculate_size(self, signal, portfolio):
        """Calculate position size using Kelly formula."""
        # Kelly formula: f* = (p * b - (1 - p)) / b
        # Where f* is the fraction of the bankroll to wager
        # p is the probability of winning
        # b is the odds received on the wager (how much you win per dollar wagered)
        
        # Calculate Kelly percentage
        kelly_pct = (self.win_rate * self.win_loss_ratio - (1 - self.win_rate)) / self.win_loss_ratio
        
        # Apply fraction to make more conservative
        kelly_pct = kelly_pct * self.fraction
        
        # Limit to reasonable range (0-50%)
        kelly_pct = max(0, min(0.5, kelly_pct))
        
        # Calculate dollar amount
        dollar_amount = portfolio.equity * kelly_pct
        
        # Convert to shares
        price = signal.price if hasattr(signal, 'price') else 100  # Default price if not available
        num_shares = dollar_amount / price
        
        # Adjust direction based on signal
        if hasattr(signal, 'signal_type'):
            direction = signal.signal_type.value
        elif hasattr(signal, 'signal'):
            direction = signal.signal
        else:
            direction = 1  # Default to buy
            
        if direction < 0:
            num_shares = -num_shares
            
        return num_shares


class RiskManager:
    """
    Manages risk controls and limits for trading.
    """
    
    def __init__(self, max_position_pct=0.25, max_drawdown_pct=0.10, max_concentration_pct=None,
                 use_stop_loss=False, stop_loss_pct=0.05, use_take_profit=False, take_profit_pct=0.10):
        """
        Initialize with risk parameters.
        
        Args:
            max_position_pct: Maximum position size as percentage of portfolio
            max_drawdown_pct: Maximum allowable drawdown from peak equity
            max_concentration_pct: Maximum allocation to a single instrument
            use_stop_loss: Whether to use stop losses
            stop_loss_pct: Stop loss percentage from entry
            use_take_profit: Whether to use take profits
            take_profit_pct: Take profit percentage from entry
        """
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_concentration_pct = max_concentration_pct or max_position_pct
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.use_take_profit = use_take_profit
        self.take_profit_pct = take_profit_pct
        self.peak_equity = 0
        self.stops = {}  # symbol -> {stop_price, take_profit_price}
        
    def check_signal(self, signal, portfolio):
        """
        Check if a signal should be acted upon given risk constraints.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            bool: True if signal passes risk checks, False otherwise
        """
        # Update peak equity
        if portfolio.equity > self.peak_equity:
            self.peak_equity = portfolio.equity
            
        # Check drawdown
        current_drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if current_drawdown > self.max_drawdown_pct:
            return False  # Exceeded max drawdown
            
        # Get symbol from signal
        symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
            
        # Check existing position concentration
        if symbol in portfolio.positions:
            position = portfolio.positions[symbol]
            position_pct = abs(position.market_value / portfolio.equity)
            
            # If adding to position that's already large, reject
            if position_pct > self.max_position_pct and (
                (position.quantity > 0 and signal.signal_type.value > 0) or
                (position.quantity < 0 and signal.signal_type.value < 0)
            ):
                return False
        
        # Check total portfolio concentration
        if self.max_concentration_pct < 1.0:
            # Calculate total allocation to this instrument family
            # (Would need instrument metadata for more sophisticated checks)
            pass
        
        return True
        
    def adjust_position_size(self, symbol, size, portfolio):
        """
        Adjust position size to comply with risk limits.
        
        Args:
            symbol: Instrument symbol
            size: Calculated position size
            portfolio: Current portfolio state
            
        Returns:
            float: Adjusted position size
        """
        # Skip if size is zero
        if size == 0:
            return 0
            
        # Get current position if exists
        current_position = portfolio.positions.get(symbol)
        position_value = 0
        
        if current_position:
            # If adding to existing position
            if (size > 0 and current_position.quantity > 0) or (size < 0 and current_position.quantity < 0):
                position_value = abs(current_position.market_value + (size * current_position.avg_price))
            # If reducing position, no need to check limits
            elif (size > 0 and current_position.quantity < 0) or (size < 0 and current_position.quantity > 0):
                if abs(size) <= abs(current_position.quantity):
                    return size  # Reducing position, no adjustment needed
        else:
            # New position
            avg_price = portfolio.positions.get(symbol, 0).avg_price if symbol in portfolio.positions else 100
            position_value = abs(size * avg_price)
            
        # Check against max position size
        max_position_value = portfolio.equity * self.max_position_pct
        if position_value > max_position_value:
            # Scale down to max allowed size
            size = (max_position_value / position_value) * size
            
        return size
        
    def get_risk_metrics(self, portfolio):
        """
        Get current risk metrics for the portfolio.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            dict: Risk metrics
        """
        # Calculate current drawdown
        current_drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Calculate position concentrations
        concentrations = {
            symbol: abs(position.market_value / portfolio.equity)
            for symbol, position in portfolio.positions.items()
        }
        
        return {
            'current_drawdown': current_drawdown,
            'peak_equity': self.peak_equity,
            'position_concentrations': concentrations,
            'stops': self.stops
        }
    
    def set_stop_loss(self, symbol, entry_price, direction):
        """
        Set stop loss and take profit levels for a new position.
        
        Args:
            symbol: Instrument symbol
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
        """
        if not self.use_stop_loss and not self.use_take_profit:
            return
            
        if symbol not in self.stops:
            self.stops[symbol] = {}
            
        # Set stop loss
        if self.use_stop_loss:
            if direction > 0:  # Long position
                stop_price = entry_price * (1 - self.stop_loss_pct)
            else:  # Short position
                stop_price = entry_price * (1 + self.stop_loss_pct)
                
            self.stops[symbol]['stop_price'] = stop_price
            
        # Set take profit
        if self.use_take_profit:
            if direction > 0:  # Long position
                take_profit_price = entry_price * (1 + self.take_profit_pct)
            else:  # Short position
                take_profit_price = entry_price * (1 - self.take_profit_pct)
                
            self.stops[symbol]['take_profit_price'] = take_profit_price
    
    def check_stops(self, symbol, current_price, direction):
        """
        Check if stop loss or take profit levels have been hit.
        
        Args:
            symbol: Instrument symbol
            current_price: Current market price
            direction: Position direction (1 for long, -1 for short)
            
        Returns:
            bool: True if stop or take profit hit, False otherwise
        """
        if symbol not in self.stops:
            return False
            
        stops = self.stops[symbol]
        
        # Check stop loss
        if 'stop_price' in stops:
            if direction > 0 and current_price <= stops['stop_price']:  # Long position stop
                return True
            elif direction < 0 and current_price >= stops['stop_price']:  # Short position stop
                return True
                
        # Check take profit
        if 'take_profit_price' in stops:
            if direction > 0 and current_price >= stops['take_profit_price']:  # Long position take profit
                return True
            elif direction < 0 and current_price <= stops['take_profit_price']:  # Short position take profit
                return True
                
        return False
    
    def reset(self):
        """Reset risk manager state."""
        self.peak_equity = 0
        self.stops = {}


class AllocationStrategy(ABC):
    """Base class for portfolio allocation strategies."""
    
    @abstractmethod
    def adjust_allocation(self, symbol, size, portfolio):
        """
        Adjust position size based on portfolio allocation constraints.
        
        Args:
            symbol: Instrument symbol
            size: Calculated position size
            portfolio: Current portfolio state
            
        Returns:
            float: Adjusted position size
        """
        pass


class EqualAllocationStrategy(AllocationStrategy):
    """
    Allocate capital equally across instruments.
    
    This strategy ensures that no instrument uses more than its fair share
    of the portfolio, based on the maximum number of simultaneous positions.
    """
    
    def __init__(self, max_instruments=10):
        """
        Initialize with maximum number of instruments.
        
        Args:
            max_instruments: Maximum number of simultaneous positions
        """
        self.max_instruments = max_instruments
        
    def adjust_allocation(self, symbol, size, portfolio):
        """Adjust position size for equal allocation."""
        # Skip if size is zero
        if size == 0:
            return 0
        
        # Calculate maximum allocation per instrument
        max_allocation_pct = 1.0 / self.max_instruments
        
        # Current allocation to this instrument
        current_position = portfolio.positions.get(symbol)
        current_allocation = 0
        
        if current_position:
            current_allocation = abs(current_position.market_value / portfolio.equity)
        
        # Calculate target allocation
        avg_price = current_position.avg_price if current_position else 100  # Default if no position
        target_allocation = abs(size * avg_price / portfolio.equity)
        
        # If target exceeds max allocation, adjust size
        if target_allocation > max_allocation_pct:
            max_size = max_allocation_pct * portfolio.equity / avg_price
            # Preserve direction
            return max_size if size > 0 else -max_size
            
        return size


class VolatilityParityAllocation(AllocationStrategy):
    """
    Allocate capital based on relative volatility of instruments.
    
    This strategy allocates more capital to less volatile instruments
    and less capital to more volatile ones, targeting equal risk contribution.
    """
    
    def __init__(self, lookback_period=20, target_portfolio_vol=0.01):
        """
        Initialize with volatility parameters.
        
        Args:
            lookback_period: Period for calculating volatility
            target_portfolio_vol: Target portfolio volatility
        """
        self.lookback_period = lookback_period
        self.target_portfolio_vol = target_portfolio_vol
        self.volatilities = {}  # symbol -> volatility
        self.price_history = {}  # symbol -> deque of prices
        
    def update_volatility(self, symbol, price):
        """
        Update volatility estimate for an instrument.
        
        Args:
            symbol: Instrument symbol
            price: Current price
        """
        # Initialize price history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_period)
            
        # Add price to history
        self.price_history[symbol].append(price)
        
        # Calculate volatility if we have enough history
        if len(self.price_history[symbol]) >= 2:
            prices = list(self.price_history[symbol])
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = np.std(returns) if returns else 0.01
            self.volatilities[symbol] = volatility
        else:
            # Default volatility if not enough history
            self.volatilities[symbol] = 0.01
        
    def adjust_allocation(self, symbol, size, portfolio):
        """Adjust position size for volatility parity."""
        # Skip if size is zero
        if size == 0:
            return 0
            
        # Get or calculate volatility
        price = portfolio.positions.get(symbol, 0).avg_price if symbol in portfolio.positions else 100
        self.update_volatility(symbol, price)
        vol = self.volatilities.get(symbol, 0.01)
        
        # Calculate risk contribution
        dollar_vol = abs(size) * price * vol
        
        # Calculate inverse volatility weight
        total_inverse_vol = sum(1/v for v in self.volatilities.values() if v > 0)
        weight = (1/vol) / total_inverse_vol if vol > 0 and total_inverse_vol > 0 else 0
        
        # Calculate target position size
        target_size = (weight * portfolio.equity * self.target_portfolio_vol) / (price * vol) if vol > 0 else 0
        
        # Preserve direction
        target_size = target_size if size > 0 else -target_size
        
        return target_size


class PositionManager:
    """
    Manages position sizing, risk, and allocation decisions.
    """
    
    def __init__(self, sizing_strategy=None, risk_manager=None, allocation_strategy=None):
        """
        Initialize the position manager.
        
        Args:
            sizing_strategy: Strategy for determining position size
            risk_manager: Risk management controls
            allocation_strategy: Portfolio allocation strategy
        """
        self.sizing_strategy = sizing_strategy or PercentOfEquitySizing(0.02)  # Default 2%
        self.risk_manager = risk_manager or RiskManager()
        self.allocation_strategy = allocation_strategy or EqualAllocationStrategy()
        
    def calculate_position_size(self, signal, portfolio):
        """
        Calculate the appropriate position size for a signal.
        
        Args:
            signal: The trading signal
            portfolio: Current portfolio state
            
        Returns:
            float: Position size (positive for buy, negative for sell, 0 for no trade)
        """
        # Skip if signal doesn't pass risk checks
        if not self.risk_manager.check_signal(signal, portfolio):
            return 0
            
        # Calculate base position size
        size = self.sizing_strategy.calculate_size(signal, portfolio)
        
        # Apply allocation constraints
        symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
        size = self.allocation_strategy.adjust_allocation(
            symbol, size, portfolio
        )
        
        # Apply final risk limits
        size = self.risk_manager.adjust_position_size(
            symbol, size, portfolio
        )
        
        # Set stop loss if applicable
        if size != 0 and hasattr(signal, 'price'):
            direction = 1 if size > 0 else -1
            self.risk_manager.set_stop_loss(symbol, signal.price, direction)
        
        return size
    
    def get_risk_metrics(self, portfolio):
        """
        Get current risk metrics for the portfolio.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            dict: Risk metrics
        """
        return self.risk_manager.get_risk_metrics(portfolio)
    
    def update_stops(self, bar_data, portfolio):
        """
        Update and check stop losses and take profits.
        
        Args:
            bar_data: Current bar data
            portfolio: Current portfolio state
            
        Returns:
            dict: Positions to close due to stops {symbol: reason}
        """
        positions_to_close = {}
        
        for symbol, position in portfolio.positions.items():
            # Get current price
            if isinstance(bar_data, dict):
                # Multi-symbol bar data
                if symbol in bar_data:
                    current_price = bar_data[symbol]['Close']
                else:
                    # Skip if price not available
                    continue
            else:
                # Single-symbol bar data, assume it's for the current symbol
                current_price = bar_data['Close']
                
            # Check if stops are hit
            direction = 1 if position.quantity > 0 else -1
            if self.risk_manager.check_stops(symbol, current_price, direction):
                reason = "stop_loss" if current_price < position.avg_price else "take_profit"
                positions_to_close[symbol] = reason
                
        return positions_to_close
    
    def reset(self):
        """Reset position manager state."""
        # Reset risk manager
        if hasattr(self.risk_manager, 'reset'):
            self.risk_manager.reset()
            
        # Reset sizing strategy if needed
        if hasattr(self.sizing_strategy, 'reset'):
            self.sizing_strategy.reset()
            
        # Reset allocation strategy if needed
        if hasattr(self.allocation_strategy, 'reset'):
            self.allocation_strategy.reset()

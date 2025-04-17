"""
Type definitions and enums for the risk management module.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np


class RiskToleranceLevel(Enum):
    """Enum for different risk tolerance levels."""
    CONSERVATIVE = auto()
    MODERATE = auto()
    AGGRESSIVE = auto()


class ExitReason(Enum):
    """Enum for different exit reasons."""
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()
    TIME_EXIT = auto()
    STRATEGY_EXIT = auto()
    UNKNOWN = auto()


@dataclass
class TradeMetrics:
    """Data class for storing trade metrics."""
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    mae_pct: Optional[float] = None  # Maximum Adverse Excursion
    mfe_pct: Optional[float] = None  # Maximum Favorable Excursion
    duration: Optional[timedelta] = None
    duration_bars: Optional[int] = None
    is_winner: Optional[bool] = None
    exit_reason: Optional[ExitReason] = None
    price_path: Optional[List[Dict[str, Any]]] = None


@dataclass
class RiskParameters:
    """Data class for storing risk management parameters."""
    stop_loss_pct: float
    take_profit_pct: Optional[float] = None
    trailing_stop_activation_pct: Optional[float] = None
    trailing_stop_distance_pct: Optional[float] = None
    max_duration: Optional[Union[int, timedelta]] = None
    risk_reward_ratio: Optional[float] = None
    expected_win_rate: Optional[float] = None
    risk_tolerance: Optional[RiskToleranceLevel] = RiskToleranceLevel.MODERATE
    
    # Position sizing parameters
    position_sizer_type: Optional[str] = "risk_based"  # "fixed", "percent_equity", "volatility", etc.
    position_sizer_params: Optional[Dict[str, Any]] = None
    max_position_pct: Optional[float] = 0.25  # Maximum position size as % of equity
    risk_per_trade_pct: Optional[float] = 1.0  # Risk percentage per trade (1.0 = 1%)
    
    def __post_init__(self):
        """Initialize optional dictionary fields if they are None."""
        if self.position_sizer_params is None:
            self.position_sizer_params = {}


@dataclass
class RiskAnalysisResults:
    """Data class for storing risk analysis results."""
    mae_stats: Dict[str, float]
    mfe_stats: Dict[str, float]
    etd_stats: Dict[str, Any]
    trade_count: int
    win_rate: float
    risk_setups: Optional[List[Dict[str, Any]]] = None

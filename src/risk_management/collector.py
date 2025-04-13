"""
Data collection module for risk management metrics.

This module provides tools for tracking and collecting Maximum Adverse Excursion (MAE),
Maximum Favorable Excursion (MFE), and Entry-To-Exit Duration (ETD) metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

from .types import TradeMetrics, ExitReason


class RiskMetricsCollector:
    """
    Collects and stores MAE, MFE, and ETD metrics for analyzing trading performance.
    
    This class tracks the complete lifecycle of trades, including the price path
    between entry and exit to calculate MAE, MFE, and other risk metrics.
    """
    
    def __init__(self):
        """Initialize the risk metrics collector."""
        self.trade_metrics: List[TradeMetrics] = []
        self.current_trades: Dict[str, TradeMetrics] = {}  # Keyed by trade_id
        self.price_paths: Dict[str, List[Dict[str, Any]]] = {}  # Keyed by trade_id
    
    def start_trade(self, trade_id: str, entry_time: datetime, entry_price: float, 
                   direction: str) -> None:
        """
        Start tracking a new trade.
        
        Args:
            trade_id: Unique identifier for the trade
            entry_time: Time of trade entry
            entry_price: Price at entry
            direction: Trade direction ('long' or 'short')
        """
        if direction not in ('long', 'short'):
            raise ValueError(f"Direction must be 'long' or 'short', got {direction}")
            
        self.current_trades[trade_id] = TradeMetrics(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction
        )
        self.price_paths[trade_id] = []
    
    def update_price_path(self, trade_id: str, bar_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Update the price path for a trade with new bar data.
        
        Args:
            trade_id: Unique identifier for the trade
            bar_data: Bar data containing at minimum 'high', 'low', and 'timestamp'
            
        Returns:
            Optional tuple of (mae_pct, mfe_pct) for the current trade
        """
        if trade_id not in self.current_trades:
            return None
        
        # Store the bar data
        self.price_paths[trade_id].append(bar_data)
        
        # Update MAE and MFE
        trade = self.current_trades[trade_id]
        price_path = self.price_paths[trade_id]
        
        if trade.direction == 'long':
            # For long trades
            lowest_price = min(bar['low'] for bar in price_path)
            highest_price = max(bar['high'] for bar in price_path)
            
            mae_pct = (trade.entry_price - lowest_price) / trade.entry_price * 100
            mfe_pct = (highest_price - trade.entry_price) / trade.entry_price * 100
        else:
            # For short trades
            lowest_price = min(bar['low'] for bar in price_path)
            highest_price = max(bar['high'] for bar in price_path)
            
            mae_pct = (highest_price - trade.entry_price) / trade.entry_price * 100
            mfe_pct = (trade.entry_price - lowest_price) / trade.entry_price * 100
        
        # Update the trade object
        trade.mae_pct = mae_pct
        trade.mfe_pct = mfe_pct
        
        return mae_pct, mfe_pct
    
    def end_trade(self, trade_id: str, exit_time: datetime, exit_price: float, 
                 exit_reason: ExitReason = ExitReason.UNKNOWN) -> Optional[TradeMetrics]:
        """
        End tracking for a trade and calculate final metrics.
        
        Args:
            trade_id: Unique identifier for the trade
            exit_time: Time of trade exit
            exit_price: Price at exit
            exit_reason: Reason for the exit
            
        Returns:
            The completed TradeMetrics object or None if trade_id not found
        """
        if trade_id not in self.current_trades:
            return None
            
        trade = self.current_trades[trade_id]
        price_path = self.price_paths[trade_id]
        
        # Calculate final metrics
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.duration = exit_time - trade.entry_time
        trade.duration_bars = len(price_path)
        trade.exit_reason = exit_reason
        
        # Calculate return percentage
        if trade.direction == 'long':
            trade.return_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
            trade.is_winner = exit_price > trade.entry_price
        else:  # short
            trade.return_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
            trade.is_winner = exit_price < trade.entry_price
        
        # Store completed trade
        self.trade_metrics.append(trade)
        
        # Clean up current trade tracking
        del self.current_trades[trade_id]
        
        # Optionally keep price path for detailed analysis or remove to save memory
        # If keeping, assign it to the trade object
        trade.price_path = self.price_paths[trade_id]
        del self.price_paths[trade_id]
        
        return trade
    
    def track_completed_trade(self, entry_time: datetime, entry_price: float, direction: str,
                             exit_time: datetime, exit_price: float, 
                             price_path: Optional[List[Dict[str, Any]]] = None,
                             exit_reason: ExitReason = ExitReason.UNKNOWN) -> TradeMetrics:
        """
        Record a completed trade with all information in one call.
        
        Args:
            entry_time: Time of trade entry
            entry_price: Price at entry
            direction: Trade direction ('long' or 'short')
            exit_time: Time of trade exit
            exit_price: Price at exit
            price_path: Optional list of bar data between entry and exit
            exit_reason: Reason for the exit
            
        Returns:
            The newly created and stored TradeMetrics object
        """
        # Create base trade object
        trade = TradeMetrics(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            exit_time=exit_time,
            exit_price=exit_price,
            duration=exit_time - entry_time,
            exit_reason=exit_reason
        )
        
        # Process price path if provided
        if price_path:
            trade.price_path = price_path
            trade.duration_bars = len(price_path)
            
            # Calculate MAE and MFE
            if direction == 'long':
                lowest_price = min(bar['low'] for bar in price_path)
                highest_price = max(bar['high'] for bar in price_path)
                
                trade.mae_pct = (entry_price - lowest_price) / entry_price * 100
                trade.mfe_pct = (highest_price - entry_price) / entry_price * 100
            else:  # short
                lowest_price = min(bar['low'] for bar in price_path)
                highest_price = max(bar['high'] for bar in price_path)
                
                trade.mae_pct = (highest_price - entry_price) / entry_price * 100
                trade.mfe_pct = (entry_price - lowest_price) / entry_price * 100
        
        # Calculate return percentage and winner status
        if direction == 'long':
            trade.return_pct = (exit_price - entry_price) / entry_price * 100
            trade.is_winner = exit_price > entry_price
        else:  # short
            trade.return_pct = (entry_price - exit_price) / entry_price * 100
            trade.is_winner = exit_price < entry_price
        
        # Store the completed trade
        self.trade_metrics.append(trade)
        
        return trade
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to pandas DataFrame for analysis.
        
        Returns:
            DataFrame containing all trade metrics
        """
        # Convert list of TradeMetrics to dictionary of lists
        data = {
            'entry_time': [],
            'entry_price': [],
            'direction': [],
            'exit_time': [],
            'exit_price': [],
            'return_pct': [],
            'mae_pct': [],
            'mfe_pct': [],
            'duration': [],
            'duration_bars': [],
            'is_winner': [],
            'exit_reason': []
        }
        
        for trade in self.trade_metrics:
            data['entry_time'].append(trade.entry_time)
            data['entry_price'].append(trade.entry_price)
            data['direction'].append(trade.direction)
            data['exit_time'].append(trade.exit_time)
            data['exit_price'].append(trade.exit_price)
            data['return_pct'].append(trade.return_pct)
            data['mae_pct'].append(trade.mae_pct)
            data['mfe_pct'].append(trade.mfe_pct)
            # Convert timedelta to seconds, hours, or days as needed
            if trade.duration:
                data['duration'].append(trade.duration.total_seconds())
            else:
                data['duration'].append(None)
            data['duration_bars'].append(trade.duration_bars)
            data['is_winner'].append(trade.is_winner)
            if trade.exit_reason:
                data['exit_reason'].append(trade.exit_reason.name)
            else:
                data['exit_reason'].append(None)
        
        return pd.DataFrame(data)
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save the collected metrics to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.get_metrics_dataframe()
        df.to_csv(filepath, index=False)
    
    def load_from_csv(self, filepath: str) -> None:
        """
        Load metrics from a CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        df = pd.read_csv(filepath, parse_dates=['entry_time', 'exit_time'])
        
        self.trade_metrics = []
        
        for _, row in df.iterrows():
            # Convert exit_reason string back to enum
            if pd.notna(row['exit_reason']):
                exit_reason = ExitReason[row['exit_reason']]
            else:
                exit_reason = None
                
            # Calculate duration as timedelta
            if pd.notna(row['duration']):
                duration = timedelta(seconds=row['duration'])
            else:
                duration = None
            
            trade = TradeMetrics(
                entry_time=row['entry_time'],
                entry_price=row['entry_price'],
                direction=row['direction'],
                exit_time=row['exit_time'],
                exit_price=row['exit_price'],
                return_pct=row['return_pct'],
                mae_pct=row['mae_pct'],
                mfe_pct=row['mfe_pct'],
                duration=duration,
                duration_bars=row['duration_bars'] if pd.notna(row['duration_bars']) else None,
                is_winner=row['is_winner'] if pd.notna(row['is_winner']) else None,
                exit_reason=exit_reason
            )
            
            self.trade_metrics.append(trade)
    
    def clear(self) -> None:
        """Clear all stored trade metrics."""
        self.trade_metrics = []
        self.current_trades = {}
        self.price_paths = {}

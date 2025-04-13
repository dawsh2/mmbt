"""
Analysis engine for risk management metrics.

This module provides tools for analyzing Maximum Adverse Excursion (MAE),
Maximum Favorable Excursion (MFE), and Entry-To-Exit Duration (ETD) metrics
to derive optimal risk management parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from .types import RiskAnalysisResults


class RiskAnalysisEngine:
    """
    Analyzes MAE, MFE, and ETD metrics to derive insights for risk management.
    
    This class processes trade metrics to calculate statistical properties
    that can be used to optimize risk management parameters.
    """
    
    def __init__(self, metrics_df: pd.DataFrame):
        """
        Initialize the risk analysis engine.
        
        Args:
            metrics_df: DataFrame containing trade metrics including mae_pct, mfe_pct, etc.
        """
        self.metrics_df = metrics_df
        
        # Verify required columns exist
        required_columns = ['mae_pct', 'mfe_pct', 'duration', 'is_winner', 'return_pct']
        missing_columns = [col for col in required_columns if col not in metrics_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in metrics_df: {missing_columns}")
    
    def analyze_mae(self) -> Dict[str, float]:
        """
        Analyze MAE distribution and characteristics.
        
        Returns:
            Dictionary of MAE statistics
        """
        stats = {}
        
        # Overall MAE statistics
        stats['mean_mae'] = self.metrics_df['mae_pct'].mean()
        stats['median_mae'] = self.metrics_df['mae_pct'].median()
        stats['std_mae'] = self.metrics_df['mae_pct'].std()
        stats['max_mae'] = self.metrics_df['mae_pct'].max()
        
        # MAE by outcome
        winners = self.metrics_df[self.metrics_df['is_winner'] == True]
        losers = self.metrics_df[self.metrics_df['is_winner'] == False]
        
        stats['mae_winners_mean'] = winners['mae_pct'].mean() if not winners.empty else np.nan
        stats['mae_losers_mean'] = losers['mae_pct'].mean() if not losers.empty else np.nan
        
        # MAE percentiles for stop-loss considerations
        for p in [50, 75, 80, 85, 90, 95]:
            stats[f'mae_percentile_{p}'] = self.metrics_df['mae_pct'].quantile(p/100)
            
        # Analyze MAE to return correlation
        stats['mae_return_correlation'] = self.metrics_df[['mae_pct', 'return_pct']].corr().iloc[0, 1]
        
        return stats
    
    def analyze_mfe(self) -> Dict[str, float]:
        """
        Analyze MFE distribution and characteristics.
        
        Returns:
            Dictionary of MFE statistics
        """
        stats = {}
        
        # Overall MFE statistics
        stats['mean_mfe'] = self.metrics_df['mfe_pct'].mean()
        stats['median_mfe'] = self.metrics_df['mfe_pct'].median()
        stats['std_mfe'] = self.metrics_df['mfe_pct'].std()
        stats['max_mfe'] = self.metrics_df['mfe_pct'].max()
        
        # MFE by outcome
        winners = self.metrics_df[self.metrics_df['is_winner'] == True]
        losers = self.metrics_df[self.metrics_df['is_winner'] == False]
        
        stats['mfe_winners_mean'] = winners['mfe_pct'].mean() if not winners.empty else np.nan
        stats['mfe_losers_mean'] = losers['mfe_pct'].mean() if not losers.empty else np.nan
        
        # MFE percentiles for take-profit considerations
        for p in [25, 50, 65, 75, 80, 90]:
            stats[f'mfe_percentile_{p}'] = self.metrics_df['mfe_pct'].quantile(p/100)
            
        # Calculate MFE capture ratio (how much of potential profit was captured)
        avg_return = self.metrics_df['return_pct'].mean()
        avg_mfe = stats['mean_mfe']
        stats['mfe_capture_ratio'] = avg_return / avg_mfe if avg_mfe > 0 else 0
        
        # Calculate MFE to MAE ratio
        avg_mae = self.metrics_df['mae_pct'].mean()
        stats['mfe_mae_ratio'] = avg_mfe / avg_mae if avg_mae > 0 else np.inf
            
        return stats
    
    def analyze_etd(self) -> Dict[str, Any]:
        """
        Analyze trade duration characteristics.
        
        Returns:
            Dictionary of ETD statistics
        """
        stats = {}
        
        # Basic duration statistics
        stats['mean_duration'] = self.metrics_df['duration'].mean()
        stats['median_duration'] = self.metrics_df['duration'].median()
        stats['std_duration'] = self.metrics_df['duration'].std()
        stats['max_duration'] = self.metrics_df['duration'].max()
        stats['min_duration'] = self.metrics_df['duration'].min()
        
        # Duration by outcome
        winners = self.metrics_df[self.metrics_df['is_winner'] == True]
        losers = self.metrics_df[self.metrics_df['is_winner'] == False]
        
        stats['duration_winners_mean'] = winners['duration'].mean() if not winners.empty else np.nan
        stats['duration_losers_mean'] = losers['duration'].mean() if not losers.empty else np.nan
        
        # Duration percentiles
        for p in [25, 50, 75, 90, 95]:
            stats[f'duration_percentile_{p}'] = self.metrics_df['duration'].quantile(p/100)
        
        # Analyze duration vs. return correlation
        stats['duration_return_correlation'] = self.metrics_df[['duration', 'return_pct']].corr().iloc[0, 1]
        
        # Calculate win rate by duration quartiles
        self.metrics_df['duration_quartile'] = pd.qcut(self.metrics_df['duration'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        duration_win_rates = self.metrics_df.groupby('duration_quartile')['is_winner'].mean()
        
        # Convert Series to dict for JSON serialization
        stats['win_rate_by_quartile'] = duration_win_rates.to_dict()
        
        return stats
    
    def analyze_exit_reasons(self) -> Dict[str, Any]:
        """
        Analyze the distribution and performance of different exit reasons.
        
        Returns:
            Dictionary of exit reason statistics
        """
        if 'exit_reason' not in self.metrics_df.columns:
            return {'error': 'exit_reason column not found in metrics dataframe'}
        
        stats = {}
        
        # Count by exit reason
        exit_counts = self.metrics_df['exit_reason'].value_counts()
        stats['counts'] = exit_counts.to_dict()
        
        # Win rate by exit reason
        win_rates = self.metrics_df.groupby('exit_reason')['is_winner'].mean()
        stats['win_rates'] = win_rates.to_dict()
        
        # Average return by exit reason
        avg_returns = self.metrics_df.groupby('exit_reason')['return_pct'].mean()
        stats['avg_returns'] = avg_returns.to_dict()
        
        # Average MAE by exit reason
        avg_mae = self.metrics_df.groupby('exit_reason')['mae_pct'].mean()
        stats['avg_mae'] = avg_mae.to_dict()
        
        # Average MFE by exit reason
        avg_mfe = self.metrics_df.groupby('exit_reason')['mfe_pct'].mean()
        stats['avg_mfe'] = avg_mfe.to_dict()
        
        return stats
    
    def calculate_mad_ratio(self) -> float:
        """
        Calculate the MAE/MFE Adjusted Duration (MAD) ratio.
        
        This is a custom metric that evaluates the quality of trades by comparing
        how quickly favorable moves happen relative to adverse moves.
        
        Returns:
            MAD ratio value
        """
        # Add calculated columns
        self.metrics_df['mfe_mae_ratio'] = self.metrics_df['mfe_pct'] / self.metrics_df['mae_pct'].replace(0, 0.01)
        self.metrics_df['return_duration_ratio'] = self.metrics_df['return_pct'] / self.metrics_df['duration']
        
        # Calculate correlation between return and duration
        mad_ratio = self.metrics_df['return_duration_ratio'].median() * self.metrics_df['mfe_mae_ratio'].median()
        
        return mad_ratio
    
    def analyze_all(self) -> RiskAnalysisResults:
        """
        Run all analyses and return comprehensive results.
        
        Returns:
            RiskAnalysisResults object containing all analysis results
        """
        # Run individual analyses
        mae_stats = self.analyze_mae()
        mfe_stats = self.analyze_mfe()
        etd_stats = self.analyze_etd()
        
        # Calculate overall trade statistics
        trade_count = len(self.metrics_df)
        win_rate = self.metrics_df['is_winner'].mean()
        
        # Create and return the combined results
        return RiskAnalysisResults(
            mae_stats=mae_stats,
            mfe_stats=mfe_stats,
            etd_stats=etd_stats,
            trade_count=trade_count,
            win_rate=win_rate
        )
    
    @classmethod
    def from_collector(cls, collector, clear_price_paths: bool = True):
        """
        Create an analyzer directly from a RiskMetricsCollector.
        
        Args:
            collector: A RiskMetricsCollector instance
            clear_price_paths: Whether to clear price paths to save memory
            
        Returns:
            RiskAnalysisEngine instance
        """
        # Get metrics dataframe
        metrics_df = collector.get_metrics_dataframe()
        
        # Optionally clear price paths to free memory
        if clear_price_paths:
            for trade in collector.trade_metrics:
                trade.price_path = None
        
        return cls(metrics_df)

"""
Parameter optimization for risk management rules.

This module derives optimal risk management parameters from analyzed trade metrics
including stop-loss levels, take-profit targets, trailing stops, and time exits.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

from src.risk_management.types import RiskAnalysisResults, RiskToleranceLevel, RiskParameters


class RiskParameterOptimizer:
    """
    Derives optimal risk management parameters from analyzed trade metrics.
    
    This class uses the statistical properties of MAE, MFE, and ETD to create
    data-driven risk management rules that align with the strategy's characteristics.
    """
    
    def __init__(self, analysis_results: RiskAnalysisResults):
        """
        Initialize the risk parameter optimizer.
        
        Args:
            analysis_results: Results from the RiskAnalysisEngine
        """
        self.mae_stats = analysis_results.mae_stats
        self.mfe_stats = analysis_results.mfe_stats
        self.etd_stats = analysis_results.etd_stats
        self.trade_count = analysis_results.trade_count
        self.win_rate = analysis_results.win_rate
    
    def optimize_stop_loss(self, risk_tolerance: Union[str, RiskToleranceLevel] = 'moderate') -> Dict[str, Any]:
        """
        Derive optimal stop-loss parameters based on MAE analysis.
        
        Args:
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
                            (or corresponding RiskToleranceLevel enum)
            
        Returns:
            Dictionary of optimized stop-loss parameters
        """
        # Convert string to enum if needed
        if isinstance(risk_tolerance, str):
            risk_tolerance = getattr(RiskToleranceLevel, risk_tolerance.upper())
        
        # Map risk tolerance to MAE percentiles
        percentile_map = {
            RiskToleranceLevel.CONSERVATIVE: 90,  # Wider stop that covers 90% of normal adverse moves
            RiskToleranceLevel.MODERATE: 85,      # Medium stop that covers 85% of normal adverse moves
            RiskToleranceLevel.AGGRESSIVE: 75     # Tighter stop that only covers 75% of normal adverse moves
        }
        
        percentile = percentile_map.get(risk_tolerance, 75)
        stop_level = self.mae_stats.get(f'mae_percentile_{percentile}', 
                                        self.mae_stats.get('median_mae', 5.0))
        
        # Add safety margin
        if risk_tolerance == RiskToleranceLevel.CONSERVATIVE:
            margin = 1.5
        elif risk_tolerance == RiskToleranceLevel.MODERATE:
            margin = 1.2
        else:
            margin = 1.0
            
        final_stop = stop_level * margin
        
        return {
            'stop_loss_pct': final_stop,
            'percentile_used': percentile,
            'risk_tolerance': risk_tolerance.name,
            'expected_win_rate': percentile / 100,  # Approximate expected win rate
            'mae_stats_used': {k: v for k, v in self.mae_stats.items() if 'percentile' in k}
        }
    
    def optimize_take_profit(self, profit_target_style: Union[str, RiskToleranceLevel] = 'moderate') -> Dict[str, Any]:
        """
        Derive optimal take-profit parameters based on MFE analysis.
        
        Args:
            profit_target_style: 'conservative', 'moderate', or 'aggressive'
                                 (or corresponding RiskToleranceLevel enum)
            
        Returns:
            Dictionary of optimized take-profit parameters
        """
        # Convert string to enum if needed
        if isinstance(profit_target_style, str):
            profit_target_style = getattr(RiskToleranceLevel, profit_target_style.upper())
        
        # Map profit target style to MFE percentiles
        percentile_map = {
            RiskToleranceLevel.CONSERVATIVE: 50,  # Lower target that's easier to hit (median MFE)
            RiskToleranceLevel.MODERATE: 65,      # Medium target
            RiskToleranceLevel.AGGRESSIVE: 75     # Higher target that maximizes gains but harder to reach
        }
        
        percentile = percentile_map.get(profit_target_style, 50)
        take_profit_level = self.mfe_stats.get(f'mfe_percentile_{percentile}', 
                                              self.mfe_stats.get('median_mfe', 5.0))
        
        return {
            'take_profit_pct': take_profit_level,
            'percentile_used': percentile,
            'target_style': profit_target_style.name,
            'probability': 1 - (percentile / 100),  # Approximate probability of reaching target
            'mfe_stats_used': {k: v for k, v in self.mfe_stats.items() if 'percentile' in k}
        }
    
    def optimize_trailing_stop(self) -> Dict[str, float]:
        """
        Derive optimal trailing stop parameters based on MAE and MFE analysis.
        
        Returns:
            Dictionary of optimized trailing stop parameters
        """
        # Calculate optimal activation threshold based on average MFE
        activation_threshold = self.mfe_stats.get('mean_mfe', 5.0) * 0.5  # Activate at 50% of average MFE
        
        # Calculate optimal trailing distance based on average intra-trade volatility
        trail_distance = self.mae_stats.get('mean_mae', 2.0) * 0.8  # 80% of average MAE
        
        # Ensure trailing stop isn't too tight
        min_trail = trail_distance * 0.8
        trail_distance = max(trail_distance, min_trail)
        
        return {
            'activation_threshold_pct': activation_threshold,
            'trail_distance_pct': trail_distance,
            'mfe_used': self.mfe_stats.get('mean_mfe'),
            'mae_used': self.mae_stats.get('mean_mae')
        }
    
    def optimize_time_exit(self) -> Dict[str, Any]:
        """
        Derive optimal time-based exit parameters based on ETD analysis.
        
        Returns:
            Dictionary of optimized time-based exit parameters
        """
        # Calculate optimal time exit based on average duration of winning trades
        avg_winner_duration = self.etd_stats.get('duration_winners_mean')
        median_duration = self.etd_stats.get('median_duration')
        
        if avg_winner_duration and not np.isnan(avg_winner_duration):
            # If winning trades tend to be shorter, use that as a guide
            if avg_winner_duration < median_duration:
                # Set exit at 1.5x the average winning trade duration
                time_exit_threshold = avg_winner_duration * 1.5
            else:
                # Otherwise set at 2x the average winning trade duration
                time_exit_threshold = avg_winner_duration * 2
        else:
            # Fallback to using median duration of all trades
            time_exit_threshold = median_duration * 2 if median_duration else 10  # default fallback
        
        # Get 90th percentile as a sanity check
        percentile_90 = self.etd_stats.get('duration_percentile_90', time_exit_threshold * 1.2)
        
        # If our calculated threshold exceeds the 90th percentile, cap it
        time_exit_threshold = min(time_exit_threshold, percentile_90)
        
        return {
            'max_duration': time_exit_threshold,
            'units': 'same as input duration',  # e.g., bars, minutes, hours, days
            'duration_winners_mean': avg_winner_duration,
            'duration_percentile_90': percentile_90,
            'win_rate_by_quartile': self.etd_stats.get('win_rate_by_quartile', {})
        }
    
    def calculate_risk_reward_setups(self) -> List[Dict[str, Any]]:
        """
        Calculate various risk-reward setups based on the optimized parameters.
        
        Returns:
            List of different risk management setups with expected metrics
        """
        setups = []
        
        # Generate different combinations of stop-loss and take-profit
        for risk in [RiskToleranceLevel.CONSERVATIVE, RiskToleranceLevel.MODERATE, RiskToleranceLevel.AGGRESSIVE]:
            for target in [RiskToleranceLevel.CONSERVATIVE, RiskToleranceLevel.MODERATE, RiskToleranceLevel.AGGRESSIVE]:
                stop_dict = self.optimize_stop_loss(risk)
                take_profit_dict = self.optimize_take_profit(target)
                
                stop_loss_pct = stop_dict['stop_loss_pct']
                take_profit_pct = take_profit_dict['take_profit_pct']
                
                risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else float('inf')
                expected_win_rate = stop_dict['expected_win_rate']
                
                # Calculate expectancy using simplified formula: (win_rate * RR) - (1 - win_rate)
                expectancy = (expected_win_rate * risk_reward_ratio) - (1 - expected_win_rate)
                
                # Calculate Kelly criterion for position sizing (simplified)
                kelly = expected_win_rate - ((1 - expected_win_rate) / risk_reward_ratio)
                kelly = max(0, min(kelly, 0.5))  # Cap between 0% and 50%
                
                setups.append({
                    'name': f"{risk.name.lower()}_risk_{target.name.lower()}_target",
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'risk_reward_ratio': risk_reward_ratio,
                    'expected_win_rate': expected_win_rate,
                    'expectancy': expectancy,
                    'kelly_criterion': kelly,
                    'risk_tolerance': risk.name,
                    'profit_target_style': target.name
                })
        
        # Sort by expectancy (highest first)
        setups.sort(key=lambda x: x['expectancy'], reverse=True)
        
        return setups
    
    def get_optimal_parameters(self, risk_tolerance: Union[str, RiskToleranceLevel] = 'moderate',
                              include_trailing_stop: bool = True,
                              include_time_exit: bool = True) -> RiskParameters:
        """
        Get a complete set of optimal risk parameters.
        
        Args:
            risk_tolerance: Overall risk tolerance level
            include_trailing_stop: Whether to include trailing stop parameters
            include_time_exit: Whether to include time-based exit parameters
            
        Returns:
            RiskParameters object with optimized values
        """
        # Convert string to enum if needed
        if isinstance(risk_tolerance, str):
            risk_tolerance_enum = getattr(RiskToleranceLevel, risk_tolerance.upper())
        else:
            risk_tolerance_enum = risk_tolerance
        
        # Get stop-loss parameters
        stop_loss_dict = self.optimize_stop_loss(risk_tolerance_enum)
        stop_loss_pct = stop_loss_dict['stop_loss_pct']
        
        # Get take-profit parameters
        take_profit_dict = self.optimize_take_profit(risk_tolerance_enum)
        take_profit_pct = take_profit_dict['take_profit_pct']
        
        # Calculate risk-reward ratio
        risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else float('inf')
        
        # Initialize parameters
        params = RiskParameters(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            risk_reward_ratio=risk_reward_ratio,
            expected_win_rate=stop_loss_dict['expected_win_rate'],
            risk_tolerance=risk_tolerance_enum
        )
        
        # Add trailing stop if requested
        if include_trailing_stop:
            trailing_stop_dict = self.optimize_trailing_stop()
            params.trailing_stop_activation_pct = trailing_stop_dict['activation_threshold_pct']
            params.trailing_stop_distance_pct = trailing_stop_dict['trail_distance_pct']
        
        # Add time exit if requested
        if include_time_exit:
            time_exit_dict = self.optimize_time_exit()
            params.max_duration = time_exit_dict['max_duration']
        
        return params
    
    def get_balanced_parameters(self) -> RiskParameters:
        """
        Get a balanced set of risk parameters that maximizes expectancy.
        
        This method evaluates different combinations and returns the one with
        the highest expected value.
        
        Returns:
            RiskParameters object with balanced optimal values
        """
        # Calculate various setups
        setups = self.calculate_risk_reward_setups()
        
        # Get the setup with the highest expectancy
        if setups:
            best_setup = setups[0]  # Already sorted by expectancy
            
            # Get risk tolerance and profit target style from the best setup
            risk_level = getattr(RiskToleranceLevel, best_setup['risk_tolerance'])
            
            # Create parameters with the best setup values
            params = RiskParameters(
                stop_loss_pct=best_setup['stop_loss_pct'],
                take_profit_pct=best_setup['take_profit_pct'],
                risk_reward_ratio=best_setup['risk_reward_ratio'],
                expected_win_rate=best_setup['expected_win_rate'],
                risk_tolerance=risk_level
            )
            
            # Add trailing stop
            trailing_stop_dict = self.optimize_trailing_stop()
            params.trailing_stop_activation_pct = trailing_stop_dict['activation_threshold_pct']
            params.trailing_stop_distance_pct = trailing_stop_dict['trail_distance_pct']
            
            # Add time exit
            time_exit_dict = self.optimize_time_exit()
            params.max_duration = time_exit_dict['max_duration']
            
            return params
        else:
            # Return default parameters if no setups could be calculated
            return self.get_optimal_parameters()

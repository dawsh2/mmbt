"""
Strategy Factory Module

This module provides the StrategyFactory class that simplifies the creation
of strategy instances.
"""

from typing import Dict, List, Any, Optional, Union, Type
from .strategy_base import Strategy
from .strategy_registry import StrategyRegistry
from .weighted_strategy import WeightedStrategy
from .ensemble_strategy import EnsembleStrategy
from .regime_strategy import RegimeStrategy
from .topn_strategy import TopNStrategy

class StrategyFactory:
    """Factory for creating strategy instances.
    
    This class provides methods for creating various types of strategies
    from configuration dictionaries or parameters.
    """
    
    @staticmethod
    def create_strategy(strategy_type: Union[str, Type[Strategy]], 
                        params: Optional[Dict[str, Any]] = None) -> Strategy:
        """Create a strategy instance.
        
        Args:
            strategy_type: Name or class of the strategy
            params: Parameters for strategy initialization
            
        Returns:
            Strategy: Initialized strategy instance
        """
        params = params or {}
        
        if isinstance(strategy_type, str):
            # Get strategy class from registry
            strategy_class = StrategyRegistry.get_strategy_class(strategy_type)
            return strategy_class(**params)
        else:
            # Assume strategy_type is a class
            return strategy_type(**params)
    
    @staticmethod
    def create_weighted_strategy(rules: List[Any], 
                               weights: Optional[List[float]] = None, 
                               buy_threshold: float = 0.5, 
                               sell_threshold: float = -0.5, 
                               name: Optional[str] = None) -> WeightedStrategy:
        """Create a weighted strategy.
        
        Args:
            rules: List of rule objects
            weights: Optional weights for each rule
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            name: Strategy name
            
        Returns:
            WeightedStrategy: Initialized weighted strategy
        """
        return WeightedStrategy(
            rules=rules,
            weights=weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            name=name
        )
    
    @staticmethod
    def create_ensemble_strategy(strategies: Dict[str, Strategy], 
                               combination_method: str = 'voting', 
                               weights: Optional[Dict[str, float]] = None, 
                               name: Optional[str] = None) -> EnsembleStrategy:
        """Create an ensemble strategy.
        
        Args:
            strategies: Dictionary of strategies
            combination_method: Method for combining signals
            weights: Optional weights for each strategy
            name: Strategy name
            
        Returns:
            EnsembleStrategy: Initialized ensemble strategy
        """
        return EnsembleStrategy(
            strategies=strategies,
            combination_method=combination_method,
            weights=weights,
            name=name
        )
    
    @staticmethod
    def create_regime_strategy(regime_detector: Any, 
                             regime_strategies: Dict[Any, Strategy], 
                             default_strategy: Optional[Strategy] = None, 
                             name: Optional[str] = None) -> RegimeStrategy:
        """Create a regime strategy.
        
        Args:
            regime_detector: Regime detection object
            regime_strategies: Dictionary mapping regimes to strategies
            default_strategy: Default strategy when no regime-specific one exists
            name: Strategy name
            
        Returns:
            RegimeStrategy: Initialized regime strategy
        """
        return RegimeStrategy(
            regime_detector=regime_detector,
            regime_strategies=regime_strategies,
            default_strategy=default_strategy,
            name=name
        )
    
    @staticmethod
    def create_topn_strategy(rule_objects: List[Any],
                           name: Optional[str] = None) -> TopNStrategy:
        """Create a TopN strategy.
        
        Args:
            rule_objects: List of rule objects
            name: Strategy name
            
        Returns:
            TopNStrategy: Initialized TopN strategy
        """
        return TopNStrategy(
            rule_objects=rule_objects,
            name=name
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Strategy:
        """Create a strategy from a configuration dictionary.
        
        Args:
            config: Strategy configuration
            
        Returns:
            Strategy: Initialized strategy
            
        Example config:
        {
            'type': 'WeightedStrategy',
            'params': {
                'rules': [rule1, rule2, rule3],
                'weights': [0.5, 0.3, 0.2],
                'buy_threshold': 0.4,
                'sell_threshold': -0.4
            }
        }
        """
        strategy_type = config.get('type')
        params = config.get('params', {})
        
        if not strategy_type:
            raise ValueError("Strategy configuration must include 'type'")
        
        # Handle nested strategies for Ensemble and Regime strategies
        if strategy_type == 'EnsembleStrategy' and 'strategies' in params:
            # Convert nested strategy configs to strategy objects
            strategies = {}
            for name, strategy_config in params['strategies'].items():
                strategies[name] = StrategyFactory.create_from_config(strategy_config)
            params['strategies'] = strategies
            
        elif strategy_type == 'RegimeStrategy' and 'regime_strategies' in params:
            # Convert nested strategy configs to strategy objects
            regime_strategies = {}
            for regime, strategy_config in params['regime_strategies'].items():
                regime_strategies[regime] = StrategyFactory.create_from_config(strategy_config)
            params['regime_strategies'] = regime_strategies
            
            # Handle default strategy if present
            if 'default_strategy' in params and isinstance(params['default_strategy'], dict):
                params['default_strategy'] = StrategyFactory.create_from_config(
                    params['default_strategy']
                )
        
        return StrategyFactory.create_strategy(strategy_type, params)

"""
Regime manager for adapting trading strategies based on market regimes.
"""

from .regime_type import RegimeType

class RegimeManager:
    """
    Manages trading strategies based on detected market regimes.
    
    This class uses a regime detector to identify the current market regime
    and selects the appropriate strategy accordingly.
    """
    
    def __init__(self, regime_detector, strategy_factory, rule_objects=None, data_handler=None):
        """
        Initialize the regime manager.
        
        Args:
            regime_detector: RegimeDetector object for identifying regimes
            strategy_factory: Factory for creating strategies
            rule_objects: List of trading rule objects (passed to the factory)
            data_handler: Optional data handler for optimization
        """
        self.regime_detector = regime_detector
        self.strategy_factory = strategy_factory
        self.rule_objects = rule_objects or []
        self.data_handler = data_handler
        self.current_regime = RegimeType.UNKNOWN
        self.regime_strategies = {}  # Mapping from regime to strategy
        
        # Create default strategy
        if hasattr(self.strategy_factory, 'create_default_strategy') and self.rule_objects:
            self.default_strategy = self.strategy_factory.create_default_strategy(self.rule_objects)
        else:
            self.default_strategy = None
    
    def optimize_regime_strategies(self, regimes_to_optimize=None, optimization_metric='sharpe', verbose=True):
        """
        Optimize strategies for different market regimes.
        
        Args:
            regimes_to_optimize: List of regimes to optimize for (or None for all)
            optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
            verbose: Whether to print optimization progress
            
        Returns:
            dict: Mapping from regime to optimized parameters
        """
        if self.data_handler is None:
            raise ValueError("Data handler must be provided for optimization")
        
        if regimes_to_optimize is None:
            # Use all regimes except UNKNOWN
            regimes_to_optimize = list(RegimeType)
            regimes_to_optimize.remove(RegimeType.UNKNOWN)
        
        # First, identify bars in each regime using the full dataset
        regime_bars = self._identify_regime_bars()
        
        # Initialize the dictionary to store optimal parameters
        optimal_params = {}
        
        for regime in regimes_to_optimize:
            if regime in regime_bars and len(regime_bars[regime]) >= 30:
                if verbose:
                    print(f"\nOptimizing strategy for {regime.name} regime "
                          f"({len(regime_bars[regime])} bars) using {optimization_metric} metric")
                
                # Create a regime-specific data handler
                regime_specific_data = self._create_regime_specific_data(regime_bars[regime])
                
                # Optimize parameters for this regime
                try:
                    # Import here to avoid circular imports
                    from genetic_optimizer import GeneticOptimizer
                    
                    optimizer = GeneticOptimizer(
                        data_handler=regime_specific_data,
                        rule_objects=self.rule_objects,
                        population_size=15,
                        num_generations=30,
                        optimization_metric=optimization_metric
                    )
                    optimal_params[regime] = optimizer.optimize(verbose=verbose)
                    
                    # Create and store the optimized strategy
                    self.regime_strategies[regime] = self.strategy_factory.create_strategy(
                        regime, self.rule_objects, optimal_params[regime]
                    )
                    
                    if verbose:
                        print(f"Optimized parameters for {regime.name}: {optimal_params[regime]}")
                except ImportError:
                    if verbose:
                        print("GeneticOptimizer not available, skipping optimization.")
            else:
                if verbose and regime in regime_bars:
                    print(f"Insufficient data for {regime.name} regime "
                          f"({len(regime_bars.get(regime, []))} bars). Using default strategy.")
                elif verbose:
                    print(f"No bars found for {regime.name} regime. Using default strategy.")
                
                # Ensure a strategy exists for this regime (use default)
                self.regime_strategies.setdefault(regime, self.default_strategy)
        
        return optimal_params
    
    def _identify_regime_bars(self):
        """
        Identify which bars belong to each regime.
        
        Returns:
            dict: Mapping from regime to list of (index, bar) tuples
        """
        regime_bars = {}
        self.regime_detector.reset()
        self.data_handler.reset_train()
        
        bar_index = 0
        while True:
            bar = self.data_handler.get_next_train_bar()
            if bar is None:
                break
                
            regime = self.regime_detector.detect_regime(bar)
            
            if regime not in regime_bars:
                regime_bars[regime] = []
                
            regime_bars[regime].append((bar_index, bar))
            bar_index += 1
            
        return regime_bars
    
    def _create_regime_specific_data(self, regime_bars):
        """
        Create a mock data handler with only bars from a specific regime.
        
        Args:
            regime_bars: List of (index, bar) tuples for the regime
            
        Returns:
            object: A data handler-like object for the specific regime
        """
        class RegimeSpecificDataHandler:
            def __init__(self, bars):
                self.bars = [bar for _, bar in bars]
                self.index = 0
            
            def get_next_train_bar(self):
                if self.index < len(self.bars):
                    bar = self.bars[self.index]
                    self.index += 1
                    return bar
                return None
            
            def get_next_test_bar(self):
                return self.get_next_train_bar()
            
            def reset_train(self):
                self.index = 0
            
            def reset_test(self):
                self.index = 0
                
        return RegimeSpecificDataHandler(regime_bars)
    
    def get_strategy_for_regime(self, regime):
        """
        Get the optimized strategy for a specific regime.
        
        Args:
            regime: RegimeType to get strategy for
            
        Returns:
            object: Strategy instance for the regime
        """
        return self.regime_strategies.get(regime, self.default_strategy)
    
    def on_bar(self, event):
        """
        Process a bar and generate trading signals using the appropriate strategy.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            object: Signal information from the strategy
        """
        bar = event.bar
        self.current_regime = self.regime_detector.detect_regime(bar)
        strategy = self.get_strategy_for_regime(self.current_regime)
        
        if strategy and hasattr(strategy, 'on_bar'):
            return strategy.on_bar(event)
            
        return None
    
    def reset(self):
        """Reset the regime manager and its components."""
        self.regime_detector.reset()
        
        if self.default_strategy and hasattr(self.default_strategy, 'reset'):
            self.default_strategy.reset()
            
        for strategy in self.regime_strategies.values():
            if strategy and hasattr(strategy, 'reset'):
                strategy.reset()
                
        self.current_regime = RegimeType.UNKNOWN

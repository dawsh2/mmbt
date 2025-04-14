"""
Enhanced OptimizerManager with integrated grid search capabilities.
"""
import numpy as np
import time
import gc
from enum import Enum, auto
from src.optimization import OptimizationMethod  # Import from the package
from src.optimization.evaluators import RuleEvaluator, RegimeDetectorEvaluator
from src.optimization.grid_search import GridOptimizer
from src.optimization.components import RuleFactory, RegimeDetectorFactory
from src.optimization.genetic_optimizer import GeneticOptimizer
from src.strategies.weighted_strategy import WeightedStrategy as WeightedRuleStrategy
from src.regime_detection import RegimeManager

class OptimizationMethod(Enum):
    """Enumeration of supported optimization methods."""
    GENETIC = auto()
    GRID_SEARCH = auto()  # Added grid search
    BAYESIAN = auto()
    RANDOM_SEARCH = auto()
    JOINT = auto()

class OptimizerManager:
    """
    Enhanced manager for coordinating different optimization approaches.
    """
    
    def __init__(self, data_handler, rule_objects=None):
        """
        Initialize the optimizer manager.
        
        Args:
            data_handler: The data handler for accessing market data
            rule_objects: Optional list of pre-initialized rule objects
        """
        self.data_handler = data_handler
        self.rule_objects = rule_objects or []
        
        # Component registry
        self.components = {
            'rule': {},
            'regime_detector': {}
        }
        
        # Optimization results
        self.optimized_components = {}
        
        # Register any provided rule objects
        if self.rule_objects:
            for i, rule in enumerate(self.rule_objects):
                rule_class = rule.__class__
                self.register_component(
                    name=f"rule_{i}",
                    component_type='rule', 
                    component_class=rule_class,
                    instance=rule
                )
    
    def register_component(self, name, component_type, component_class, params_range=None, instance=None):
        """
        Register a component for optimization.
        
        Args:
            name: Unique identifier for the component
            component_type: Type of component ('rule', 'regime_detector', etc.)
            component_class: Class of the component
            params_range: Optional parameter ranges for optimization
            instance: Optional pre-initialized instance
        """
        if component_type not in self.components:
            self.components[component_type] = {}
            
        self.components[component_type][name] = {
            'class': component_class,
            'params': params_range,
            'instance': instance
        }
    
    def register_rule(self, name, rule_class, params_range=None, instance=None):
        """Register a trading rule."""
        self.register_component(name, 'rule', rule_class, params_range, instance)
    
    def register_regime_detector(self, name, detector_class, params_range=None, instance=None):
        """Register a regime detector."""
        self.register_component(name, 'regime_detector', detector_class, params_range, instance)

    def optimize(self, component_type, method=OptimizationMethod.GRID_SEARCH, 
                 components=None, metrics='return', verbose=True, **kwargs):
        """
        Optimize components of a specific type.

        Args:
            component_type: Type of component to optimize ('rule', 'regime_detector', etc.)
            method: Optimization method to use
            components: List of component names to optimize (or None for all registered)
            metrics: Performance metric(s) to optimize for
            verbose: Whether to print progress information
            **kwargs: Additional parameters for specific optimization methods
                - top_n: Number of top components to select
                - genetic: Dictionary of genetic algorithm parameters
                    - population_size: Size of population
                    - num_generations: Number of generations to run
                    - mutation_rate: Rate of mutation
                    - num_parents: Number of parents to select
                    - cv_folds: Number of cross-validation folds
                    - regularization_factor: Strength of regularization
                    - optimize_thresholds: Whether to optimize thresholds
                - sequence: Optional optimization sequence to use
                - regime_detector: Optional regime detector for regime-based optimization

        Returns:
            dict or object: Optimized components or strategy depending on method and component type
        """
        # Handle specified optimization sequence if provided
        if 'sequence' in kwargs:
            sequence = kwargs.pop('sequence')
            regime_detector = kwargs.pop('regime_detector', None)

            if sequence == OptimizationSequence.RULES_FIRST:
                return self._optimize_rules_first(method, metrics, regime_detector, kwargs, verbose)
            elif sequence == OptimizationSequence.REGIMES_FIRST:
                return self._optimize_regimes_first(method, metrics, regime_detector, kwargs, verbose)
            elif sequence == OptimizationSequence.ITERATIVE:
                return self._optimize_iterative(method, metrics, regime_detector, kwargs, verbose)
            elif sequence == OptimizationSequence.JOINT:
                return self._optimize_joint(method, metrics, regime_detector, kwargs, verbose)

        # Validate component type
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")

        # Get components to optimize
        if components is None:
            components_to_optimize = list(self.components[component_type].keys())
        else:
            components_to_optimize = components

        # Collect configs for optimization
        configs = []
        for name in components_to_optimize:
            if name not in self.components[component_type]:
                raise ValueError(f"Unknown component: {name}")

            component_data = self.components[component_type][name]

            if component_data['params'] is None:
                if verbose:
                    print(f"Skipping {name}: No parameter ranges defined")
                continue

            configs.append((component_data['class'], component_data['params']))

        # Choose factory and evaluator based on component type
        if component_type == 'rule':
            factory = RuleFactory()
            evaluator = RuleEvaluator.evaluate
        elif component_type == 'regime_detector':
            factory = RegimeDetectorFactory()
            evaluator = RegimeDetectorEvaluator.evaluate
        elif component_type == 'strategy':
            factory = StrategyFactory()
            evaluator = StrategyEvaluator.evaluate
        else:
            raise ValueError(f"Unsupported component type for optimization: {component_type}")

        # Run optimization based on selected method
        if method == OptimizationMethod.GRID_SEARCH:
            # Grid search optimization
            if verbose:
                print(f"Running grid search for {len(configs)} {component_type} components")

            top_n = kwargs.get('top_n', 5)
            optimizer = GridOptimizer(factory, evaluator, top_n=top_n)
            optimized = optimizer.optimize(configs, self.data_handler, metrics, verbose)

        elif method == OptimizationMethod.GENETIC:
            # Genetic algorithm optimization
            if verbose:
                print(f"Running genetic optimization for {len(configs)} {component_type} components")

            # Extract genetic parameters
            genetic_params = kwargs.get('genetic', {})
            population_size = genetic_params.get('population_size', 20)
            num_generations = genetic_params.get('num_generations', 50)
            mutation_rate = genetic_params.get('mutation_rate', 0.1)
            num_parents = genetic_params.get('num_parents', 8)
            cv_folds = genetic_params.get('cv_folds', 3)
            regularization_factor = genetic_params.get('regularization_factor', 0.2)
            optimize_thresholds = genetic_params.get('optimize_thresholds', True)

            # For rules, extract rule instances if they're registered components
            if component_type == 'rule':
                if not self.rule_objects and components_to_optimize:
                    self.rule_objects = [
                        self.components[component_type][name]['instance'] 
                        for name in components_to_optimize
                        if self.components[component_type][name]['instance'] is not None
                    ]

                # If we have rule objects, use the traditional genetic optimizer
                if self.rule_objects:
                    if verbose:
                        print(f"Using traditional genetic optimizer with {len(self.rule_objects)} rule objects")

                    # Import from original location for backward compatibility
                    from genetic_optimizer import GeneticOptimizer as TraditionalGeneticOptimizer

                    optimizer = TraditionalGeneticOptimizer(
                        data_handler=self.data_handler,
                        rule_objects=self.rule_objects,
                        population_size=population_size,
                        num_parents=num_parents,
                        num_generations=num_generations,
                        mutation_rate=mutation_rate,
                        optimization_metric=metrics,
                        cv_folds=cv_folds,
                        regularization_factor=regularization_factor,
                        optimize_thresholds=optimize_thresholds
                    )

                    # Run optimization
                    best_weights = optimizer.optimize(verbose=verbose)

                    # If best_weights is a dict with thresholds, extract the weights and thresholds
                    if isinstance(best_weights, dict) and 'weights' in best_weights:
                        weights = best_weights['weights']
                        buy_threshold = best_weights.get('buy_threshold', 0.5)
                        sell_threshold = best_weights.get('sell_threshold', -0.5)
                    else:
                        weights = best_weights
                        buy_threshold = 0.5
                        sell_threshold = -0.5

                    # Create and return strategy with optimized weights
                    from optimization.strategies import WeightedComponentStrategy
                    strategy = WeightedComponentStrategy(
                        components=self.rule_objects,
                        weights=weights,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold
                    )

                    # Store and return the strategy rather than components
                    self.optimized_strategy = strategy
                    return strategy

            # For other component types or if no rule objects, use the new genetic optimizer
            from optimization.genetic_search import GeneticOptimizer

            optimizer = GeneticOptimizer(
                component_factory=factory,
                evaluation_method=evaluator,
                top_n=kwargs.get('top_n', 5),
                population_size=population_size,
                num_generations=num_generations,
                mutation_rate=mutation_rate,
                num_parents=num_parents,
                cv_folds=cv_folds,
                regularization_factor=regularization_factor,
                optimize_thresholds=optimize_thresholds
            )

            optimized = optimizer.optimize(configs, self.data_handler, metrics, verbose)

            # If this is rule optimization, create a weighted strategy
            if component_type == 'rule' and optimized:
                from optimization.strategies import WeightedComponentStrategy

                # Get component objects and optimal weights
                optimized_components = list(optimized.values())
                weights = optimizer.best_weights if hasattr(optimizer, 'best_weights') else np.ones(len(optimized_components)) / len(optimized_components)

                # Get thresholds if available
                buy_threshold = optimizer.best_thresholds[0] if hasattr(optimizer, 'best_thresholds') and optimizer.best_thresholds is not None else 0.5
                sell_threshold = optimizer.best_thresholds[1] if hasattr(optimizer, 'best_thresholds') and optimizer.best_thresholds is not None else -0.5

                # Create the weighted strategy
                strategy = WeightedComponentStrategy(
                    components=optimized_components,
                    weights=weights,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold
                )

                # Store and return the strategy
                self.optimized_strategy = strategy
                return strategy

        elif method == OptimizationMethod.BAYESIAN:
            # Bayesian optimization (placeholder)
            if verbose:
                print("Bayesian optimization not yet implemented - using grid search instead")

            top_n = kwargs.get('top_n', 5)
            optimizer = GridOptimizer(factory, evaluator, top_n=top_n)
            optimized = optimizer.optimize(configs, self.data_handler, metrics, verbose)

        elif method == OptimizationMethod.RANDOM_SEARCH:
            # Random search optimization (placeholder)
            if verbose:
                print("Random search not yet implemented - using grid search instead")

            top_n = kwargs.get('top_n', 5)
            optimizer = GridOptimizer(factory, evaluator, top_n=top_n)
            optimized = optimizer.optimize(configs, self.data_handler, metrics, verbose)

        else:
            raise ValueError(f"Optimization method {method} not implemented")

        # Store optimized components
        self.optimized_components[component_type] = optimized

        return optimized

    def get_optimized_components(self, component_type):
        """Get optimized components of a specific type."""
        return self.optimized_components.get(component_type, {})
    
    def optimize_regime_specific_rules(self, regime_detector, optimization_method=OptimizationMethod.GRID_SEARCH, 
                                       optimization_metric='return', verbose=True):
        """
        Optimize rules specifically for different market regimes.
        
        Args:
            regime_detector: The regime detector to use
            optimization_method: Method to use for optimization
            optimization_metric: Metric to optimize for
            verbose: Whether to print progress
            
        Returns:
            dict: Mapping from regime to optimized rules
        """
        # First, identify bars in each regime
        regime_bars = self._identify_regime_bars(regime_detector)
        
        # Results storage
        regime_specific_rules = {}
        
        # Optimize for each regime
        for regime, bars in regime_bars.items():
            if len(bars) < 30:  # Need enough data
                if verbose:
                    print(f"Insufficient data for {regime.name}, skipping")
                continue
                
            if verbose:
                print(f"\nOptimizing rules for {regime.name} regime ({len(bars)} bars)")
                
            # Create regime-specific data handler
            regime_data = self._create_regime_specific_data(bars)
            
            # Optimize rules for this regime
            if optimization_method == OptimizationMethod.GRID_SEARCH:
                # Get rule configs
                rule_configs = []
                for name, data in self.components['rule'].items():
                    if data['params']:
                        rule_configs.append((data['class'], data['params']))
                
                # Run grid search
                factory = RuleFactory()
                optimizer = GridOptimizer(factory, RuleEvaluator.evaluate, top_n=5)
                optimized_rules = optimizer.optimize(
                    rule_configs, regime_data, optimization_metric, verbose)
                
                regime_specific_rules[regime] = optimized_rules
                
            elif optimization_method == OptimizationMethod.GENETIC:
                # Use genetic optimizer for rules in this regime
                genetic_optimizer = GeneticOptimizer(
                    data_handler=regime_data,
                    rule_objects=self.rule_objects,
                    optimization_metric=optimization_metric
                )
                
                weights = genetic_optimizer.optimize(verbose=verbose)
                
                # Create strategy with optimized weights
                strategy = WeightedRuleStrategy(
                    rule_objects=self.rule_objects,
                    weights=weights
                )
                
                regime_specific_rules[regime] = strategy
        
        return regime_specific_rules

    def _identify_regime_bars(self, regime_detector):
        """Identify which bars belong to each regime."""
        regime_bars = {}
        regime_detector.reset()
        self.data_handler.reset_train()
        
        bar_index = 0
        while True:
            bar = self.data_handler.get_next_train_bar()
            if bar is None:
                break
                
            regime = regime_detector.detect_regime(bar)
            
            if regime not in regime_bars:
                regime_bars[regime] = []
                
            regime_bars[regime].append((bar_index, bar))
            bar_index += 1
            
        return regime_bars

    def _create_regime_specific_data(self, regime_bars):
        """Create a data handler with only bars from a specific regime."""
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


    def _optimize_rules_first(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Optimize rule weights first, then optimize for regimes.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if verbose:
            print("Step 1: Optimizing rule weights...")

        # Optimize rule weights using the unified method
        optimized_rules = self.optimize(
            component_type='rule',
            method=method,
            metrics=metrics,
            verbose=verbose,
            **optimization_params
        )

        # Create optimized rule strategy
        if method == OptimizationMethod.GENETIC:
            # For genetic, we already have a strategy
            optimized_strategy = self.optimized_strategy
        else:
            # For other methods, create a strategy with the optimized rules
            optimized_rule_objects = list(optimized_rules.values())
            weights = np.ones(len(optimized_rule_objects)) / len(optimized_rule_objects)

            # Use the new WeightedComponentStrategy if available, fallback to original if not
            try:
                from optimization.strategies import WeightedComponentStrategy
                optimized_strategy = WeightedComponentStrategy(
                    components=optimized_rule_objects,
                    weights=weights
                )
            except ImportError:
                # Fallback to original WeightedRuleStrategy for backward compatibility
                from genetic_optimizer import WeightedRuleStrategy
                optimized_strategy = WeightedRuleStrategy(
                    rule_objects=optimized_rule_objects,
                    weights=weights
                )

        # Optimize for regimes if detector provided
        if regime_detector:
            if verbose:
                print("\nStep 2: Optimizing regime-specific strategies...")

            # Create strategy factory
            try:
                from optimization.strategies import WeightedComponentStrategy
                from strategy import WeightedRuleStrategyFactory
                strategy_factory = WeightedRuleStrategyFactory()
            except ImportError:
                # Fallback for backward compatibility
                from strategy import WeightedRuleStrategyFactory
                strategy_factory = WeightedRuleStrategyFactory()

            # Create regime manager with optimized rules
            rule_objects = self.rule_objects if method == OptimizationMethod.GENETIC else optimized_rule_objects

            self.regime_manager = RegimeManager(
                regime_detector=regime_detector,
                strategy_factory=strategy_factory,
                rule_objects=rule_objects,
                data_handler=self.data_handler
            )

            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)

            # Set default strategy to the initially optimized strategy
            self.regime_manager.default_strategy = optimized_strategy

            return self.regime_manager
        else:
            return optimized_strategy

 

    def _optimize_regimes_first(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Optimize for regimes first, then optimize rule weights for each regime.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if not regime_detector:
            raise ValueError("Regime detector must be provided for regimes-first optimization")

        if verbose:
            print("Step 1: Identifying market regimes...")

        # Create strategy factory
        from strategy import WeightedRuleStrategyFactory
        strategy_factory = WeightedRuleStrategyFactory()

        # Create regime manager
        self.regime_manager = RegimeManager(
            regime_detector=regime_detector,
            strategy_factory=strategy_factory,
            rule_objects=self.rule_objects,
            data_handler=self.data_handler
        )

        # Optimize each regime using unified optimization
        optimized_regime_rules = self.optimize_regime_specific_rules(
            regime_detector=regime_detector,
            optimization_method=method,
            optimization_metric=metrics,
            verbose=verbose
        )

        # Update regime manager with optimized strategies
        for regime, rule_objects in optimized_regime_rules.items():
            if isinstance(rule_objects, dict):
                # Convert rule dict to list for strategy
                rule_list = list(rule_objects.values())
                weights = np.ones(len(rule_list)) / len(rule_list)
                strategy = WeightedRuleStrategy(
                    rule_objects=rule_list,
                    weights=weights
                )
            else:
                # Already a strategy
                strategy = rule_objects

            self.regime_manager.regime_strategies[regime] = strategy

        # Create default strategy with equal weights
        default_strategy = WeightedRuleStrategy(
            rule_objects=self.rule_objects,
            weights=np.ones(len(self.rule_objects)) / len(self.rule_objects)
        )
        self.regime_manager.default_strategy = default_strategy

        return self.regime_manager

    def _optimize_iterative(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Iteratively optimize rules and regimes in multiple passes.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if not regime_detector:
            raise ValueError("Regime detector must be provided for iterative optimization")

        iterations = optimization_params.get('iterations', 3)

        # Initial optimization of rules
        optimized_rules = None

        for i in range(iterations):
            if verbose:
                print(f"\nIteration {i+1}/{iterations}")
                print("Step 1: Optimizing rule weights...")

            # Optimize rule weights
            optimized_rules = self.optimize(
                component_type='rule',
                method=method,
                metrics=metrics,
                verbose=verbose,
                **optimization_params
            )

            # Create weighted strategy
            if method == OptimizationMethod.GENETIC:
                # For genetic, we already have a strategy
                optimized_strategy = self.optimized_strategy
            else:
                # For other methods, create a strategy with the optimized rules
                optimized_rule_objects = list(optimized_rules.values())
                weights = np.ones(len(optimized_rule_objects)) / len(optimized_rule_objects)
                optimized_strategy = WeightedRuleStrategy(
                    rule_objects=optimized_rule_objects,
                    weights=weights
                )

            if verbose:
                print("Step 2: Optimizing regime-specific strategies...")

            # Create strategy factory
            from strategy import WeightedRuleStrategyFactory
            strategy_factory = WeightedRuleStrategyFactory()

            # Create regime manager if needed
            if not hasattr(self, 'regime_manager') or self.regime_manager is None:
                self.regime_manager = RegimeManager(
                    regime_detector=regime_detector,
                    strategy_factory=strategy_factory,
                    rule_objects=list(optimized_rules.values()) if optimized_rules else self.rule_objects,
                    data_handler=self.data_handler
                )

            # Update rule objects in regime manager
            if optimized_rules:
                self.regime_manager.rule_objects = list(optimized_rules.values())

            # Set default strategy to current optimized strategy
            self.regime_manager.default_strategy = optimized_strategy

            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)

        return self.regime_manager


    def _optimize_joint(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Jointly optimize rule weights and regime parameters.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if verbose:
            print("Joint optimization of rules and regimes")
            print("Note: This is an advanced method that may take significant time")

        # Currently only genetic algorithm is supported for joint optimization
        if method != OptimizationMethod.GENETIC and method != OptimizationMethod.JOINT:
            raise ValueError("Joint optimization currently only supports genetic algorithm")

        # First, optimize rules using the unified optimize method
        if verbose:
            print("Step 1: Optimizing rule weights globally")

        rule_result = self.optimize(
            component_type='rule',
            method=method,
            metrics=metrics,
            verbose=verbose,
            **optimization_params
        )

        # For genetic optimization, we get a strategy directly
        if method == OptimizationMethod.GENETIC:
            optimized_strategy = rule_result  # This is the actual strategy
        else:
            # For other methods, create a strategy with optimized rule objects
            optimized_rule_objects = list(rule_result.values())
            weights = np.ones(len(optimized_rule_objects)) / len(optimized_rule_objects)
            optimized_strategy = WeightedRuleStrategy(
                rule_objects=optimized_rule_objects,
                weights=weights
            )

        if regime_detector:
            if verbose:
                print("Step 2: Optimizing regime-specific strategies")

            # Create strategy factory
            from strategy import WeightedRuleStrategyFactory
            strategy_factory = WeightedRuleStrategyFactory()

            # Create regime manager
            self.regime_manager = RegimeManager(
                regime_detector=regime_detector,
                strategy_factory=strategy_factory,
                rule_objects=self.rule_objects if method == OptimizationMethod.GENETIC else optimized_rule_objects,
                data_handler=self.data_handler
            )

            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)

            # Set optimized global strategy as default
            self.regime_manager.default_strategy = optimized_strategy

            return self.regime_manager
        else:
            return optimized_strategy

    def _optimize_joint(self, method, metrics, regime_detector, optimization_params, verbose):
        """
        Jointly optimize rule weights and regime parameters.

        Args:
            method: Optimization method
            metrics: Performance metric to optimize
            regime_detector: Regime detector to use
            optimization_params: Additional parameters
            verbose: Whether to print progress
        """
        if verbose:
            print("Joint optimization of rules and regimes")
            print("Note: This is an advanced method that may take significant time")

        # Currently only genetic algorithm is supported for joint optimization
        if method != OptimizationMethod.GENETIC and method != OptimizationMethod.JOINT:
            raise ValueError("Joint optimization currently only supports genetic algorithm")

        # First, optimize rules
        self.optimized_rule_weights = self._optimize_rule_weights(
            method, metrics, optimization_params, verbose
        )

        if regime_detector:
            # Create strategy factory
            from strategy import WeightedRuleStrategyFactory
            strategy_factory = WeightedRuleStrategyFactory()

            # Create regime manager with default rule weights
            self.regime_manager = RegimeManager(
                regime_detector=regime_detector,
                strategy_factory=strategy_factory,
                rule_objects=self.rule_objects,
                data_handler=self.data_handler
            )

            # Optimize regime-specific strategies
            self.regime_manager.optimize_regime_strategies(verbose=verbose)

            # Set optimized global strategy as default
            optimized_strategy = WeightedRuleStrategy(
                rule_objects=self.rule_objects,
                weights=self.optimized_rule_weights
            )
            self.regime_manager.default_strategy = optimized_strategy

            return self.regime_manager
        else:
            # Create and return strategy with optimized weights
            return WeightedRuleStrategy(
                rule_objects=self.rule_objects,
                weights=self.optimized_rule_weights
            )



    def validate(self, validation_method, component_type='rule', method=OptimizationMethod.GENETIC, 
                 components=None, metrics='sharpe', verbose=True, **kwargs):
        """
        Validate optimization of components using cross-validation or walk-forward.

        Args:
            validation_method: Validation method to use ('cross_validation', 'walk_forward', etc.)
            component_type: Type of component to optimize ('rule', 'regime_detector', etc.)
            method: Optimization method to use
            components: List of component names to optimize (or None for all registered)
            metrics: Performance metric(s) to optimize for
            verbose: Whether to print progress information
            **kwargs: Additional parameters
                - validation_params: Dictionary of validation-specific parameters
                    - window_size: Size of windows for walk-forward (default: 252)
                    - step_size: Step size for walk-forward (default: 63)
                    - train_pct: Train percentage for walk-forward (default: 0.7)
                    - n_folds: Number of folds for cross-validation (default: 5)
                - ... (other optimization parameters)

        Returns:
            dict: Validation results
        """
        # Get validation parameters
        validation_params = kwargs.pop('validation_params', {})

        # Choose factory and evaluator based on component type
        if component_type == 'rule':
            factory = RuleFactory()
            evaluator = RuleEvaluator.evaluate
        elif component_type == 'regime_detector':
            factory = RegimeDetectorFactory()
            evaluator = RegimeDetectorEvaluator.evaluate
        # ... other component types ...

        # Create validator based on method
        if validation_method == 'walk_forward':
            from optimization.validation import WalkForwardValidator
            validator = WalkForwardValidator(
                window_size=validation_params.get('window_size', 252),
                step_size=validation_params.get('step_size', 63),
                train_pct=validation_params.get('train_pct', 0.7),
                top_n=validation_params.get('top_n', 5),
                plot_results=validation_params.get('plot_results', True)
            )
        elif validation_method == 'cross_validation':
            from optimization.validation import CrossValidator
            validator = CrossValidator(
                n_folds=validation_params.get('n_folds', 5),
                top_n=validation_params.get('top_n', 5),
                plot_results=validation_params.get('plot_results', True)
            )
        elif validation_method == 'nested_cv':
            from optimization.validation import NestedCrossValidator
            validator = NestedCrossValidator(
                outer_folds=validation_params.get('outer_folds', 5),
                inner_folds=validation_params.get('inner_folds', 3),
                top_n=validation_params.get('top_n', 5),
                plot_results=validation_params.get('plot_results', True)
            )
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")

        # Get component configs
        if components is None:
            components_to_validate = list(self.components[component_type].keys())
        else:
            components_to_validate = components

        configs = []
        for name in components_to_validate:
            component_data = self.components[component_type][name]
            if component_data['params'] is not None:
                configs.append((component_data['class'], component_data['params']))

        # Run validation
        validation_results = validator.validate(
            component_factory=factory,
            optimization_method=method,
            data_handler=self.data_handler,
            configs=configs,
            metric=metrics,
            verbose=verbose,
            **kwargs
        )

        # Store results
        self.validation_results = validation_results

        return validation_results

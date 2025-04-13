"""
Evaluator classes for different component types in the optimization framework.
"""

import numpy as np

class RuleEvaluator:
    """Evaluator for trading rules."""
    
    @staticmethod
    def evaluate(rule, data_handler, metric='return'):
        """
        Evaluate a rule's performance on historical data.
        
        Args:
            rule: The rule to evaluate
            data_handler: Data handler providing market data
            metric: Performance metric ('return', 'sharpe', 'win_rate')
            
        Returns:
            float: Evaluation score
        """
        from strategy import TopNStrategy
        from backtester import Backtester
        
        # Reset states for clean evaluation
        if hasattr(rule, 'reset'):
            rule.reset()
        data_handler.reset_train()
        
        # Create strategy with just this rule
        strategy = TopNStrategy(rule_objects=[rule])
        
        # Backtest
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=False)
        
        # Check if we have enough trades
        if results['num_trades'] > 0:
            if metric == 'return':
                return results['total_log_return']
            elif metric == 'sharpe':
                return backtester.calculate_sharpe()
            elif metric == 'win_rate':
                win_count = sum(1 for trade in results['trades'] if trade[5] > 0)
                return win_count / results['num_trades']
            elif metric == 'risk_adjusted':
                # Custom risk-adjusted return
                if results['num_trades'] >= 5:
                    returns = [trade[5] for trade in results['trades']]
                    mean_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else float('inf')
                    downside = sum(r for r in returns if r < 0) if any(r < 0 for r in returns) else -0.0001
                    return mean_return / (std_return * abs(downside)) if abs(downside) > 0 else 0
        
        # Return very negative score if no trades or invalid metric
        return -float('inf')


class RegimeDetectorEvaluator:
    """Evaluator for regime detectors."""
    
    @staticmethod
    def evaluate(detector, data_handler, metric='stability'):
        """
        Evaluate a regime detector's performance.
        
        Args:
            detector: The regime detector to evaluate
            data_handler: Data handler providing market data
            metric: Performance metric ('stability', 'accuracy', 'strategy')
            
        Returns:
            float: Evaluation score
        """
        # Reset states
        if hasattr(detector, 'reset'):
            detector.reset()
        data_handler.reset_train()
        
        regime_changes = 0
        last_regime = None
        regime_durations = []
        current_duration = 0
        
        # Process all bars
        while True:
            bar = data_handler.get_next_train_bar()
            if bar is None:
                break
                
            regime = detector.detect_regime(bar)
            
            # Track regime changes and durations
            if last_regime is not None:
                if regime != last_regime:
                    regime_changes += 1
                    regime_durations.append(current_duration)
                    current_duration = 0
                else:
                    current_duration += 1
                    
            last_regime = regime
        
        # Add final duration
        if current_duration > 0:
            regime_durations.append(current_duration)
            
        # Calculate metrics
        avg_duration = sum(regime_durations) / len(regime_durations) if regime_durations else 0
        
        if metric == 'stability':
            # Balance between fewer regime changes and reasonable duration
            return avg_duration - (regime_changes * 0.1)  # Penalize frequent changes
            
        elif metric == 'strategy':
            # This would evaluate a strategy that uses this detector
            # Implementation depends on your regime-based strategy setup
            from backtester import Backtester
            from regime_detection import RegimeManager
            from strategy import WeightedRuleStrategyFactory
            
            # Use a default set of rules
            from optimizer_manager import OptimizerManager
            
            # Get rule objects from manager or use default
            if hasattr(data_handler, 'rule_objects'):
                rule_objects = data_handler.rule_objects
            else:
                # Fallback to some default rules if available
                rule_objects = []
            
            if not rule_objects:
                # Cannot evaluate without rules
                return -float('inf')
            
            # Create a regime manager with this detector
            factory = WeightedRuleStrategyFactory()
            regime_manager = RegimeManager(
                regime_detector=detector,
                strategy_factory=factory,
                rule_objects=rule_objects,
                data_handler=data_handler
            )
            
            # Run a basic backtest
            backtester = Backtester(data_handler, regime_manager)
            results = backtester.run(use_test_data=False)
            
            if results['num_trades'] > 0:
                if metric == 'strategy_return':
                    return results['total_log_return']
                elif metric == 'strategy_sharpe':
                    return backtester.calculate_sharpe()
                else:
                    return results['total_log_return']
            
            return -float('inf')
            
        # Default fallback
        return -regime_changes  # Fewer changes is better


class StrategyEvaluator:
    """Evaluator for complete trading strategies."""
    
    @staticmethod
    def evaluate(strategy, data_handler, metric='sharpe'):
        """
        Evaluate a complete strategy's performance.
        
        Args:
            strategy: The strategy to evaluate
            data_handler: Data handler providing market data
            metric: Performance metric
            
        Returns:
            float: Evaluation score
        """
        from backtester import Backtester
        
        # Reset states
        if hasattr(strategy, 'reset'):
            strategy.reset()
        data_handler.reset_train()
        
        # Run backtest
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=False)
        
        # Calculate metrics
        if results['num_trades'] > 0:
            if metric == 'return':
                return results['total_log_return']
            elif metric == 'sharpe':
                return backtester.calculate_sharpe()
            elif metric == 'win_rate':
                win_count = sum(1 for trade in results['trades'] if trade[5] > 0)
                return win_count / results['num_trades']
            elif metric == 'calmar':
                # Approximate Calmar ratio calculation
                max_dd = backtester.calculate_max_drawdown() if hasattr(backtester, 'calculate_max_drawdown') else 0.1
                return results['total_log_return'] / max_dd if max_dd > 0 else 0
        
        return -float('inf')

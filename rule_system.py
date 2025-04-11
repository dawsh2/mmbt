import itertools
import numpy as np
from backtester import Backtester  # Assuming your Backtester is in backtester.py
from strategy import TopNStrategy  # Assuming TopNStrategy is in strategy.py

class EventDrivenRuleSystem:
    def __init__(self, rules_config, top_n=5):
        """
        Args:
            rules_config (list of tuples): Each tuple contains (RuleClass, list_of_parameter_dicts).
                                            Example: [(Rule0, [{'fast_window': [5, 10], 'slow_window': [20, 30]}]), ...]
            top_n (int): Number of top-performing rules to select.
        """
        self.rules_config = self._expand_param_grid(rules_config)
        self.top_n = top_n
        self.best_params = {}  # Rule index -> best params
        self.best_scores = {}  # Rule index -> best score
        self.trained_rule_objects = {} # Rule index -> instance of the rule with best params

    def _expand_param_grid(self, rules_config):
        """
        Expands the parameter grid for each rule.
        """
        expanded_config = []
        for rule_class, param_ranges in rules_config:
            print(f"Expanding parameters for {rule_class.__name__}:")
            print(f"  Original param ranges: {param_ranges}")

            param_names = param_ranges.keys()
            param_values = param_ranges.values()
            param_combinations = list(itertools.product(*param_values))
            parameter_sets = [dict(zip(param_names, combo)) for combo in param_combinations]

            print(f"  Generated {len(parameter_sets)} parameter sets")
            print(f"  First few sets: {parameter_sets[:3]}")

            expanded_config.append((rule_class, parameter_sets))
        return expanded_config

    # Metric for rule optmization is defined here
    def train_rules(self, data_handler):
        all_rule_performances = {}

        for i, (rule_class, param_sets) in enumerate(self.rules_config):
            rule_performances = {}
            print(f"Training Rule {rule_class.__name__} with {len(param_sets)} parameter sets...")
            for params in param_sets:
                print(f"Creating {rule_class.__name__} with params: {params}")
                rule_instance = rule_class(params)
                strategy = TopNStrategy(rule_objects=[rule_instance])
                backtester = Backtester(data_handler, strategy)
                results = backtester.run(use_test_data=False) # Ensure training uses train data
                if results['num_trades'] > 0:
                    optimization_metric = results['total_log_return']
                    rule_performances[tuple(params.items())] = optimization_metric
                else:
                    rule_performances[tuple(params.items())] = -np.inf
                rule_instance.reset()
                strategy.reset()
                backtester.reset()

            if rule_performances:
                best_params_tuple = max(rule_performances, key=rule_performances.get)
                best_score = rule_performances[best_params_tuple]
                best_params = dict(best_params_tuple)
                all_rule_performances[i] = (best_params, best_score, rule_class)
                print(f"Rule {rule_class.__name__} - Best Params: {best_params}, Total Log Return: {best_score:.4f}")
            else:
                print(f"Rule {rule_class.__name__} - No valid parameter sets found.")

        # Select the top N rules based on their best Total Log Return
        sorted_rules = sorted(all_rule_performances.items(), key=lambda item: item[1][1], reverse=True)
        top_rules = sorted_rules[:self.top_n]

        self.best_params = {rule_index: data[0] for rule_index, data in top_rules}
        self.best_scores = {rule_index: data[1] for rule_index, data in top_rules}
        self.trained_rule_objects = {rule_index: data[2](self.best_params[rule_index]) for rule_index, data in top_rules}

    def get_top_n_strategy(self):
        return TopNStrategy(rule_objects=list(self.trained_rule_objects.values()))

    def reset(self):
        for rule in self.trained_rule_objects.values():
            rule.reset()
    
 

    def get_top_n_strategy(self):
        """
        Returns a TopNStrategy instance with the top performing rules.
        """
        return TopNStrategy(rule_objects=list(self.trained_rule_objects.values()))

    def reset(self):
        for rule in self.trained_rule_objects.values():
            rule.reset()

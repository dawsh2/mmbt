from src.optimization.components import ComponentFactory
from src.rules.rule_factory import RuleFactory as ComprehensiveRuleFactory

class RuleFactoryAdapter(ComponentFactory):
    def __init__(self):
        self.factory = ComprehensiveRuleFactory()
        
    def create(self, rule_class, params):
        rule_name = rule_class.__name__
        return self.factory.create_rule(rule_name=rule_name, params=params)

from events import EventBus
from data_handler import CSVDataHandler
from strategies import WeightedStrategy
from engine import Backtester
from config import ConfigManager

def initialize_system(config_file=None):
    """Initialize the trading system with proper component order.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Initialized components
    """
    # 1. Create configuration
    config = ConfigManager(config_file=config_file)
    
    # 2. Create event bus
    event_bus = EventBus()
    
    # 3. Create data handler
    data_handler = CSVDataHandler(
        event_bus=event_bus,
        csv_filepath=config.get('data.csv_filepath'),
        train_fraction=config.get('data.train_fraction', 0.8),
        close_positions_eod=config.get('backtester.close_positions_eod', True)
    )
    
    # 4. Create rules and strategies
    rule_objects = create_rules(config)
    strategy = WeightedStrategy(
        rules=rule_objects,
        weights=config.get('strategy.weights', None),
        buy_threshold=config.get('strategy.buy_threshold', 0.5),
        sell_threshold=config.get('strategy.sell_threshold', -0.5)
    )
    
    # 5. Create execution components
    backtester = Backtester(config, data_handler, strategy)
    
    # 6. Set up event handlers
    register_event_handlers(event_bus, strategy, backtester)
    
    return {
        'config': config,
        'event_bus': event_bus,
        'data_handler': data_handler,
        'strategy': strategy,
        'backtester': backtester
    }

def register_event_handlers(event_bus, strategy, backtester):
    """Register event handlers with the event bus.
    
    Args:
        event_bus: Event bus
        strategy: Strategy instance
        backtester: Backtester instance
    """
    # Register bar handler for strategy
    from events import EventHandler, EventType
    
    class BarHandler(EventHandler):
        def __init__(self, strategy):
            super().__init__([EventType.BAR])
            self.strategy = strategy
            
        def _process_event(self, event):
            self.strategy.on_bar(event)
    
    bar_handler = BarHandler(strategy)
    event_bus.register(EventType.BAR, bar_handler)
    
    # Register other handlers as needed
    # ...

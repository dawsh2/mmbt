classDiagram
    class EventBus {
        +emit(event)
        +register(event_type, handler)
    }

    class Event {
        +event_type
        +data
        +timestamp
    }

    class BarEvent {
        +get_data()
        +get_price()
        +get_symbol()
        +get_timestamp()
    }

    class SignalEvent {
        +BUY: 1
        +SELL: -1
        +NEUTRAL: 0
        +get_signal_value()
        +get_signal_name()
        +get_price()
        +get_symbol()
        +get_rule_id()
        +is_active()
    }

    class Strategy {
        +name
        +on_bar(event) SignalEvent
        +process_signals(bar_event) SignalEvent
    }

    class Rule {
        +name
        +on_bar(event) SignalEvent
        +generate_signal(bar_event) SignalEvent
    }

    class CompositeStrategy {
        +rules
        +process_signals(bar_event) SignalEvent
    }

    class TechnicalRule {
        +generate_signal(bar_event) SignalEvent
    }

    Strategy <|-- CompositeStrategy
    Rule <|-- TechnicalRule
    Event <|-- BarEvent
    Event <|-- SignalEvent
    
    EventBus --> Event: emits
    Strategy ..> SignalEvent: creates
    Rule ..> SignalEvent: creates
    Strategy --> BarEvent: processes
    Rule --> BarEvent: analyzes

    note for Rule "Analyzes market data and decides WHEN to generate signals"
    note for Strategy "Processes signals from rules"
    note for SignalEvent "Standard event class that represents all trading signals"
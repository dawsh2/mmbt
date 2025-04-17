Error reading src/.#optimization.py: [Errno 2] No such file or directory: 'src/.#optimization.py'
Error reading src/regime_detection/detectors/.#regime_manager.py: [Errno 2] No such file or directory: 'src/regime_detection/detectors/.#regime_manager.py'
Error reading src/optimization/.#evaluators.py: [Errno 2] No such file or directory: 'src/optimization/.#evaluators.py'
Error reading src/optimization/.#validation.py: [Errno 2] No such file or directory: 'src/optimization/.#validation.py'

==== Code Standardization Finder Results ====

Summary:
  Found issues in 24 files
  generate_signals_method: 0 occurrences
  dict_bar_data: 7 occurrences
  dict_access: 73 occurrences
  create_signal_helper: 5 occurrences
  direct_signal_creation: 68 occurrences
  direct_bar_event_creation: 14 occurrences
  dict_to_barevent: 4 occurrences

Detailed Results:

== data/data_connectors.py ==
  dict_access:
    Line 364: timestamps = chart_data['timestamp']

== data/data_handler.py ==
  dict_bar_data:
    Line 418: def create_bar_event(self, bar_data: Dict[str, Any]) -> BarEvent:
    Line 451: def process_bar(self, bar_data: Dict[str, Any]) -> None:
  dict_access:
    Line 282: timestamps = self.full_data['timestamp'].values
    Line 289: closest_idx = self.full_data['timestamp'].searchsorted(next_day)
    Line 506: return self.full_data[self.full_data['symbol'] == symbol].copy()
  direct_bar_event_creation:
    Line 345: return BarEvent(bar_dict)
    Line 359: return BarEvent(bar_dict)
    Line 428: return BarEvent(bar_data)
  dict_to_barevent:
    Line 428: return BarEvent(bar_data)

== engine/backtester.py ==
  direct_bar_event_creation:
    Line 154: bar_event = BarEvent(bar_data)
  dict_to_barevent:
    Line 154: bar_event = BarEvent(bar_data)

== engine/execution_engine.py ==
  dict_access:
    Line 217: price = bar_data['Close']
    Line 252: price = bar_data['Close']

== engine/position_manager.py ==
  dict_access:
    Line 669: current_price = bar_data['Close']

== events/event_emitters.py ==
  dict_access:
    Line 115: bar_data['symbol'] = self.default_symbol
  direct_bar_event_creation:
    Line 116: bar_event = BarEvent(bar_data, bar_event.timestamp)

== events/event_manager.py ==
  direct_bar_event_creation:
    Line 126: bar_event = BarEvent(bar_data)
    Line 231: bar_event = BarEvent(bar_data)
  dict_to_barevent:
    Line 126: bar_event = BarEvent(bar_data)
    Line 231: bar_event = BarEvent(bar_data)

== events/event_types.py ==
  dict_bar_data:
    Line 126: def __init__(self, bar_data: Dict[str, Any], timestamp: Optional[datetime] = None):
  dict_access:
    Line 294: return self.data['symbol']
    Line 360: return self.data['symbol']
  direct_bar_event_creation:
    Line 123: class BarEvent(Event):
    Line 177: return f"BarEvent({symbol} @ {timestamp}, O:{self.get_open():.2f}, H:{self.get_high():.2f}, L:{self.get_low():.2f}, C:{self.get_price():.2f})"

== events/event_utils.py ==
  dict_bar_data:
    Line 111: def create_bar_event(bar_data: Dict[str, Any], timestamp=None) -> BarEvent:
    Line 459: # def create_bar_event(bar_data: Dict[str, Any], timestamp=None) -> BarEvent:
  dict_access:
    Line 106: return event.data['symbol']
  create_signal_helper:
    Line 125: def create_signal(
    Line 191: return create_signal(
    Line 473: # def create_signal(
  direct_bar_event_creation:
    Line 121: return BarEvent(bar_data, timestamp)
    Line 469: #     return BarEvent(bar_data, timestamp)

== features/price_features.py ==
  dict_access:
    Line 271: highs = data['High']
    Line 272: lows = data['Low']
    Line 273: closes = data['Close']
    Line 483: closes = data['Close']
    Line 484: volumes = data['Volume']
    Line 624: prices = data['Close']

== features/technical_features.py ==
  dict_access:
    Line 113: close_price = data['Close'][-1] if isinstance(data['Close'], (list, np.ndarray, pd.Series)) else data['Close']
    Line 230: prices = data['Close']
    Line 361: highs = data['High']
    Line 362: lows = data['Low']
    Line 363: closes = data['Close']
    Line 387: if 'Volume' in data and isinstance(data['Volume'], (list, np.ndarray, pd.Series)) and len(data['Volume']) >= lookback:
    Line 388: levels = self._find_volume_levels(highs[-lookback:], lows[-lookback:], closes[-lookback:], data['Volume'][-lookback:])
    Line 724: close = data['Close'][-1] if isinstance(data['Close'], (list, np.ndarray, pd.Series)) else data['Close']
    Line 819: prices = data['Close']
    Line 1189: if 'Close' in data and isinstance(data['Close'], (list, np.ndarray, pd.Series)) and len(data['Close']) >= 5:
    Line 1192: recent_prices = data['Close'][-5:]
    Line 1363: prices = data['Close']
    Line 1428: prices = [bar['Close'] for bar in history[-self.slow_window:]] + [bar_data['Close']]

== features/time_features.py ==
  dict_access:
    Line 78: timestamp = data['timestamp']
    Line 199: timestamp = data['timestamp']
    Line 299: timestamp = data['timestamp']
    Line 409: timestamps = data['timestamp']
    Line 410: prices = data['Close']
    Line 601: timestamp = data['timestamp']

== position_management/__init__.py ==
  direct_signal_creation:
    Line 79: return Signal(

== risk_management/collector.py ==
  dict_bar_data:
    Line 51: def update_price_path(self, trade_id: str, bar_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:

== rules/crossover_rules.py ==
  dict_access:
    Line 253: #         close = data['Close']
    Line 442: close = data['Close']
    Line 616: close = data['Close']
    Line 811: close = data['Close']
    Line 972: close = data['Close']
    Line 1150: high = data['High']
    Line 1151: low = data['Low']
    Line 1152: close = data['Close']
  create_signal_helper:
    Line 128: return self.create_signal(
    Line 151: return self.create_signal(
  direct_signal_creation:
    Line 235: #             return Signal(
    Line 312: #                     return Signal(
    Line 329: #                 return Signal(
    Line 342: #         return Signal(
    Line 427: return Signal(
    Line 500: return Signal(
    Line 514: return Signal(
    Line 600: return Signal(
    Line 699: return Signal(
    Line 713: return Signal(
    Line 796: return Signal(
    Line 864: return Signal(
    Line 877: return Signal(
    Line 957: return Signal(
    Line 1032: return Signal(
    Line 1047: return Signal(
    Line 1132: return Signal(
    Line 1226: return Signal(
    Line 1241: return Signal(

== rules/oscillator_rules.py ==
  dict_access:
    Line 110: close = data['Close']
    Line 357: high = data['High']
    Line 358: low = data['Low']
    Line 359: close = data['Close']
    Line 619: high = data['High']
    Line 620: low = data['Low']
    Line 621: close = data['Close']
    Line 830: close = data['Close']
  direct_signal_creation:
    Line 94: return Signal(
    Line 122: return Signal(
    Line 137: return Signal(
    Line 238: return Signal(
    Line 339: return Signal(
    Line 369: return Signal(
    Line 393: return Signal(
    Line 408: return Signal(
    Line 505: return Signal(
    Line 601: return Signal(
    Line 631: return Signal(
    Line 713: return Signal(
    Line 813: return Signal(
    Line 840: return Signal(
    Line 871: return Signal(
    Line 963: return Signal(

== rules/rule_base.py ==
  direct_bar_event_creation:
    Line 113: bar_event = BarEvent(event.data)

== rules/trend_rules.py ==
  dict_access:
    Line 97: high = data['High']
    Line 98: low = data['Low']
    Line 99: close = data['Close']
    Line 327: high = data['High']
    Line 328: low = data['Low']
    Line 329: close = data['Close']
    Line 557: high = data['High']
    Line 558: low = data['Low']
    Line 559: close = data['Close']
  direct_signal_creation:
    Line 82: return Signal(
    Line 109: return Signal(
    Line 147: return Signal(
    Line 208: return Signal(
    Line 311: return Signal(
    Line 340: return Signal(
    Line 456: return Signal(
    Line 543: return Signal(
    Line 569: return Signal(
    Line 602: return Signal(
    Line 658: return Signal(

== rules/volatility_rules.py ==
  dict_access:
    Line 103: close = data['Close']
    Line 104: high = data['High']
    Line 105: low = data['Low']
    Line 328: high = data['High']
    Line 329: low = data['Low']
    Line 330: close = data['Close']
    Line 561: high = data['High']
    Line 562: low = data['Low']
    Line 563: close = data['Close']
    Line 827: high = data['High']
    Line 828: low = data['Low']
    Line 829: close = data['Close']
  direct_signal_creation:
    Line 87: return Signal(
    Line 114: return Signal(
    Line 216: return Signal(
    Line 312: return Signal(
    Line 340: return Signal(
    Line 364: return Signal(
    Line 447: return Signal(
    Line 545: return Signal(
    Line 578: return Signal(
    Line 706: return Signal(
    Line 811: return Signal(
    Line 839: return Signal(
    Line 863: return Signal(
    Line 889: return Signal(
    Line 968: return Signal(

== signals/signal_processing.py ==
  direct_signal_creation:
    Line 55: return Signal(
    Line 104: return f"Signal({self.signal_type}, rule={self.rule_id}, confidence={self.confidence:.2f}, price={self.price})"

== strategies/ensemble_strategy.py ==
  direct_signal_creation:
    Line 165: self.last_signal = Signal(

== strategies/regime_strategy.py ==
  direct_signal_creation:
    Line 84: self.last_signal = Signal(

== strategies/strategy_base.py ==
  dict_bar_data:
    Line 91: def update_indicators(self, bar_data: Dict[str, Any]) -> None:
  direct_bar_event_creation:
    Line 64: bar_event = BarEvent(event.data, event.timestamp)
    Line 70: bar_event = BarEvent(bar_data, event.timestamp)

== tests/test_strategies.py ==
  direct_signal_creation:
    Line 25: return Signal(
    Line 56: self.mock_rule.on_bar.return_value = Signal(

==== Standardization TODO List ====


2. Replace dictionary usage with BarEvent in these files:
   - data/data_connectors.py
   - data/data_handler.py
   - engine/backtester.py
   - engine/execution_engine.py
   - engine/position_manager.py
   - events/event_emitters.py
   - events/event_manager.py
   - events/event_types.py
   - events/event_utils.py
   - features/price_features.py
   - features/technical_features.py
   - features/time_features.py
   - risk_management/collector.py
   - rules/crossover_rules.py
   - rules/oscillator_rules.py
   - rules/trend_rules.py
   - rules/volatility_rules.py
   - strategies/strategy_base.py

3. Replace signal helper methods with direct SignalEvent creation:
   - events/event_utils.py
   - position_management/__init__.py
   - rules/crossover_rules.py
   - rules/oscillator_rules.py
   - rules/trend_rules.py
   - rules/volatility_rules.py
   - signals/signal_processing.py
   - strategies/ensemble_strategy.py
   - strategies/regime_strategy.py
   - tests/test_strategies.py

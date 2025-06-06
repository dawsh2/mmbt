from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, List
# from backtester import BarEvent



class SignalType(Enum):
    """Enumeration of different signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


@dataclass
class Signal:
    """
    Class representing a trading signal.
    
    This class provides a standardized structure for trading signals
    generated by rules and strategies in the trading system.
    """
    
    timestamp: datetime
    signal_type: SignalType
    price: float
    rule_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Ensure metadata is initialized
        if self.metadata is None:
            self.metadata = {}
        # Ensure confidence is between 0 and 1
        self.confidence = min(max(self.confidence, 0.0), 1.0)

    def copy(self):
        """Create a copy of the signal."""
        return Signal(
            timestamp=self.timestamp,
            signal_type=self.signal_type,
            price=self.price,
            rule_id=self.rule_id,
            confidence=self.confidence,
            metadata=self.metadata.copy() if self.metadata else None
        )

    @classmethod
    def from_numeric(cls, timestamp: datetime, signal_value: int, price: float,
                     rule_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a Signal object from a numeric signal value (-1, 0, 1).
        
        Args:
            timestamp: Signal timestamp
            signal_value: Numeric signal value (-1, 0, 1)
            price: Price at signal generation
            rule_id: Optional rule identifier
            metadata: Additional signal data
            
        Returns:
            Signal: A new Signal object
        """
        if signal_value == 1:
            signal_type = SignalType.BUY
        elif signal_value == -1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
            
        return cls(timestamp, signal_type, price, rule_id, 1.0, metadata)

    @classmethod
    def from_dict(cls, signal_dict: Dict[str, Any]):
        """
        Create a Signal object from a dictionary.
        
        Args:
            signal_dict: Dictionary containing signal data
            
        Returns:
            Signal: A new Signal object
        """
        timestamp = signal_dict["timestamp"]
        
        # Extract signal type
        if "signal" in signal_dict:
            numeric_signal = signal_dict["signal"]
            if numeric_signal == 1:
                signal_type = SignalType.BUY
            elif numeric_signal == -1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL
        else:
            signal_type = SignalType.NEUTRAL
            
        price = signal_dict["price"]
        
        # Extract additional fields
        rule_id = signal_dict.get("rule_id")
        confidence = signal_dict.get("confidence", 1.0)
        
        # Extract any additional data as metadata
        metadata = {k: v for k, v in signal_dict.items()
                    if k not in ["timestamp", "signal", "price", "rule_id", "confidence"]}
        
        return cls(timestamp, signal_type, price, rule_id, confidence, metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Signal to a dictionary representation.
        
        Returns:
            dict: Dictionary representation of the signal
        """
        result = {
            "timestamp": self.timestamp,
            "signal": self.signal_type.value,
            "price": self.price,
        }
        
        if self.rule_id:
            result["rule_id"] = self.rule_id
            
        if self.confidence != 1.0:
            result["confidence"] = self.confidence
            
        # Add any metadata
        result.update(self.metadata)
        
        return result

    def get_numeric_signal(self) -> int:
        """
        Get the numeric representation of the signal type.
        
        Returns:
            int: 1 for BUY, -1 for SELL, 0 for NEUTRAL
        """
        return self.signal_type.value

    def __str__(self) -> str:
        """String representation of the signal."""
        return f"Signal({self.signal_type.name}, {self.timestamp}, ${self.price:.2f})"

    def __repr__(self) -> str:
        """Detailed representation of the signal."""
        return (f"Signal(timestamp={self.timestamp}, type={self.signal_type.name}, "
                f"price={self.price:.2f}, rule_id={self.rule_id}, "
                f"confidence={self.confidence:.2f})")


class SignalCollection:
    """
    Collection of signals with utility methods for aggregation and analysis.
    
    This class manages multiple signals, particularly useful for combining
    signals from different rules in a strategy.
    """
    
    def __init__(self, signals: Optional[List[Signal]] = None):
        """
        Initialize a signal collection.
        
        Args:
            signals: Optional initial list of signals
        """
        self.signals = signals or []
    
    def add(self, signal: Signal):
        """
        Add a signal to the collection.
        
        Args:
            signal: Signal to add
        """
        self.signals.append(signal)
    
    def clear(self):
        """Clear all signals from the collection."""
        self.signals = []
    
    def get_signals_by_type(self, signal_type: SignalType) -> List[Signal]:
        """
        Get all signals of a specific type.
        
        Args:
            signal_type: Type of signals to retrieve
        
        Returns:
            list: List of matching signals
        """
        return [s for s in self.signals if s.signal_type == signal_type]
    
    def get_signals_by_rule(self, rule_id: str) -> List[Signal]:
        """
        Get all signals generated by a specific rule.
        
        Args:
            rule_id: ID of the rule
        
        Returns:
            list: List of matching signals
        """
        return [s for s in self.signals if s.rule_id == rule_id]
    
    def get_weighted_consensus(self) -> SignalType:
        """
        Calculate a weighted consensus signal based on signal confidences.
        
        Returns:
            SignalType: The consensus signal type
        """
        if not self.signals:
            return SignalType.NEUTRAL
        
        weighted_sum = sum(s.get_numeric_signal() * s.confidence for s in self.signals)
        avg = weighted_sum / sum(s.confidence for s in self.signals)
        
        if avg > 0.2:  # Threshold for buy
            return SignalType.BUY
        elif avg < -0.2:  # Threshold for sell
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the signal collection to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all signals
        """
        if not self.signals:
            return pd.DataFrame()
            
        data = []
        for signal in self.signals:
            row = {
                'timestamp': signal.timestamp,
                'type': signal.signal_type.name,
                'numeric_signal': signal.get_numeric_signal(),
                'price': signal.price,
                'rule_id': signal.rule_id,
                'confidence': signal.confidence
            }
            # Add any metadata as columns
            for k, v in signal.metadata.items():
                row[k] = v
            data.append(row)
            
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        """Get the number of signals in the collection."""
        return len(self.signals)
    
    def __getitem__(self, index) -> Signal:
        """Get a signal by index."""
        return self.signals[index]
    
    def __iter__(self):
        """Iterate through signals."""
        return iter(self.signals)


class SignalRouter:
    """
    A signal router that standardizes signal flow between rules and strategies.
    
    This class acts as an adapter and router for signals, ensuring consistent formats
    throughout the trading system regardless of how individual rules generate signals.
    """
    def __init__(self, rules, rule_ids=None):
        """
        Initialize the signal router with a list of rules.
        
        Args:
            rules: List of rule objects that generate signals
            rule_ids: Optional list of rule identifiers. If not provided, 
                     rule indices will be used as identifiers.
        """
        self.rules = rules
        self.rule_ids = rule_ids or [f"rule_{i}" for i in range(len(rules))]
        self.signal_collection = SignalCollection()
    
    def on_bar(self, event):
        """
        Process a bar event through all rules and standardize outputs.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            dict: Standardized signal information with all rule signals
        """
        bar = event.bar
        timestamp = bar["timestamp"]
        price = bar["Close"]
        
        # Clear previous signals
        self.signal_collection.clear()
        
        # Process the bar through each rule and collect signals
        for i, rule in enumerate(self.rules):
            try:
                # Get the raw signal from the rule - passing just the bar, not the event
                rule_signal = rule.on_bar(bar)
                
                # Convert different signal formats to standardized Signal objects
                if rule_signal is None:
                    # Skip None signals
                    continue
                    
                elif isinstance(rule_signal, Signal):
                    # If already a Signal object, ensure it has a rule_id
                    if rule_signal.rule_id is None:
                        rule_signal.rule_id = self.rule_ids[i]
                    self.signal_collection.add(rule_signal)
                    
                elif isinstance(rule_signal, dict) and 'signal' in rule_signal:
                    # If rule returns a dictionary with signal key
                    signal = Signal.from_dict(rule_signal)
                    if signal.rule_id is None:
                        signal.rule_id = self.rule_ids[i]
                    self.signal_collection.add(signal)
                    
                elif isinstance(rule_signal, (int, float)):
                    # If rule returns a numeric signal
                    signal = Signal.from_numeric(
                        timestamp=timestamp,
                        signal_value=int(rule_signal),
                        price=price,
                        rule_id=self.rule_ids[i]
                    )
                    self.signal_collection.add(signal)
                    
                else:
                    # Try to extract useful information from other formats
                    if hasattr(rule_signal, 'signal_type') or hasattr(rule_signal, 'type'):
                        # Some custom object with a signal_type attribute
                        signal_type = getattr(rule_signal, 'signal_type', 
                                            getattr(rule_signal, 'type', SignalType.NEUTRAL))
                        
                        signal = Signal(
                            timestamp=timestamp,
                            signal_type=signal_type,
                            price=price,
                            rule_id=self.rule_ids[i]
                        )
                        self.signal_collection.add(signal)
            except Exception as e:
                # Log error and continue with other rules
                print(f"Error processing rule {self.rule_ids[i]}: {str(e)}")
                continue
        
        # Return a standardized format with the signal collection
        return {
            "timestamp": timestamp,
            "price": price,
            "signals": self.signal_collection
        }
    
    def reset(self):
        """Reset the router state and all rules."""
        self.signal_collection.clear()
        for rule in self.rules:
            if hasattr(rule, 'reset'):
                rule.reset()

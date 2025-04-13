"""
Composite regime detectors.
"""

from collections import Counter
from ..detector_base import DetectorBase
from ..regime_type import RegimeType
from ..detector_registry import registry

@registry.register(category="composite")
class CompositeDetector(DetectorBase):
    """
    Composite detector that combines multiple regime detectors.
    
    This detector aggregates the outputs of multiple detectors using
    different combination methods (majority vote, consensus, etc.).
    """
    
    def __init__(self, name=None, config=None, detectors=None, combination_method='majority'):
        """
        Initialize the composite detector.
        
        Args:
            name: Optional name for the detector
            config: Optional configuration dictionary
            detectors: List of detector instances to combine
            combination_method: Method to combine detector outputs:
                - majority: Use the most common regime
                - consensus: Use a regime only if all detectors agree
                - weighted: Use weighted voting (requires weights in config)
        """
        super().__init__(name=name, config=config)
        
        self.detectors = detectors or []
        self.combination_method = self.config.get('combination_method', combination_method)
        self.weights = self.config.get('weights', {})
        
        # Ensure weights exist for all detectors if using weighted method
        if self.combination_method == 'weighted' and not self.weights:
            # Default to equal weights
            self.weights = {i: 1 for i in range(len(self.detectors))}
    
    def add_detector(self, detector):
        """
        Add a detector to the composite.
        
        Args:
            detector: Detector instance to add
        """
        self.detectors.append(detector)
        
        # Update weights if using weighted method
        if self.combination_method == 'weighted':
            self.weights[len(self.detectors) - 1] = 1
    
    def detect_regime(self, bar_data):
        """
        Detect the current market regime based on combined detector outputs.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            RegimeType: The combined market regime
        """
        # Collect regime predictions from all detectors
        regimes = []
        for i, detector in enumerate(self.detectors):
            regime = detector.detect_regime(bar_data)
            
            if self.combination_method == 'weighted':
                # Add weighted votes
                weight = self.weights.get(i, 1)
                regimes.extend([regime] * weight)
            else:
                regimes.append(regime)
        
        # If no regimes, return UNKNOWN
        if not regimes:
            return RegimeType.UNKNOWN
        
        # Combine regimes based on method
        if self.combination_method == 'consensus':
            # Check if all regimes are the same
            if all(r == regimes[0] for r in regimes):
                self.current_regime = regimes[0]
            else:
                self.current_regime = RegimeType.UNKNOWN
        else:  # 'majority' or 'weighted'
            # Use most common regime
            regime_counter = Counter(regimes)
            self.current_regime = regime_counter.most_common(1)[0][0]
        
        return self.current_regime
    
    def reset(self):
        """Reset the detector and all sub-detectors."""
        super().reset()
        for detector in self.detectors:
            detector.reset()

# Structure to implement
import copy

# Dictionary mapping rule types to their required parameters
RULE_REQUIRED_PARAMS = {
    "RSIRule": ["rsi_period", "overbought", "oversold", "signal_type"],
    "SMAcrossoverRule": ["fast_window", "slow_window"],
    # Add more rule types as needed
}

def validate_parameters(params, required_params):
    """Validate that all required parameters are present"""
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    return True
    
def ensure_rule_parameters(rule_class_name, params):
    """Add missing parameters with default values for known rule types"""
    # Make a deep copy to avoid modifying the original
    params_copy = copy.deepcopy(params) if params else {}
    
    # Add missing parameters for RSI rule
    if rule_class_name == "RSIRule" and "signal_type" not in params_copy:
        params_copy["signal_type"] = "levels"
    
    return params_copy

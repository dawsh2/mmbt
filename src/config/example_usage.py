# example_usage.py

"""
Example usage of the ConfigManager.
"""

from config import ConfigManager
import json
import os

def main():
    # Create a config manager with default settings
    config = ConfigManager()
    print("Default configuration loaded.")
    
    # Get a specific value using dot notation
    slippage_model = config.get('backtester.market_simulation.slippage_model')
    initial_capital = config.get('backtester.initial_capital')
    print(f"Default slippage model: {slippage_model}")
    print(f"Default initial capital: {initial_capital}")
    
    # Update some configuration values
    config.set('backtester.initial_capital', 200000)
    config.set('position_management.position_sizing.method', 'volatility')
    config.set('position_management.risk_management.use_stop_loss', True)
    
    # Display the updated values
    print("\nUpdated configuration values:")
    print(f"Initial capital: {config.get('backtester.initial_capital')}")
    print(f"Position sizing method: {config.get('position_management.position_sizing.method')}")
    print(f"Use stop loss: {config.get('position_management.risk_management.use_stop_loss')}")
    
    # Save configuration to JSON file
    config_dir = "config_examples"
    os.makedirs(config_dir, exist_ok=True)
    json_path = os.path.join(config_dir, "trading_config.json")
    config.save(json_path)
    print(f"\nConfiguration saved to {json_path}")
    
    # Load configuration from file
    print("\nLoading configuration from file...")
    loaded_config = ConfigManager(config_file=json_path)
    
    # Verify loaded values
    print("Verification of loaded configuration:")
    print(f"Initial capital: {loaded_config.get('backtester.initial_capital')}")
    print(f"Position sizing method: {loaded_config.get('position_management.position_sizing.method')}")
    print(f"Use stop loss: {loaded_config.get('position_management.risk_management.use_stop_loss')}")
    
    # Example of overriding with custom dictionary
    custom_config = {
        'backtester': {
            'initial_capital': 500000,
            'market_simulation': {
                'slippage_model': 'volume',
                'price_impact': 0.2
            }
        },
        'regime_detection': {
            'detector_type': 'composite'
        }
    }
    
    print("\nCreating configuration with custom overrides...")
    custom_loaded_config = ConfigManager(config_dict=custom_config)
    
    # Verify the merged configuration
    print("Verification of custom configuration:")
    print(f"Initial capital: {custom_loaded_config.get('backtester.initial_capital')}")
    print(f"Slippage model: {custom_loaded_config.get('backtester.market_simulation.slippage_model')}")
    print(f"Price impact: {custom_loaded_config.get('backtester.market_simulation.price_impact')}")
    print(f"Detector type: {custom_loaded_config.get('regime_detection.detector_type')}")
    print(f"Fee model (from defaults): {custom_loaded_config.get('backtester.market_simulation.fee_model')}")
    
    # Example of validation failure
    try:
        print("\nTesting validation with invalid value...")
        invalid_config = ConfigManager()
        invalid_config.set('backtester.initial_capital', 500)  # Below minimum of 1000
    except ValueError as e:
        print(f"Validation error caught: {e}")

if __name__ == "__main__":
    main()

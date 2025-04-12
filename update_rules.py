#!/usr/bin/env python3
"""
Script to automatically update rule classes to use the new Signal object approach.
This script also fixes the issue with rule_system.py using TopNStrategy as a passthrough.

Usage:
    python update_rules.py

The script will:
1. Update all rule classes in strategy.py to use Signal objects
2. Update rule_system.py to properly use TopNStrategy
3. Create backup files of the original files before making changes
"""

import os
import re
import shutil
import sys
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def update_rule_class(rule_class_content):
    """
    Update a single rule class to use Signal objects.

    Args:
        rule_class_content (str): The content of the rule class

    Returns:
        str: Updated rule class content
    """
    # Add imports at the top if not present
    import_statement = "from signals import Signal, SignalType\n"

    # Parse class name from the content
    class_name_match = re.search(r'class\s+([A-Za-z0-9_]+)\s*:', rule_class_content)
    if not class_name_match:
        return rule_class_content  # Not a class definition

    class_name = class_name_match.group(1)

    # Check if this is a rule class by looking for on_bar method
    if not re.search(r'def\s+on_bar\s*\(', rule_class_content):
        return rule_class_content  # Not a rule class

    # Extract the __init__ method
    init_match = re.search(r'def\s+__init__\s*\(\s*self\s*,\s*params\s*\):(.*?)(?=\n\s*def|\Z)',
                            rule_class_content, re.DOTALL)

    # Modify initialization to use SignalType
    updated_init = None
    if init_match:
        init_body = init_match.group(1)
        # Add SignalType and rule_id
        if 'self.current_signal' in init_body:
            rule_id_string = '        self.rule_id = "' + class_name + '"'
            init_body = init_body.replace('self.current_signal = 0',
                                            'self.current_signal_type = SignalType.NEUTRAL\n' + rule_id_string)
        else:
            rule_id_string = '        self.rule_id = "' + class_name + '"'
            init_body += '\n        self.current_signal_type = SignalType.NEUTRAL\n' + rule_id_string

        updated_init = "def __init__(self, params):" + init_body

    # Extract the on_bar method
    on_bar_match = re.search(r'def\s+on_bar\s*\(\s*self\s*,\s*bar\s*\):(.*?)(?=\n\s*def|\Z)',
                                 rule_class_content, re.DOTALL)

    if not on_bar_match:
        return rule_class_content  # Couldn't find on_bar method

    on_bar_body = on_bar_match.group(1)

    # Look for current_signal assignments and update them to current_signal_type
    updated_body = on_bar_body
    if 'self.current_signal = 1' in updated_body:
        updated_body = updated_body.replace('self.current_signal = 1', 'self.current_signal_type = SignalType.BUY')
    if 'self.current_signal = -1' in updated_body:
        updated_body = updated_body.replace('self.current_signal = -1', 'self.current_signal_type = SignalType.SELL')
    if 'self.current_signal = 0' in updated_body:
        updated_body = updated_body.replace('self.current_signal = 0', 'self.current_signal_type = SignalType.NEUTRAL')

    # Replace the return statement
    return_match = re.search(r'return\s+(.*?)(?=\n|\Z)', updated_body)
    if return_match:
        return_value = return_match.group(1).strip()
        if return_value == 'self.current_signal':
            # Update to return Signal object
            new_return = """
        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )"""
            updated_body = re.sub(r'return\s+.*?(?=\n|\Z)', new_return, updated_body)
        else:
            # Handle other return values
            new_return = """
        # Determine signal type from return value
        signal_value = """ + return_value + """
        if signal_value == 1:
            signal_type = SignalType.BUY
        elif signal_value == -1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={}
        )"""
            updated_body = re.sub(r'return\s+.*?(?=\n|\Z)', new_return, updated_body)

    # Extract the reset method
    reset_match = re.search(r'def\s+reset\s*\(\s*self\s*\):(.*?)(?=\n\s*def|\Z)',
                            rule_class_content, re.DOTALL)

    updated_reset = None
    if reset_match:
        reset_body = reset_match.group(1)
        # Update current_signal to current_signal_type
        if 'self.current_signal = 0' in reset_body:
            reset_body = reset_body.replace('self.current_signal = 0',
                                            'self.current_signal_type = SignalType.NEUTRAL')

        updated_reset = "def reset(self):" + reset_body
    else:
        # Create reset method if it doesn't exist
        updated_reset = """
    def reset(self):
        # Reset rule state
        # Reset any state variables here
        self.current_signal_type = SignalType.NEUTRAL
"""

    # Assemble the updated class
    updated_class = """
class """ + class_name + """:
    \"\"\"
    """ + class_name + """ using Signal objects directly.
    \"\"\"
    """ + (updated_init if updated_init else "def __init__(self, params):\n        pass") + """

    def on_bar(self, bar):""" + updated_body + """

    """ + updated_reset + """
"""

    return import_statement + updated_class

def update_strategy_py(file_path):
    """Update the strategy.py file to use Signal objects in all rule classes."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Add required imports at the top of the file
    if 'from signals import Signal, SignalType' not in content:
        import_section = "from collections import deque\nimport pandas as pd\nimport numpy as np\nfrom signals import Signal, SignalType\n\n"
        content = re.sub(r'^(from|import).*?\n\n', import_section, content, count=1, flags=re.DOTALL)

    # Find all rule class definitions
    rule_class_pattern = r'class\s+Rule[0-9]+.*?(?=class|\Z)'
    rule_classes = re.findall(rule_class_pattern, content, re.DOTALL)

    # Update each rule class
    for rule_class in rule_classes:
        updated_class = update_rule_class(rule_class)
        content = content.replace(rule_class, updated_class)

    # Update TopNStrategy to use the new SignalRouter
    top_n_strategy_pattern = r'class\s+TopNStrategy.*?(?=class|\Z)'
    top_n_strategy_match = re.search(top_n_strategy_pattern, content, re.DOTALL)

    if top_n_strategy_match:
        new_top_n_strategy = """
class TopNStrategy:
    \"\"\"
    A strategy that combines signals from multiple rules using a simple voting mechanism.

    This implementation focuses on its core responsibility: combining signals from
    top-performing rules to generate trading decisions.
    \"\"\"
    def __init__(self, rule_objects):
        \"\"\"
        Initialize the TopN strategy with rule objects.

        Args:
            rule_objects: List of rule instances to use
        \"\"\"
        from signals import SignalRouter, SignalType

        self.rules = rule_objects
        self.router = SignalRouter(rule_objects)
        self.last_signal = None

    def on_bar(self, event):
        \"\"\"
        Process a bar and generate a signal by combining rule signals.

        Args:
            event: Bar event containing market data

        Returns:
            dict: Signal information including timestamp, signal value, and price
        \"\"\"
        # Get standardized signals from all rules via the router
        router_output = self.router.on_bar(event)
        signal_collection = router_output["signals"]

        # Use SignalCollection's weighted consensus to determine the overall signal
        consensus_signal_type = signal_collection.get_weighted_consensus()
        consensus_signal_value = consensus_signal_type.value

        # Create the final output signal
        self.last_signal = {
            "timestamp": router_output["timestamp"],
            "signal": consensus_signal_value,
            "price": router_output["price"]
        }

        return self.last_signal

    def reset(self):
        \"\"\"Reset the strategy and all rules.\"\"\"
        self.router.reset()
        self.last_signal = None
"""
        content = content.replace(top_n_strategy_match.group(0), new_top_n_strategy)

    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Updated {file_path}")
    return True

def update_rule_system_py(file_path):
    """Update rule_system.py to fix the TopNStrategy passthrough issue."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Update the get_top_n_strategy method
    original_method = """    def get_top_n_strategy(self):
        \"\"\"
        Returns a TopNStrategy instance with the top performing rules.
        \"\"\"
        return TopNStrategy(rule_objects=list(self.trained_rule_objects.values()))"""

    updated_method = """    def get_top_n_strategy(self):
        \"\"\"
        Returns a TopNStrategy instance with the top performing rules.
        This method creates a strategy that combines signals from the top rules.
        \"\"\"
        # Create the TopNStrategy with the trained rule objects
        strategy = TopNStrategy(rule_objects=list(self.trained_rule_objects.values()))
        return strategy"""

    content = content.replace(original_method, updated_method)

    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Updated {file_path}")
    return True

def main():
    """Main function to run the script."""
    # Define file paths
    strategy_py = "strategy.py"
    rule_system_py = "rule_system.py"

    # Check if files exist
    if not os.path.exists(strategy_py):
        print(f"Error: {strategy_py} not found")
        return False

    if not os.path.exists(rule_system_py):
        print(f"Error: {rule_system_py} not found")
        return False

    # Create backups
    backup_file(strategy_py)
    backup_file(rule_system_py)

    # Update files
    update_strategy_py(strategy_py)
    update_rule_system_py(rule_system_py)

    print("\nUpdate completed successfully!")
    print("The rule classes now use Signal objects and the TopNStrategy has been updated.")
    print("Backups of the original files were created before making changes.")
    return True

if __name__ == "__main__":
    main()

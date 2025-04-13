# test_imports.py in root directory
def test_basic_imports():
    try:
        from src.config.config_manager import ConfigManager
        print("✓ ConfigManager import successful")
        
        from src.engine.backtester import Backtester
        print("✓ Backtester import successful")
        
        from src.features.feature_base import Feature
        print("✓ Feature import successful")
        
        print("\nAll basic imports successful!")
    except ImportError as e:
        print(f"Import error: {e}")

if __name__ == "__main__":
    test_basic_imports()

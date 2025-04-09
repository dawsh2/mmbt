import pandas as pd
import numpy as np
from metrics import calculate_metrics, print_metrics

# Step 1: Create flat price dataset
n_rows = 500  # 500 minutes
data = pd.DataFrame({
    'Close': 100.0,
    'Open': 100.0,
    'High': 100.0,
    'Low': 100.0
}, index=pd.date_range(start="2024-01-01", periods=n_rows, freq="T"))

# Step 2: Add log returns (should all be 0)
data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)

# Step 3: Generate random signals
np.random.seed(42)
data['Signal'] = np.random.choice([-1, 0, 1], size=n_rows)

# Step 4: Apply strategy returns
data['StrategyReturn'] = data['Signal'].shift(1) * data['LogReturn']
data['StrategyReturn'] = data['StrategyReturn'].fillna(0)

# Step 5: Evaluate metrics
metrics = calculate_metrics(data['StrategyReturn'])

print("\n🧪 Test with Flat Price & Random Signals:")
print_metrics(metrics, label="Random Strategy on Flat Market")

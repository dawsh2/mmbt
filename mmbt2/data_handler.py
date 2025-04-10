import pandas as pd

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.bars = []

    def load_data(self):
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        df = df.dropna(subset=["Close"])  # Ensure no missing price data
        self.bars = df.to_dict(orient="records")
        return self.bars

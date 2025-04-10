import pandas as pd

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.bars = []
        self.current_index = 0
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        df = df.dropna(subset=["Close"])  # Ensure no missing price data
        self.bars = df.to_dict(orient="records")
        return self.bars

    def get_next_bar(self):
        if self.current_index < len(self.bars):
            bar = self.bars[self.current_index]
            self.current_index += 1
            return bar
        return None

    def reset(self):
        self.current_index = 0

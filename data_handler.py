import pandas as pd

class CSVDataHandler:
    def __init__(self, csv_filepath, train_fraction=0.8):
        self.csv_filepath = csv_filepath
        self.train_fraction = train_fraction
        self.full_df = self._load_and_preprocess_data()
        self.train_df = None
        self.test_df = None
        self._split_data()
        self.current_train_index = 0
        self.current_test_index = 0

    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_filepath, parse_dates=['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df

    def _split_data(self):
        train_size = int(self.train_fraction * len(self.full_df))
        self.train_df = self.full_df[:train_size]
        self.test_df = self.full_df[train_size:]
        print(f"In-sample data size: {len(self.train_df)}")
        print(f"Out-of-sample data size: {len(self.test_df)}")

    def get_next_train_bar(self):
        if self.current_train_index < len(self.train_df):
            bar = self.train_df.iloc[self.current_train_index].to_dict()
            self.current_train_index += 1
            return bar
        return None

    def get_next_test_bar(self):
        if self.current_test_index < len(self.test_df):
            bar = self.test_df.iloc[self.current_test_index].to_dict()
            self.current_test_index += 1
            return bar
        return None

    def reset_train(self):
        self.current_train_index = 0

    def reset_test(self):
        self.current_test_index = 0

    def get_full_data(self): # Optional: to access the full DataFrame if needed
        return self.full_df

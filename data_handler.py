import pandas as pd

class CSVDataHandler:
    def __init__(self, csv_filepath, train_fraction=0.8):
        self.csv_filepath = csv_filepath
        self.train_fraction = train_fraction
        self.full_length = self._get_full_length()
        self.train_size = int(self.train_fraction * self.full_length)
        self.train_iterator = self._bar_generator(0, self.train_size)
        self.test_iterator = self._bar_generator(self.train_size, self.full_length)
        self._current_train_bar = None
        self._current_test_bar = None
        self.reset_train()
        self.reset_test()

    def _get_full_length(self):
        with open(self.csv_filepath, 'r') as f:
            return sum(1 for line in f) - 1

    def _bar_generator(self, start, end):
        with pd.read_csv(self.csv_filepath, parse_dates=['timestamp'], skiprows=range(1, start + 1), chunksize=1) as reader:
            for i, chunk in enumerate(reader):
                if start + i < end:
                    yield chunk.iloc[0].to_dict()
                else:
                    break

    def get_next_train_bar(self):
        try:
            self._current_train_bar = next(self.train_iterator)
            return self._current_train_bar
        except StopIteration:
            return None

    def get_next_test_bar(self):
        try:
            self._current_test_bar = next(self.test_iterator)
            return self._current_test_bar
        except StopIteration:
            return None

    def reset_train(self):
        self.train_iterator = self._bar_generator(0, self.train_size)
        self._current_train_bar = None

    def reset_test(self):
        self.test_iterator = self._bar_generator(self.train_size, self.full_length)
        self._current_test_bar = None

    def get_full_data(self):
        return pd.read_csv(self.csv_filepath, parse_dates=['timestamp'])

# data_handler.py

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

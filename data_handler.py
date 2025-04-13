import pandas as pd

class CSVDataHandler:
    def __init__(self, csv_filepath, train_fraction=0.8, close_positions_eod=True):
        self.csv_filepath = csv_filepath
        self.train_fraction = train_fraction
        self.close_positions_eod = close_positions_eod  # New flag
        self.full_df = self._load_and_preprocess_data()
        self.train_df = None
        self.test_df = None
        self._split_data()
        self.current_train_index = 0
        self.current_test_index = 0
        
        # Track current day for EOD detection
        self.current_train_day = None
        self.current_test_day = None


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
        if self.current_train_index < len(self.train_df):
            bar = self.train_df.iloc[self.current_train_index].to_dict()
            self.current_train_index += 1
            
            # Check if this is a new day
            new_day = False
            if self.close_positions_eod:
                current_date = pd.to_datetime(bar['timestamp']).date()
                if self.current_train_day is not None and current_date != self.current_train_day:
                    new_day = True
                self.current_train_day = current_date
                
                # Add a flag to indicate end of day
                bar['is_eod'] = new_day
            
            return bar
        return None

    def get_next_test_bar(self):
        if self.current_test_index < len(self.test_df):
            bar = self.test_df.iloc[self.current_test_index].to_dict()
            self.current_test_index += 1
            
            # Check if this is a new day
            new_day = False
            if self.close_positions_eod:
                current_date = pd.to_datetime(bar['timestamp']).date()
                if self.current_test_day is not None and current_date != self.current_test_day:
                    new_day = True
                self.current_test_day = current_date
                
                # Add a flag to indicate end of day
                bar['is_eod'] = new_day
            
            return bar
        return None

    def reset_train(self):
        self.current_train_index = 0
        self.current_train_day = None  # Reset day tracking

    def reset_test(self):
        self.current_test_index = 0
        self.current_test_day = None  # Reset day tracking
                

    def get_full_data(self):
        return pd.read_csv(self.csv_filepath, parse_dates=['timestamp'])


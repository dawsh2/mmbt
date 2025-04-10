# rule_strategy.py - Enhance the strategy

class RuleBasedStrategy(EventDrivenStrategy):
    def __init__(self, event_queue, symbols, use_weights=True, top_n=5):
        super().__init__(event_queue, symbols)
        self.use_weights = use_weights
        self.top_n = top_n
        
        # Create your existing rule system
        self.rule_system = RuleSystem(top_n=top_n, use_weights=use_weights)
        
        # Parameters will be loaded or trained
        self.rules_trained = False
        
        # Lookback period for training/signal generation
        self.lookback_period = 100  # Adjust as needed
    
    def train_rules(self, symbol):
        """Train rules on historical data"""
        print(f"Training rules for {symbol}...")
        
        # Get training data
        training_data = self.market_data[symbol].tail(self.lookback_period)
        
        # Train the rules using your existing methods
        params, scores, indices = self.rule_system.train_rules(training_data)
        self.rules_trained = True
        
        print(f"Rules trained successfully. Top rule score: {scores[0] if scores else 'N/A'}")
    
    def calculate_signals(self, event):
        """Generate signals using rule system"""
        if event.type != EventType.MARKET:
            return
        
        symbol = event.symbol
        
        # Update internal data
        self.update_data(event)
        
        # Wait until we have enough data
        if len(self.market_data[symbol]) < self.lookback_period:
            return
        
        # Train rules if not already done
        if not self.rules_trained:
            self.train_rules(symbol)
        
        # Generate signals using existing rule system
        latest_data = self.market_data[symbol].tail(self.lookback_period)
        result_df = self.rule_system.generate_signals(latest_data)
        
        # Extract signal
        latest_signal = result_df['Signal'].iloc[-1] if isinstance(result_df, pd.DataFrame) else result_df.iloc[-1]
        
        # Generate signal event for non-zero signals
        if latest_signal != 0:
            signal_event = SignalEvent(
                symbol=symbol,
                datetime=latest_data.index[-1],
                signal_type=latest_signal
            )
            self.event_queue.put(signal_event)

class BaseStrategyBacktest:
    def __init__(self, returns_df, predictions):
        self.returns_df = returns_df
        self.predictions = predictions
        self.results = self._assemble_df()
    
    def _assemble_df(self):
        results_df = self.returns_df.copy().to_frame()
        results_df.columns = ['return']
        results_df['long_only_hold'] = results_df.cumsum()
        results_df['predictions'] = self.predictions
        return results_df
        
    def process_signals(self):
        pass
    
    def plot(self):
        self.results[['long_only_hold', 'strategy_return']].plot()
        return
    
class LongOnlyBacktest(BaseStrategyBacktest):
    def __init__(self, returns_df, predictions):
        super().__init__(returns_df, predictions)
        self._process_signals()
        
    def _process_signals(self):
        results_df = self.results.copy()
        # Initialise required columns as zeroes
        results_df['holding_begin_period'] = 0
        results_df['holding_end_period'] = 0
        results_df['strategy_return'] = 0
        # Apply the holdings
        results_df['holding_end_period'] = results_df.apply(self._set_holdings, axis=1)
        # Update holdings begin period
        results_df['holding_begin_period'] = results_df['holding_end_period'].shift(1)
        # Apply summation
        results_df['strategy_return'] = results_df.apply(self._sum_return, axis=1)
        results_df['strategy_return'] = results_df['strategy_return'].cumsum()
        # Have the first holding period be nil
        results_df['holding_begin_period'].iloc[0] = 0
        self.results = results_df
    
    def _set_holdings(self, row):
        # If the prediction is 1
        if row['predictions'] == 1:
            # If the holding is 1
            # Hold
            #If the holding is 0
            # Buy
            return 1
        # If the prediction is 0
        else:
            # If the holding is 1
            # Sell
            # If the holding is 0
            # Do nothing
            return 0
    
    def _sum_return(self, row):
        # if the state at beginning of period is 1 add return to running total
        if row['holding_begin_period'] == 1:
            return row['return']
        # If the state at beginning of period is 0 do not add return
        else:
            return 0
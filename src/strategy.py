import matplotlib.pyplot as plt


class BaseStrategyBacktest:
    """
    A base class with utility functions for assembing common dataframes across all strategies and defining plotting utilities.

    ...

    Attributes
    ----------
    returns_df : Series
            The daily returns pandas series
    predictions : list
            The predictiosn list in the format 0 or 1
    val_start : str or DatetimeIndex
            The start date for the validation dataset in pandas DatetimeIndex or a valid data time string
    test_start : str or DatetimeIndex
            The start date for the test dataset in pandas DatetimeIndex or a valid data time string
            
    Methods
    -------
    _assemble_df():
        Assembles the beginnings of the results df
    plot():
        Plots the results of the strategy against a benchmark long hold.
    """

    def __init__(self, returns_df, predictions, val_start, test_start):
        """
        Parameters
        ----------
        returns_df : Series
            The daily returns pandas series
        predictions : list
            The predictions list in the format 0 or 1
        val_start : str or DatetimeIndex
            The start date for the validation dataset in pandas DatetimeIndex or a valid data time string
        test_start : str or DatetimeIndex
            The start date for the test dataset in pandas DatetimeIndex or a valid data time string
        """
        self.returns_df = returns_df
        self.predictions = predictions
        self.val_start = val_start
        self.test_start = test_start
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
        train_results = self.results[:self.val_start]
        val_results = self.results[self.val_start:self.test_start]
        test_results = self.results[self.test_start:]
        plt.plot(self.results['long_only_hold'], label='long_only_hold')
        plt.plot(train_results['strategy_return'],
                 label='strategy_return_train_data')
        plt.plot(val_results['strategy_return'], c='y',
                 label='strategy_return_val_data')
        plt.plot(test_results['strategy_return'], c='r',
                 label='strategy_return_test_data')
        plt.ylabel('Return')
        plt.xlabel('Date')
        plt.legend()
        plt.show()
        return


class LongOnlyBacktest(BaseStrategyBacktest):
    """
    A long only strategy. 
    ...
    
    Attributes
    ----------
    returns_df : Series
            The daily returns pandas series
    predictions : list
            The predictiosn list in the format 0 or 1
    val_start : str or DatetimeIndex
            The start date for the validation dataset in pandas DatetimeIndex or a valid data time string
    test_start : str or DatetimeIndex
            The start date for the test dataset in pandas DatetimeIndex or a valid data time string
            
    Methods
    -------
    _process_signals():
        Processes the predictions and the logic contained in _set_holdings and _sum_returns
    _set_holdings():
        Set the holdings at each step according to the strategy
    _sum_returns():
        Add return to running total if you hold the security at the beginning of the period
    """

    def __init__(self, returns_df, predictions, val_start, test_start):
        super().__init__(returns_df, predictions, val_start, test_start)
        self._process_signals()

    def _process_signals(self):
        results_df = self.results.copy()
        # Initialise required columns as zeroes
        results_df['holding_begin_period'] = 0
        results_df['holding_end_period'] = 0
        results_df['strategy_return'] = 0
        # Apply the holdings
        results_df['holding_end_period'] = results_df.apply(
            self._set_holdings, axis=1)
        # Update holdings begin period
        results_df['holding_begin_period'] = results_df['holding_end_period'].shift(
            1)
        # Apply summation
        results_df['strategy_return'] = results_df.apply(
            self._sum_return, axis=1)
        results_df['strategy_return'] = results_df['strategy_return'].cumsum()
        # Have the first holding period be nil
        results_df['holding_begin_period'].iloc[0] = 0
        self.results = results_df

    def _set_holdings(self, row):
        # If the prediction is 1
        if row['predictions'] == 1:
            # If the holding is 1
            # Hold
            # If the holding is 0
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

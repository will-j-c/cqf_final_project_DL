import numpy as np
import pandas as pd
    

def create_features(df, open_col, close_col, high_col, low_col, vol_col):
    # Original reference Kannan Singaravelu, with some tweaks
    df = df.copy()
    multiplier = 2
    # Features
    # Open close - % difference between closing and opening price on any particular day
    df['OC'] = df[close_col] / df[open_col] - 1
    # High low - % difference between high price and low price on the day
    df['HL'] = df[low_col] / df[low_col] - 1
    # Gap - % difference between prior day close and current day open
    df['GAP'] = df[open_col] / df[close_col].shift(1) - 1
    # Log return - natural log of % change in closing
    df['RET'] = np.log(df[close_col] / df[close_col].shift(1))
    # Sigh of return
    df['SIGN'] = np.where(df['RET'] < 0, -1, 1)
    # Create features for different time periods (numbers represent days)
    periods = [6, 8, 12, 24, 48, 72]
    for i in periods:
        # % change in closing price
        df['PCHG' + str(i)] = df[close_col].pct_change(i)
        # % change in volume traded
        df['VCHG' + str(i)] = df[vol_col].pct_change(i)
        # Sum of log return over period
        df['RET' + str(i)] = df['RET'].rolling(i).sum()
        # Price moving average
        df['MA' + str(i)] = df[close_col].rolling(i).mean()
        # Price EMA
        df['EMA' + str(i)] = df[close_col].ewm(span=i, adjust=False).mean()
        # % change volume moving average
        df['VMA' + str(i)] = df[vol_col] / df[vol_col].rolling(i).mean()
        # open close mean
        df['OC' + str(i)] = df['OC'].rolling(i).mean()
        # high low mean
        df['HL' + str(i)] = df['HL'].rolling(i).mean()
        # Gap mean
        df['GAP' + str(i)] = df['GAP'].rolling(i).mean()
        # Standard deviation of log returns over period
        df['STD' + str(i)] = df['RET'].rolling(i).std()
        # Upper bollinger band mean +- standard deviation x multiplier
        df['UB' + str(i)] = df[close_col].rolling(i).mean() + \
            df[close_col].rolling(i).std() * multiplier
        # Lower bollinger band
        df['LB' + str(i)] = df[close_col].rolling(i).mean() - \
            df[close_col].rolling(i).std() * multiplier
        # Momentum
        df['MOM' + str(i)] = df[close_col] - df[close_col].shift(i)
    # MACD
    df['MACD'] = df[close_col].ewm(span=12, adjust=False).mean() - df[close_col].ewm(span=24, adjust=False).mean()
    # Fast stochastic
    df['F_STOCH'] = (df[close_col] - df[low_col].rolling(14).max()) / (df[high_col].rolling(14).max() - df[low_col].rolling(14).max())
    # Slow stochastic
    df['S_STOCH'] = df['F_STOCH'].rolling(3).mean()
    # On balance volume
    df['OBV'] = (np.sign(df[close_col].diff()) * df[vol_col]).fillna(0).cumsum()
    # Reorder the columns into alphabetical order for easier analysis and visualization
    new_column_order = df.columns.sort_values()
    df = df[new_column_order]
    # Drop NaN values and other features that we won't use
    df.dropna(inplace=True)
    return df
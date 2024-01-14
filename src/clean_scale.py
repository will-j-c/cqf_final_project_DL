# Imports
from helpers import *
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

# Set seeds for reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# Load the raw price data
df = pd.read_csv('data/raw/Gemini_ETHUSD_1h.csv', skiprows=[0])
# Covert unix seconds to ms
df['unix'] = df['unix'].apply(convert_unix_to_ms)
# Parse unix timestamp as UTC dates and save as index
df['unix'] = pd.to_datetime(df['unix'], unit='ms', utc=True)
df.set_index(df['unix'], inplace=True)
# Drop date, symbol and Volume USD columns
df.drop(['unix', 'date', 'symbol', 'Volume ETH'], axis=1, inplace=True)
# Rename
df.rename(columns={'Volume USD': 'volume'}, inplace=True)
# Sort date ascending
df.sort_index(inplace=True)

# Load the raw fear and greed index data
df_fear_greed = pd.read_csv('data/raw/crypto_greed_fear_index.csv', parse_dates=True, index_col='timestamp')
# Drop unneeded columns
df_fear_greed.drop(['time_until_update', 'timestamp.1'], axis=1, inplace=True)
# Rename columns
df_fear_greed.columns = ['fg_value', 'fg_value_classification']
# Put classification to lower case
df_fear_greed['fg_value_classification'] = df_fear_greed['fg_value_classification'].str.lower()

# Join the fear and greed data to price data
df = df.join(df_fear_greed)
# As index is only published once a day, forward fill for the remainder of the day
df['fg_value'].ffill(inplace=True)
df['fg_value_classification'].ffill(inplace=True)
# Drop the leading dates where no fear and greed index is available
df.dropna(inplace=True)

# Feature engineering
# Create the technical indicators
df.ta.study(cores=0)
# Add hours, days, months to investigate seasonality
df['hour'] = df.index.hour
df['day_of_week'] = df.index.day_of_week
df['month'] = df.index.month

# One Hot Encoding of categorical data
encoder = OneHotEncoder(sparse_output=False)
onehot = encoder.fit_transform(df[['fg_value_classification', 'hour', 'day_of_week', 'month']])
feature_names = encoder.get_feature_names_out()
df[feature_names] = onehot
df.drop(['fg_value_classification', 'hour', 'day_of_week', 'month'], axis=1, inplace=True)

# Clean the data

# Drop all the columns with all NaN
df = df.dropna(axis=1, how='all')
# Remove columns that do not have at least 40000 of data
df = df.dropna(axis=1, thresh=40000)
# Remove the leading rows of the data with NaN
df.dropna(axis=0, inplace=True)
# Remove VIDYA_14 (calculates to inf)
df.drop('VIDYA_14', axis=1, inplace=True)
# Take out columns with no variance (i.e. all the same value)
vt = VarianceThreshold()
df = pd.DataFrame(vt.fit_transform(df), columns=vt.get_feature_names_out(), index=df.index)

# Compute the outliers
# Calculate what columns have outliers based on a threshold
# Initialise an empty list
outliers_arr = []
threshold = 10
cdl_cols = [col for col in df if col.startswith('CDL_')]
outlier_df = df.drop(cdl_cols, axis=1)
# Cycle through all columns
for col in outlier_df.columns:
    try:
        binary = outlier_df[col].isin([0, 1]).all()
        if binary:
            # break loop and go again
            continue
        Q1 = outlier_df[col].quantile(0.25)
        Q3 = outlier_df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Check if the datapoint is an outlier based on a threshold
        outliers = outlier_df[col][(outlier_df[col] < Q1 - threshold * IQR) | (outlier_df[col] > Q3 + threshold * IQR)].count()
        if outliers > 0:
            outliers_arr.append((col, outliers))
    except:
        continue
    
outlier_df = pd.DataFrame(outliers_arr, columns=['feature', 'outlier_count'])

robust_df = outlier_df 
robust_df['scaler'] = 'RobustScaler'
min_max_df = pd.DataFrame()
min_max_df['feature'] = df.drop(robust_df['feature'], axis=1).columns
min_max_df['scaler'] = 'MinMaxScaler'
scalers_df = pd.concat([robust_df, min_max_df])
scalers_df.sort_values('feature', inplace=True)
scalers_df.reset_index(inplace=True, drop=True)
scalers_df.set_index('scaler', drop=True, inplace=True)
scalers_df.drop('outlier_count', axis=1, inplace=True)

# Create the labels
df['label'] = np.where(df['LOGRET_1'].shift(-1) > 0.005, 1, 0)

# Save a copy of the unscaled data and outliers
scalers_df.to_csv('static/scalers.csv')
df.to_csv('data/unscaled_clean_data.csv')

# Split into features and labels
y = df['label']
X = df.drop('label', axis=1)

# Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False)

# Save column names
X_cols = X_train.columns
# Scaling
ct = ColumnTransformer(
    [('robust', RobustScaler(), robust_df['feature'].values),
     ('minmax', MinMaxScaler(), min_max_df['feature'].values)])
X_train_arr = ct.fit_transform(X_train)
X_test_arr = ct.transform(X_test)
X_val_arr = ct.transform(X_val)

# Save the scaled final dataframes for future use
X_train = pd.DataFrame(X_train_arr, columns=X_cols, index=X_train.index)
X_test = pd.DataFrame(X_test_arr, columns=X_cols, index=X_test.index)
X_val = pd.DataFrame(X_val_arr, columns=X_cols, index=X_val.index)
X_train.to_csv('data/train/scaled_X_train.csv')
y_train.to_csv('data/train/y_train.csv')
X_test.to_csv('data/test/scaled_X_test.csv')
y_test.to_csv('data/test/y_test.csv')
X_val.to_csv('data/val/scaled_X_val.csv')
y_val.to_csv('data/val/y_val.csv')
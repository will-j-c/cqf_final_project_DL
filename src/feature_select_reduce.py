# Import helps
from helpers import *

# Import base
import pandas as pd
import numpy as np

# Preprocessing

# Feature selection
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# tensorflow
import tensorflow as tf

# Set seeds for reproducibility
set_seeds()

# Reload the saved scaled data
X_train = pd.read_csv('data/train/scaled_X_train.csv', parse_dates=True, index_col='unix')
y_train = pd.read_csv('data/train/y_train.csv', parse_dates=True, index_col='unix')
X_test = pd.read_csv('data/test/scaled_X_test.csv', parse_dates=True, index_col='unix')
y_test = pd.read_csv('data/test/y_test.csv', parse_dates=True, index_col='unix')

# Calculate the weights for the imbalanced classes

# Reshape the data into the correct format

# Baseline model
inputs = tf.keras.Input()
x = tf.keras.layers.LSTM(16)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
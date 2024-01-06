# Import helps
from helpers import *

# Import base
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

# Preprocessing
from sklearn.pipeline import Pipeline

# Feature selection
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Dimentionality reduction
from umap import UMAP

# tensorflow
import tensorflow as tf

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
set_seeds()

# Clear any backend
tf.keras.backend.clear_session()

# Reload the saved scaled data
X_train = pd.read_csv('data/train/scaled_X_train.csv',
                      parse_dates=True, index_col='unix')
y_train = pd.read_csv('data/train/y_train.csv',
                      parse_dates=True, index_col='unix')
X_test = pd.read_csv('data/test/scaled_X_test.csv',
                     parse_dates=True, index_col='unix')
y_test = pd.read_csv('data/test/y_test.csv',
                     parse_dates=True, index_col='unix')
X_val = pd.read_csv('data/val/scaled_X_val.csv',
                    parse_dates=True, index_col='unix')
y_val = pd.read_csv('data/val/y_val.csv', parse_dates=True, index_col='unix')

# Calculate the weights for the imbalanced classes
y = pd.concat([y_train, y_val, y_test])
weights = cwts(y.values.flatten())

# Metrics
binary_accuracy = tf.keras.metrics.BinaryAccuracy()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
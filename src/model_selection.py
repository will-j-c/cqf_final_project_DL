# Import helps
from helpers import *

# Import base
import pandas as pd
import numpy as np
import sys

# Preprocessing
from sklearn.pipeline import Pipeline

# Feature selection
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# tensorflow
import tensorflow as tf

# Keras tuner
import keras_tuner

# Warnings
import warnings
warnings.filterwarnings('ignore')

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

# Run the pipeline
corr = RemoveCorPairwiseTransform()
pipe = Pipeline([('pairwisecorr', corr)], verbose=True)

X_train_pipe = pipe.fit_transform(X_train, y_train.values.ravel())
X_val_pipe = pipe.transform(X_val)

# Calculate feature length
featurelen = X_train_pipe.shape[-1]
    
# Begin defining models
# Baseline model
def baseline(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', name=f'lstm-baseline')(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-baseline')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Baseline model
def baseline_dropout(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5, name=f'lstm-baseline-dropout')(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-baseline-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Two layer model
def two_layer(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', return_sequences=True, name=f'lstm-1-twolayer')(inputs)
    x = tf.keras.layers.LSTM(units=36, activation='relu', name=f'lstm-2-twolayer')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-twolayer')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Two layer model with dropout
def two_layer_dropout(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5,return_sequences=True, name=f'lstm-1-twolayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5,name=f'lstm-2-twolayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-twolayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Three layer model
def three_layer(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', return_sequences=True, name=f'lstm-1-threelayer')(inputs)
    x = tf.keras.layers.LSTM(units=36, activation='relu', return_sequences=True, name=f'lstm-2-threelayer')(x)
    x = tf.keras.layers.LSTM(units=36, activation='relu', name=f'lstm-3-threelayer')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-threelayer')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Three mayer model with dropout
def three_layer_dropout(inputs):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5,return_sequences=True, name=f'lstm-1-threelayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5, return_sequences=True, name=f'lstm-2-threelayer-dropout')(x)
    x = tf.keras.layers.LSTM(units=36, activation='relu', dropout=0.5, recurrent_dropout=0.5,name=f'lstm-3-threelayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-threelayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Collect the models into an array

models = [[baseline, 'baseline'],
          [baseline_dropout, 'baseline_dropout'],
          [two_layer, 'two_layer'], 
          [two_layer_dropout, 'two_layer_dropout'], 
          [three_layer, 'three_layer'], 
          [three_layer_dropout, 'three_layer_dropout']]

for model_func, name in models:

    # Define the sequence length and reshape the data into the correct array
    seqlens = [1, 6, 12, 24]
    
    for seqlen in seqlens:
        # Define the tensors
        train_tensors = tf.keras.utils.timeseries_dataset_from_array(
            X_train_pipe, y_train.iloc[seqlen:], seqlen)
        val_tensors = tf.keras.utils.timeseries_dataset_from_array(
            X_val_pipe, y_val.iloc[seqlen:], seqlen)

        # Define the input
        inputs = tf.keras.Input(shape=(seqlen, featurelen))

        # Define the file paths
        filepath = f'./tensorboard/model_testing/{name}_{seqlen}_hour'
        modelpath = f'./models/model_testing/{name}_{seqlen}_hour'

        # Define the callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, monitor='val_binary_accuracy', mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir=filepath, histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='val_binary_accuracy', save_best_only=True, mode='max')]

        # Initialise the model
        model = model_func(inputs)

        # Fit the models
        model.fit(x=train_tensors, epochs=1000, validation_data=val_tensors,
            class_weight=weights, callbacks=callbacks)
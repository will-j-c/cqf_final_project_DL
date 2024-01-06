# Import helps
from src.helpers import *

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
vif = VIFTransform(threshold=5)
rf = RandomForestClassifier(n_jobs=-1, class_weight=weights)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, perc=90)
pipe = Pipeline([('vif', vif), ('boruta', boruta)], verbose=True)

X_train_pipe = pipe.fit_transform(X_train, y_train.values.ravel())
X_val_pipe = pipe.transform(X_val)

# Calculate feature length
featurelen = X_train_pipe.shape[-1]

# An extension of the keras tuner HyperModel that allows for iterating over different model functions
class IterableHyperModel(keras_tuner.HyperModel):
    def __init__(self, inputs, model_func, name=None, tunable=True):
        self.inputs = inputs
        self.model_func = model_func

    def build(self, hp):
        # Define the hyperparameters
        # Units
        units_1 = hp.Int('units_1', min_value=16, max_value=512, step=16)
        units_2 = hp.Int('units_2', min_value=16, max_value=512, step=16)
        units_3 = hp.Int('units_3', min_value=16, max_value=512, step=16)
        # Dropout rate
        dr = hp.Float('dropout_rate', min_value=0.0, max_value=0.9)
        # Learning rate
        lr = hp.Float('learning_rate', min_value=0.05, max_value=0.5)
        # Optimizer
        hp_optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'adagrad'])
        if hp_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif hp_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif hp_optimizer == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

        # Activations
        activation_1 = hp.Choice(
            'activation_1', ['relu', 'elu', 'tanh', 'sigmoid', 'selu'])
        activation_2 = hp.Choice(
            'activation_2', ['relu', 'elu', 'tanh', 'sigmoid', 'selu'])
        activation_3 = hp.Choice(
            'activation_3', ['relu', 'elu', 'tanh', 'sigmoid', 'selu'])

        model = self.model_func(self.inputs, units_1, units_2, units_3,
                                dr, lr, optimizer, activation_1, activation_2, activation_3)
        return model
    
# Begin defining models
# Baseline model
def baseline(inputs, units_1, units_2, units_3, dr, lr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units_1, activation=activation_1, name=f'lstm-baseline')(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-baseline')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Two layer model
def two_layer(inputs, units_1, units_2, units_3, dr, lr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units_1, activation=activation_1, return_sequences=True, name=f'lstm-1-twolayer')(inputs)
    x = tf.keras.layers.LSTM(units_2, activation=activation_2, name=f'lstm-2-twolayer')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-twolayer')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Two layer model with dropout
def two_layer_dropout(inputs, units_1, units_2, units_3, dr, lr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units_1, activation=activation_1, dropout=dr, recurrent_dropout=dr,return_sequences=True, name=f'lstm-1-twolayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units_2, activation=activation_2, dropout=dr, recurrent_dropout=dr,name=f'lstm-2-twolayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-twolayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Three layer model
def three_layer(inputs, units_1, units_2, units_3, dr, lr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units_1, activation=activation_1, return_sequences=True, name=f'lstm-1-threelayer')(inputs)
    x = tf.keras.layers.LSTM(units_2, activation=activation_2, return_sequences=True, name=f'lstm-2-threelayer')(x)
    x = tf.keras.layers.LSTM(units_3, activation=activation_3, name=f'lstm-3-threelayer')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-threelayer')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Three mayer model with dropout
def three_layer_dropout(inputs, units_1, units_2, units_3, dr, lr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units_1, activation=activation_1, dropout=dr, recurrent_dropout=dr,return_sequences=True, name=f'lstm-1-threelayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units_2, activation=activation_2, dropout=dr, recurrent_dropout=dr, return_sequences=True, name=f'lstm-2-threelayer-dropout')(x)
    x = tf.keras.layers.LSTM(units_3, activation=activation_3, dropout=dr, recurrent_dropout=dr,name=f'lstm-3-threelayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense-threelayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model

# Collect the models into an array

models = [[baseline, 'baseline'],
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
                patience=5, monitor='val_precision', mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir=filepath, histogram_freq=1)]

        # Initialise tuner and
        tuner = keras_tuner.Hyperband(IterableHyperModel(inputs, model_func), objective=keras_tuner.Objective(
            'val_precision', direction='max'), max_epochs=30, overwrite=True, project_name=modelpath, seed=42)

        tuner.search(train_tensors, validation_data=val_tensors,
                         callbacks=callbacks, epochs=1000, use_multiprocessing=True, workers=6)
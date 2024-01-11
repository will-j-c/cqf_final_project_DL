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

# Dimentionality reduction
from umap import UMAP

# tensorflow
import tensorflow as tf

# Keras tuner
import keras_tuner

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Clear any backend
tf.keras.backend.clear_session()

# Set seeds for reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

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
rf = RandomForestClassifier(n_jobs=-1, class_weight=weights)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, perc=90)
umap = UMAP(n_neighbors=10)
pipe = Pipeline([('pairwisecorr', corr), ('boruta', boruta), ('umap', umap)], verbose=True)

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
        # Learning rate
        lr = hp.Float('learning_rate', min_value=0.05, max_value=0.5)
        # Dropout rate
        dr = hp.Float('dropout_rate', min_value=0.01, max_value=0.8)
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

        model = self.model_func(self.inputs, units_1, units_2, units_3, dr, optimizer, activation_1, activation_2, activation_3)
        return model

# Begin defining models
# Three mayer model with dropout
def three_layer_dropout(inputs, units_1, units_2, units_3, dr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=units_1, activation=activation_1, dropout=dr, recurrent_dropout=dr,return_sequences=True, name='lstm-1-threelayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units=units_2, activation=activation_2, dropout=dr, recurrent_dropout=dr, return_sequences=True, name='lstm-2-threelayer-dropout')(x)
    x = tf.keras.layers.LSTM(units=units_3, activation=activation_3, dropout=dr, recurrent_dropout=dr,name='lstm-3-threelayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name='dense-threelayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      weighted_metrics=[binary_accuracy, precision, recall])
    return model


# Define the sequence length and reshape the data into the correct array
seqlen = 1
name = 'three_layer_dropout'

# Define the tensors
train_tensors = tf.keras.utils.timeseries_dataset_from_array(
    X_train_pipe, y_train.iloc[seqlen-1:], seqlen)
val_tensors = tf.keras.utils.timeseries_dataset_from_array(
    X_val_pipe, y_val.iloc[seqlen-1:], seqlen)

# Define the input
inputs = tf.keras.Input(shape=(seqlen, featurelen))

# Define the file paths
filepath = f'./tensorboard/model_tuning/{name}_{seqlen}_hour'
modelpath = f'./models/model_tuning/{name}_{seqlen}_hour'

# Define the callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10, monitor='val_binary_accuracy', mode='max'),
    tf.keras.callbacks.TensorBoard(log_dir=filepath, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='val_binary_accuracy', save_best_only=True, mode='max')]

# Initialise tuner and search

tuner = keras_tuner.Hyperband(IterableHyperModel(inputs, three_layer_dropout), objective=keras_tuner.Objective(
    'val_binary_accuracy', direction='max'), max_epochs=30, overwrite=True, directory=modelpath, seed=42, project_name=f'{name}_{seqlen}_hour')

tuner.search(train_tensors, validation_data=val_tensors,
             callbacks=callbacks, epochs=1000, class_weight=weights, use_multiprocessing=True, workers=6)

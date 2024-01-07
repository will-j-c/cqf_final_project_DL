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

# Define the various feature selection methods
rf = RandomForestClassifier(n_jobs=-1, class_weight=weights)
vif = VIFTransform(threshold=5)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, perc=90)
umap = UMAP(n_neighbors=10)
corr = RemoveCorPairwiseTransform()

# Define data pipelines
pipelines = [
    'none',
    Pipeline([('pairwisecorr', corr)], verbose=True),
    Pipeline([('pairwisecorr', corr), ('boruta', boruta)], verbose=True),
    Pipeline([('pairwisecorr', corr), ('boruta', boruta), ('umap', umap)], verbose=True),
    Pipeline([('vif', vif)], verbose=True),
    Pipeline([('vif', vif), ('boruta', boruta)], verbose=True),
    Pipeline([('boruta', boruta)], verbose=True),
    Pipeline([('boruta', boruta), ('umap', umap)], verbose=True),
    Pipeline([('umap', umap)], verbose=True),
    Pipeline([('vif', vif), ('umap', umap)], verbose=True)
]

for pipe in pipelines:
    # Time the run
    start = time.time()

    # Create file path for run
    filepath = './tensorboard/feature_selection/run'
    run_name = ''
    time_str = datetime.now().strftime('%m-%d-%Y-%H:%M:%S')
    if pipe == 'none':
        run_name += '_all_'
    else:
        for key in pipe.named_steps.keys():
            run_name += f'_{key}_'
    filepath += run_name
    filepath += time_str
    
    # Log the output in a log file
    modelpath = f'./models/feature_selection/model{run_name}{time_str}.keras' 
    print(f'Starting {run_name} run')
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_binary_accuracy', mode='max'),
        tf.keras.callbacks.TensorBoard(log_dir=filepath, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='val_binary_accuracy', save_best_only=True, mode='max')]
    if pipe == 'none':
        X_train_pipe = X_train.copy()
        X_val_pipe = X_val.copy()
    # Create the output of the pipeline
    # If Boruta is first (as it can't handle a df input)
    elif list(pipe.named_steps.keys())[0] == 'boruta':
        X_train_pipe = pipe.fit_transform(X_train.values, y_train.values.ravel())
        X_val_pipe = pipe.transform(X_val.values)
    else:
        X_train_pipe = pipe.fit_transform(X_train, y_train.values.ravel())
        X_val_pipe = pipe.transform(X_val)
    # Convert the output to tensors
    seqlen = 6 # Number of time steps in the past to consider, the past day
    featurelen = X_train_pipe.shape[-1]
    train_tensors = tf.keras.utils.timeseries_dataset_from_array(
        X_train_pipe, y_train.iloc[seqlen:], seqlen)
    val_tensors = tf.keras.utils.timeseries_dataset_from_array(
        X_val_pipe, y_val.iloc[seqlen:], seqlen)
    # Baseline model
    inputs = tf.keras.Input(shape=(seqlen, featurelen))
    x = tf.keras.layers.LSTM(36, activation='relu', name=f'lstm{run_name}')(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'dense{run_name}')(x)
    model = tf.keras.Model(inputs, outputs)
    # Summary
    model.summary()
    # Compile baseline classifier model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  weighted_metrics=[binary_accuracy, precision, recall])
    # Fit the models
    model.fit(x=train_tensors, epochs=1000, validation_data=val_tensors,
              class_weight=weights, callbacks=callbacks)
    # Clean up logging
    end = time.time()
    duration = '{0:.1f}'.format(end - start)
    print(f'Duration of pipeline: {duration} seconds')

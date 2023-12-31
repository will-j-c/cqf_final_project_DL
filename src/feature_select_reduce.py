# Import helps
from helpers import *

# Import base
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
X_train = pd.read_csv('data/train/scaled_X_train.csv', parse_dates=True, index_col='unix')
y_train = pd.read_csv('data/train/y_train.csv', parse_dates=True, index_col='unix')
X_test = pd.read_csv('data/test/scaled_X_test.csv', parse_dates=True, index_col='unix')
y_test = pd.read_csv('data/test/y_test.csv', parse_dates=True, index_col='unix')
X_val = pd.read_csv('data/val/scaled_X_val.csv', parse_dates=True, index_col='unix')
y_val = pd.read_csv('data/val/y_val.csv', parse_dates=True, index_col='unix')

# Calculate the weights for the imbalanced classes
y = pd.concat([y_train, y_val, y_test])
weights = cwts(y.values.flatten())

# Reshape the data into the correct format
seqlen = 1
featurelen = X_train.shape[-1]
train_tensors = tf.keras.utils.timeseries_dataset_from_array(X_train, y_train, seqlen)
val_tensors = tf.keras.utils.timeseries_dataset_from_array(X_val, y_val, seqlen)
test_tensors = tf.keras.utils.timeseries_dataset_from_array(X_test, y_test, seqlen)

# Baseline model
inputs = tf.keras.Input(shape=(seqlen, featurelen))
x = tf.keras.layers.LSTM(36)(inputs)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Compile baseline classifier model
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Define the various feature selection methods
rf = RandomForestClassifier(n_jobs=-1, class_weight=weights)
vif = VIFTransform()
boruta = BorutaPy(rf, n_estimators='auto', verbose=2)
umap = UMAP(n_neighbors=5)

# Define data pipelines
pipelines = [
    'none',
    Pipeline([('vif', vif)], verbose=True),
    Pipeline([('vif', vif), ('boruta', boruta)], verbose=True),
    Pipeline([('boruta', boruta)], verbose=True),
    Pipeline([('boruta', boruta), ('umap', umap)], verbose=True)
]

for pipe in pipelines:
    # Time the run
    start = time.time()
    # Create file path for run
    filepath = './tensorboard/feature_selection/run'
    time_str = datetime.now().strftime('%m-%d-%Y-%H:%M:%S')
    if pipe == 'none':
        filepath += '_all_'
    else:
        for key in pipe.named_steps.keys():
            filepath += f'_{key}_'
    filepath += time_str
    
    # Callbacks
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.TensorBoard(log_dir=filepath, histogram_freq=1)]
    
    if pipe == 'none':
        model.fit(x=train_tensors, epochs=100, validation_data=val_tensors,
                    class_weight=weights, callbacks=callbacks)
        
        end = time.time()
        duration = '{0:.5f}'.format(end - start)
        print(f'Duration of pipeline: {duration} seconds')
        # Continue the loop
        continue
    
    # Create the output of the pipeline
    X_train_pipe = pipe.fit_transform(X_train, y_train.values.ravel())
    X_val_pipe = pipe.transform(X_val)

    # Convert the output to tensors
    seqlen = 1
    featurelen = X_train.shape[-1]
    train_tensors = tf.keras.utils.timeseries_dataset_from_array(X_train_pipe, y_train, seqlen)
    val_tensors = tf.keras.utils.timeseries_dataset_from_array(X_val_pipe, y_val, seqlen)
    # test_tensors = tf.keras.utils.timeseries_dataset_from_array(X_test_pipe, y_test, seqlen)
    # Fit the models
    model.fit(x=train_tensors, epochs=100, validation_data=val_tensors,
                    class_weight=weights, callbacks=callbacks)
    end = time.time()
    duration = '{0:.5f}'.format(end - start)
    print(f'Duration of pipeline: {duration} seconds')
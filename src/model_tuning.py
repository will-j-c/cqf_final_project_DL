# Import helps
from helpers import *

# Import base
import pandas as pd
from datetime import datetime
import glob

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

# Tidy up all the folders first
delete_all('./tensorboard/model_tuning')
delete_all('./models/model_tuning')

# Delete the baseline model
try:
    model_to_remove = glob.glob('./models/final_model_*.keras')[0]
    os.rmdir('./models/model_tuning')
    os.unlink(model_to_remove)
except IndexError:
    pass

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

# Define the various feature selection methods
vif = VIFTransform(threshold=5)
umap = UMAP(n_neighbors=10)

pipe = Pipeline([('vif', vif), ('umap', umap)], verbose=True)

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
        dr = hp.Float('dropout_rate', min_value=0.01, max_value=0.8)
        
        # Adam optimizer
        # Learning rate
        lr = hp.Float('learning_rate', min_value=0.0005, max_value=0.)
        beta_1 = hp.Float('learning_rate', min_value=0.5, max_value=0.99)
        beta_2 = hp.Float('learning_rate', min_value=0.5, max_value=0.9999)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
        
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
def two_layer_dropout(inputs, units_1, units_2, units_3, dr, optimizer, activation_1, activation_2, activation_3):
            
    # Initialise layers
    x = tf.keras.layers.LSTM(units=units_1, activation=activation_1, dropout=dr, recurrent_dropout=dr,return_sequences=True, name='lstm-1-twolayer-dropout')(inputs)
    x = tf.keras.layers.LSTM(units=units_2, activation=activation_2, dropout=dr, recurrent_dropout=dr,name='lstm-3-twolayer-dropout')(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name='dense-twolayer-dropout')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[binary_accuracy, precision, recall])
    return model


# Define the sequence length and reshape the data into the correct array
seqlen = 1
name = 'two_layer_dropout'

# Define the tensors
train_tensors = tf.keras.utils.timeseries_dataset_from_array(
    X_train_pipe, y_train.iloc[seqlen-1:], seqlen)
val_tensors = tf.keras.utils.timeseries_dataset_from_array(
    X_val_pipe, y_val.iloc[seqlen-1:], seqlen)

# Define the input
inputs = tf.keras.Input(shape=(seqlen, featurelen))

# Define the file paths
filepath = f'./tensorboard/model_tuning/{name}_{seqlen}_hour'
project_path = f'./models/model_tuning/'

# Define the callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10, monitor='val_precision', mode='max'),
    tf.keras.callbacks.TensorBoard(log_dir=filepath)]

# Initialise tuner and search
tuner = keras_tuner.Hyperband(IterableHyperModel(inputs, two_layer_dropout), objective=keras_tuner.Objective(
    'val_precision', direction='max'), max_epochs=30, seed=42, directory=project_path, project_name=f'{name}_{seqlen}_hour')

tuner.search(train_tensors, validation_data=val_tensors,
             callbacks=callbacks, epochs=1000, class_weight=weights)

model = tuner.get_best_models(num_models=1)[0]
time_str = datetime.now().strftime('%m-%d-%Y-%H:%M:%S')
model.save(f'./models/final_model_{time_str}.keras')

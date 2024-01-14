import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import shutil
import glob
# Clear any backend
tf.keras.backend.clear_session()

# Reproducibility
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
# Transformers
class VIFTransform(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5, dropna=False):
        self.threshold = threshold
        self.dropna = dropna
        self.vif = None
        self.vif_features = None
        self.fit_transform_run = False
        
    def _calc_vif(self, X):
        vif = pd.DataFrame()
        vif["Features"] = X.columns 
        vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif.sort_values('VIF Factor', ascending=False, inplace=True)
        vif.reset_index(inplace=True, drop=True)
        return vif
    
    def fit_transform(self, X, y=0):
        X = X.copy()
        print('Calculating VIF Factors')
        self.vif = self._calc_vif(X)
        print('Calculating VIF Factors - Complete')
        # Filter the desired vif features based on the threshold
        # Drop NaN features (happens when feature set is all zeros)
        if self.dropna:
            self.vif = self.vif.dropna()
        else:
            # Assume the factor is nil
            self.vif = self.vif.fillna(0)
        self.vif_features = self.vif['Features'][(self.vif['VIF Factor']) < self.threshold].values
        # Return only the features that are greater than the threshold
        X =  X[self.vif_features]
        # Update bool for run
        self.fit_transform_run = True
        return X.values
        
    def transform(self, X):
        # Transform the data based on the 
        if self.fit_transform_run:
            X = X.copy()
            X =  X[self.vif_features]
            return X.values
        else:
            # If you haven't run fit_transform yet to the training data
            raise Exception('Please run fit_transform on training data')
        
    def fit(self, X, y=0):
        return self.fit_transform(X)
    
    def summary(self):
        vif = self.vif
        return vif

# Transformers
class RemoveCorPairwiseTransform(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.correlated_features = None
        self.fit_transform_run = False
        
    def _calc_features(self, X):
        col_corr = set()
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr
        
    def fit_transform(self, X, y=0):
        X = X.copy()
        self.correlated_features = self._calc_features(X)
        # Drop the features
        X = X.drop(self.correlated_features, axis=1)
        # Update bool for run
        self.fit_transform_run = True
        return X.values
        
    def transform(self, X):
        # Transform the data based on the 
        if self.fit_transform_run:
            X = X.copy()
            X = X.drop(self.correlated_features, axis=1)
            return X.values
        else:
            # If you haven't run fit_transform yet to the training data
            raise Exception('Please run fit_transform on training data')
        
    def fit(self, X, y=0):
        return self.fit_transform(X)
    
    def get_removed(self):
        return self.correlated_features

# Calculate class weights. Returns dict of class weights
# Credit to Kannan from Advanced ML I, CQF
def cwts(y):
    c0, c1 = np.bincount(y)
    w0=(1/c0)*(len(y))/2 
    w1=(1/c1)*(len(y))/2 
    return {0: w0, 1: w1}

# Convert unix time from seconds to ms
def convert_unix_to_ms(num):
    num = str(num)
    if len(num) != 13:
        num += '000'
    return int(num)

# Delete all files and folders in a folder
def delete_all(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Collect tensorboard max scalers into a dataframe          
def present_tensorboard_logs(stage='feature_selection'):
    # Create dict of values
    df_dict = {
        'run_name': [],
        'max_binary_accuracy': [],
        'max_binary_accuracy_epoch': [],
        'max_precision': [],
        'max_precision_epoch': [],
        'max_recall': [],
        'max_recall_epoch': [],
        'max_f1': [],
        'max_f1_epoch': []
    }

    if stage == 'feature_selection':
    # Collect the validation filepaths
        filepaths = glob.glob('./tensorboard/feature_selection/*/validation/*')
    elif stage == 'model_selection':
        filepaths = glob.glob('./tensorboard/model_testing/*/validation/*')
    elif stage == 'model_tuning':
        filepaths = glob.glob('./tensorboard/model_tuning/*/*/*/validation/*')

    # Iterate over the filepaths
    for filepath in filepaths:
        if stage == 'model_tuning':
            string_arr = run_name = filepath.split('/')
            run_name = f'{string_arr[3]}_{string_arr[4]}'
        else:
            run_name = filepath.split('/')[3]
        # Set the empty arrays
        epoch_binary_accuracy_arr = []
        epoch_precision_arr = []
        epoch_recall_arr = []
        epoch_f1_arr = []

        # Initialise tf iterator to iterate over the log file
        for event in tf.compat.v1.train.summary_iterator(filepath):
            for value in event.summary.value:
                if value.tag == 'epoch_binary_accuracy':
                    num = tf.make_ndarray(value.tensor).ravel()[0]
                    epoch_binary_accuracy_arr.append(num)
                elif value.tag == 'epoch_precision':
                    num = tf.make_ndarray(value.tensor).ravel()[0]
                    epoch_precision_arr.append(num)
                elif value.tag == 'epoch_recall':
                    num = tf.make_ndarray(value.tensor).ravel()[0]
                    epoch_recall_arr.append(num)

        # Compute max and epoch max
        epoch_binary_accuracy_max = np.max(epoch_binary_accuracy_arr)
        epoch_binary_accuracy_max_epoch = epoch_binary_accuracy_arr.index(epoch_binary_accuracy_max)
        epoch_precision_max = np.max(epoch_precision_arr)
        epoch_precision_max_epoch = epoch_precision_arr.index(epoch_precision_max)
        epoch_recall_max = np.max(epoch_recall_arr)
        epoch_recall_max_epoch = epoch_recall_arr.index(epoch_recall_max)
        
        # Compute F1 score for each epoch and select the max and the epoch it relates to
        for precision, recall in zip(epoch_precision_arr, epoch_recall_arr):
            f1_score = 2 * (precision * recall) / (precision + recall)
            if np.isnan(f1_score):
                epoch_f1_arr.append(0)
            else:
                epoch_f1_arr.append(f1_score)

        epoch_f1_max = np.max(epoch_f1_arr)
        epoch_f1_max_epoch = epoch_f1_arr.index(epoch_f1_max)
        
        # Update dict
        df_dict['run_name'].append(run_name)
        df_dict['max_binary_accuracy'].append(epoch_binary_accuracy_max)
        df_dict['max_binary_accuracy_epoch'].append(epoch_binary_accuracy_max_epoch)
        df_dict['max_precision'].append(epoch_precision_max)
        df_dict['max_precision_epoch'].append(epoch_precision_max_epoch)
        df_dict['max_recall'].append(epoch_recall_max)
        df_dict['max_recall_epoch'].append(epoch_recall_max_epoch)
        df_dict['max_f1'].append(epoch_f1_max)
        df_dict['max_f1_epoch'].append(epoch_f1_max_epoch)

    # Create dataframe
    df = pd.DataFrame.from_dict(df_dict)
    df.set_index('run_name', inplace=True)
    df.sort_values('max_f1', ascending=False, inplace=True)
    return df
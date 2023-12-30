import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Reproducibility
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
# Transformers
class VifTransform(BaseEstimator, TransformerMixin):
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
    
    def fit_transform(self, X):
        X = X.copy()
        self.vif = self._calc_vif(X)
        
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
        return X
        
    def transform(self, X):
        # Transform the data based on the 
        if self.fit_transform_run:
            X = X.copy()
            X =  X[self.vif_features]
            return X
        else:
            # If you haven't run fit_transform yet to the training data
            raise Exception('Please run fit_transform on training data')
        
    
    def summary(self):
        vif = self.vif
        return vif
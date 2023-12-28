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
    def __init__(self, threshold=5):
        self.threshold = threshold
        
    def _calc_vif(self, X):
        vif = pd.DataFrame()
        vif["Features"] = X.columns 
        vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        return vif
        
    def transform(self, X):
        return X
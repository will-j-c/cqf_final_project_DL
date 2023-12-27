import random
# import tensorflow as tf
import numpy as np

# Reproducibility
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    

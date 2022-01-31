import numpy as np
from sklearn.datasets import make_blobs
from tensorflow.keras.utils import to_categorical

def load_sample(n, p, std):

    c_1 = np.ones(p)
    c_2 = - c_1.copy()
    centers = [list(c_1), list(c_2)]
    x, y = make_blobs(n_samples=n, 
                      n_features=p, 
                      centers=centers, 
                      cluster_std=std, 
                      shuffle=True)
    y = to_categorical(y)
    return x, y
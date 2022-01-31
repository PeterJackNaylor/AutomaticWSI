import sys
import numpy as np
from simulation import load_sample

n_train = 224
n_val = 56
n_test = 5000
p = 256
std = float(sys.argv[1])

X_train, y_train = load_sample(n_train, p, std)
np.savez("Xy_train.npz", X=X_train, Y=y_train)

X_val, y_val = load_sample(n_val, p, std)
np.savez("Xy_val.npz", X=X_val, Y=y_val)

X_test, y_test = load_sample(n_test, p, std)
np.savez("Xy_test.npz", X=X_test, Y=y_test)
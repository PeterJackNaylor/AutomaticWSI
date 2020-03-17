
import numpy as np

from glob import glob
from tqdm import tqdm

means = []
sizes = []
for file in tqdm(glob('*.npy')):
    tmp = np.load(file)
    tmp = tmp.mean(axis=0)
    means.append(tmp)
    sizes.append(len(tmp))

means = np.array(means)
sizes = np.array(means)

mean = means * sizes / sizes.sum()
np.save('mean.npy', mean)
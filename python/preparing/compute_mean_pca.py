
import numpy as np

from glob import glob
from tqdm import tqdm

res = []
shapes = []
all_files = glob('*.npy')
for file in tqdm(all_files):
    tmp = np.load(file)
    shapes.append(tmp.shape[0])
    tmp = tmp.mean(axis=0)
    res.append(tmp)
n_tot = np.sum(shapes)
n = len(all_files)
for i in range(n):
    res[i] = shapes[i] * n / n_tot * res[i]
res_npy = np.vstack(res).mean(axis=0)
np.save('mean.npy', res_npy)
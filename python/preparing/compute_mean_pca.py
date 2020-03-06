
import numpy as np

from glob import glob
from tqdm import tqdm

res = []
for file in tqdm(glob('*.npy')):
    tmp = np.load(file)
    tmp = tmp.mean(axis=0)
    res.append(tmp)

res_npy = np.vstack(res).mean(axis=0)
np.save('mean.npy', res_npy)
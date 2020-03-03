#%% code written by tristan lazard
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
from joblib import load, dump
import seaborn as sns
 
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default=".", help="path to the files of tiles")
    args = parser.parse_args()
    files = glob(args.path)
    ipca = IncrementalPCA()

    for path in tqdm(files[0:5]):
        mat = np.load(path)
        ipca.partial_fit(X=mat)

    msg = " ----------------  RESULTS -------------------- \n"
    for i,o in enumerate(ipca.explained_variance_ratio_, 1):
        msg += "Dimension {} explains {}% of the variance \n".format(i, o*100)
    msg += "----------------------------------------------"

    ## Saving
    with open('results.txt', 'w') as f:
        f.write(msg)

    dump(ipca, 'pca_tiles.joblib')

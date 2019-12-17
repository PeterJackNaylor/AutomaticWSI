
import numpy as np
import h5py

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='image format')
    parser.add_argument('--method', required=True,
                        metavar="str", type=str,
                        help='method to use')
    parser.add_argument('--nber_ds', required=True,
                        metavar="str", type=int,
                        help='number of patches per tissue')
    parser.add_argument('--npy', required=True,
                        metavar="str", type=str,
                        help='npy file for tissue')
    args = parser.parse_args()
    return args 

def load(path):
    """
    Loads a tissue bag.
    Parameters
    ----------
    path: string, 
        path related to numpy array object
    Returns
    -------
    A numpy array where each line is en encoded bag of the tissue
    """
    mat = np.load(path)
    return mat

def subsample(npy, n, method="uniform", weight=False):
    """
    Subsampling function. Takes in input a numpy array and returns 
    a downsampled numpy array of size n.
    It can use a number of methods.
    Parameters
    ----------
    npy: numpy array, 
        2D numpy array where each line is an encoded tile
    n: int,
        maximum number of samples to keep.
    method: string,
        Can be several: 
            - 'uniform', uniform downsampling
            - 'kmeans', kmeans downsampling for a partition
                view of the tissue.
    weight: bool,
        Whether to weight the each sample in the downsampled
        bag.
    Returns
    -------
    A numpy array corresponding to the downsampled bag given
    in input.
    """
    ns = npy.shape[0]
    if method == "uniform":
        size = min(n, ns)
        index = np.random.choice(range(ns), size=size, replace=False)
        
    elif method == "kmeans":


        div = 40 if ns > 40 else 1
        init_size_div = 20 if ns > 40 else 1

        model = MiniBatchKMeans(n_clusters=ns // div, init_size=ns // init_size_div)
        #model = KMeans(n_clusters=n // 40, n_jobs=8)

        x = npy.copy()
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        y_npy = model.fit_predict(x)

        samples_per_cluster = max(n // (ns // div), 1)
        kept_samples = []
        weights = []

        for cluster in range(ns // div):
            x_coord = np.where(y_npy == cluster)[0]
            n_c = len(x_coord)
            if n_c != 0:
                size = min(samples_per_cluster, n_c)
                subindex = np.random.choice(x_coord, size=size, replace=False)

                kept_samples.append(subindex)

                w_scaler = n_c / size
                weights.append(np.zeros(size) + w_scaler)
        index = np.concatenate(kept_samples)
        weights = np.concatenate(weights)
    smaller_npy = npy[index]

    if weight:
        nsp, psp = smaller_npy.shape
        new_smaller_npy = np.zeros((nsp, psp + 1))
        new_smaller_npy[:, :-1] = smaller_npy
        new_smaller_npy[:, -1] = weights
        smaller_npy = new_smaller_npy
    return smaller_npy

def main():
    
    options = get_options()
    print('Loading data, might be sloww...')
    npy_array = load(options.npy)
    print('See I managed to load it :D \n Starting subsampling wish me luck! ;-)')
    weight = False

    if options.method == "wkmeans":
        weight = True
        options.method = "kmeans"
    # try:
    sub_npy_array = subsample(npy_array, 
                              options.nber_ds, 
                              options.method, weight)
    # except OverflowError as error:
    #     print('oh well... overflow error, mostly something to do with the matrice size.')
    #     if weight:
    #         subh5_array = np.hstack([h5_array, np.ones((h5_array.shape[0], 1))])
    #     else:
    #         subh5_array = h5_array
    # print('Oh well, not much left to do... \n What to do what to do... SAVING!')
    
    output_name =  options.npy.replace('.npy', '_ds.npy')
    np.save(output_name, sub_npy_array)

if __name__ == '__main__':
    main()

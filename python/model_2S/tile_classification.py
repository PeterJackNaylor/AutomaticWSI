
import os
import pickle

from glob import glob
from time import time
from tqdm import tqdm

import numpy as np
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='dispatches a set of files into folds')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='path to npy files')
    parser.add_argument('--clustering_method', required=True,
                        metavar="str", type=str,
                        help='clustering method')
    parser.add_argument('--n_c', required=True,
                        metavar="str", type=int,
                        help='number of clusters')  
    parser.add_argument('--seed', required=False, default=42, 
                        metavar="str", type=int,
                        help='seed')
    parser.add_argument('--cpus', required=False, default=1,
                        metavar="str", type=int,
                        help='number of available cpus') 
    args = parser.parse_args()
    return args


def load_all_patients_ds(path):
    """
    Loading all patient downsample numpy array corresponding
    to their encoded bag of features.
    Parameters
    ----------
    path: string, 
        path to the folder containing all the numpy array of every patients
    Returns
    -------
    A 2D numpy array consisting of the concatenated bag of tissue tiles.
    """
    tissues = glob(os.path.join(path,'*.npy'))
    feat_list = []
    for tissue in tqdm(tissues):
        if "_ds" in tissue:
            tissue_feat = np.load(tissue)
        feat_list.append(tissue_feat)
    data_feat = np.concatenate(feat_list)
    return data_feat

def tile_cluster(mat, nclass, method="KMeans",
                 seed=None, cpus=1, scaler=True):
    """
    Creates a tile classification model in a unsupervised manner.

    Parameters
    ----------
    mat: 2D numpy array, 
        Concatenated bag of encoded tissue tiles of all tissues.
    nclass: int, 
        number of classes to cluster to.
    method: string, 
        method name, indication the unsupervised method to apply between:
            - "KMeans"
            - "UMAP"
    seed: int, 
        fixed seed.
    cpus: int, 
        number of cpus to use during training.
    scaler: bool, 
        Whether to mean substract and scale.
    Returns
    -------
    A model to perform clustering and to infer.
    """
    begin_time = time()
    models = []

    print("Creating cluster...")
    if scaler:
        scaler_f = scaler_function(mat)
        mat = scaler_f.transform(mat)
        models.append(scaler_f)
    else:
        scaler_f = None

    if method == "KMeans":
        model = KMeans(n_clusters=nclass, random_state=seed, n_jobs=cpus).fit(mat)
    elif method == "UMAP":
        model = umap_model(2, nclass, seed, cpus)
        model.fit(mat)        
    else:
        raise ValueError('Clustering method -- {} --  unknow'.format(method))

    models.append(model)

    def pred_function(X):
        if scaler:
            X = scaler_f.transform(X)
        return model.predict(X)
    
    print("Model finished training.")
    elapsed_time = time() - begin_time
    print("\t%02i:%02i:%02i" % (elapsed_time / 3600, (elapsed_time % 3600) / 60, elapsed_time % 60))
    return models, pred_function

def scaler_function(data):
    """
    Parameters
    ----------
    data: 2D numpy array.
    Returns
    -------
    A fitted scaler function.
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

class umap_model:
    """
    UMAP model object. Performs a UMAP unsupervised clustering
    where we project the data with a UMAP and then cluster the 
    resulting projection with K-Means.
    ----------
    data: 2D numpy array.
    n_components: int,
        number of axes to project the data to.
    n_clusters: int,
        number of clusters for the unsupervised method
    seed: int,
    cpu: int,
        number of cpus to use during training.
    Returns
    -------
    A UMAP object which can fit, predict and plot results.
    """
    def __init__(self, n_components, n_clusters, seed, cpu):
        print("Reducing p down to 50.")
        self.pca = PCA(n_components=50)
        self.umap = umap.UMAP(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=cpu)

    def fit(self, table):
        self.pca.fit(table)
        pca_table = self.pca.transform(table)
        self.umap.fit(pca_table)
        umap_res = self.umap.transform(pca_table)
        self.kmeans.fit(umap_res)

    def predict(self, table):
        pca_table = self.pca.transform(table)
        umap_res = self.umap.transform(pca_table)
        kmeans_res = self.kmeans.predict(umap_res)
        return kmeans_res

    def plot(self, table):

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pylab as plt
        import pandas as pd
        import seaborn as sns

        table = self.pca.transform(table)
        umap_table = self.umap.transform(table)
        cluster = self.kmeans.predict(umap_table)
        import pdb; pdb.set_trace()
        to_plot = pd.DataFrame(umap_table)
        to_plot["hue"] = cluster
        sns.pairplot(to_plot, hue="hue")
        plt.savefig("test" + ".png")

def create_features_for_all_tissue(tissues, func, nclass):
    """
    Projects every tiles into each class and computes
    the average of vector Z_i = mean(one_hot_encoding_class_j) 
    where j goes from 1 to the number of tiles of patient i.
    ----------
    tissues: list of strings,
        each element is a path to a patient's tissue.
    func: function,
        tile class assignement.
    n_c: int,
        number of clusters for the unsupervised method
    seed: int,
    Returns
    -------
    A 2D numpy array, 'vector' of Z_i describing the tissue profiles
    for each patient.
    """

    z_i = []
    order = []
    for tissue in tissues:
        mat = np.load(tissue)
        tile_assigned = func(mat)
        onehot_yji = one_hot(tile_assigned, nclass)
        zi = np.mean(onehot_yji, axis=0)
        z_i.append(zi)
        order.append(os.path.basename(tissue).replace(".npy", ""))
    result = np.concatenate(z_i, axis=0)
    return result, order

def one_hot(a, nclass):
    b = np.zeros((a.size, nclass))
    b[np.arange(a.size),a] = 1
    return b

def main():
    
    options = get_options()

    #load downsampled version of patients 
    all_tissue_ds = load_all_patients_ds(options.path)

    # create tile clusters
    mod, pred_function = tile_cluster(all_tissue_ds, options.n_c, 
                                      method=options.clustering_method,
                                      seed=options.seed, cpus=options.cpus, 
                                      scaler=True)
    
    # create Z_i by transforming each tissue tile into the one_hot_encoding
    # class assignement followed by taking the average.
    all_tissue = [f for f in glob(os.path.join(options.path,'*.npy')) if "_ds" not in f]
    tissue_profiles, order = create_features_for_all_tissue(all_tissue, pred_function,
                                                            options.n_c)
    # saving
    np.save("tissue_zi.npy", tissue_profiles)

    check_or_create("models")
    if len(mod) == 2:
        name_scale = "models/scaler.pickle"
        pickle.dump(mod[0], open(name_scale, 'wb'))
        del mod[0]
    
    name_model = "models/{}.pickle".format(options.clustering_method)
    pickle.dump(mod[0], open(name_model, 'wb'))
    name_model = "order_zi.pickle"
    pickle.dump(order, open(name_model, 'wb'))

if __name__ == '__main__':
    main()

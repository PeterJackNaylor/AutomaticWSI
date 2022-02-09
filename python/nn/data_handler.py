import h5py
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.array as da
import os
import ipdb
import sys
import threading

from keras.utils import to_categorical
from keras.utils import Sequence

from sklearn.model_selection import StratifiedKFold


def load(img_path, input_depth):
    """[summary]
    
    Parameters
    ----------
    img_path : str 
        path to the image (= one patient) to load. Must be h5 files of npy files (results of the Tiling Encoding process)
    input_depth : int
        number of variable of the encoding space to use.
    
    Returns
    -------
    dask.array
        dask array containing a whole encoded slide.
   
    Returns
    -------
    [type]
        [description]
    """

    
    if img_path.endswith('.npy'):
        img = np.load(img_path)[:,:input_depth]
        img = img.astype('float32')
    elif img_path.endswith('.h5'):
        h5f = h5py.File(img_path, 'r')
        img = h5f['dataset_1'][:, :input_depth]
        h5f.close()
    else:
        raise TypeError("The format of the input images must be either h5 or npy.")
    #mat = da.from_array(img, chunks=("auto", -1))
    mat = img
    return mat

def parse_table_name(name):
    return name.name.split('__')[-1].split('.')[0]

def add_index(table, index):
    """ Adds the start and end index (in the concatenated object of the dataset) to results_table.
        Harmonize the name of the files between table and the `*.h5` names (may change).
    
    Parameters
    ----------
    table : pd.DataFrame
        Results dataframe containing the information about each patients files.
    index : dict
        maps a file name (`"ny__$number.h5"`) to an entry in `table` ($number)
    
    Returns
    -------
    pd.DataFrame
        dataframe table with two new columns, start and ends, corresponding to each index[file].
    """
    index_table = pd.DataFrame.from_dict(index).T
    index_table.columns = ["start", "end"]
    index_table["Biopsy"] = index_table.apply(lambda x: x.name.split('.')[0].split('/')[-1], axis=1)
    table = pd.merge(table, index_table, on='Biopsy')
    return table

def set_table(table, fold_test, inner_number_folds, index_table, y_name):
    """ Set the table containing the data information

    Set the table by adding to each entry (patient) its start and end indexes in the concatenated data object.
    In fact each patients i is composed by `n_i` tiles so that for example patient 0 will have as starts and ends indices 0 and `n_0`.
    It then separates the dataset into test and train sets (according to `fold_test`).
    Finally, several splits of the train sets are done for cross validation, preserving relative class frequency.
    
    Obviously, dataset is shuffled, and splitted at the patient level, so that the indexes returned are the table indexes,
    not the concatenated object indexes.

    Parameters
    ----------
    table : pd.DataFrame
        data information.
    fold_test : int
        number of the fold which will be used for testing.
    inner_number_folds : int
        number of splits used in the cross validation.
    index_table : dict
        maps each file (key) to its start and end index in the data object (concatenated encoded bags)
    y_name : str
        or "y_interest", is the name of the target variable.
    
    Returns
    -------
    pd.DataFrame, list(tuple), list
        returns 1: the table DataFrame augmented with start and end indexes
                2: The `inner_number_folds` splits for cross_validation, each containing (list(train_indexes), list(val_indexes)).
                3: List containing indexes of the test dataset.
    """
    ## add index_table to table so that all the info is in table 
    table = add_index(table, index_table)
    train_table = table[table["fold"] != fold_test]
    test_index = table[table["fold"] == fold_test].index
    stratified_variable = train_table[y_name].round(0)
    skf = StratifiedKFold(n_splits=inner_number_folds, shuffle=True) # Assures that relative class frequency is preserve in each folds.
    obj = skf.split(train_table.index, stratified_variable)
    # index_folds = [(train_index, val_index) for train_index, val_index in obj]
    index_folds = [(np.array(train_table.index[train_index]), np.array(train_table.index[val_index])) for train_index, val_index in obj]
    # import pdb; pdb.set_trace()
    return table, index_folds, test_index



def load_concatenated_data(files, depth, mean):
    """ Loads a concatenated matrix from a list of files.

    Loads the matrix of the encoded WSI of files in a single matrix of size NxM with N the total number of tiles
    of all the WSI and M the choosen depth.
    
    Parameters
    ----------
    files : list
        list of path to the WSI encoded files (npy or h5, h5 for the moment)
    depth : int
        number of variables of the encoded tiles to keep.
    mean : np.array
        mean matrix for normalisation
    
    Returns
    -------
    np.array, dict
        1: concatenated matrix of all the tiles
        2: index_file dictionnary, indicating start and end indexes of each patients in the concatenated matrix.
    """
    index_file = {}
    list_table = {} 
    last_index = 0
    for f in tqdm(files):
        mat = load(f, depth) - mean[:depth]
        n_i = mat.shape[0]
        index_file[f] = [last_index, last_index + n_i]
        last_index += n_i
        list_table[f] = mat

    # I removed np concatenate, so that the following code is better optimised in RAM
    n, p = mat.shape
    data = np.zeros(shape=(last_index, p), dtype='float32')
    for f in tqdm(files):
        b_idx, l_idx = index_file[f]
        data[b_idx:l_idx] = list_table[f]
        del list_table[f]

    return data, index_file
    
def file_is_in_labels(files, table):
    """ Returns the list of files that are also keys of the table (in the labels.csv file)
    
    Parameters
    ----------
    files : list[str]
        list to the path of the files.
    table : pd.DataFrame
        names of files available in table (labels.csv)
    
    Returns
    -------
    list
        list of path.
    """
    names = []
    for f in files:
        name = os.path.basename(f).split('.')[0]
        names.append(name)
    files = zip(names, files)
    labels_name = set(table['Biopsy'])
    filtered_files = [x[1] for x in files if x[0] in labels_name]
    return filtered_files


class data_handler:
    """Instantiate a data_handler Class. Creates iterators with the chosen characteristics with the functions
    dg_train, dg_val, dg_test.
    In particular, this class allows the histological data to only be loaded once and by one object.
    In particular, once loaded, you can access different training, validation and test generators
    for a subsequent neural network training with custom data.
    """
    def __init__(self, path, fold_test, table_name, 
                 inner_fold, batch_size, 
                 mean, options):
        files = glob(path)
        depth = options.input_depth
        mean = np.load(mean).astype("float32")
        table = pd.read_csv(table_name)
        files = file_is_in_labels(files, table)
        data, index_file = load_concatenated_data(files=files, depth=depth, mean=mean)

        self.data = data
        table, obj, test_index = set_table(table, fold_test, 
                                           inner_fold, 
                                           index_file, options.y_interest)
        self.table = table
        self.obj = obj
        self.batch_size = batch_size
        self.test_index = test_index
        self.size = options.size
        self.y_interest = options.y_interest
        

    def dg_train(self, fold):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. In particular, we specify the fold value
        in order to know which fold to remove from the training set.
        Parameters
        ----------
        fold : int
            fold to remove from training set
        Returns
        -------
        h5_Sequencer from the Keras package,
            training data generator for a Keras neural network module.
        """
        train_index = self.obj[fold][0] 
        return h5_Sequencer(train_index, 
                            self.table, 
                            self.data, 
                            self.batch_size, 
                            self.size, 
                            y_name=self.y_interest)

    def dg_val(self, fold):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. In particular, we specify the fold value
        in order to know which fold to generate as a validation set.
        Parameters
        ----------
        fold : int
            fold to return for the validation set
        Returns
        -------
        h5_Sequencer from the Keras package,
            validation data generator for a Keras neural network module.
        """
        val_index = self.obj[fold][1]
        return h5_Sequencer(val_index, 
                            self.table, 
                            self.data, 
                            self.batch_size,
                            self.size,  
                            y_name=self.y_interest)

    def dg_test(self):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. This module returns the data generator for
        the test set.
        Returns
        -------
        h5_Sequencer from the Keras package,
            testing data generator for a Keras neural network module.
        """
        return h5_Sequencer(self.test_index, 
                            self.table, 
                            self.data, 
                            1, 
                            self.size, 
                            shuffle=False,
                            y_name=self.y_interest)

    def dg_test_index_table(self):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. This module returns the data generator for
        the test set.
        Returns
        -------
        A tuple of three elements:
            - h5_Sequencer from the Keras package,
                testing data generator for a Keras neural network module.
            - list[str] representing the test index from the original table
            - pd.DataFrame, the original table
        """
        return h5_Sequencer(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            y_name=self.y_interest), self.test_index, self.table
    def weights(self):
        """
        Returns a dictionnary corresponding to the weights
        given to each class in order to weight the contribution
        of each separate class. Does not work for fully convolutional
        layers.
        """
        n_0 = self.table[self.table[self.y_interest] == 0].shape[0]
        n_1 = self.table[self.table[self.y_interest] == 1].shape[0]
        p_0 = n_0 / (n_0 + n_1)
        p_1 = n_1 / (n_0 + n_1)
        scaler = 1 / min(p_0, p_1)
        dic_weights = {0: p_0 * scaler, 1: p_1 * scaler}
        return dic_weights

def sample(start, end, size):
    return np.sort(np.random.randint(start, high=end, size=size))

def pj_fetch(name, table, size):
    """Samples the tiles indexes for a slice.

    If the slide has a number of tiles different than slide, then we sample uniformly `size` tiles
    among the available tiles. --> Why not sample all the cases in a Uniform ?
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!                                                                   !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!      Changer name par index                                       !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!                                                                   !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters
    ----------
    name : int
        index of the patient in the table.
    table : pdf.DataFrame
        description table of the dataset
    size : int 
        fixed number of tiles per slide
    
    Returns
    -------
    list
        list of indexes corresponding to the tiles, in the concatenated data object.
    """

    start = table.iloc[name]["start"]
    end = table.iloc[name]["end"]
    if end - start < size:
        slice_index = sample(start, end, size)
    elif (end - start) == size:
        slice_index = range(start, end)
    else:
        slice_index = sample(start, end, size)

    return slice_index

class h5_Sequencer(Sequence):

    def __init__(self, index_patient, table, data, batch_size=1, size=10000, shuffle=True, y_name="RCB_class"):
        n_classes = len(np.unique(table[y_name]))
        if shuffle:
            np.random.shuffle(index_patient)
        self.index_patient = index_patient
        self.y_name = y_name
        self.table = table
        self.y_onehot = to_categorical(self.return_labels(), n_classes)

        self.n_size = len(self.index_patient)
        self.batch_size = batch_size
        self.size = size
        self.data = data
        print("Initializing generator", flush=True)
        self.lock = threading.Lock()   #Set self.lock
        
    def return_labels(self):
        return np.array(self.table.iloc[self.index_patient][self.y_name])

    def __len__(self):
        return self.n_size // self.batch_size

    def __getitem__(self, idx):
        """Samples a batch
        
        Each bacth is the aggregation of `batch_size` number of patients, each patients being `size` tiles.

        Parameters
        ----------
        idx : int
            index of the sample
        
        Returns
        -------
        np.array, np.array
            1: return_1[i] is an array containing the `size` encoded tiles of patient `i`
            2: return_2[i] is an array containing the `size` number of labels of patient `i`
        """

        with self.lock:                #Use self.lock
            batch_x = self.index_patient[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y_onehot[idx * self.batch_size:(idx + 1) * self.batch_size]

            def f(name, table, data, size):
                index = pj_fetch(name, table, size)
                npy_array = data[index]
                return npy_array

            list_f_x = [f(name, self.table, self.data, self.size) for name in batch_x]
            batch_x = np.array(list_f_x).astype(float)
            batch_y = np.array(batch_y)
            return (batch_x, batch_y)


def fill_table(row, ar=None, y=None):
    start = int(row["start"])
    end = int(row["end"])
    ind = row.name
    ar[start:end] = np.tile(y[ind], (end-start,1))

class h5_Sequencer_HL(Sequence):

    def __init__(self, index_patient, table, data, batch_size=1, size=10000, shuffle=True, y_name="RCB_class"):
        n_classes = len(np.unique(table[y_name]))
        if shuffle:
            np.random.shuffle(index_patient)
        self.index_patient = index_patient
        self.y_name = y_name
        self.table = table
        self.y_onehot = to_categorical(self.return_biop_labels(), n_classes)

        self.n_size = len(self.index_patient)
        self.batch_size = batch_size
        self.size = size
        self.data = data
        print("check patient index")
        new_indices = []
        corresp_samples = []
        new_y = []
        for i, pat in enumerate(list(index_patient)):
            start = self.table.loc[pat, "start"]
            end = self.table.loc[pat, "end"]
            new_indices.append(np.array(range(start,end)))
            corresp_samples.append(np.repeat(pat, end-start))
            new_y.append(np.tile(self.y_onehot[i], (end-start,1)))

        self.new_indices = np.concatenate(new_indices)
        self.corresp_samples = np.concatenate(corresp_samples)
        self.new_x = self.data[self.new_indices]

        self.new_y = np.vstack(new_y)
        self.n_size = self.new_x.shape[0]
        if shuffle:
            idx = np.array(range(self.n_size))
            np.random.shuffle(idx)
            self.new_x = self.new_x[idx]
            self.new_y = self.new_y[idx]
            self.corresp_samples = self.corresp_samples[idx]
        print("Initializing generator", flush=True)
        self.lock = threading.Lock()   #Set self.lock
        
    def return_labels(self):
        return self.new_y
    def return_biop_labels(self):
        return np.array(self.table.iloc[self.index_patient][self.y_name]
    def __len__(self):
        return self.n_size // self.batch_size

    def __getitem__(self, idx):
        """Samples a batch
        
        Each bacth is the aggregation of `batch_size` number of patients, each patients being `size` tiles.

        Parameters
        ----------
        idx : int
            index of the sample
        
        Returns
        -------
        np.array, np.array
            1: return_1[i] is an array containing the `size` encoded tiles of patient `i`
            2: return_2[i] is an array containing the `size` number of labels of patient `i`
        """

        with self.lock:                #Use self.lock
            batch_x = self.new_x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.new_y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return (batch_x, batch_y)

class data_handler_hardlabel(data_handler):

    def dg_train(self, fold):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. In particular, we specify the fold value
        in order to know which fold to remove from the training set.
        Parameters
        ----------
        fold : int
            fold to remove from training set
        Returns
        -------
        h5_Sequencer from the Keras package,
            training data generator for a Keras neural network module.
        """
        train_index = self.obj[fold][0] 
        return h5_Sequencer_HL(train_index, 
                            self.table, 
                            self.data, 
                            self.batch_size, 
                            self.size, 
                            y_name=self.y_interest)

    def dg_val(self, fold):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. In particular, we specify the fold value
        in order to know which fold to generate as a validation set.
        Parameters
        ----------
        fold : int
            fold to return for the validation set
        Returns
        -------
        h5_Sequencer from the Keras package,
            validation data generator for a Keras neural network module.
        """
        val_index = self.obj[fold][1]
        return h5_Sequencer_HL(val_index, 
                            self.table, 
                            self.data, 
                            self.batch_size,
                            self.size,  
                            y_name=self.y_interest)

    def dg_test(self):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. This module returns the data generator for
        the test set.
        Returns
        -------
        h5_Sequencer from the Keras package,
            testing data generator for a Keras neural network module.
        """
        return h5_Sequencer_HL(self.test_index, 
                            self.table, 
                            self.data, 
                            1, 
                            self.size, 
                            shuffle=False,
                            y_name=self.y_interest)

    def dg_test_index_table(self):
        """
        Returns a h5_sequencer following the object template given
        by Keras in order to feed a neural network with a custom
        data generator. This module returns the data generator for
        the test set.
        Returns
        -------
        A tuple of three elements:
            - h5_Sequencer from the Keras package,
                testing data generator for a Keras neural network module.
            - list[str] representing the test index from the original table
            - pd.DataFrame, the original table
        """
        dg = h5_Sequencer_HL(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            y_name=self.y_interest)
        dg = h5_Sequencer_HL(self.test_index, self.table, self.data, 
                            dg.n_size, self.size, shuffle=False,
                            y_name=self.y_interest)
        return dg, dg.new_indices, self.table
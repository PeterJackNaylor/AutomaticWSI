import h5py
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.array as da
import os
import ipdb
import sys

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

def label_mapping(string):
    group = None
    if string == "pCR":
        group = 0
    elif string == "RCB":
        group = 1
    return group

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
        or "y_variable", is the name of the target variable.
    
    Returns
    -------
    pd.DataFrame, list(tuple), list
        returns 1: the table DataFrame augmented with start and end indexes
                2: The `inner_number_folds` splits for cross_validation, each containing (list(train_indexes), list(val_indexes)).
                3: List containing indexes of the test dataset.
    """
    ## add index_table to table so that all the info is in table 
    if y_name == "RCB_class":
        table[y_name] = list(map(label_mapping, table[y_name]))
    table = add_index(table, index_table)
    train_table = table[table["fold"] != fold_test]
    test_index = table[table["fold"] == fold_test].index
    stratified_variable = train_table[y_name].round(0)
    skf = StratifiedKFold(n_splits=inner_number_folds, shuffle=True) # Assures that relative class frequency is preserve in each folds.
    obj = skf.split(train_table.index, stratified_variable)
    index_folds = [(train_index, val_index) for train_index, val_index in obj]
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
    list_table = []
    last_index = 0
    for f in tqdm(files):
        mat = load(f, depth) - mean
        n_i = mat.shape[0]
        # index_table[f] = mat
        index_file[f] = [last_index, last_index + n_i]
        last_index += n_i
        list_table.append(mat)
    data = np.concatenate(list_table, axis=0)
    return data, index_file
    
def file_is_in_labels(files, table):
    """ Returns the lsit of files that are also keys of the table (in the labels.csv file)
    
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
    """Instanciate a data_handler Class. Creates iterators with the choosen characteristics with the functions
    dg_train, dg_val, dg_test.
    """
    def __init__(self, path, fold_test, table_name, 
                 inner_cross_validation_number, batch_size, 
                 mean, options):
        files = glob(path)
        depth = options.input_depth
        mean = np.load(mean)
        table = pd.read_csv(table_name)
        files = file_is_in_labels(files, table)
        data, index_file = load_concatenated_data(files=files, depth=depth, mean=mean)
        self.data = data
        table, obj, test_index = set_table(table, fold_test, 
                                           inner_cross_validation_number, 
                                           index_file, options.y_variable)
        self.table = table
        self.obj = obj
        self.batch_size = batch_size
        self.test_index = test_index
        self.size = options.size
        self.y_variable = options.y_variable
       # if self.y_variable in ["RCB_score", "ee_grade"]:
       #     self.categorical = "categorical"
       # elif self.y_variable in ["stroma", "til"]:
       #     self.table[self.y_variable] = self.table[self.y_variable] / 100
       #     self.categorical = "regression_ce"
       # else:
       #     self.categorical = "regression"

    def dg_train(self, sub_fold_val):
        """
        """
        
        train_index = self.obj[sub_fold_val][0] 
        return h5_Sequencer(train_index, self.table, self.data, self.batch_size, 
                            self.size, 
                            y_name=self.y_variable)
    def dg_val(self, sub_fold_val):
        val_index = self.obj[sub_fold_val][1]
        return h5_Sequencer(val_index, self.table, self.data, 1, 
                            self.size,  
                            y_name=self.y_variable)

    def dg_test(self):
        return h5_Sequencer(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            y_name=self.y_variable)
    def dg_test_index_table(self):
        return h5_Sequencer(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            y_name=self.y_variable), self.test_index, self.table
    def weights(self):
        if self.y_variable == "RCB_class":
            n_0 = self.table[self.table["RCB_class"] == 0].shape[0]
            n_1 = self.table[self.table["RCB_class"] == 1].shape[0]
            p_0 = n_0 / (n_0 + n_1)
            p_1 = n_1 / (n_0 + n_1)
            scale = 1 / min(p_0, p_1)
            dic_weights = {0: p_0 * scale, 1: p_1 * scale}
            return dic_weights
        else:
            return None

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
        n_classes = len(np.unique(table["RCB_class"]))
        if shuffle:
            np.random.shuffle(index_patient)
        self.index_patient = index_patient
       # if categorical == "categorical":
       #     self.y_onehot = to_categorical(table.ix[self.index_patient, y_name], 
       #                                    num_classes=n_classes)
       # elif categorical == "regression_ce":
       #     shape = (table.ix[self.index_patient, y_name].shape[0], 2)
       #     self.y_onehot = np.zeros(shape=shape)
       #     self.y_onehot[:, 0] = table.ix[self.index_patient, y_name]
       #     self.y_onehot[:, 1] = 1 - self.y_onehot[:, 0]
       # else:
        self.y_onehot = np.array(table.iloc[self.index_patient][y_name])
        self.n_size = len(self.index_patient)
        self.batch_size = batch_size
        self.size = size
        self.data = data
        self.table = table
        print("Initializing generator", flush=True)

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
        batch_x = self.index_patient[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_onehot[idx * self.batch_size:(idx + 1) * self.batch_size]

        def f(name, table=self.table, size=self.size):
            data = self.data
            index = pj_fetch(name, table, size)
            npy_array = data[index]
            return npy_array

        list_f_x = [f(name) for name in batch_x]
        batch_x = np.array(list_f_x).astype(float)
        batch_y = np.array(batch_y)

        return (batch_x, batch_y)







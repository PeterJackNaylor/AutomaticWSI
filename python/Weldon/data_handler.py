import h5py
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.array as da

from keras.utils import to_categorical
from keras.utils import Sequence

from sklearn.model_selection import StratifiedKFold


def load(img_path, input_depth):
    h5f = h5py.File(img_path, 'r')
    x = h5f['dataset_1'][:, :input_depth]
    mat = da.from_array(x, chunks=("auto", -1))
    h5f.close()
    return mat

def parse_table_name(name):
    return int(name.name.split('__')[-1].split('.')[0])

def add_index(table, index):
    index_table = pd.DataFrame.from_dict(index).T
    index_table.columns = ["start", "end"]
    index_table["Biopsy"] = index_table.apply(lambda x: int(x.name.split('.')[0].split('__')[-1]), axis=1)
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
    ## add index_table to table so that all the info is in table 
    if y_name == "RCB_class":
        table[y_name] = list(map(label_mapping, table[y_name]))
    table = add_index(table, index_table)
    train_table = table[table["fold"] != fold_test]
    test_index = table[table["fold"] == fold_test].index
    stratefied_variable = train_table["RCB_score"].round(0)
    skf = StratifiedKFold(n_splits=inner_number_folds, shuffle=True)
    obj = skf.split(train_table.index, stratefied_variable)
    index_folds = [(train_index, val_index) for train_index, val_index in obj]
    return table, index_folds, test_index

class data_handler:
    def __init__(self, path, fold_test, table_name, 
                 inner_cross_validation_number, batch_size, 
                 mean, options):
        files = glob(path)
        index_file = {}
        # index_table = {}
        list_table = []
        last_index = 0
        mean = np.load(mean)
        for f in tqdm(files):
            mat = load(f, options.input_depth) - mean
            n_i = mat.shape[0]
            # index_table[f] = mat
            index_file[f] = [last_index, last_index + n_i]
            last_index += n_i
            list_table.append(mat)
            # if n_i < size_min:
        data = da.concatenate(list_table, axis=0)
        # import pdb; pdb.set_trace()
        # data = np.zeros(shape=(last_index, options.input_depth))
        # for key, value in tqdm(index_table.items()):
        #     i, j = index_file[key]
        #     data[i:j] = value
        self.data = data

        table = pd.read_csv(table_name)
        table, obj, test_index = set_table(table, fold_test, 
                                           inner_cross_validation_number, 
                                           index_file, options.y_variable)
        self.table = table
        self.obj = obj
        self.batch_size = batch_size
        self.test_index = test_index
        self.size = options.size
        self.y_variable = options.y_variable
        if self.y_variable in ["RCB_score", "ee_grade"]:
            self.categorical = "categorical"
        elif self.y_variable in ["stroma", "til"]:
            self.table[self.y_variable] = self.table[self.y_variable] / 100
            self.categorical = "regression_ce"
        else:
            self.categorical = "regression"

    def dg_train(self, sub_fold_val):
        train_index = self.obj[sub_fold_val][0] # Corresponds aux index des 9 subfolds d'entrainement ou bien du subfold de validation ??!
        return h5_Sequencer(train_index, self.table, self.data, self.batch_size, 
                            self.size, categorical=self.categorical, 
                            y_name=self.y_variable)
    def dg_val(self, sub_fold_val):
        val_index = self.obj[sub_fold_val][1]
        return h5_Sequencer(val_index, self.table, self.data, 1, 
                            self.size, categorical=self.categorical, 
                            y_name=self.y_variable)

    def dg_test(self):
        return h5_Sequencer(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            categorical=self.categorical, 
                            y_name=self.y_variable)
    def dg_test_index_table(self):
        return h5_Sequencer(self.test_index, self.table, self.data, 
                            1, self.size, shuffle=False,
                            categorical=self.categorical, 
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
    start = table.ix[name, "start"]
    end = table.ix[name, "end"]
    if end - start < size:
        slice_index = sample(start, end, size)
    elif (end - start) == size:
        slice_index = range(start, end)
    else:
        slice_index = sample(start, end, size)

    return slice_index

class h5_Sequencer(Sequence):

    def __init__(self, index_patient, table, data, batch_size=1, size=10000, shuffle=True,
                 categorical="categorical", y_name="RCB_class"):
        n_classes = len(np.unique(table["RCB_class"]))
        if shuffle:
            np.random.shuffle(index_patient)
        self.index_patient = index_patient
        if categorical == "categorical":
            self.y_onehot = to_categorical(table.ix[self.index_patient, y_name], 
                                           num_classes=n_classes)
        elif categorical == "regression_ce":
            shape = (table.ix[self.index_patient, y_name].shape[0], 2)
            self.y_onehot = np.zeros(shape=shape)
            self.y_onehot[:, 0] = table.ix[self.index_patient, y_name]
            self.y_onehot[:, 1] = 1 - self.y_onehot[:, 0]
        else:
            self.y_onehot = np.array(table.ix[self.index_patient, y_name])
        self.n_size = len(self.index_patient)
        self.batch_size = batch_size
        self.size = size
        self.data = data
        self.table = table
    def __len__(self):
        return self.n_size // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.index_patient[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_onehot[idx * self.batch_size:(idx + 1) * self.batch_size]
        def f(name, table=self.table, data=self.data, size=self.size):
            # old_size = npy_array.shape[0]
            # number_zero = (npy_array.sum(axis=1) == 0).sum()
            index = pj_fetch(name, table, size)
            npy_array = data[index]
            # new_size = npy_array.shape[0]
            # print("size_array in mem: {} with {} lines of 0 and is now {}".format(old_size, number_zero, new_size))
#            return self.datagen.random_transform(npy_array)
            return npy_array

        list_f_x = [f(name).compute() for name in batch_x]

        batch_x = np.array(list_f_x).astype(float)
        batch_y = np.array(batch_y)
        return batch_x, batch_y







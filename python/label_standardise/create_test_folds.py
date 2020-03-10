import numpy as np
from glob import glob 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')

    parser.add_argument('--table',
                        metavar="str", type=str,
                        help='input table')

    parser.add_argument('--output_name', 
                        metavar="str", type=str,
                        help='output csv table name')

    parser.add_argument('--folds', 
                        metavar="int", type=int,
                        help='number of folds')
    args = parser.parse_args()
    return args

def createfolds(table, num, strat_var):

    skf = StratifiedKFold(n_splits=num, shuffle=True)
    obj = skf.split(table.index, table[strat_var])
    i = 0
    for _, test_index in obj:
        table.ix[test_index, "fold"] = i
        i += 1
    return table

def main():

    options = get_options()
    table = pd.read_csv(options.table)
    import pdb; pdb.set_trace()
    table = createfolds(table, options.folds, 'RCB_class')
    table.to_csv(options.output_name, index=False)







if __name__ == '__main__':
    main()


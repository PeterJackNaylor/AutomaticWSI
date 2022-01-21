
import os

from glob import glob
import pandas as pd

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='takes a folder with ')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='folder where the result files can be found')
    parser.add_argument('--variable_to_report', required=True,
                        metavar="str", type=str,
                        help='folder where the result files can be found')
    parser.add_argument('--name', required=True,
                        metavar="str", type=str,
                        help='name of the output')
    args = parser.parse_args()
    return args

def remove_percentage(tab, perc):
    max_val = tab['val_auc_roc'].max()
    tab = tab[0.1 > max_val - tab['val_auc_roc']]
    return tab

def main():
    only_keep_10_highest = True
    options = get_options()
    files = glob(os.path.join(options.path, 'neural_networks_model_fold_number*.csv'))
    if options.variable_to_report == 'auc':
        valv = 'val_auc_roc'
        testv = 'test_auc_roc'
    tab_global = []
    for f in files:
        table = pd.read_csv(f)
        tab_global.append(table)
    tab_best = []
    for f in files:
        table = pd.read_csv(f)
        idx = table.groupby(['validation_fold'])['val_auc_roc'].transform(max) == table['val_auc_roc']
        best_val_idx = table[idx]
        if only_keep_10_highest:
            best_val_idx = remove_percentage(best_val_idx, 10)
        tab_best.append(best_val_idx)

    table = pd.concat(tab_global, axis=0)
    table_best = pd.concat(tab_best, axis=0)
    name_ = options.name + '_all.csv'
    name_all = options.name + '_best.csv'
    table.to_csv(name_)
    table_best.to_csv(name_all)


if __name__ == '__main__':
    main()
 
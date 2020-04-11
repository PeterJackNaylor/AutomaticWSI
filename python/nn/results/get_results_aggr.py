
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
    parser.add_argument('--name', required=True,
                        metavar="str", type=str,
                        help='name of the output')
    args = parser.parse_args()
    return args

def fres(st):
    return st.split('at_res_')[1].split('___be')[0]

def fmodel(st):
    return st.split('_for_')[0]

def fy(st):
    return st.split('_for_')[1].split('_at_')[0]

def ftype(st):
    return st.split('___best')[1]

def main():
    options = get_options()
    files = glob(os.path.join(options.path, '*best.csv'))
    stats = pd.DataFrame()
    for f in files:
        table = pd.read_csv(f)
        table = table.drop('Unnamed: 0', axis=1)
        table['counts'] = table.shape[0]
        table['mean'] = table.shape[0]

        col = os.path.basename(f).split('.')[0]
        stats[col + "mean"] = table.mean()
        stats[col + "Std.Dev"] = table.std()
    #     stats[col + "Var"] = table.var()
    stats = stats.T
    stats['res'] = stats.apply(lambda x: fres(x.name), axis=1)
    stats['model'] = stats.apply(lambda x: fmodel(x.name), axis=1)
    stats['y'] = stats.apply(lambda x: fy(x.name), axis=1)
    stats['type'] = stats.apply(lambda x: ftype(x.name), axis=1)
    import pdb; pdb.set_trace()
    stats = stats.set_index(['y', 'model', 'res', 'type'])
    stats.to_csv(options.name)

if __name__ == '__main__':
    main()
 
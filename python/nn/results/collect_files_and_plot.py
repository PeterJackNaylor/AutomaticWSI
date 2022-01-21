import os
import pandas as pd
from get_results_aggr import fres, fmodel, ftype, fmodel, fy
from glob import glob
from ploting import plot
import matplotlib as mpl
mpl.use('Agg')


RES_MAX = 3
TASK = "Residual"
folder = "/mnt/data3/pnaylor/AutomaticWSI/outputs/nature_3-0"
PATH = ["{}/Model_NN_R{}/res_aggr/{}".format(folder, el, TASK) for el in range(RES_MAX)]
 
BASELINE = ["{}/nature_3-0/naive_rf_{}/{}".format(folder, el, TASK) for el in range(RES_MAX)]

MODEL_2S = ["{}/model_2S_R{}/tissue_classification/{}/results".format(folder, el, TASK) for el in range(RES_MAX)]



def get_files():
    files = []
    for f in PATH:
        print(len(glob(os.path.join(f, '*best.csv'))))
        files += glob(os.path.join(f, '*best.csv'))
    return files 

def main():
    
    files = get_files()
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
    # stats = stats.set_index(['y', 'model', 'res', 'type'])
    stats.to_csv('results.csv')

    table = stats
    
    table = table[table['y'] == TASK]
    plot(table, "final_output_{}.png".format(TASK))

if __name__ == '__main__':
    main()
 
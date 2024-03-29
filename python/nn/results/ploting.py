
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='takes a folder with ')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='file name')
    parser.add_argument('--y', required=True,
                        metavar="str", type=str,
                        help='y variable')
    parser.add_argument('--output', required=True,
                        metavar="str", type=str,
                        help='folder where the result files can be found')
    args = parser.parse_args()
    return args
def plot(table, name):
    
    width = 0.25         # the width of the bars
    inch = 1
    models =  ["model_1S_a", "model_1S_b", "model_1S_c", "model_1S_d", "owkin", "weldon_plus_a", "weldon_plus_b", "weldon_plus_c", "weldon_plus_d", "conan_a", "conan_b", "conan_c", "conan_d"]
    N = len(models)

    ind = np.arange(N)    # the x locations for the groups
    variable_names = ['test_acc', 'test_f1', 'test_auc_roc']
    for var in variable_names:
        fig, ax = plt.subplots(figsize=(20, 12))
        for res in [0,1,2]:
            means = []
            err = []
            for model in models:
                try:
                    mean = table.ix[(table['model'] == model) & (table['res'] == str(res)) & (table['type'] == 'mean'), var].values[0]
                    std_err = table.ix[(table['model'] == model) & (table['res'] == str(res)) & (table['type'] == 'Std.Dev'), var].values[0]
                except:
                    mean = 0
                    std_err = 0
                means.append(mean)
                err.append(std_err)
            ax.bar(ind + res * width, means, width, bottom=0, yerr=err, label=str(res))



        ax.set_title('Scores by resolution and model')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(tuple(models))
        if var == "test_f1":
            base_y = 0
        else:
            base_y = 0.5    
        ax.set_ylim((base_y,1))
        ax.legend()
        ax.yaxis.set_units(inch)
        ax.autoscale_view()
        plt.savefig(name.split('.')[0] + "__{}.png".format(var))
        plt.close()
    # plt.show()
    

def main():
    options = get_options()
    table = pd.read_csv(options.path)
    table = table[table['y'] == options.y]
    plot(table, options.output)

if __name__ == "__main__":
    main()
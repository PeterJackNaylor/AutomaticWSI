
import os
import numpy as np

from glob import glob
import pandas as pd
from sklearn.metrics import roc_auc_score


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='takes a folder with ')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='folder where the result files can be found')
    args = parser.parse_args()
    return args

def main():
    options = get_options()
    files = glob(os.path.join(options.path, 'neural_networks_model_fold_number*_rep_*.csv'))
    tab_global = []
    for f in files:
        table = pd.read_csv(f)
        table['real_path'] = os.path.realpath(f)
        tab_global.append(table)
    
    tab = pd.concat(tab_global, axis=0)
    #import pdb; pdb.set_trace()
    recap = []
    for i in range(5):
        min_tab = tab.loc[tab['fold_test'] == i]
        cv_mean = min_tab.groupby([ 'hidden_btleneck',
                                    'hidden_fcn',
                                    'drop_out',
                                    'learning_rate',
                                    'weight_decay']).mean()
        hbt, hfcn, do, lr, wd = cv_mean["val_auc_roc"].idxmax()
        full_exp = min_tab.loc[(min_tab['hidden_btleneck'] == hbt)&(min_tab['hidden_fcn'] == hfcn)&(min_tab['drop_out'] == do)&(min_tab['learning_rate'] == lr)&(min_tab['weight_decay'] == wd)]
        recap.append(full_exp)
    recap = pd.concat(recap, axis=0)



    # ensemble predictions
    prediction_for_auc = []
    truth_for_auc = []
    score_cv = np.zeros(5)
    for i in range(5):
        min_recap = recap[recap['fold_test'] == i]
        folder_path = min_recap['real_path']
        path = os.path.dirname(folder_path[folder_path.index[0]])
        preds = None
        for idx in min_recap.index:
            line = min_recap.loc[idx]
            run_number = line["run_number"]
            fold = line["validation_fold"]
            prediction_file = os.path.join(path, "predictions_run_{}__fold_test_{}.csv".format(run_number, fold))
            if preds is None:
                preds = pd.read_csv(prediction_file)['y_test']
            else:
                preds += pd.read_csv(prediction_file)['y_test']
        preds /= 5
        y_true = pd.read_csv(prediction_file)['y_true']
        score_cv[i] = roc_auc_score(y_true, preds)
        prediction_for_auc.append(preds)
        truth_for_auc.append(y_true)

    output = pd.DataFrame(score_cv).T
    folder = os.path.realpath(options.path)
    predict_type = folder.split('/')[-2]
    res = folder.split('Model_NN_R')[1].split('/')[0]
    model = folder.split('/')[-3]
    output['prediction'] = predict_type
    output['resolution'] = res
    output['model'] = model
    p = "/cbio/donnees/pnaylor/finishing_tmi/results"
    output.to_csv(os.path.join(p, '{}_{}_{}.csv'.format(predict_type, model, res)), index=False)
    p_2 =  "/cbio/donnees/pnaylor/finishing_tmi/results_for_auc"
    pd.DataFrame({'y_true': np.concatenate(truth_for_auc, axis=0),
                  'y_pred': np.concatenate(prediction_for_auc, axis=0)}).to_csv(os.path.join(p_2, '{}_{}_{}.csv'.format(predict_type, model, res)), index=False)

if __name__ == '__main__':
    main()
 
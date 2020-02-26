

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np
import pickle as pkl

import umap

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_options():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create cluster')
    parser.add_argument('--main_name', required=True,
                        metavar="str", type=str,
                        help='main name for saving')  
    parser.add_argument('--method', required=True,
                        metavar="str", type=str,
                        help='method to apply')
    parser.add_argument('--label', required=True,
                        metavar="str", type=str,
                        help='table where we have the labels')
    parser.add_argument('--y_interest', required=True,
                        metavar="str", type=str,
                        help='variable to classify from label')
    parser.add_argument('--dataset', required=True,
                        metavar="str", type=str,
                        help='dataset containing each patient')
    parser.add_argument('--cpu', required=True,
                        metavar="str", type=int,
                        help='number of available cpu for computation')
    parser.add_argument('--seed', required=False, default=42,
                        metavar="int", type=int,
                        help='seed')
    parser.add_argument('--inner_fold', required=True,
                        metavar="int", type=int,
                        help='number of innerfolds to perform.') 
    parser.add_argument('--order', required=True,
                        metavar="str", type=str,
                        help='order of something, change TODO') 
    args = parser.parse_args()
    return args

def load(name, y_interest):
    """
    Loads label csv and returns the y variable and fold
    with the identifier as index.
    Parameters
    ----------
    name: string, 
        path to csv file
        maximum number of samples to keep.
    y_interest: string,
        Can be several: 
            - "Residual"
            - "Prognostic"
            - RCB 
    Returns
    -------
    A tuple where the first element corresponds to the 
    y and the second to the fold. They are both series
    with the identifier as index.
    """
    mat =  pd.read_csv(name)
    mat = mat.set_index('Biopsy')
    y = mat[y_interest]
    fold = mat["fold"]

    return y, fold
    

def load_randomforest(cpu):
    """
    Loads a random forest model with a dictionnary of 
    hyperparameters.
    Parameters
    ----------
    cpu: int, 
        Number of parallel processes used with n_jobs
    Returns
    -------
    A tuple where the first element corresponds to the 
    random forest model and the second to a dictionnary of
    hyper parameters for the random forest.
    """
    param_grid = {"n_estimators": [10, 100], 
                  "criterion":    ["gini", "entropy"]}
    model = RandomForestClassifier(n_jobs=cpu)
    param_grid["class_weight"] = ['balanced']
    return model, param_grid 

def load_linear(cpu):
    """
    Loads a linear logistic regression model with a dictionnary of 
    hyperparameters.
    Parameters
    ----------
    cpu: int, 
        Number of parallel processes used with n_jobs
    Returns
    -------
    A tuple where the first element corresponds to the 
    logistic regression model and the second to a dictionnary 
    of hyper parameters for the logistic regression.
    """
    model = LogisticRegression(n_jobs=cpu)
    param_grid = {"penalty" : ['l1', 'l2'],
                    "fit_intercept": [True, False],
                    "multi_class": ["ovr"],
                    "class_weight" : ['balanced']
                    }
    return model, param_grid 

def load_svm(cpu, prediction_type):
    """
    Loads a SVM model with a dictionnary of 
    hyperparameters.
    Parameters
    ----------
    cpu: int, 
        Number of parallel processes used with n_jobs
    Returns
    -------
    A tuple where the first element corresponds to the 
    SVM model and the second to a dictionnary of hyper
    parameters for the SVM.
    """
    model = SVC()
    param_grid = {"C" : [10**i for i in range(-3, 3)],
                    "kernel": ["rbf", "linear", "poly"],
                    "degree": [3, 4, 5, 6],
                    "decision_function_shape": ["ovr", "ovo"],
                    "class_weight" : ['balanced']
                    }
    return model, param_grid 


def model_training(x_, y_, fold, method, cpu, inner_fold, class_type):
    """
    Performs nested cross validation 
    Parameters
    ----------
    name: name of the numpy matrix containing 
        tissue level descriptors.
    order: pickle list containing the identifier
        for each line in z_i numpy matrix.
    Returns
    -------
    A dataframe containing the tissue level
    descriptors of our cohort.
    """



    cv = int(fold.max() + 1)
    full_y = np.zeros_like(y_)

    def score_function(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        if class_type in ["Residual", "Prognostic"]:
            labels = [0, 1]
        elif class_type == "RCB":
            labels = ['pCR', 'RCB-I', 'RCB-II', 'RCB-III']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return acc, cm


    fold_scores = []
    for i in range(cv):
        if method == "RandomForest":
            model, param_grid = load_randomforest(cpu)
        elif method == "LinearModel":
            model, param_grid = load_linear(cpu)
        elif method == "SVM":
            model, param_grid = load_svm(cpu)
        else:
            raise ValueError('Method -- {} --  unknown'.format(method))
        train_fold = x_[(fold != i).values]
        test_fold = x_[(fold == i).values]
        train_y = y_[(fold != i).values]
        test_y = y_[(fold == i).values]
        model_cv = GridSearchCV(model, param_grid, cv=int(inner_fold), n_jobs=cpu, refit=True)
        model_cv.fit(train_fold, train_y)
        full_y[fold == i] = model_cv.predict(test_fold).flatten()
        fold_scores.append(score_function(test_y, full_y[fold == i]))
    final_scores = score_function(y_, full_y)
    full_y = pd.DataFrame(full_y, index=x_.index)
    return final_scores, fold_scores, full_y

def save_result(final_scores, fold_scores, y, name):
    """
    Saving the results and .. 
    Parameters
    ----------
    name: name of the numpy matrix containing 
        tissue level descriptors.
    order: pickle list containing the identifier
        for each line in z_i numpy matrix.
    Returns
    -------
    A dataframe containing the tissue level
    descriptors of our cohort.
    """
    fold_name = name + "__fold_scores.pickle"
    pkl.dump(fold_scores, open(fold_name, 'wb'))
    err_name = name + "__error_scores.txt"
    y_name = name + "__pred_y.csv"
    y.to_csv(y_name)
    err = final_scores[0]
    cm = final_scores[1]
    cm_name = name + "__cm_scores.txt"
    np.savetxt(cm_name, cm, fmt='%1.0f')
    file = open(err_name, "w") 
    file.write(str(err)) 
    file.close() 

def load_zi(name, order, y):
    """
    Loads zi scores for all 
    Parameters
    ----------
    name: name of the numpy matrix containing 
        tissue level descriptors.
    order: pickle list containing the identifier
        for each line in z_i numpy matrix.
    Returns
    -------
    A dataframe containing the tissue level
    descriptors of our cohort.
    """
    mat = np.load(name)
    order = pkl.load(open(order,'rb'))
    # mat = np.reshape(mat, (len(order), 4))
    # import pdb; pdb.set_trace()
    tab = pd.DataFrame(mat, index=order)
    tab = tab.ix[y.index]
    return tab
    
def plot_umap(zi, y, name):
    """
    Does a umap projection and plots it.
    Parameters
    ----------
    name: name of the numpy matrix containing 
        tissue level descriptors.
    order: pickle list containing the identifier
        for each line in z_i numpy matrix.
    Returns
    -------
    A dataframe containing the tissue level
    descriptors of our cohort.
    """
    
    pi = umap.UMAP(n_components=2, random_state=42, 
                      n_neighbors=10, min_dist=0.).fit_transform(zi)    

    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    pi1 = pi[:, 0]
    pi2 = pi[:, 1]
    data = pd.DataFrame({'pi1':pi1, 'pi2':pi2, 'label':y})
    sns.lmplot('pi1', 'pi2', data, hue="label", fit_reg=False, scatter_kws={"s": 1.0})
    name = name + "__umap.png"
    plt.savefig(name)

def main():
    options = get_options()
    y, fold = load(options.label, options.y_interest)
    zi = load_zi(options.dataset, options.order, y)
    
    final_scores, fold_scores, pred_y = model_training(zi, y, fold, 
                                                       options.method, options.cpu,
                                                       options.inner_fold,
                                                       options.y_interest)
    save_result(final_scores, fold_scores, pred_y, 
                options.main_name)
    plot_umap(zi, y, options.main_name)

if __name__ == '__main__':
    main()

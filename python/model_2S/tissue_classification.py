
import pandas as pd
import numpy as np
import pickle as pkl

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
    parser.add_argument('--cpus', required=True,
                        metavar="str", type=int,
                        help='number of available cpus for computation')
    parser.add_argument('--seed', required=False, default=42,
                        metavar="int", type=int,
                        help='seed')
    parser.add_argument('--inner_fold', required=True,
                        metavar="int", type=int,
                        help='number of innerfolds to perform.') 
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
    

def load_randomforest(cpus, prediction_type):
    param_grid = {"n_estimators": [10, 100], 
                  "criterion":    ["gini", "entropy"]}
    if prediction_type == "classification":
        model = RandomForestClassifier(n_jobs=cpus)
        param_grid["class_weight"] = ['balanced']
    elif prediction_type == "regression":
        model = RandomForestRegressor(n_jobs=cpus)
        param_grid["criterion"] = ['mse', 'mae']
    return model, param_grid 

def load_linear(cpus, prediction_type):
    if prediction_type == "classification":
        model = LogisticRegression(n_jobs=cpus)
        param_grid = {"penalty" : ['l1', 'l2'],
                      "fit_intercept": [True, False],
                      "multi_class": ["ovr"],
                      "class_weight" : ['balanced']
                      }
                      
    elif prediction_type == "regression":
        param_grid = {"fit_intercept" : [True, False]}
        model = LinearRegression(n_jobs=cpus)
    return model, param_grid 

def load_svm(cpus, prediction_type):
    if prediction_type == "classification":
        model = SVC()
        param_grid = {"C" : [10**i for i in range(-3, 3)],
                      "kernel": ["rbf", "linear", "poly"],
                      "degree": [3, 4, 5, 6],
                      "decision_function_shape": ["ovr", "ovo"],
                      "class_weight" : ['balanced']
                      }
    elif prediction_type == "regression":
        model = SVR()
        param_grid = {"C" : [10**i for i in range(-3, 3)],
                      "kernel": ["rbf", "linear", "poly"],
                      "degree": [3, 4, 5, 6]}
    return model, param_grid 


def model_training(x_, y_, fold, method, cpus, prediction_type, class_type):

    cv = int(fold.max() + 1)
    full_y = np.zeros_like(y_)

    if prediction_type == "classification":
        def score_function(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            if class_type in ["residuum", "prognostic"]:
                labels = ['pCR', 'RCB']
            elif class_type == "four_classes":
                labels = ['pCR', 'RCB-I', 'RCB-II', 'RCB-III']
                
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            return acc, cm
    elif prediction_type == "regression":
        def score_function(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)

    fold_scores = []
    for i in range(cv):
        if method == "RandomForest":
            model, param_grid = load_randomforest(cpus, prediction_type)
        elif method == "LinearModel":
            model, param_grid = load_linear(cpus, prediction_type)
        elif method == "SVM":
            model, param_grid = load_svm(cpus, prediction_type)
        else:
            raise ValueError('Method -- {} --  unknow'.format(method))
        train_fold = x_[(fold != i).values]
        test_fold = x_[(fold == i).values]
        train_y = y_[(fold != i).values]
        test_y = y_[(fold == i).values]
        model_cv = GridSearchCV(model, param_grid, cv=int(cv), n_jobs=cpus, refit=True)
        model_cv.fit(train_fold, train_y)
        full_y[fold == i] = model_cv.predict(test_fold).flatten()
        fold_scores.append(score_function(test_y, full_y[fold == i]))
    final_scores = score_function(y_, full_y)
    full_y = pd.DataFrame(full_y, index=x_.index)
    return final_scores, fold_scores, full_y

def save_result(final_scores, fold_scores, y, name, prediction):
    fold_name = name + "__fold_scores.pickle"
    pkl.dump(fold_scores, open(fold_name, 'wb'))
    err_name = name + "__error_scores.txt"
    y_name = name + "__pred_y.csv"
    y.to_csv(y_name)
    if prediction == "classification":
        err = final_scores[0]
        cm = final_scores[1]
        cm_name = name + "__cm_scores.txt"
        np.savetxt(cm_name, cm, fmt='%1.0f')
    elif prediction == "regression":
        err = final_scores

    file = open(err_name, "w") 
    file.write(str(err)) 
    file.close() 

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

def load_zi(name, order):
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
    


def main():
    options = get_options()
    y, fold = load(options.label, options.y_interest)
    zi = load_zi(options.dataset)
    table_feat, table_y, folds = load(options.table,
                                      options.detail,
                                      options.prediction_type)
    final_scores, fold_scores, pred_y = model_training(table_feat, table_y, folds, 
                                                       options.method, options.cpus,
                                                       options.prediction_type,
                                                       options.class_type)
    save_result(final_scores, fold_scores, pred_y, 
                options.main_name, options.prediction_type)

if __name__ == '__main__':
    main()

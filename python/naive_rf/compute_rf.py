  
from glob import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, recall_score, 
                             roc_auc_score, precision_score)
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR

FEATURES = list(range(2048))


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    C = C.astype('float')
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Classification from marick features')
    parser.add_argument('--label', required=True,
                        metavar="str", type=str,
                        help='table containing fold information')
    parser.add_argument('--inner_fold', required=True,
                        metavar="int", type=int,
                        help='number of subfolds')
    parser.add_argument('--cpu', required=False,
                        metavar="int", type=int,
                        help='number of cpus')
    parser.add_argument('--y_interest', required=True,
                        metavar="str", type=str,
                        help='variable to classify from label')
    args = parser.parse_args()
    return args

def load_custom_table():
    feats = []
    index = []
    for patient in glob('*_mean.npy'):
        feat = np.load(patient)
        feats.append(feat)
        num = patient.split('_')[0]
        index.append(num)

    df_features = pd.DataFrame(feats, index=index)
    return df_features

def load_custom_csv(path):
    df_folds = pd.read_csv(path)
    return df_folds

def load(path_fold_label):
    x_feat = load_custom_table()
    fold_pat = load_custom_csv(path_fold_label)
    df_data = x_feat.join(fold_pat.set_index('Biopsy'))

    # df_data = df_data.dropna() because of substrat merging with fabien,
    # marick features are incomplete with respect to substrat and vice versa

    return df_data

def model_config(df_data, y_interest, cv, cpu):
    result = df_data.copy()

    for y_fold in range(int(df_data['fold'].max() + 1)):
        df_tr = df_data[df_data['fold'] != y_fold]
        df_te = df_data[df_data['fold'] == y_fold]

        x_tr = df_tr[FEATURES].as_matrix()
        x_te = df_te[FEATURES].as_matrix()


        scoring = 'accuracy'
        y_tr = df_tr[y_interest]
        # train here

        hyper_parameters = {'max_depth' : randint(low=3, high=6),
                            'n_estimators' : randint(low=10, high=50)}


        grid_search_rf = RandomizedSearchCV(RandomForestClassifier(),
                                            cv=cv, 
                                            param_distributions=hyper_parameters,
                                            scoring=scoring,
                                            n_iter=50,
                                            verbose=1)

        grid_search_rf.fit(x_tr, y_tr)

        rf_model = grid_search_rf.best_estimator_
        # pred = rf_model.predict(x_te)
        prob = rf_model.predict_proba(x_te)[:,1]
        pred = prob.round(0)
        
        result.ix[df_data['fold'] == y_fold, "y_predicted"] = pred
        result.ix[df_data['fold'] == y_fold, "y_prob"] = prob
        score_val = grid_search_rf.best_score_
        print('Fold {} we get a {} score of {}'.format(y_fold, scoring, score_val))
    y_pred = result["y_predicted"]
    y_true = result[y_interest]
    y_prob = result["y_prob"]
    return y_true, y_pred, y_prob

def compute_score(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)    
    cm_ = confusion_matrix(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred)
    recall_ = recall_score(y_true, y_pred)
    auc_ = roc_auc_score(y_true, y_prob)
    prec_ = precision_score(y_true, y_pred)
    return acc, f1_, recall_, auc_, prec_, cm_

def save_result(y_true, y_pred, y_prob):
    scores = list(compute_score(y_true, y_pred, y_prob))
    cm = scores[-1]
    del scores[-1]
    for i, sc in enumerate(["acc", "f1", "recall", "auc", "prec"]):
        file = open("{}.txt".format(sc), "w") # acc score 
        score = scores[i]
        file.write(str(score)) 
        file.close() 
    np.savetxt("cm.txt", cm, fmt='%1.0f') # confusion matrix
    y_pred.to_csv("y_pred.csv")
    y_prob.to_csv("y_prob.csv")
def main():
    options = get_options()

    df_data = load(options.label)
    y_true, y_pred, y_proba = model_config(df_data,
                                           options.y_interest,
                                           options.inner_fold,
                                           options.cpu)

    save_result(y_true, y_pred, y_proba)

if __name__ == '__main__':
    main()


from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             log_loss, f1_score, precision_score,
                              recall_score)
import pandas as pd
def evaluate_model(model, dg, 
                        max_queue_size=1, 
                        workers=1, 
                        use_multiprocessing=False, 
                        verbose=0):

    y_true = dg.return_labels()[:(len(dg)*dg.batch_size)]
    y_pred = model.predict_generator(dg, max_queue_size=max_queue_size,
                                     workers=workers, 
                                     use_multiprocessing=use_multiprocessing, 
                                     verbose=verbose)
    acc_ = accuracy_score(y_true, y_pred[:,1].round(0))
    auc_ = roc_auc_score(y_true, y_pred[:,1]) 
    l_loss = log_loss(y_true, y_pred) 
    recall = recall_score(y_true, y_pred[:,1].round(0)) 
    precision = precision_score(y_true, y_pred[:,1].round(0)) 
    f1 = f1_score(y_true, y_pred[:,1].round(0)) 
    scores = [l_loss, acc_, recall, precision, f1, auc_]
    
    return scores, y_pred
    

             
def evaluate_model_hardlabel(model, dg, 
                        max_queue_size=1, 
                        workers=1, 
                        use_multiprocessing=False, 
                        verbose=0):


    # y_true = dg.return_labels()[:(len(dg)*dg.batch_size)].argmax(axis=1)
    y_pred = model.predict_generator(dg, max_queue_size=max_queue_size,
                                     workers=workers, 
                                     use_multiprocessing=use_multiprocessing, 
                                     verbose=verbose)
    y_corres = dg.corresp_samples[:(len(dg)*dg.batch_size)]
    y_bio = dg.return_biop_labels()
    df = pd.DataFrame({'corresp':y_corres, 'pred':y_pred[:,1]}).groupby('corresp').mean()
    df = df.join(y_bio)
    df.columns = ['pred', 'true']
    #regroup pred by y_corresp
    y_pred = df['pred']
    y_true = df['true']

    acc_ = accuracy_score(y_true, y_pred.round(0))
    auc_ = roc_auc_score(y_true, y_pred) 
    l_loss = log_loss(y_true, y_pred) 
    recall = recall_score(y_true, y_pred.round(0)) 
    precision = precision_score(y_true, y_pred.round(0)) 
    f1 = f1_score(y_true, y_pred.round(0)) 
    scores = [l_loss, acc_, recall, precision, f1, auc_]
    
    return scores, y_pred
    
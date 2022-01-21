from numpy import array, random, round
from pandas import DataFrame
from options_py import get_options
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys
from data_handler import data_handler
from model_definition import load_model
from evaluate_nn import evaluate_model
import ipdb

import tensorflow as tf
from keras import backend as K


def log_sample(a, b):
    assert a <= b
    unit = random.randint(1, 10)
    power = random.randint(a, b)
    return unit * 10 ** power


def sample_hyperparameters(options, validation_fold):
    """ Samples hyperparameters: 
    learning_rate, weight_decay, gaussian_noise, drop_out, hidden_fcn, hidden_btleneck, validation fold.
    See commentary for validation fold. 

    
    Parameters
    ----------
    options : NameSpace
        Parameters sampled by argParse
    validation_fold : int
        number of the validation fold used.
    
    Returns
    -------
    dict
        dictionary containing all the values cited above. Keys are the name written in the description.
    """
    dic = {}
    dic["learning_rate"] = 0.04
    dic["weight_decay"] = 0.0007
    dic["gaussian_noise"] = 0
    dic["drop_out"] = 0.3006
    dic["hidden_fcn"] = 128
    dic["hidden_btleneck"] = 32
    return dic

def call_backs(options):
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   mode='auto')  
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   min_delta=0, 
                                   patience=20, 
                                   verbose=0, 
                                   mode='auto', 
                                   baseline=None,    
                                   restore_best_weights=True)
    callbacks = [early_stopping]
    # return [lr_reducer]
    return [early_stopping]

def train_model(model, dg_train, dg_val, class_weight, options):
    callbacks_list = call_backs(options)
    history = model.fit_generator(dg_train, 
                                  validation_data=dg_val,
                                  epochs=options.epochs, callbacks=callbacks_list, 
                                  class_weight=class_weight, 
                                  max_queue_size=options.max_queue_size, workers=options.workers, 
                                  use_multiprocessing=options.use_multiprocessing)#,
                                  #verbose=2)
    # model.save('my_model_run_number_{}.h5'.format(options.run))

    return model, history


def evaluate_model_generator(dg, index, model, options, repeat=5):
    final_scores = None
    final_predictions = None
    for i in range(repeat):
        scores, predictions = evaluate_model(model, dg, 
                                     max_queue_size=options.max_queue_size, 
                                     workers=options.workers, 
                                     use_multiprocessing=options.use_multiprocessing, 
                                     verbose=0)
        if final_scores is None:
            final_scores = array(scores)
            final_predictions = array(predictions)
        else:
            final_scores += array(scores)
            final_predictions += array(predictions)

    final_scores = final_scores / repeat
    final_predictions = final_predictions / repeat

    y_true = dg.return_labels()[:(len(dg)*dg.batch_size)]

    lbl_predictions = DataFrame({"y_true": y_true, "y_test": final_predictions[:,1]}, index=index)
                          
    return list(final_scores), lbl_predictions

def trunc_if_possible(el):
    try:
        return round(el, decimals=4)
    except:
        return el

def fill_table(train_scores, val_scores, test_scores, table, parameter_dic, validation_number, options):
    train_values = ['loss', 'acc', 'recall', 'precision', 'f1', 'auc_roc']
    val_values = ['val_' + el for el in train_values]
    test_values = ['test_' + el for el in train_values]
    parameters_values = list(parameter_dic.keys())
    model_values = ['k', 'pooling', 'batch_size', 'size', 
                    'input_depth', 'fold_test', "run_number"]
    vec_train = [trunc_if_possible(el) for el in train_scores]
    vec_validation = [trunc_if_possible(el) for el in val_scores]
    vec_test = [trunc_if_possible(el) for el in test_scores]
    vec_parameters = list(parameter_dic.values())
    vec_model = [options.k, options.pool, 
                 options.batch_size, options.size, 
                 options.input_depth,
                 options.fold_test-1, options.run]

    name_columns = train_values + val_values + test_values + parameters_values + model_values
    val_columns = vec_train + vec_validation + vec_test + vec_parameters + vec_model
    table.iloc[options.run][name_columns] = val_columns
    table.iloc[options.run]["model"] = options.model 
    table.iloc[options.run]["validation_fold"] = validation_number 
    return table

def main():

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))

    options = get_options()
    mean = options.mean_name
    path = options.path
    fold_test = options.fold_test - 1 
    table_name = options.table
    batch_size = options.batch_size


    ### data business
    data = data_handler(path, fold_test, 
                        table_name, options.inner_folds, 
                        batch_size, mean, options)
    columns = ['loss', 'acc', 'recall', 'precision', 'f1', 'auc_roc',
               'val_loss', 'val_acc', 'val_recall', 'val_precision', 'val_f1', 
               'val_auc_roc',
                'test_loss', 'test_acc', 'test_recall', 'test_precision', 
                'test_f1', 'test_auc_roc',
                'hidden_btleneck', 'hidden_fcn', 'drop_out', 
                'validation_fold', 'learning_rate', 'weight_decay', 
                'gaussian_noise', 'k', 'model', 'pooling', 'batch_size', 
                'size', 'input_depth', 'fold_test', 'run_number']

    results_table = DataFrame(index=range(options.repeat*(options.inner_folds)), columns=columns) # Link with the previous table ? Or just result_table ?
    options.run = 0
    for i in range(options.repeat):
        for j in range(options.inner_folds): # Defines which fold will be the validation fold
            parameter_dic = sample_hyperparameters(options, j)

            model = load_model(parameter_dic, options)
            print("Model loaded")

            dg_train = data.dg_train(j)
            dg_val = data.dg_val(j)
            class_weight = data.weights()
            dg_test, test_index, _ = data.dg_test_index_table()
            print("Data Loaded")

            print("begin Training", flush=True)
            model, history = train_model(model, dg_train, dg_val, class_weight, options)
            print("end Training")
            print("Evaluating..")
            scores_train, _ = evaluate_model_generator(dg_train, None, model, 
                                                     options, repeat=10)
            scores_val, _ = evaluate_model_generator(dg_val, None, model, 
                                                     options, repeat=10)
            scores_test, predictions = evaluate_model_generator(dg_test, test_index, model, 
                                                           options, repeat=10)
            print("Cleaning up")
            results_table = fill_table(scores_train, scores_val, scores_test, results_table, parameter_dic, j, options)

            K.clear_session()
            del model
            results_table.to_csv(options.output_name, index=False)
            predictions.to_csv("predictions_run_{}_fold_test_{}.csv".format(options.run, options.fold_test))
            options.run += 1


if __name__ == '__main__':
    main()

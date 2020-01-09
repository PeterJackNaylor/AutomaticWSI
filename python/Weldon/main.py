  
from numpy import array, random, round
from pandas import DataFrame
from options_py import get_options
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_handler import data_handler
from model_definition import load_model

from keras import backend as K

def log_sample(a, b):
    unit = random.randint(1, 10)
    power = random.randint(a, b)
    return unit * 10 ** power


def sample_hyperparameters(options, validation_fold):
    dic = {}
    dic["learning_rate"] = log_sample(options.learning_rate_start, options.learning_rate_stop)
    dic["weight_decay"] = log_sample(options.weight_decay_start, options.weight_decay_stop)
    dic["gaussian_noise"] = random.uniform() if options.gaussian_noise else 0
    dic["drop_out"] = random.uniform(low=0.0, high=0.5)
    dic["hidden_fcn"] = random.choice(options.hidden_fcn_list)
    dic["hidden_btleneck"] = random.choice(options.hidden_btleneck_list)
    fold_test = options.fold_test - 1
    possible_val_fold = [el for el in range(options.n_fold) if fold_test != el]
    dic["validation_fold"] = possible_val_fold[validation_fold]
    # dic["validation_fold"] = random.choice(possible_val_fold)
    return dic

def call_backs(options):
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   mode='auto')  
    # return [lr_reducer]
    return []

def train_model(model, data, parameter_dic, options):
    callbacks_list = call_backs(options)
    validation_fold = parameter_dic["validation_fold"]
    history = model.fit_generator(data.dg_train(validation_fold), 
                                  validation_data=data.dg_val(validation_fold),
                                  epochs=options.epochs, callbacks=callbacks_list, 
                                  class_weight=data.weights(), 
                                  max_queue_size=options.max_queue_size, workers=options.workers, 
                                  use_multiprocessing=options.use_multiprocessing)
    model.save('my_model_run_number_{}.h5'.format(options.run))

    return model, history

def evaluate_test(model, data, options, repeat=5):
    final_scores = None
    dg_test, test_index, table = data.dg_test_index_table()
    for i in range(repeat):
        scores = model.evaluate_generator(dg_test, 
                                          # steps=data.test_size(), 
                                          max_queue_size=options.max_queue_size, 
                                          workers=options.workers, 
                                          use_multiprocessing=options.use_multiprocessing, 
                                          verbose=0)
        if final_scores is None:
            final_scores = array(scores)
        else:
            final_scores += array(scores)
    final_scores = final_scores / repeat

    y_test = model.predict_generator(dg_test)[:,0]
    y_true = table.iloc[test_index][options.y_variable]
    new_table = DataFrame({"y_true": y_true, "y_test": y_test}, index=test_index)
    return list(final_scores), new_table
def trunc_if_possible(el):
    try:
        return round(el, decimals=2)
    except:
        return el

def fill_table(history, scores, table, parameter_dic, options):
    if options.y_variable in ["RCB_class", "ee_grade"]:
        train_values = ['loss', 'acc', 'recall', 'precision', 'f1']
        val_values = ['val_loss', 'val_acc', 'val_recall', 'val_precision', 'val_f1']
        test_values = ['test_loss', 'test_acc', 'test_recall', 'test_precision', 'test_f1']
    else:
        train_values = ['loss', 'mean_squared_error']
        val_values = ['val_loss', 'val_mean_squared_error']
        test_values = ['test_loss', 'test_mean_squared_error']
    parameters_values = list(parameter_dic.keys())
    model_values = ['k', 'model', 'pooling', 'batch_size', 'size', 
                    'input_depth', 'fold_test', "run_number"]
    vec_train = [trunc_if_possible(history.history[key][-1]) for key in train_values]
    vec_validation = [trunc_if_possible(history.history[key][-1]) for key in val_values]
    vec_test = [trunc_if_possible(el) for el in scores]
    vec_parameters = list(parameter_dic.values())
    vec_model = [options.k, options.pool, 
                 options.batch_size, options.size, options.input_depth,
                 options.fold_test-1, options.run]

    name_columns = train_values + val_values + test_values + parameters_values + model_values
    val_columns = vec_train + vec_validation + vec_test + vec_parameters + vec_model
    table.ix[options.run, name_columns] = val_columns
    return table

def main():
    options = get_options()
    mean = options.mean_name
    path = options.path
    fold_test = options.fold_test - 1 
    table_name = options.table
    inner_cross_validation_number = options.inner_cross_validation_number
    batch_size = options.batch_size
    class_type = options.class_type
    seed = options.seed
    

    ### data business
    data = data_handler(path, fold_test, 
                        table_name, options.n_fold, 
                        batch_size, mean, options)
    if options.y_variable in ["RCB_class", "ee_grade"]:
        columns = ['loss', 'acc', 'recall', 'precision', 'f1', 'val_loss', 
                   'val_acc', 'val_recall', 'val_precision', 'val_f1', 
                   'test_loss', 'test_acc', 'test_recall', 'test_precision', 
                   'test_f1', 'hidden_btleneck', 'hidden_fcn', 'drop_out', 
                   'validation_fold', 'learning_rate', 'weight_decay', 
                   'gaussian_noise', 'k', 'model', 'pooling', 'batch_size', 
                   'size', 'input_depth', 'fold_test', 'run_number']
    else:
        columns = ['loss', 'mean_squared_error', 'val_loss', 
                   'val_mean_squared_error', 'test_loss', 'test_mean_squared_error', 
                   'hidden_btleneck', 'hidden_fcn', 'drop_out', 
                   'validation_fold', 'learning_rate', 'weight_decay', 
                   'gaussian_noise', 'k', 'model', 'pooling', 'batch_size', 
                   'size', 'input_depth', 'fold_test', 'run_number']
    table = DataFrame(index=range(options.repeat), columns=columns)

    options.run = 0
    for i in range(options.repeat):
        options.run += 1
        for j in range(options.n_fold - 1): # Defines which fold will be the validation fold
            parameter_dic = sample_hyperparameters(options, j)
            model = load_model(parameter_dic, options)
            model, history = train_model(model, data, parameter_dic, options)
            scores, predictions = evaluate_test(model, data, options)
            table = fill_table(history, scores, table, parameter_dic, options)

            K.clear_session()
            del model
            table.to_csv(options.output_name, index=False)
            predictions.to_csv("predictions_run_{}_fold_val_{}.csv".format(options.run, j))


if __name__ == '__main__':
    main()

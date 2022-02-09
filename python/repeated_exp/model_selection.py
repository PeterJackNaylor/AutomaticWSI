
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from simulation import load_sample
from nn_model import load_model, performance, repeated_experiment
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import accuracy_score

randint = np.random.randint

def fold_generator(data, k):
    x, y = data
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    for train_idx, val_idx in skf.split(x, y.argmax(axis=1)):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield (x_train, y_train), (x_val, y_val)

def hp_generator(lr_range, wd_range, do_range):
    lr_min, lr_max = lr_range
    random_lr = randint(1, 10) * 10 ** randint(lr_min, lr_max) 
    wd_min, wd_max = wd_range
    random_wd = randint(1, 10) * 10 ** randint(wd_min, wd_max) 
    do_min, do_max = do_range
    random_do = np.random.uniform(do_min, do_max)
    return random_lr, random_wd, random_do


def sample_hp(d):
    return hp_generator(d["lr_range"], d["wd_range"], d["do_range"])


def ncv_single(std, repeat):

    k = 5
    n_train = 224
    n_val = 56
    n_loads = 5000
    p = 256



    hp = {'lr_range': (-4, -1),
          'wd_range': (-5, -3),
          'do_range': (0.3, 0.7)}
    # name = f"bncv_sco.csv"
    out_names = ["ncv", "bncv", "bncv_top_3", "bncv_top_5", "bfcv", "bfcv_top_3", "bfcv_top_5"]

    final_results = pd.DataFrame(columns=["name", "score", "std"])

    line_counter = 0
    
    data_loads = load_sample(n_loads, p, std)
    data_cv = load_sample(n_train+n_val, p, std)
    outputs = cv(data_cv, hp, k, repeat, data_loads)
    for name, score in zip(out_names, outputs):
        final_results.loc[line_counter, "name"] = name
        final_results.loc[line_counter, "score"] = score
        final_results.loc[line_counter, "std"] = std
        line_counter += 1
    final_results.to_csv(f"bncv_vs_ncv_{std}_{repeat}.csv", index=False)



def ncv():
    k = 5
    n_train = 224
    n_val = 56
    n_loads = 5000
    p = 256
    list_std = list(range(1, 21))
    repeats_hp = 20
    repeats_simulated_data = 20
    hp = {'lr_range': (-4, -1),
          'wd_range': (-5, -3),
          'do_range': (0.3, 0.7)}
    # name = f"bncv_sco.csv"
    out_names = ["ncv", "bncv", "bncv_top_3", "bncv_top_5"]

    final_results = pd.DataFrame(columns=["name", "score", "std"])

    line_counter = 0
    for std in tqdm(list_std):
        data_loads = load_sample(n_loads, p, std)
        for _ in tqdm(range(repeats_simulated_data)):
            data_cv = load_sample(n_train+n_val, p, std)
            outputs = cv(data_cv, hp, k, repeats_hp, data_loads)
            for name, score in zip(out_names, outputs):
                final_results.loc[line_counter, "name"] = name
                final_results.loc[line_counter, "score"] = score
                final_results.loc[line_counter, "std"] = std
                line_counter += 1
    final_results.to_csv("bncv_vs_ncv.csv", index=False)

def cv(data, hp, k, repeats, data_test):
    max_epoch = 400
    epoch_patience = 10
    fcn = 256
    columns = ["wd", "do", "lr"] + list(range(k))
    results = pd.DataFrame(columns=columns)
    test_predictions = {}
    for _ in range(repeats):
        lr, wd, do = sample_hp(hp)
        param = [fcn, wd, do, lr]
        fold_count = 0
        ensembled_p = {}
        results.loc[_, ["wd", "do", "lr"]] = [wd, do, lr]
        for data_train, data_val in fold_generator(data, k):
            
            model = load_model(data_train[0].shape[1], param)

            es = EarlyStopping(monitor='val_loss', mode='min', 
                       verbose=1, patience=epoch_patience)
            h = model.fit(x=data_train[0], y=data_train[1], 
                        batch_size=16, 
                        epochs=max_epoch, 
                        verbose=0,
                        validation_data=data_val,
                        shuffle=True,
                        callbacks=[es])
            pred = model.predict(data_test[0])
            ensembled_p[fold_count] = pred
            results.loc[_, fold_count] = h.history["val_accuracy"][-1]
            del model
            K.clear_session()
            fold_count += 1
        test_predictions[_] = ensembled_p
    
    mean_val_idx = results[list(range(k))].mean(axis=1).argmax()

    # cv score
    param = [fcn] + [results.loc[mean_val_idx, el] for el in ["wd", "do", "lr"]]
    model = load_model(data_train[0].shape[1], param)
    model.fit(x=data[0], y=data[1], 
                batch_size=16, 
                epochs=max_epoch, 
                verbose=0,
                shuffle=True)
    score = performance(model, data_test)

    # bncv

    validation_scores = results.loc[mean_val_idx, list(range(k))]
    validation_scores = validation_scores.sort_values(ascending=False)

    bncv1 = aggre(validation_scores, test_predictions[mean_val_idx], 1, data_test[1])
    bncv3 = aggre(validation_scores, test_predictions[mean_val_idx], 3, data_test[1])
    bncv5 = aggre(validation_scores, test_predictions[mean_val_idx], 5, data_test[1])

    bfcv1 = best_fold_aggre(results, test_predictions, 1, k, data_test[1])
    bfcv3 = best_fold_aggre(results, test_predictions, 3, k, data_test[1])
    bfcv5 = best_fold_aggre(results, test_predictions, 5, k, data_test[1])

    return score, bncv1, bncv3, bncv5, bfcv1, bfcv3, bfcv5


def aggre(series_scores, dic_pred_test, order, y_test):
    results = np.zeros_like(dic_pred_test[0])
    for i in range(order):
        results += dic_pred_test[i]
    results /= order
    y_ens = results.argmax(axis=1)
    return accuracy_score(y_test.argmax(axis=1), y_ens)

def best_fold_aggre(res, test_pred, n, k, y_true):
    fold_val = []
    idx_val = []
    res = res.astype(float)
    for i in range(k):
        best_idx_val = res[i].argmax()
        idx_val.append(best_idx_val)
        fold_val.append(res.loc[best_idx_val, i])
    
    idx_sort = np.argsort(fold_val)[::-1]
    pred = np.zeros_like(test_pred[0][0])
    for i in range(n):
        f_num = idx_sort[i]
        pred += test_pred[idx_val[f_num]][f_num]
    pred /= n
    y_ens = pred.argmax(axis=1)
    return accuracy_score(y_true.argmax(axis=1), y_ens)

    




def repeat(num):
    # n = 336 => 350
    # NCV: 5 folds in 5 folds => test_fold ~ 350 / 5 => 70
    # val fold => (350 - 70) / 5 = 56
    # train fold => 224
    if num == "1":
        repeats = 1000
        repeats_simulated_data = 1
    elif num == "2":
        repeats = 10
        repeats_simulated_data = 10
    n_train = 224
    n_val = 56
    n_loads = 5000
    p = 256
    list_std = list(range(1, 21))
    name = f"repeated_experiment_sco_{num}.csv"
    e_name = f"repeated_experiment_epo_{num}.csv"


    results = pd.DataFrame(columns=list_std)
    e_results = pd.DataFrame(columns=list_std)

    for std in tqdm(list_std):
        x_loads, y_loads = load_sample(n_loads, p, std)
        score_std = []
        epochs = []
        for _ in tqdm(range(repeats_simulated_data)):
            x_train, y_train = load_sample(n_train, p, std)
            x_val, y_val = load_sample(n_val, p, std)
            for _ in tqdm(range(repeats)):
                score, end_epoch = repeated_experiment((x_train, y_train),
                                                      (x_val, y_val),
                                                      (x_loads, y_loads))
                score_std.append(score)
                epochs.append(end_epoch)
        results[std] = score_std
        e_results[std] = epochs 
    results.to_csv(name, index=False)
    e_results.to_csv(e_name, index=False)

def plot(name, out):
    table = pd.read_csv(name)
    N = table.shape[1]
    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    fig = go.Figure(data=[go.Box(y=table[str(i)], marker_color=c[i-1]) for i in range(1, int(N+1))])
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
    )
    fig.write_html(out)

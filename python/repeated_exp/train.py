import sys
import pandas as pd
import numpy as np

from nn_model import repeated_experiment

params = sys.argv[1]
std = int(params.split("--")[0].split("=")[1])
repeat_d = int(params.split("--")[1].split("=")[1])
repeat = sys.argv[2]
repeats_by_data = int(sys.argv[3])
early_stopping = sys.argv[4] == "1"


file = f"{std}_rep_{repeat}_exp_{repeat_d}.csv"

def load_npz(path):
    data = np.load(path)
    X = data["X"]
    y = data["Y"]
    return X, y

print("loading train")
x_train, y_train = load_npz("Xy_train.npz")
print("loading val")
x_val, y_val = load_npz("Xy_val.npz")
print("loading test")
x_test, y_test = load_npz("Xy_test.npz")

score_std = []
epochs = []
list_std = []
list_early_stopping = []
indexes = []
data_rep = []
for _ in range(repeats_by_data):
    score, end_epoch = repeated_experiment((x_train, y_train),
                                            (x_val, y_val),
                                            (x_test, y_test),
                                            early_stopping)
    score_std.append(score)
    epochs.append(end_epoch)
    list_std.append(std)
    list_early_stopping.append(sys.argv[4])
    indexes.append(_)
    data_rep.append(repeat_d)

res_dic = {"std": list_std,
            "score": score_std,
            "epoch": epochs,
            "early_stopping": list_early_stopping,
            "data_rep": repeat_d}
results = pd.DataFrame(res_dic, index=indexes)
results.to_csv(file, index=False)
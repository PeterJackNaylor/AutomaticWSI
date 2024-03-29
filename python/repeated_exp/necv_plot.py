import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from simulation import load_sample
import sys

colors = {"ncv": ("rgb(31,120,180)", "rgb(166, 206, 227)", "rgba(166, 206, 227,0.1)"),
          "bncv": ("rgb(51,160,44)", "rgb(178,223,138)", "rgba(178,223,138,0.1)"),
          "bncv_top_3": ("rgb(227,26,28)", "rgb(251,154,153)", "rgba(251,154,153,0.1)"),
          "bncv_top_5": ("rgb(255,127,0)", "rgb(253,191,111)", "rgba(253,191,111,0.1)"),
          "bayes": ("rgb(106,61,154)", "rgb(106,61,154)", "rgba(106,61,154,0.1)")}



name_dic = {
    'ncv': 'NCV',
    'bncv': 'NECV-1',
    'bncv_top_3': 'NECV-3',
    'bncv_top_5': 'NECV'
}


def naive_bayes():
    list_std = list(range(1, 22))
    acc_list = []
    for std in list_std:
        x, y = load_sample(10000, 256, std)
        answers = (x.sum(axis=1) > 0).astype(int)
        acc = accuracy_score(y.argmax(axis=1), 1 - answers)
        acc_list.append(acc * 100)
    return acc_list
    


def continuous_line_plots(df):
    x = "std"
    df["score"] = df["score"] * 100
    variables = list(df["name"].unique())
    x_axis = list(df["std"].unique())
    x_axis.sort()
    children = []
    n = df.loc[(df["name"] == "ncv") & (df["std"] == 1)].copy().shape[0]
    k = 1.96 / (n ** 0.5)
    for name in variables:
        score_std_name = []
        ave_var = []
        std_var = []
        for s in x_axis:
            tmp = df.loc[(df["name"] == name) & (df["std"] == s)].copy()
            ave_var.append(tmp["score"].mean())
            std_var.append(tmp["score"].std())
        ave_var = np.array(ave_var)
        std_var = np.array(std_var)
        children += [go.Scatter(
                            name=name_dic[name],
                            x=x_axis,
                            y=ave_var,
                            mode='lines',
                            line=dict(width=0.5, color=colors[name][0]),
                        )]
        children += [go.Scatter(
                            name=f'{name} Upper Bound',
                            x=x_axis,
                            y=ave_var+k*std_var,
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0.2, color=colors[name][1]),
                            showlegend=False
                        )]
        children += [go.Scatter(
                        name=f'{name} Lower Bound',
                        x=x_axis,
                        y=ave_var-k*std_var,
                        marker=dict(color="#444"),
                        line=dict(width=0.2, color=colors[name][1]),
                        mode='lines',
                        fillcolor=colors[name][2],
                        fill='tonexty',
                        showlegend=False
                    )]
    score_bayes = naive_bayes()
    children += [go.Scatter(
                    name="Naive bayes",
                    x=x_axis,
                    y=score_bayes,
                    mode='lines',
                    line=dict(width=0.5, color=colors["bayes"][0]),
                )]
    fig = go.Figure(children)
    fig.update_layout(
        yaxis_title='Accuracy (%) (with 95% confidence intervals)',
        title='NECV vs NCV',
        xaxis_title='Standard deviation of simulated data',
        xaxis={'tickformat': ',d'},
        template='simple_white',
    )
    fig.write_html(sys.argv[2])


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    continuous_line_plots(df)

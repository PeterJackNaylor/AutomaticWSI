

import sys

import numpy as np
from numpy.lib.function_base import diff
import pandas as pd

import plotly.graph_objects as go
from tqdm import tqdm, trange

def mean_normalise(table):
    N = table.shape[0]
    n = np.arange(start=0, stop=N / 100, dtype=int)
    groups = np.repeat(n, 100)
    table['groups'] = groups
    group_mean = table.groupby('groups').mean()
    for std_es, grp in tqdm(group_mean.groupby(['std', 'early_stopping'])):
        std, es = std_es
        true_mean = grp['score'].mean()
        diff_mapping = grp['score'] - true_mean
        group_mean.loc[diff_mapping.index, "correction"] = diff_mapping

    for i in trange(0, int(N / 100)):
        ind_grp = table["groups"] == i
        table.loc[ind_grp, 'corrected_score'] = table.loc[ind_grp, "score"] + group_mean.loc[i, "correction"]
    return table


def group_mean_normalise(table):
    N = table.shape[0]
    for std_es, grp in tqdm(table.groupby(['std', 'early_stopping'])):
        std, es = std_es
        data_mean = grp.groupby('data_rep').mean()
        true_mean = data_mean['score'].mean()

        diff_mapping = data_mean['score'] - true_mean
        correct_group = grp.groupby('data_rep')
        for grp_i, grp2 in tqdm(correct_group):
            table.loc[grp2.index, "corrected_score"] =  table.loc[grp2.index, "score"] + diff_mapping.loc[grp_i]
            
    return table


def plot(name, name2=None, out=None, out_only_variance=None):
    table = pd.read_csv(name)
    if False:
        table = mean_normalise(table)
        table["score"] = table["corrected_score"]

    if True:
        table = group_mean_normalise(table)
        table["score"] = table["corrected_score"]
    N = table["std"].max()
    if name2 is not None:
        table2 = pd.read_csv(name2)
        N2 = table2.shape[1]
    else:
        table2 = table.loc[table["early_stopping"] == 0]
        N2 = table2.shape[1]
        table = table.loc[table["early_stopping"] == 1]
        N = table.shape[1]
        N = table["std"].max()


    print(N)
    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
    wiskers = []
    std_es = []
    std_nes = []
    for i in range(1, int(N+1)):
        y1 = table.loc[table["std"] == i, "score"]
        y2 = table2.loc[table2["std"] == i, "score"]
        if True:
            y1 = y1 - y1.mean() + y2.mean()
        wiskers.append(go.Box(y=y1 * 100, marker_color=c[i-1], boxmean=True))
        wiskers.append(go.Box(y=y2 * 100, marker_color=c[i-1], boxmean=True))
        std_es.append(np.std(y1*100))
        std_nes.append(np.std(y2*100))
    fig = go.Figure(data=wiskers)



    tikz = list(range(0, N))
    tikz_vals = [i + 0.5 for i in range(0, 2*N, 2)]
    print(tikz_vals)
    fig.update_layout(
        xaxis=dict(showgrid=False, 
                   zeroline=False, 
                   showticklabels=True,
                   tickvals = tikz_vals,
                   ticktext = [f"{i+1}" for i in tikz]),
        yaxis=dict(zeroline=False, gridcolor='white'),
        template='simple_white',
        showlegend=False,
        yaxis_title='Accuracy (%)',
        xaxis_title='Standard deviation of simulated data',
        yaxis_range=[60,101]
        # paper_bgcolor='rgb(233,233,233)',
        # plot_bgcolor='rgb(233,233,233)',
    )
    for val in tikz_vals:
        fig.add_vline(x=val+1, line_width=1, line_dash="dash", line_color="gray")
    # fig.show()
    fig.write_html(out)


    fig = go.Figure()
    var_es = [el*el for el in std_es]
    var_nes = [el*el for el in std_nes]
    fig.add_trace(go.Bar(
        x=tikz,
        y=var_es,
        name='With early stopping',
        marker_color='rgba(27,158,119,0.5)'
        #marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=tikz,
        y=var_nes,
        name='No early stopping',
        marker_color='rgba(217,95,2,0.5)'
        #marker_color='lightsalmon'
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group')
    fig.update_layout(
        template="ggplot2",
        title=None,
        xaxis_title="&#963; standard deviation parameter for the dataset ",
        yaxis_title="Variance",
        legend_title="Legend:",
        font=dict(size=18),
    )
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.write_html(out_only_variance)



def main():

    plot(sys.argv[1], name2=sys.argv[2], out=sys.argv[3], out_only_variance=sys.argv[4])

def main2():

    plot(sys.argv[1], name2=None, out=sys.argv[2], out_only_variance=sys.argv[3])



if __name__ == "__main__":
    main2()
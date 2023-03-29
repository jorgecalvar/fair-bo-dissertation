
import numpy as np
import pandas as pd
import plotly.express as px
import torch


class ResultExplorer:

    def __init__(self,
                 experiment_dir):

        self.experiment_dicts = []

        i = 0
        while True:
            d = experiment_dir / f'iter{i}.pt'
            if not d.exists():
                break
            self.experiment_dicts.append(torch.load(d))
            i += 1

        self.n_experiments  = len(self.experiment_dicts)

    def plot_experiment(self, i=0):

        d = self.experiment_dicts[i]

        hv = list(d['bo_hv'].numpy())
        random_hv = list(d['random_hv'].numpy())

        df = pd.DataFrame({
            'iter': [i for i in range(len(hv))] + [i for i in range(len(random_hv))],
            'hypervolume': hv + random_hv ,
            'model': ['bo'] * len(hv) + ['random'] * len(random_hv)
        })

        fig = px.line(
            df,
            x='iter',
            y='hypervolume',
            color='model'
        )
        fig.show()



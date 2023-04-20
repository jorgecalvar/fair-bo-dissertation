
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


class ResultExplorer:

    def __init__(self,
                 experiment_dir):

        self.experiment_dicts = []

        i = 0
        while True:
            d = experiment_dir / f'iter{i}.pt'
            if not d.exists():
                break
            loaded_dict = torch.load(d)
            for k, v in loaded_dict.items():
                if isinstance(v, torch.Tensor):
                    loaded_dict[k] = v.cpu()
            self.experiment_dicts.append(loaded_dict)
            i += 1

        self.n_experiments  = len(self.experiment_dicts)


    def find_hv_percentage(self):

        rp = torch.zeros((2,))

        hv_bo = torch.zeros((0,))
        hv_random = torch.zeros((0,))

        for i in range(self.n_experiments):
            d = self.experiment_dicts[i]
            hv_bo_i = DominatedPartitioning(ref_point=rp, Y=d['bo_y']).compute_hypervolume()
            hv_random_i = DominatedPartitioning(ref_point=rp, Y=d['random_y']).compute_hypervolume()
            hv_bo = torch.cat((hv_bo, hv_bo_i.unsqueeze(dim=0)))
            hv_random = torch.cat((hv_random, hv_random_i.unsqueeze(dim=0)))

        return ((hv_bo > hv_random).float().mean())


    def find_p(self):

        bo_wins = 0
        for d in self.experiment_dicts:
            if d['bo_hv'][-1].item() > d['random_hv'][-1].item():
                bo_wins += 1


        est_p = bo_wins / self.n_experiments
        err_p = 1.96 * (est_p * (1 - est_p) / self.n_experiments) ** 0.5

        print(f'Estimated p: {est_p:1.4f} +- {err_p:1.4f} = [{est_p - err_p:1.4f}, {est_p + err_p:1.4f}]')



    def plot_hv(self, i=0):

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

    def plot_scatter(self, i=0):

        d = self.experiment_dicts[i]

        bo_y = d['bo_y'].numpy()
        random_y = d['random_y'].numpy()

        rp = torch.zeros((2,))
        hv_bo = DominatedPartitioning(ref_point=rp, Y=d['bo_y']).compute_hypervolume()
        hv_random = DominatedPartitioning(ref_point=rp, Y=d['random_y']).compute_hypervolume()

        df = pd.DataFrame({
            'accuracy': list(bo_y[:, 0]) + list(random_y[:, 0]),
            'fairness': list(bo_y[:, 1]) + list(random_y[:, 1]),
            'model': ['bo'] * len(bo_y) + ['random'] * len(random_y)
        })

        fig = px.scatter(
            df,
            x='accuracy',
            y='fairness',
            color='model',
            title=f'Experiment {i+1} | Hypervolume (BO: {hv_bo:.5f} / Random: {hv_random:.5f})',
            template='plotly_white',
            range_x=[0,1.05],
            range_y=[0,1.05]
        )
        fig.show()


    def plot_several_experiments(self, min_i, max_i, title=None):

        assert (max_i - min_i) % 2 == 0

        hvs_bo = []
        hvs_random = []
        for i in range(min_i, max_i):
            d = self.experiment_dicts[i]
            rp = torch.zeros((2,))
            hv_bo = DominatedPartitioning(ref_point=rp, Y=d['bo_y']).compute_hypervolume()
            hv_random = DominatedPartitioning(ref_point=rp, Y=d['random_y']).compute_hypervolume()
            hvs_bo.append(hv_bo)
            hvs_random.append(hv_random)

        fig = make_subplots(
            cols=2, rows=(max_i-min_i)//2,
            subplot_titles=[f'Experiment {i+1} | Hypervolume (BO: {hvs_bo[row]:.5f} / Random: {hvs_random[row]:.5f}) ' for row, i in enumerate(range(min_i, max_i))]
        )

        for row, i in enumerate(range(min_i, max_i)):

            d = self.experiment_dicts[i]
            bo_y = d['bo_y'].numpy()
            random_y = d['random_y'].numpy()

            fig.add_trace(
                go.Scatter(
                    x=list(bo_y[:, 0]),
                    y=list(bo_y[:, 1]),
                    mode='markers',
                    marker=dict(color=plotly.colors.DEFAULT_PLOTLY_COLORS[0]),
                    showlegend=(row == 1),
                    name='BO'
                ),
                row=row // 2 + 1,
                col=row % 2 + 1
            )

            fig.add_trace(
                go.Scatter(
                    x=list(random_y[:, 0]),
                    y=list(random_y[:, 1]),
                    mode='markers',
                    marker=dict(color=plotly.colors.DEFAULT_PLOTLY_COLORS[1]),
                    showlegend=(row == 1),
                    name='Random'
                ),
                row=row // 2 + 1,
                col=row % 2 + 1
            )

        fig.update_layout(
            width=1200,
            height=200 + 600 * (max_i - min_i) // 2,
            template='plotly_white'
        )
        fig.update_xaxes(range=[0.6,1.05], title='accuracy')
        fig.update_yaxes(range=[0.8,1.05], title='fairness')

        if title:
            fig.update_layout(title=title)

        fig.show()






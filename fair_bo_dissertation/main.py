
from argparse import ArgumentParser
import logging
from pathlib import Path
import torch

from fair_bo_dissertation.bo import MOBO_Experiment
from fair_bo_dissertation.models import AutomaticTrainer
from fair_bo_dissertation.viz import ResultExplorer


parser = ArgumentParser()



def plot_results():
    rx = ResultExplorer(Path('experiments/experiment8'))
    # print(rx.find_hv_percentage())
    # quit()

    for i in range(5):
        rx.plot_scatter(i)
    quit()

def explore():
    rx = ResultExplorer(Path('experiments/experiment8'))
    all_bo_hv = torch.zeros((0,))
    all_random_hv = torch.zeros((0,))
    for i in range(30):
        all_bo_hv = torch.cat((all_bo_hv, rx.experiment_dicts[i]['bo_hv']))
        all_random_hv = torch.cat((all_random_hv, rx.experiment_dicts[i]['random_hv']))

    print((all_bo_hv > all_random_hv).float().mean())
    print(len(all_bo_hv))

    quit()


def explore2():
    rx = ResultExplorer(Path('experiments/experiment9'))
    rx.plot_several_experiments(0, 6, title='Dataset: Adult | Input: n1, n2')
    quit()


if __name__ == '__main__':

    explore2()

    # plot_results()

    target_function = AutomaticTrainer(calculate_epoch_metrics=False,
                                       input_vars=['n1', 'n2'])

    # metrics =  target_function(torch.tensor([0.5, 0.5]))
    # print(metrics)
    # quit()

    experiment = MOBO_Experiment(target_function)
    experiment.run_multiple(n_experiments=30)

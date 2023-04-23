
from argparse import ArgumentParser
import logging
from pathlib import Path
import torch

from fair_bo_dissertation.bo import MOBO_Experiment
from fair_bo_dissertation.models import AutomaticTrainer
from fair_bo_dissertation.viz import ResultExplorer


parser = ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train')
train_parser.add_argument('--dataset', nargs='+', type=str,
                          choices=['adult_census', 'german_credit'],
                          help='Datasets to use')
train_parser.add_argument('--acquisition', nargs='+', type=str, default='qehvi',
                          choices=['qehvi', 'nqehvi', 'qnparego'],
                          help='Acquisition function')



def plot_results():
    rx = ResultExplorer(Path('experiments/experiment8'))
    # print(rx.find_hv_percentage())
    # quit()

    for i in range(5):
        rx.plot_scatter(i)
    quit()

def explore():

    n_points = 3

    rx = ResultExplorer(Path('experiments/experiment11'))
    all_bo_hv = torch.zeros((0,))
    all_random_hv = torch.zeros((0,))
    for i in range(5):
        all_bo_hv = torch.cat((all_bo_hv, rx.experiment_dicts[i]['bo_hv'][2:]))
        all_random_hv = torch.cat((all_random_hv, rx.experiment_dicts[i]['random_hv'][2:]))

    print((all_bo_hv > all_random_hv).float().mean())
    print(len(all_bo_hv))

    quit()


def explore2():
    rx = ResultExplorer(Path('experiments/experiment11'))
    rx.plot_several_experiments(0, 4, title='Dataset: Adult | Input: lr, dropout')
    quit()


def find_p():
    rx = ResultExplorer(Path('experiments/experiment1'))
    rx.find_p()
    quit()


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    quit()

    find_p()

    # explore()

    # explore2()

    # plot_results()

    target_function = AutomaticTrainer(dataset='adult_census',
                                       calculate_epoch_metrics=False,
                                       input_vars=['n1', 'n2'])

    # metrics =  target_function(torch.tensor([0.5, 0.5]))
    # print(metrics)
    # quit()

    experiment = MOBO_Experiment(target_function,
                                 init_points=5,
                                 n_iterations=15,
                                 dir=Path('experiments/experiment0'))
    experiment.run_multiple(n_experiments=36)

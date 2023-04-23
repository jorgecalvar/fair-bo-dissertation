
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
train_parser.add_argument('--dataset', nargs='+', type=str, default='all',
                          choices=['adult_census', 'german_credit'],
                          help='Datasets to use')
train_parser.add_argument('--acquisition', nargs='+', type=str, default='all',
                          choices=['qehvi', 'nqehvi', 'qnparego'],
                          help='Acquisition function')
train_parser.add_argument('--device', type=str, default='auto',
                          choices=['cpu', 'cuda', 'auto'],
                          help='Whether to run on CPU or GPU')
train_parser.add_argument('--input_vars', nargs='+', type=str, default=['lr', 'dropout'],
                          choices=['lr', 'dropout', 'n1', 'n2'],
                          help='Hyperparameters that will be optimized')
train_parser.add_argument('--n_experiments', type=int, default=100,
                          help='How many identical experiments to run')
train_parser.add_argument('--init_points', type=int, default=5,
                          help='How many initial random points to generate before starting the experiment')
train_parser.add_argument('--n_iterations', type=int, default=15,
                          help='How many iterations to run')

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

    if args.command == 'train':

        # Dataset
        if args.dataset == 'all':
            datasets = AutomaticTrainer.VALID_DATASETS
        else:
            datasets = args.dataset

        # Device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device

        # Acquisition
        if args.acquisition == 'all':
            acquisitions = MOBO_Experiment.VALID_ACQUISITIONS
        else:
            acquisitions = args.acquisition


        # RUN

        for dataset in datasets:

            for acquisition in acquisitions:

                dir = MOBO_Experiment.create_new_dir()

                target_function = AutomaticTrainer(dataset=dataset,
                                                   calculate_epoch_metrics=False,
                                                   input_vars=args.input_vars,
                                                   device=device)

                experiment = MOBO_Experiment(target_function,
                                             init_points=args.init_points,
                                             n_iterations=args.n_iterations,
                                             acquisition=acquisition,
                                             dir=dir,
                                             device=device)

                experiment.run_multiple(n_experiments=args.n_experiments)


    print(args)

    quit()

    # find_p()

    # explore()

    # explore2()

    # plot_results()


    # metrics =  target_function(torch.tensor([0.5, 0.5]))
    # print(metrics)
    # quit()



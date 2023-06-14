
from argparse import ArgumentParser
from pathlib import Path
import torch
import yaml

from fair_bo_dissertation.bo import MOBO_Experiment
from fair_bo_dissertation.models import AutomaticTrainer
from fair_bo_dissertation.viz import ResultsExplorer, ExperimentExplorer


parser = ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

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
train_parser.add_argument('--n_points', type=int, default=1,
                          help='How many new candidates to obtain in each iteration')
train_parser.add_argument('--resume', type=str,
                          help='Resume the training of an experiment')


visualize_parser = subparsers.add_parser('visualize')
visualize_parser.add_argument('experiment', type=str,
                              help='Name of the dir where the experiment you want to visualize is')


def plot_results():
    rx = ExperimentExplorer(Path('experiments/experiment8'))
    # print(rx.find_hv_percentage())
    # quit()

    for i in range(5):
        rx.plot_scatter(i)
    quit()


def explore():

    n_points = 3

    rx = ExperimentExplorer(Path('experiments/experiment11'))
    all_bo_hv = torch.zeros((0,))
    all_random_hv = torch.zeros((0,))
    for i in range(5):
        all_bo_hv = torch.cat((all_bo_hv, rx.experiment_dicts[i]['bo_hv'][2:]))
        all_random_hv = torch.cat((all_random_hv, rx.experiment_dicts[i]['random_hv'][2:]))

    print((all_bo_hv > all_random_hv).float().mean())
    print(len(all_bo_hv))

    quit()


def explore2():
    rx = ExperimentExplorer(Path('experiments/experiment11'))
    rx.plot_several_experiments(0, 4, title='Dataset: Adult | Input: lr, dropout')
    quit()



def run_dir(experiment_dir: Path):

    with (experiment_dir / 'config.yaml').open() as f:
        config = yaml.safe_load(f)

    # Device
    if config['device'] in ('auto', 'cuda'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    target_function = AutomaticTrainer(dataset=config['dataset'],
                                       calculate_epoch_metrics=False,
                                       input_vars=config['input_vars'],
                                       device=device)

    experiment = MOBO_Experiment(target_function,
                                 init_points=config['init_points'],
                                 n_iterations=config['n_iterations'],
                                 acquisition=config['acquisition'],
                                 n_points=config['n_points'],
                                 dir=experiment_dir,
                                 device=device)

    # Check
    rx = ExperimentExplorer(experiment_dir)
    experiments_to_run = config['n_experiments'] - rx.n_experiments

    experiment.run_multiple(n_experiments=experiments_to_run)



if __name__ == '__main__':

    args = parser.parse_args()

    if args.command == 'train':

        # In resume mode, we run the experiment and resume
        if args.resume is not None:
            run_dir(MOBO_Experiment.EXPERIMENTS_DIR / args.resume)
            quit()

        # Dataset
        if args.dataset == 'all':
            datasets = AutomaticTrainer.VALID_DATASETS
        else:
            datasets = args.dataset

        # Acquisition
        if args.acquisition == 'all':
            acquisitions = MOBO_Experiment.VALID_ACQUISITIONS
        else:
            acquisitions = args.acquisition

        # CREATE DIRS WITH CONFIGS

        new_experiment_dirs = []

        for dataset in datasets:

            for acquisition in acquisitions:

                experiment_dir = MOBO_Experiment.create_new_dir()

                config = {
                    'device': args.device,
                    'dataset': dataset,
                    'acquisition': acquisition,
                    'input_vars': args.input_vars,
                    'init_points': args.init_points,
                    'n_iterations': args.n_iterations,
                    'n_experiments': args.n_experiments,
                    'n_points': args.n_points
                }

                with (experiment_dir / 'config.yaml').open('w') as f:
                    yaml.dump(config, f)

                new_experiment_dirs.append(experiment_dir)


        # RUN

        for experiment_dir in new_experiment_dirs:
            run_dir(experiment_dir)

    elif args.command == 'visualize':

        rx = ExperimentExplorer(Path(f'experiments/{args.experiment}'))
        print(rx.experiment_dicts[0])
        rx.plot_hv()

    else:

        # trainer = AutomaticTrainer(dataset='communities_crime')
        # print(trainer(torch.tensor([0.5, 0.5])))
        results_explorer = ResultsExplorer(Path('experiments'))
        results_explorer.plot_points({'dataset': 'adult_census',
                                       'input_vars': ['lr', 'dropout']})


    print(args)

    quit()

    # find_p()

    # explore()

    # explore2()

    # plot_results()


    # metrics =  target_function(torch.tensor([0.5, 0.5]))
    # print(metrics)
    # quit()



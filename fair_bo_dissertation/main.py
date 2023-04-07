
from argparse import ArgumentParser
import logging
from pathlib import Path
import torch

from fair_bo_dissertation.bo import MOBO_Experiment
from fair_bo_dissertation.models import AutomaticTrainer
from fair_bo_dissertation.viz import ResultExplorer

parser = ArgumentParser()





if __name__ == '__main__':

    rx = ResultExplorer(Path('experiments/experiment7'))
    for i in range(20):
        rx.plot_experiment(i)
    quit()

    target_function = AutomaticTrainer(dataset='german_credit',
                                       calculate_epoch_metrics=False)
    # target_function(torch.tensor([0.5, 0.5]))

    experiment = MOBO_Experiment(target_function)
    experiment.run_multiple(n_experiments=20)

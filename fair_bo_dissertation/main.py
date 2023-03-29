
from argparse import ArgumentParser
import logging
from pathlib import Path

from fair_bo_dissertation.bo import MOBO_Experiment
from fair_bo_dissertation.models import AutomaticTrainer
from fair_bo_dissertation.viz import ResultExplorer

parser = ArgumentParser()





if __name__ == '__main__':

    # rx = ResultExplorer(Path('experiments/experiment10'))
    # rx.plot_experiment(0)
    # quit()

    target_function = AutomaticTrainer()
    experiment = MOBO_Experiment(target_function)

    experiment.run()
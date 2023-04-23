
import logging
from pathlib import Path
import torch
import time

from .candidate_search import qEHVI_CandidateSearcher, qNEHVI_CandidateSearcher, qNParEGO_CandidateSearcher, RandomCandidateSearcher

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


class MOBO_Experiment:
    """Run an experiment in which we aim to maximize the objectives of the target function.
    Init point are generated randomly from a uniform U[0, 1) distribution."""

    EXPERIMENTS_DIR = Path('experiments')
    VALID_ACQUISITIONS = ('qehvi', 'nqehvi', 'qnparego', 'random')

    def __init__(self,
                 target_function,
                 n_iterations=10,
                 init_points=2,
                 n_points=1,
                 acquisition='qehvi',
                 dir=None,
                 device='cpu'
                 ):

        self.device = device
        self.target_function = target_function
        self.n_input_vars = len(target_function.input_vars)
        self.n_iterations = n_iterations
        self.init_points = init_points
        self.n_points = n_points
        self.n_objectives = len(target_function.metrics)
        self.reference_point = torch.tensor([0] * self.n_objectives, device=self.device)
        self.acquisition = acquisition
        if self.acquisition not in self.VALID_ACQUISITIONS:
            raise ValueError(f'The specified acquisition ({acquisition}) is not valid.')

        if dir is None:
            self.dir = self.create_new_dir()
        else:
            self.dir = dir

        # Bounds
        self.bounds = torch.tensor([[0.] * self.n_input_vars, [1.] * self.n_input_vars], device=self.device)

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.log(logging.DEBUG, 'Logger created')

    def _target_function(self, x):
        a = [self.target_function(i) for i in x]
        return torch.stack(a)

    def run_multiple(self,
                     n_experiments,
                     verbose=True):
        for i in range(n_experiments):
            print(f'Running experiment {i}...')
            self.run()

    def run(self):

        x = torch.rand((self.init_points, self.n_input_vars), device=self.device)
        y = self._target_function(x)
        hv = torch.zeros((self.init_points,), device=self.device)
        for i in range(len(y)):
            hv[i] = DominatedPartitioning(ref_point=self.reference_point,
                                          Y=y[:i+1]).compute_hypervolume()


        if self.acquisition == 'qehvi':
            searcher = qEHVI_CandidateSearcher(self.bounds, self.reference_point, device=self.device)
        elif self.acquisition == 'nqehvi':
            searcher = qNEHVI_CandidateSearcher(self.bounds, self.reference_point, device=self.device)
        elif self.acquisition == 'qnparego':
            searcher = qNParEGO_CandidateSearcher(self.bounds, self.reference_point, device=self.device)
        else:
            searcher = RandomCandidateSearcher()

        total_time_per_iteration = []
        get_candidates_time_per_iteration = []
        target_function_time_per_iteration = []

        for i in range(self.n_iterations):

            ts_total = time.perf_counter()

            # Get candidates
            ts_candidates = time.perf_counter()
            new_x = searcher.get_candidates(x,
                                            y,
                                            n_points=self.n_points)
            get_candidates_time_per_iteration.append(time.perf_counter() - ts_candidates)

            # Target function
            ts_target = time.perf_counter()
            new_y = self._target_function(new_x)
            target_function_time_per_iteration.append(time.perf_counter() - ts_target)

            # Concatenate
            x = torch.cat((x, new_x), dim=0)
            y = torch.cat((y, new_y), dim=0)

            # Obtain HV
            new_hv = DominatedPartitioning(ref_point=self.reference_point,
                                           Y=y).compute_hypervolume()
            hv = torch.cat((hv, new_hv.unsqueeze(0)), dim=0)

            total_time_per_iteration.append(time.perf_counter() - ts_total)

        # Save
        experiment_dict = {'x': x,
                           'y': y,
                           'hv': hv,
                           'total_time': total_time_per_iteration,
                           'get_candidates_time': get_candidates_time_per_iteration,
                           'target_function_time': target_function_time_per_iteration}

        torch.save(experiment_dict, self.get_next_iter_path(self.dir))


    @classmethod
    def create_new_dir(cls, base_name=None):
        base_name = base_name or 'experiment'
        i = 0
        while True:
            d = cls.EXPERIMENTS_DIR / f'{base_name}{i}'
            if not d.exists():
                d.mkdir()
                return d
            i += 1

    @classmethod
    def get_next_iter_path(cls, experiment_dir):
        i = 0
        while True:
            d = experiment_dir / f'iter{i}.pt'
            if not d.exists():
                return d
            i += 1



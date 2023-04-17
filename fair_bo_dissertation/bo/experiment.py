
import logging
from pathlib import Path
import torch

from .utils import get_candidates

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


class MOBO_Experiment:
    """Run an experiment in which we aim to maximize the objectives of the target function.
    Init point are generated randomly from a uniform U[0, 1) distribution."""

    EXPERIMENTS_DIR = Path('experiments')

    def __init__(self,
                 target_function,
                 n_iterations=10,
                 init_points=2,
                 n_points=1,
                 run_random=True,
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
        self.run_random = run_random
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
                     n_experiments=20):
        for _ in range(n_experiments):
            self.run()

    def run(self):

        x = torch.rand((self.init_points, self.n_input_vars), device=self.device)
        y = self._target_function(x)
        hv = torch.zeros((self.init_points,), device=self.device)
        for i in range(len(y)):
            hv[i] = DominatedPartitioning(ref_point=self.reference_point,
                                          Y=y[:i+1]).compute_hypervolume()

        if self.run_random:
            random_x = torch.clone(x.detach())
            random_y = torch.clone(y.detach())
            random_hv = torch.clone(hv.detach())

        for i in range(self.n_iterations):

            new_x = get_candidates(x,
                                   y,
                                   self.bounds,
                                   self.reference_point,
                                   n_points=self.n_points)
            new_y = self._target_function(new_x)

            x = torch.cat((x, new_x), dim=0)
            y = torch.cat((y, new_y), dim=0)

            new_hv = DominatedPartitioning(ref_point=self.reference_point,
                                           Y=y).compute_hypervolume()

            hv = torch.cat((hv, new_hv.unsqueeze(0)), dim=0)



            if self.run_random:

                new_random_x = torch.rand((self.n_points, self.n_input_vars), device=self.device)
                new_random_y = self._target_function(new_random_x)

                random_x = torch.cat([random_x, new_random_x])
                random_y = torch.cat([random_y, new_random_y])

                new_random_hv = DominatedPartitioning(ref_point=self.reference_point,
                                                      Y=random_y).compute_hypervolume()

                random_hv = torch.cat((random_hv, new_random_hv.unsqueeze(0)), dim=0)

        # Save
        experiment_dict = {'bo_x': x,
                           'bo_y': y,
                           'bo_hv': hv,
                           'init_points': self.init_points}

        if self.run_random:
            experiment_dict['random_x'] = random_x
            experiment_dict['random_y'] = random_y
            experiment_dict['random_hv'] = random_hv

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



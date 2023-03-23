
from pathlib import Path
import torch
from utils import get_candidates


class MOBO_Experiment:
    """Run an experiment in which we aim to maximize the objectives of the target function."""

    EXPERIMENTS_DIR = Path('experiments')

    def __init__(self,
                 target_function,
                 init_generator,
                 bounds,
                 n_iterations=20,
                 init_points=10,
                 n_points=1,
                 run_random=True,
                 name=None
                 ):

        self.target_function = target_function
        self.init_generator = init_generator
        self.n_iterations = n_iterations
        self.init_points = init_points
        self.n_points = n_points
        self.bounds = bounds
        self.n_objectives = self.bounds.shape[1]
        self.reference_point = torch.tensor([0] * self.n_objectives)
        self.run_random = run_random
        self.name = self.get_new_name(name)

    def _target_function(self, x):
        return torch.tensor([self.target_function(i) for i in x])

    def run(self):

        x, y = self.init_generator(self.init_points)

        if self.run_random:
            random_x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
                (self.init_points, self.n_objectives))
            random_y = self._target_function(random_x)

        for i in range(self.n_iterations):

            new_x = self.get_candidates(x,
                                        y,
                                        self.bounds,
                                        n=self.n_points)
            new_y = self._target_function(new_x)

            x = torch.cat((x, new_x), dim=0)
            y = torch.cat((y, new_y), dim=0)

            if self.run_random:

                new_random_x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
                    (self.n_point, self.n_objectives))
                new_random_y = self._target_function(new_random_x)

                random_x = torch.cat([random_x, new_random_x])
                random_y = torch.cat([random_x, new_random_y])

            # Save

    @classmethod
    def get_new_name(cls, name):
        base_name = name or ''
        i = 0
        while True:
            d = cls.EXPERIMENTS_DIR / f'{base_name}{i}.pt'
            if not d.exists():
                return d
            i += 1



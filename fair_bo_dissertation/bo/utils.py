
from abc import ABC, abstractmethod

from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_list
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

import torch


class BO_Optimizer:

    NUM_RESTARTS = 200
    RAW_SAMPLES = 512

    def __init__(self,
                 bounds,
                 reference_point,
                 n_points,
                 device='cpu'):
        self.bounds = bounds
        self.reference_point = reference_point
        self.device = device

    def get_candidates(self,
                       x,
                       y,
                       n_points=1):

        # Get models
        model = ModelListGP(*[SingleTaskGP(x.double(), y[:, i:i+1].double(), outcome_transform=Standardize(m=1)) for i in range(y.shape[1])])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Sampler
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        candidates = self._optimize()

        return candidates

    @abstractmethod
    def _optimize(self, model, x, y, sampler, n_points=1):
        pass




class qEHVI_Optimizer(BO_Optimizer):

    def _optimize(self, model, x, y, sampler, n_points=1):

        # Partitioning
        with torch.no_grad():
            pred = model.posterior(x).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.reference_point,
            Y=pred,
        )

        # Acquitisition function
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.reference_point,
            partitioning=partitioning,
            sampler=sampler,
        )

        # FastNondominatedPartitioning(ref_point=reference_point, Y=torch.Tensor([[0.2, 0.8], [0.7, 0.3]]),)

        # Optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_points,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        return candidates



class qNEHVI_Optimizer(BO_Optimizer):

    def _optimize(self, model, x, y, sampler, n_points=1):

        # Acquisition function
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.reference_point,
            X_baseline=x,
            prune_baseline=True,
            sampler=sampler

        )

        # Optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_points,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True
        )

        return candidates



class qNParEGO_Optimizer(BO_Optimizer, n_points=1):

    def _optimize(self, model, x, y, sampler, n_points=1):

        with torch.no_grad():
            pred = model.posterior(x).mean

        acq_func_list = []
        for _ in range(n_points):
            weights = sample_simplex(2, device=self.device).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)

        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates



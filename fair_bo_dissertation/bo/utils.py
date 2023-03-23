
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning


import torch


def get_candidates(x,
                   y,
                   bounds,
                   reference_point,
                   n_points=1):

    # Get models
    model = ModelListGP(*[SingleTaskGP(x, y[:, i:i+1]) for i in range(y.shape[1])])
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Sampler
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

    # Partitioning
    with torch.no_grad():
        pred = model.posterior(x).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=reference_point,
        Y=pred,
    )

    # Acquitisition function
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    # Optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=200,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    return candidates


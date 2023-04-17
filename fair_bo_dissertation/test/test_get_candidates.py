
import torch
from fair_bo_dissertation.bo.utils import get_candidates
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import plotly.express as px
import pandas as pd
import tqdm


def beale_function(x1, x2):
    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
    term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
    return term1 + term2 + term3


def beale_function_moo(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((beale_function(x1, x2),
                        beale_function(x2, x1)),
                       dim=-1)



def test_get_candidates():

    print('STARTING TESTING OF get_candidates...')

    init_points = 5
    reference_point = torch.tensor([0, 0])
    n_iterations = 10
    n_points = 1

    bounds = torch.tensor([[-4.5, -4.5], [4.5, 4.5]])

    x = torch.rand((init_points, 2))
    y = beale_function_moo(x)
    hv = torch.zeros((init_points,))
    for i in range(len(y)):
        hv[i] = DominatedPartitioning(ref_point=reference_point,
                                      Y=y[:i + 1]).compute_hypervolume()

    random_x = torch.clone(x.detach())
    random_y = torch.clone(y.detach())
    random_hv = torch.clone(hv.detach())

    for _ in tqdm.tqdm(range(n_iterations)):

        new_x = get_candidates(x,
                               y,
                               bounds,
                               reference_point,
                               n_points=n_points)
        new_y = beale_function_moo(new_x)

        x = torch.cat((x, new_x), dim=0)
        y = torch.cat((y, new_y), dim=0)

        new_hv = DominatedPartitioning(ref_point=reference_point,
                                       Y=y).compute_hypervolume()

        hv = torch.cat((hv, new_hv.unsqueeze(0)), dim=0)

        new_random_x = torch.rand((n_points, 2))
        new_random_y = beale_function_moo(new_random_x)

        random_x = torch.cat([random_x, new_random_x])
        random_y = torch.cat([random_y, new_random_y])

        new_random_hv = DominatedPartitioning(ref_point=reference_point,
                                              Y=random_y).compute_hypervolume()

        random_hv = torch.cat((random_hv, new_random_hv.unsqueeze(0)), dim=0)


    # PLOT

    df = pd.DataFrame({
        'iteration': list(range(len(hv))) + list(range(len(random_hv))),
        'hypervolume': list(torch.log(hv)) + list(torch.log(random_hv)),
        'model': ['bo'] * len(hv) + ['random'] * len(random_hv)
    })


    fig = px.line(
        df,
        x='iteration',
        y='hypervolume',
        color='model',
        template='plotly_white',
        title='Test of MOO with the Beale function'
    )
    fig.show()



if __name__ == '__main__':
    test_get_candidates()

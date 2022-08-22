import torch.linalg as la
from torch import arccosh

import torch


def poincarre_dist(x,y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return arccosh(\
    1 + 2*(\
        la.norm(x-y, ord=1)/((1-la.norm(x, ord=1))*(1-la.norm(y, ord=1)))
        )
    )
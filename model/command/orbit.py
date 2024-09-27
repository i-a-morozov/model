"""
Orbit
-----

Functionality related to (parametric) closed orbit and fixed points computation

"""
from typing import Optional
from typing import Callable

import torch
from torch import Tensor

from ndmap.pfp import fixed_point

from model.library.line import Line

def orbit(line:Line|Callable,
          guess:Tensor,
          *pars: tuple,
          limit:int=32,
          power:int=1,
          epsilon:float=1.0E-12,
          factor:float=1.0,
          alpha:float=0.0,
          solve:Optional[Callable]=None,
          roots:Optional[Tensor]=None,
          jacobian:Optional[Callable]=None):
    """
    Estimate (dynamical) fixed point

    Parameters
    ----------
    line: Line|Callable
        input line
    guess: Tensor
        initial guess
    *pars: tuple
        additional function arguments
    limit: int, positive
        maximum number of newton iterations
    power: int, positive, default=1
        function power / fixed point order
    epsilon: Optional[float], default=None
        tolerance epsilon
    factor: float, default=1.0
        step factor (learning rate)
    alpha: float, positive, default=0.0
        regularization alpha
    solve: Optional[Callable]
        linear solver(matrix, vector)
    roots: Optional[Tensor], default=None
        known roots to avoid
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    return fixed_point(limit,
                       line,
                       guess,
                       *pars,
                       power=power,
                       epsilon=epsilon,
                       factor=factor,
                       alpha=alpha,
                       solve=solve,
                       roots=roots,
                       jacobian=jacobian)

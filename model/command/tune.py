"""
Tune
----

Functionality related to computation of (parametric) tunes and chromaticities

"""
from typing import Callable
from typing import Optional

import torch
from torch import Tensor

from twiss import twiss

from model.library.line import Line

from model.command.mapping import matrix

def tune(line:Line,
         parameters:list[Tensor],
         *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
         alignment:bool=False,
         matched:bool=False,
         guess:Optional[Tensor]=None,
         limit:int=1,
         epsilon:float=None,
         solve:Optional[Callable]=None,
         jacobian:Optional[Callable]=None) -> tuple[Tensor, list[tuple[str|None, list[str], str]]]:
    """
    Compute parametric tunes

    Parameters
    ----------
    line: Line
        input line
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    alignment: bool, default=True
        flag to include the alignment parameters in the default deviation table
    matched: bool, default=False
        flag to return mapping around closed orbit
    guess: Tensor, default=None
        closed orbit guess
    limit: int, positive
        maximum number of newton iterations
    epsilon: Optional[float], default=1.0E-12
        tolerance epsilon
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    tuple[Tensor, list[tuple[str|None, list[str], str]]]

    """
    function, *_ = matrix(line,
                          0,
                          len(line) - 1,
                          *groups,
                          root=True,
                          alignment=alignment,
                          matched=matched,
                          guess=guess,
                          limit=limit,
                          epsilon=epsilon,
                          solve=solve,
                          jacobian=jacobian)
    state = torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)
    tunes, *_ = twiss(function(state, *parameters))
    return tunes


def chromaticity(line:Line,
                 parameters:list[Tensor],
                 *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
                 alignment:bool=False,
                 matched:bool=False,
                 guess:Optional[Tensor]=None,
                 limit:int=1,
                 epsilon:float=None,
                 solve:Optional[Callable]=None,
                 jacobian:Optional[Callable]=None) -> tuple[Tensor, list[tuple[str|None, list[str], str]]]:
    """
    Compute parametric chromaticities

    Parameters
    ----------
    line: Line
        input line
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    alignment: bool, default=True
        flag to include the alignment parameters in the default deviation table
    matched: bool, default=False
        flag to return mapping around closed orbit
    guess: Tensor, default=None
        closed orbit guess
    limit: int, positive
        maximum number of newton iterations
    epsilon: Optional[float], default=1.0E-12
        tolerance epsilon
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    tuple[Tensor, list[tuple[str|None, list[str], str]]]

    """

    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    dp = torch.tensor([0.0], dtype=line.dtype, device=line.device)

    def task(dp):
        return tune(line,
                    [dp, *parameters],
                    ('dp', None, None, None),
                    *groups,
                    alignment=alignment,
                    matched=matched,
                    guess=guess,
                    limit=limit,
                    epsilon=epsilon,
                    solve=solve,
                    jacobian=jacobian)

    return jacobian(task)(dp).squeeze()
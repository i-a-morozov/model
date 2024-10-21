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

from model.command.wrapper import group
from model.command.orbit import orbit

def tune(line:Line,
         parameters:list[Tensor],
         *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
         alignment:bool=False,
         matched:bool=False,
         guess:Optional[Tensor]=None,
         limit:int=1,
         epsilon:Optional[float]=None,
         solve:Optional[Callable]=None,
         jacobian:Optional[Callable]=None) -> Tensor:
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
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
        tolerance epsilon
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    guess = guess if isinstance(guess, Tensor) else torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)
    point = torch.zeros_like(guess)

    if matched:
        point, *_ = orbit(line,
                          guess,
                          parameters,
                          *groups,
                          alignment=alignment,
                          advance=False,
                          limit=limit,
                          epsilon=epsilon,
                          solve=solve,
                          jacobian=jacobian)

    mapping, *_ = group(line,
                        0,
                        len(line) - 1,
                        *groups,
                        alignment=alignment,
                        root=True)

    def wrapper(state, *parameters):
        return mapping(state + point, *parameters) - point

    state = torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)
    tunes, *_ = twiss(jacobian(wrapper)(state, *parameters))

    return tunes


def chromaticity(line:Line,
                 parameters:list[Tensor],
                 *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
                 alignment:bool=False,
                 matched:bool=False,
                 guess:Optional[Tensor]=None,
                 limit:int=1,
                 epsilon:Optional[float]=None,
                 solve:Optional[Callable]=None,
                 jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute parametric chromaticities

    Parameters
    ----------
    line: Line
        input linetune
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
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
        tolerance epsilon
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian
    dp = torch.tensor([0.0], dtype=line.dtype, device=line.device)
    def wrapper(dp):
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
    return jacobian(wrapper)(dp).squeeze()
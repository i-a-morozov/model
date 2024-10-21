"""
Twiss
-----

Functionality related to computation of (parametric) Twiss parameters

"""
from typing import Callable
from typing import Optional

import torch
from torch import Tensor

from twiss import twiss as wolski
from twiss import propagate
from twiss import wolski_to_cs

from model.library.line import Line

from model.command.wrapper import group
from model.command.orbit import orbit

def twiss(line:Line,
          parameters:list[Tensor],
          *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
          alignment:bool=False,
          convert:bool=True,
          matched:bool=False,
          guess:Optional[Tensor]=None,
          advance:bool=False,
          full:bool=True,
          limit:int=1,
          epsilon:Optional[float]=None,
          solve:Optional[Callable]=None,
          jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute Twiss parameters

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
    corvert: bool, default=True
        flag to convert Twiss matrices to CS Twiss parameters
    matched: bool, default=False
        flag to return mapping around closed orbit
    guess: Tensor, default=None
        closed orbit guess
    advance: bool, default=False
        flag to advance the parametric orbit over elements or lines
    full: bool, default=False
        flag to perform full propagation
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
    matrix = jacobian(wrapper)(state, *parameters)
    *_, tm = wolski(matrix)

    if not advance:
        return tm if not convert else wolski_to_cs(tm)

    tms = [tm]
    for i in range(len(line) - (not full)):
        mapping, *_ = group(line, i, i, *groups, alignment=alignment, root=True)
        def wrapper(state, *parameters):
            return mapping(state + point, *parameters) - point
        matrix = jacobian(wrapper)(state, *parameters)
        tm = propagate(tm, matrix)
        tms.append(tm)
        point = mapping(point, *parameters)
    tms = torch.stack(tms)

    return tms if not convert else torch.vmap(wolski_to_cs)(tms)


def chromatic_twiss(line:Line,
                    parameters:list[Tensor],
                    *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
                    alignment:bool=False,
                    convert:bool=True,
                    matched:bool=False,
                    guess:Optional[Tensor]=None,
                    advance:bool=False,
                    full:bool=True,
                    limit:int=1,
                    epsilon:Optional[float]=None,
                    solve:Optional[Callable]=None,
                    roots:Optional[Tensor]=None,
                    jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute chromatic Twiss parameters

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
    corvert: bool, default=True
        flag to convert Twiss matrices to CS Twiss parameters
    matched: bool, default=False
        flag to return mapping around closed orbit
    guess: Tensor, default=None
        closed orbit guess
    advance: bool, default=False
        flag to advance the parametric orbit over elements or lines
    full: bool, default=False
        flag to perform full propagation
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
        return twiss(line,
                    [dp, *parameters],
                    ('dp', None, None, None),
                    *groups,
                    alignment=alignment,
                    convert=convert,
                    matched=matched,
                    guess=guess,
                    advance=advance,
                    full=full,
                    limit=limit,
                    epsilon=epsilon,
                    solve=solve,
                    jacobian=jacobian)
    return jacobian(wrapper)(dp).squeeze()
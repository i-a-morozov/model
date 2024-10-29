"""
Coupling
--------

Functionality related to parametric coupling computation

"""
from typing import Callable
from typing import Optional

import torch
from torch import Tensor

from twiss import twiss
from twiss import symplectic_conjugate

from model.library.line import Line

from model.command.wrapper import group
from model.command.orbit import orbit

def coupling(line:Line,
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
    Compute parametric minimal tune separation

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
    matrix = jacobian(wrapper)(state, *parameters)

    (nux, nuy), *_ = twiss(matrix)
    mux, muy =  2.0*torch.pi*nux, 2.0*torch.pi*nuy

    mb = matrix[:2, 2:]
    mc = matrix[2:, :2]

    (m11, m12), (m21, m22) = mc + symplectic_conjugate(mb)

    return 1.0/torch.pi*(m11*m22 - m12*m21).abs().sqrt()/(mux.sin() + muy.sin()).abs()
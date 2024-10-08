"""
Orbit
-----

Functionality related to (parametric) closed orbit and fixed points computation

Parameters and functions

orbit                : compute (dynamical) closed orbit
parametric_orbit     : compute parametric closed orbit (or fixed point)
ORM                  : compute orbit response matrix
ORM_IJ               : compute IJ orbit response matrix element

"""
from typing import Optional
from typing import Callable

import torch
from torch import Tensor

from ndmap.derivative import Table
from ndmap.propagate import identity
from ndmap.propagate import propagate
from ndmap.pfp import fixed_point
from ndmap.pfp import parametric_fixed_point

from model.library.line import Line
from model.command.wrapper import group


def orbit(line:Line,
          guess:Tensor,
          parameters:list[Tensor],
          *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
          start:Optional[int|str]=None,
          alignment:bool=False,
          advance:bool=False,
          full:bool=True,
          limit:int=32,
          power:int=1,
          epsilon:float=1.0E-12,
          factor:float=1.0,
          alpha:float=0.0,
          solve:Optional[Callable]=None,
          roots:Optional[Tensor]=None,
          jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute (dynamical) closed orbit

    Note, this function is differentiable with respect to deviation groups

    Parameters
    ----------
    line: Line
        input line (one - turn)
    guess: Tensor
        initial guess
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    start: Optional[int|str], default=None
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    advance: bool, default=False
        flag to advance the parametric orbit over elements or lines
    full: bool, default=False
        flag to perform full propagation
    limit: int, positive
        maximum number of newton iterations
    power: int, positive, default=1
        function power / fixed point order
    epsilon: Optional[float], default=1.0E-12
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
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    line = line.clone()
    if start:
        line.start = start

    mapping, table, _ = group(line, 0, len(line) - 1, *groups, alignment=alignment, root=True)

    point = fixed_point(limit,
                        mapping,
                        guess,
                        *parameters,
                        power=power,
                        epsilon=epsilon,
                        factor=factor,
                        alpha=alpha,
                        solve=solve,
                        roots=roots,
                        jacobian=jacobian)

    if not advance:
        return point, table

    points = [point]

    *most, last = line.sequence
    for element in most + full*[last]:
        mapping, *_ = group(line, element.name, element.name, *groups, alignment=alignment, root=True)
        point = mapping(point, *parameters)
        points.append(point)

    return torch.stack(points), table


def parametric_orbit(line:Line,
                     point:Tensor,
                     parameters:list[Tensor],
                     *groups:tuple[int, str, list[str]|None, list[str]|None, list[str|None]],
                     start:Optional[int|str]=None,
                     alignment:bool=False,
                     advance:bool=False,
                     full:bool=True,
                     power:int=1,
                     solve:Optional[Callable]=None,
                     jacobian:Optional[Callable]=None) -> tuple[Table|list[Table], list[tuple[None, list[str], str]], list[int]]:
    """
    Compute parametric closed orbit (or fixed point)
    Use this function to construct multivariate Taylor series surrogate model for closed orbit

    Parameters
    ----------
    line: Line
        input line (one-turn)
    point: Tensor
        fixed point
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[int,str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        orders, kinds, names, clean (list of element names)
    start: Optional[int|str], default=None
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    advance: bool, default=False
        flag to advance the parametric orbit over elements or lines
    full: bool, default=False
        flag to perform full propagation
    power: int, positive, default=1
        function power / fixed point order
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    tuple[Table|list[Table], list[tuple[None, list[str], str]], list[int]]


    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    line = line.clone()
    if start:
        line.start = start

    orders, *groups = zip(*groups)
    groups = tuple(zip(*groups))

    mapping, table, _ = group(line, 0, len(line) - 1, *groups, alignment=alignment, root=True)

    orbit = parametric_fixed_point(orders,
                                   point,
                                   parameters,
                                   mapping,
                                   power=power,
                                   solve=solve,
                                   jacobian=jacobian)

    if not advance:
        return orbit, table, orders

    orbits = [orbit]
    runner = identity((0, *orders), point, parametric=orbit)

    *most, last = line.sequence
    for element in most + full*[last]:
        mapping, *_ = group(line, element.name, element.name, *groups, alignment=alignment, root=True)
        runner = propagate((len(point), *map(len, parameters)), (0, *orders), runner, parameters, mapping, jacobian=jacobian)
        orbits.append(runner)

    return orbits, table, orders


def ORM(line:Line,
        point:Tensor, *,
        exclude:Optional[list[str]]=None,
        start:Optional[int|str]=None,
        alignment:bool=False,
        limit:int=32,
        epsilon:float=1.0E-12,
        factor:float=1.0,
        alpha:float=0.0,
        solve:Optional[Callable]=None,
        roots:Optional[Tensor]=None,
        jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute orbit response matrix

    Parameters
    ----------
    line: Line
        input line (one-turn)
    point: Tensor
        fixed point
    exclude: Optional[list[str]]
        list of element names to exclude
    start: Optional[int|str], default=None
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    limit: int, positive
        maximum number of newton iterations
    epsilon: Optional[float], default=1.0E-12
        tolerance epsilon
    factor: float, default=1.0
        step factor (learning rate)
    alpha: float, positive, default=0.0
        regularization alpha
    solve: Optional[Callable]
        linear solver(matrix, vector)
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    count:int = line.describe['Corrector']
    if exclude:
        count -= len(exclude)

    cx = torch.tensor(count*[0.0], dtype=line.dtype, device=line.device)
    cy = torch.tensor(count*[0.0], dtype=line.dtype, device=line.device)

    cxy = torch.cat([cx, cy])

    def task(cxy):
        cx, cy = cxy.reshape(1 + 1, -1)
        points, _ = orbit(line,
                          point,
                          [cx, cy],
                          ('cx', ['Corrector'], None, exclude),
                          ('cy', ['Corrector'], None, exclude),
                          start=start,
                          advance=True,
                          full=False,
                          alignment=alignment,
                          limit=limit,
                          epsilon=epsilon,
                          factor=factor,
                          alpha=alpha,
                          solve=solve,
                          jacobian=jacobian)

        qx, _, qy, _ = points.T

        return torch.stack([qx, qy])

    return torch.func.jacrev(task)(cxy).reshape(-1, *cxy.shape)




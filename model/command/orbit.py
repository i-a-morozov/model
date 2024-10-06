"""
Orbit
-----

Functionality related to (parametric) closed orbit and fixed points computation

Parameters and functions

orbit                : compute (dynamical) closed orbit
parametric_orbit     : compute parametric closed orbit (or fixed point)

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

    Parameters
    ----------
    line: Line
        input line (one - turm)
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
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    jacobian = torch.func.jacrev if jacobian is None else jacobian

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

    Parameters
    ----------
    line: Line|Callable
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
    jacobian = torch.func.jacrev if jacobian is None else jacobian

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

"""
Orbit
-----

Functionality related to (parametric) closed orbit and fixed points computation

Parameters and functions

orbit                : compute (dynamical) closed orbit
parametric_orbit     : compute parametric closed orbit (or fixed point)
ORM                  : compute orbit response matrix
ORM_IJ               : compute IJ orbit response matrix element
dispersion           : compute dispersion
ORM_DP               : compute ORM derivative with respect to momentum deviation

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
          start:int=0,
          respect:bool=True,
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
          jacobian:Optional[Callable]=None) -> tuple[Tensor, list[tuple[None, list[str], str]]]:
    """
    Compute (dynamical) closed orbit

    Note, this function is differentiable with respect to deviation groups
    If orbit distortions comes only from thin correctors, limit=1 gives exact solution

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
    start: int, default=0
        start element index or name (change start)
    respect: bool, default=True
        flag to respect original element orderding (if start is changed)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    advance: bool, default=False
        flag to advance the parametric orbit over elements or lines
    full: bool, default=False
        flag to perform full propagation
    limit: int, positive, default=32
        maximum number of newton iterations
    power: int, positive, default=1
        function power / fixed point order
    epsilon: float, default=1.0E-12
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
    tuple[Tensor, list[tuple[None, list[str], str]]]

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian

    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1), driver='gels').solution.squeeze()

    if start and respect:
        with torch.no_grad():
            _, table, _ = group(line, 0, len(line) - 1, *groups, alignment=alignment, root=True)
            groups = []
            for (_, names, key) in table:
                groups.append((key, None, names, None))

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
                     start:int=0,
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
    start: int, default=0
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
        guess:Tensor,
        parameters:list[Tensor],
        *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
        exclude:Optional[list[str]]=None,
        start:int=0,
        alignment:bool=False,
        limit:int=1,
        epsilon:Optional[float]=None,
        factor:float=1.0,
        alpha:float=0.0,
        solve:Optional[Callable]=None,
        roots:Optional[Tensor]=None,
        jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute orbit response matrix

    Note, it the initial guess is correct dynamical closed orbit
    Only one iteration is sufficient (set limit=1)

    Parameters
    ----------
    line: Line
        input line (one-turn)
    guess: Tensor
        initial guess
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    exclude: Optional[list[str]]
        list of element names to exclude
    start:int, default=0
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
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
                          guess,
                          [cx, cy, *parameters],
                          ('cx', ['Corrector'], None, exclude),
                          ('cy', ['Corrector'], None, exclude),
                          *groups,
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

    return jacobian(task)(cxy).reshape(-1, *cxy.shape)


def ORM_IJ(line:Line,
           guess:Tensor,
           probe:int,
           other:int,
           parameters:list[Tensor],
           *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
           alignment:bool=False,
           limit:int=1,
           epsilon:Optional[float]=None,
           factor:float=1.0,
           alpha:float=0.0,
           solve:Optional[Callable]=None,
           roots:Optional[Tensor]=None,
           jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute ij orbit response matrix element

    Note, it the initial guess is correct dynamical closed orbit
    Only one iteration is sufficient (set limit=1)

    Parameters
    ----------
    line: Line
        input line (one-turn, flat)
    guess: Tensor
        initial guess
    probe: int
        observation location index
    other: int
        corrector location index
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
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

    cx = torch.tensor([0.0], dtype=line.dtype, device=line.device)
    cy = torch.tensor([0.0], dtype=line.dtype, device=line.device)

    cxy = torch.cat([cx, cy])

    def task(cxy):
        cx, cy = cxy.reshape(1 + 1, -1)
        point, _ = orbit(line,
                        guess,
                        [cx, cy, *parameters],
                        ('cx', None, [line.names[other]], None),
                        ('cy', None, [line.names[other]], None),
                        *groups,
                        start=probe,
                        advance=False,
                        full=False,
                        alignment=alignment,
                        limit=limit,
                        epsilon=epsilon,
                        factor=factor,
                        alpha=alpha,
                        solve=solve,
                        jacobian=jacobian)
        qx, _, qy, _ = point
        return torch.stack([qx, qy])

    return jacobian(task)(cxy)


def dispersion(line:Line,
               guess:Tensor,
               parameters:list[Tensor],
               *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
               start:int=0,
               alignment:bool=False,
               limit:int=1,
               epsilon:Optional[float]=None,
               factor:float=1.0,
               alpha:float=0.0,
               solve:Optional[Callable]=None,
               roots:Optional[Tensor]=None,
               jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute dispersion

    Note, it the initial guess is correct dynamical closed orbit
    Only one iteration is sufficient (set limit=1)

    Parameters
    ----------
    line: Line
        input line (one-turn)
    guess: Tensor
        initial guess
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    start:int, default=0
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
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

    dp = torch.tensor([0.0], dtype=line.dtype, device=line.device)

    def task(dp):
        points, _ = orbit(line,
                          guess,
                          [dp, *parameters],
                          ('dp', None, None, None),
                          *groups,
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
        qx, px, qy, py = points.T

        return torch.stack([qx, px, py, qy])

    return jacobian(task)(dp).squeeze()


def ORM_DP(line:Line,
           guess:Tensor,
           parameters:list[Tensor],
           *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
           start:int=0,
           alignment:bool=False,
           limit:int=1,
           epsilon:Optional[float]=None,
           factor:float=1.0,
           alpha:float=0.0,
           solve:Optional[Callable]=None,
           roots:Optional[Tensor]=None,
           jacobian:Optional[Callable]=None) -> Tensor:
    """
    Compute ORM derivative with respect to momentum deviation

    Note, it the initial guess is correct dynamical closed orbit
    Only one iteration is sufficient (set limit=1)

    Parameters
    ----------
    line: Line
        input line (one-turn)
    guess: Tensor
        initial guess
    parameters: list[Tensor]
        list of deviation parameters
    *groups: tuple[str,list[str]|None,list[str]|None,list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    start:int, default=0
        start element index or name (change start)
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    limit: int, positive, default=1
        maximum number of newton iterations
    epsilon: Optional[float]
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

    dp = torch.tensor([0.0], dtype=line.dtype, device=line.device)

    def task(dp):
        return ORM(line,
                   guess,
                   [dp, *parameters],
                   ('dp', None, None, None),
                   *groups,
                   start=start,
                   alignment=alignment,
                   limit=limit,
                   epsilon=epsilon,
                   factor=factor,
                   alpha=alpha,
                   solve=solve,
                   jacobian=jacobian)

    return jacobian(task)(dp).squeeze()

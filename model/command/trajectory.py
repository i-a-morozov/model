"""
Trajectory
----------

Trajectory generation at one or several observation points (main sequence)

"""
from typing import Optional
from typing import Callable

from itertools import pairwise

import torch
from torch import Tensor

from model.library.line import Line

from model.command.wrapper import group
from model.command.orbit import orbit

def trajectory(line:Line,
               locations:list[str],
               *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
               alignment:bool=False,
               matched:bool=False,
               guess:Optional[Tensor]=None,
               limit:int=1,
               epsilon:Optional[float]=None,
               solve:Optional[Callable]=None,
               jacobian:Optional[Callable]=None) -> Callable[[int, Tensor, ...], Tensor]:

    """
    Trajectory generation at one or several observation locations

    Parameters
    ----------
    line: Line
        input line
    locations: list[str]
        list of observation locations
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    alignment: bool, default=False
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
    Callable[[int, Tensor, ...], Tensor]

    """
    start, *_ = locations
    locations = locations.copy()
    locations.append(start + len(line))
    transformations = []
    for pair in pairwise(locations):
        probe, other = pair
        transformation, *_ = group(line, probe, other - 1, *groups, root=True, alignment=alignment)
        transformations.append(transformation)
    guess = guess if isinstance(guess, Tensor) else torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)
    def wrapper(n, state, *args):
        point = guess
        if matched:
            point, *_ = orbit(line, guess, [*args], *groups, alignment=alignment, full=False, limit=limit, epsilon=epsilon, solve=solve, jacobian=jacobian)
        points = [point]
        for transformation in transformations:
            point = transformation(point, *args)
            points.append(point)
        trajectory = [state]
        for _ in range(n):
            for i, transformation in enumerate(transformations):
                state = transformation(state + points[i], *args) - points[i + 1]
                trajectory.append(state)
        return torch.stack(trajectory)
    return wrapper
"""
Transformation
--------------

Functionality related to construction of (parametric) mappings between elements around (parametric) closed orbit
Construction of (parametric) transport matrices around (parametric) closed orbit

Parameters and functions

mapping              : generate mapping between elements
matrix               : generate transport matrix between elements

"""
from typing import Callable
from typing import Optional

import torch
from torch import Tensor

from model.library.line import Line

from model.command.wrapper import group
from model.command.orbit import orbit

def mapping(line:Line,
            probe:int|str,
            other:int|str,
            *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
            root:bool=True,
            name:str='LINE',
            alignment:bool=False,
            last:bool=True,
            matched:bool=False,
            origin:Optional[Tensor]=None,
            limit:int=1,
            epsilon:float=None,
            solve:Optional[Callable]=None,
            jacobian:Optional[Callable]=None) -> tuple[Callable[[Tensor, ...], Tensor], list[tuple[str|None, list[str], str]]]:

    """
    Generate mapping between elements

    Parameters
    ----------
    line: Line
        input line
    probe: int|str
        start element index or name (first occurance position)
    other: int|str
        end element index or name name (first occurance position)
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    root: bool, default=False
        flat to extract names from original line
    name: str, default='LINE'
        constructed line name
    alignment: bool, default=True
        flag to include the alignment parameters in the default deviation table
    last: bool, default=True
        flag to use last occurance position
    matched: bool, default=False
        flag to return mapping around closed orbit
    origin: Tensor, default=None
        closed orbit
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
    tuple[Callable[[Tensor, ...], Tensor], list[tuple[str|None, list[str], str]]]
    wrapper, table

    """
    if isinstance(probe, str):
        probe = line.position(probe)

    if isinstance(other, str):
        *_, other = line.positions(other) if last else [line.position(other)]

    transport, table, _ = group(line, probe, other, *groups, root=root, name=name, alignment=alignment)

    if not matched:
        return transport, table

    origin = origin if isinstance(origin, Tensor) else torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)

    _, table, _ = group(line, 0, len(line) - 1)

    ring = line.clone()
    ring.roll(probe)

    def wrapper(state, *args):
        point, *_ = orbit(ring, origin, [*args], *groups, alignment=alignment, full=False, limit=limit, epsilon=epsilon, solve=solve,jacobian=jacobian)
        return transport(state + point, *args) - transport(point, *args)

    return wrapper, table


def matrix(line:Line,
           probe:int|str,
           other:int|str,
           *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
           root:bool=True,
           name:str='LINE',
           alignment:bool=False,
           last:bool=True,
           matched:bool=False,
           origin:Optional[Tensor]=None,
           limit:int=1,
           epsilon:float=None,
           solve:Optional[Callable]=None,
           jacobian:Optional[Callable]=None) -> Tensor:

    """
    Generate transport matrix between elements

    Parameters
    ----------
    line: Line
        input line
    probe: int|str
        start element index or name (first occurance position)
    other: int|str
        end element index or name name (first occurance position)
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    root: bool, default=False
        flat to extract names from original line
    name: str, default='LINE'
        constructed line name
    alignment: bool, default=True
        flag to include the alignment parameters in the default deviation table
    last: bool, default=True
        flag to use last occurance position
    matched: bool, default=False
        flag to return mapping around closed orbit
    origin: Tensor, default=None
        closed orbit
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
    tuple[Callable[[Tensor, ...], Tensor], list[tuple[str|None, list[str], str]]]

    """
    jacobian:Callable = torch.func.jacrev if jacobian is None else jacobian
    transport, table = mapping(line, probe, other, *groups, root=root, name=name, alignment=alignment, last=last, matched=matched, origin=origin, limit=limit, epsilon=epsilon, solve=solve, jacobian=jacobian)
    return jacobian(transport), table

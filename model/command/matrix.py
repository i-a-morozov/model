"""
Matrix
------

Generate simple model input data (construction from transport matrix)
Generate parametric transport matrices between locations

"""
from __future__ import annotations

from typing import Union
from typing import Tuple
from typing import List
from typing import Callable
from typing import Optional

import torch
from torch import Tensor

from model.library.line import Line

from model.command.mapping import mapping
from model.command.mapping import matrix
from model.simple.build import Model


def resolve(line:Line,
            names:list[str],
            probe:Union[int, str],
            other:Union[int, str], *,
            invert:bool=False) -> Tuple[int, int]:
    """
    Resolve indices
    Given model location

    Parameters
    ----------
    line: Line
        input line
    names: list[str]
        model names (including HEAD and TAIL locations)
    probe: Union[int, str]
        probe index or name
    other: Union[int, str]
        other index or name
    invert: bool, default=False
        resolve direction

    Returns
    -------
    tuple[int, int]
        resolved probe and other indices

    """
    if invert:

        _, *local, _ = names
        table = {name: index for index, name in enumerate(local)}

        def translate(index:Union[int, str]) -> int:
            if isinstance(index, str):
                return table[index]
            local = index % len(line)
            shift = index // len(line)
            return table[line[local].name] + shift*len(table)

        return translate(probe), translate(other)

    _, start, *_ = names
    anchor = line.position(start)
    _, *local, _ = names
    positions = [anchor, *map(line.position, local), anchor + len(line)]

    def translate(index:Union[int, str]) -> int:
        if isinstance(index, str):
            return positions[names.index(index)]
        local = index % len(names)
        shift = index // len(names)
        return positions[local] + shift*len(line)

    return translate(probe), translate(other)


def generate(line:Line, *,
             head:Optional[str]=None,
             tail:Optional[str]=None,
             locations:Optional[List[str]]=None,
             orbit:Optional[Tensor]=None,
             alignment:bool=False,
             jacobian:Optional[Callable]=None) -> dict[str, dict[str, Union[str, float, dict, None]]]:
    """
    Generate TM model data from selected line locations (zero dispersion)

    Parameters
    ----------
    line: Line
        input line
    head: Optional[str]
        head location name
    tail: Optional[str]
        tail location name
    locations: Optional[list[str]]
        list of virtual location names
    orbit: Optional[Tensor]
        closed orbit
    alignment: bool, default=False
        alignment flag
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    dict[str, dict[str, str|float|dict|None]]
        TM model table

    """
    head = head if head else line.start
    tail = tail if tail else line.end

    line = line.clone()
    line.flatten()
    line.sequence = line[head:tail]

    names, indices = line.index(kind='BPM')
    indices = indices.tolist()
    kinds = [line[name].virtual for name in names]

    locations = locations if locations else []
    for location in locations:
        names.append(location)
        kinds.append(True)
        indices.append(line.position(location))

    indices, names, kinds = zip(*sorted(zip(indices, names, kinds)))
    locations = line.locations()[torch.tensor(indices, dtype=torch.int64)].tolist()

    orbit = orbit if isinstance(orbit, Tensor) else torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)

    elements = Model._rc_tm
    default = dict.fromkeys(Model._index['TM'])
    table = dict.fromkeys(names)

    for index, name, kind, location in zip(indices, names, kinds, locations):
        table[name] = default
        table[name]['TYPE'] = kind
        table[name]['S'] = location
        table[name]['DQX'] = 0.0
        table[name]['DPX'] = 0.0
        table[name]['DQY'] = 0.0
        table[name]['DPY'] = 0.0        
        transport, _ = matrix(line, 0, index, root=False, alignment=alignment, jacobian=jacobian)
        table[name] = {**table[name], **dict(zip(elements, transport(orbit).flatten().tolist()))}
    
    return table


def factory(line:Line,
            locations:list[str],
            *groups:tuple[str, Optional[List[str]], Optional[List[str]], list[Optional[str]]],
            state:Optional[Tensor]=None,
            errors:Optional[List[float]]=None,
            root:bool=True,
            name:str='LINE',
            alignment:bool=False,
            last:bool=True,
            matched:bool=False,
            guess:Optional[Tensor]=None,
            limit:int=1,
            epsilon:Optional[float]=None,
            solve:Optional[Callable]=None,
            jacobian:Optional[Callable]=None) -> tuple[Callable[..., Tensor], list[tuple[Optional[str], list[str], str]]]:
    """
    Generate a parametric transport-matrix factory

    Parameters
    ----------
    line: Line
        input line
    locations: list[str]
        model location names (monitor and virtual)
    *groups: tuple[str, list[str]|None, list[str]|None, list[str|None]]
        groups specification
        kinds, names, clean (list of element names)
    state: Optional[Tensor]
        initial state for matrix computation
    errors: Optional[List[float]]
        random errors std value (one value per group)
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
    tuple[Callable[..., Tensor], list[tuple[Optional[str], list[str], str]]]
        matrix callable and parameter table

    """
    state = state if isinstance(state, Tensor) else torch.tensor(4*[0.0], dtype=line.dtype, device=line.device)
    _, table = mapping(line, 0, len(line) - 1, *groups, root=True if matched else root, name=name, alignment=alignment)
    groups = tuple((key, None, names, None) for _, names, key in table)
    errors = errors if isinstance(errors, list) else len(table)*[None]

    def random() -> tuple[Tensor, ...]:
        values = []
        for (_, names, _), error in zip(table, errors):
            size = 1 if names is None else len(names)
            if error is None:
                values.append(torch.zeros(size, dtype=line.dtype, device=line.device).squeeze())
                continue
            error = error if isinstance(error, Tensor) else torch.tensor(error, dtype=line.dtype, device=line.device)
            values.append((error*torch.randn(size, dtype=line.dtype, device=line.device)).squeeze())
        return tuple(values)

    def wrapper(probe:Union[int, str],
                other:Union[int, str],
                *parameters:Tensor,
                scale:float=1.0) -> Tensor:
        probe, other = resolve(line, locations, probe, other)
        transport, _ = matrix(line,
                              probe,
                              other,
                              *groups,
                              root=root,
                              name=name,
                              alignment=alignment,
                              last=last,
                              matched=matched,
                              guess=guess,
                              limit=limit,
                              epsilon=epsilon,
                              solve=solve,
                              jacobian=jacobian)
        if not parameters:
            parameters = tuple(scale*parameter for parameter in random())
        return transport(state, *parameters)

    return wrapper, table

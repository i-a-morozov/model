"""
Wrapper
-------

Generate element/line wrappers

Parameters and functions

wrapper              : generate wrapper function for an element/line
group                : generate group wrapper (one or more elements/lines)
forward              : forward normalizaton
inverse              : inverse normalizaton
normalize            : generate normalizaton wrapper
Wrapper              : class to wrap function into torcm module
_update              : recursively update the value of the parameter (leaf key) in the nested dictionary
_select              : select a sub-dictionary from a nested dictionary given a path of keys
_construct           : construct a transformation from one element to another
_forward             : scale a tensor from [lb, ub] to [0, 1]
_inverse             : scale a tensor from [0, 1] to [lb, ub]


"""
from typing import Optional
from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import ParameterList

from model.library.element import Element
from model.library.line import Line

def wrapper(element:Element,
            *groups:tuple[list[str]|None, list[str]|None, str],
            name:bool=False,
            alignment:bool=False,
            verbose:bool=False) -> Callable[[Tensor, ...], Tensor] | tuple[Callable[[Tensor, ...], Tensor], dict]:
    """
    Generate wrapper function for an element/line

    Parameters
    ----------
    element : Element
        element to wrap
    *groups : tuple[list[str]|None, list[str]|None, str]
        groups of deviation parameters to update
    name: bool, default=False
        flag to include the element name in the default deviation table
    alignment: bool, default=False
        flag to include the alignment parameters in the default deviation table
    verbose: bool, default=False
        flag to return the deviation table

    Returns
    -------
    Callable[[Tensor, ...], Tensor] | tuple[Callable[[Tensor, ...], Tensor], dict]

    """
    data = element.data(name=name, alignment=alignment)
    def wrapper(state, *values):
        for (group, value) in zip(groups, values):
            path, names, parameter = group
            if not path and not names:
                _update(data, parameter, value.squeeze(), flag=True)
                continue
            select = _select(data, path) if path else data
            for position, name in enumerate(names):
                _update(select, parameter, value[position], flag=False, name=name)
        return element(state, data=data, alignment=alignment) if not verbose else (element(state, data=data, alignment=alignment), data)
    return wrapper


def group(line:Line,
          probe:int|str,
          other:int|str,
          *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
          root:bool=False,
          name:str='LINE',
          alignment:bool=False) -> tuple[Callable[[Tensor, ...], Tensor], list[tuple[None, list[str], str]], Line]:
    """
    Generate group wrapper (one or more elements/lines)

    Note, all elements are cloned, changes do not effect the original lattice

    String values can be passed to probe/other, in this case it is equivalent to group(line, line.position(probe:str), line.position(other:str))
    First occurrance will be matched if line contains several elements with identical names

    For integer values of probe/other, any integer values can be passed
    This can be used to construct transformation from the start of one element to the end of the other element

    For the probe > other case, inverse transformation is constructed

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

    Returns
    -------
    tuple[Callable[[Tensor, ...], Tensor], list[tuple[str|None, list[str], str]], Line]
    wrapper, tabel, line

    Details
    -------

    [QF, DF, QD, DD]
    [ 0,  1,  2,  3]

    (QF, QF) -> [QF]
    (QF, DF) -> [QF, DF]
    (QF, QD) -> [QF, DF, QD]
    (QF, DD) -> [QF, DF, QD, DD]
    ( 0,  3) -> [QF, DF, QD, DD]

    (DD, QF) -> [DD.inverse(), QD.inverse(), DF.inverse(), QF.inverse()]
    ( 3,  0) -> [DD.inverse(), QD.inverse(), DF.inverse(), QF.inverse()]

    [..., QF, DF, QD, DD, QF, DF, QD, DD, QF, DF, QD, DD, ...]
    [..., -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7, ...]
    [...,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3, ...]

    (-1,  0) -> [DD, QF]
    (-2,  5) -> [QD, DD, QF, DF, QD, DD, QF, DF]
    (-3,  2) -> [DF, QD, DD, QF, DF, QD]
    ( 2, -3) -> [QD.inverse(), DF.inverse(), QF.inverse(), DD.inverse(), QD.inverse(), DF.inverse()]

    """
    if isinstance(probe, str):
        probe = line.position(probe)

    if isinstance(other, str):
        other = line.position(other)

    sequence:list[Element|Line] = _construct(probe, other, line.sequence)

    local = Line(name=name,
                 sequence=sequence,
                 propagate=line.propagate,
                 dp=line.dp.item(),
                 exact=line.exact, output=line.output,
                 matrix=line.matrix)

    table:list[tuple[str|None, list[str], str]] = []
    value:Line = line if root else local
    for group in groups:
        key, kinds, names, clean = group
        kinds = kinds if isinstance(kinds, list) else []
        names = names if isinstance(names, list) else []
        clean = clean if isinstance(clean, list) else []
        for kind in kinds:
            for name in value.itemize(kind):
                if name not in clean:
                    names.append(name)
        table.append((None, names if names else None, key))

    return wrapper(local, *table, alignment=alignment), table, local


def forward(parameters:list[Tensor],
            bounds:list[tuple[float|None, float|None]]) -> list[Tensor]:
    """
    Forward normalizaton

    Parameters
    ----------
    parameters: list[Tensor]
        parameters
    bounds: list[tuple[float|None,float|None]]
        bounds

    Returns
    -------
    list[Tensor]

    """
    result = []
    for _, (parameter, bound) in enumerate(zip(parameters, bounds)):
        lb, ub = bound
        if all([lb, ub]):
            result.append(_forward(parameter, lb, ub))
            continue
        result.append(parameter)
    return result


def inverse(parameters:list[Tensor],
            bounds:list[tuple[float|None, float|None]]) -> list[Tensor]:
    """
    Inverse normalizaton

    Parameters
    ----------
    parameters: list[Tensor]
        parameters
    bounds: list[tuple[float|None,float|None]]
        bounds

    Returns
    -------
    list[Tensor]

    """
    result = []
    for _, (parameter, bound) in enumerate(zip(parameters, bounds)):
        lb, ub = bound
        if all([lb, ub]):
            result.append(_inverse(parameter, lb, ub))
            continue
        result.append(parameter)
    return result


def normalize(function:Callable[[Tensor, ...], Tensor],
              bounds:list[tuple[float|None, float|None]]) -> Callable[[Tensor, ...], Tensor]:
    """
    Generate normalizaton wrapper

    Parameters
    ----------
    function: Callable[[Tensor, ...], Tensor]
        function
    bounds: list[tuple[float|None, float|None]]
        bounds

    Returns
    -------
    Callable[[Tensor, ...], Tensor]

    """
    def wrapper(*parameters):
        return function(*inverse(parameters, bounds))
    return wrapper


class Wrapper(Module):
    """
    Wrap function into torcm module

    """
    def __init__(self,
                 objective:Callable[[Tensor, ...], Tensor],
                 *args:tuple[Tensor, ...]) -> None:
        """
        Initialization

        Parameters
        ----------
        objective: Callable[[Tensor, ...], Tensor]
            objective function
        *args: tuple[Tensor, ...]
            arguments to pass to the objective function

        Returns
        -------
        None

        """
        super().__init__()
        self.objective = objective
        self.args = ParameterList([Parameter(arg) for arg in args])

    def forward(self, *args):
        return self.objective(*args, *self.args)


def _update(data:dict,
            parameter:str,
            target:Tensor,
            flag:bool=True,
            name:Optional[str]=None) -> None:
    """
    Recursively update the value of the parameter (leaf key) in the nested dictionary

    Parameters
    ----------
    data: dict
        input dictionary to update
    parameter: str
        name of the parameter (leaf key)
    target: Tensor
        value of the parameter to set
    flag: bool, default=True
        update all occurancies of the parameter (true)
        update only in the sub-dictionary with the given name (false)
    name: Optional[str]
        sub-dictionary name

    Returns
    -------
    None

    """
    if flag:
        for key, value in data.items():
            if isinstance(value, dict):
                _update(value, parameter, target, flag=True)
            if key == parameter:
                data[key] = target
                return
        return
    if name in data:
        if parameter in data[name]:
            data[name][parameter] = target
            return
    for key, value in data.items():
        if isinstance(value, dict):
            _update(value, parameter, target, flag=False, name=name)


def _select(data:dict,
            path:list[str]) -> dict:
    """
    Select a sub-dictionary from a nested dictionary given a path of keys

    Note, empty dictionary is returned if the path is not found

    Parameters
    ----------
    data: dict
        nested dictionary
    path: list[str]
        path of keys to select

    Returns
    -------
    dict

    """
    for name in path:
        if name not in data:
            return {}
        data = data[name]
    return data


def _construct(probe:int,
               other:int,
               sequence:list[Element|Line], *,
               inverse:bool=False) -> list[Element|Line]:
    """
    Construct a transformation from one element to another

    Parameters
    ----------
    probe: int
        start element index
    other: int
        end element index
    sequence: list[Element|Line]
        sequence of elements or lines
    inverse: bool, default=False
        flag to invert the transformation

    Returns
    -------
    list[Element|Line]

    """
    if probe > other:
        return list(reversed(_construct(other, probe, sequence, inverse=True)))
    n = len(sequence)
    return [sequence[i % n].inverse() if inverse else sequence[i % n].clone() for i in range(probe, other + 1)]


def _forward(x:Tensor,
             lb:float,
             ub:float) -> Tensor:
    """
    Scale a tensor from [lb, ub] to [0, 1]

    Parameters
    ----------
    x: Tensor
        input tensor
    lb: float
        lower bound
    ub: float
        upper bound

    Returns
    -------
    Tensor

    """
    return (x - lb)/(ub - lb)


def _inverse(x:Tensor,
             lb:float,
             ub:float) -> Tensor:
    """
    Scale a tensor from [0, 1] to [lb, ub]

    Parameters
    ----------
    x: Tensor
        input tensor
    lb: float
        lower bound
    ub: float
        upper bound

    Returns
    -------
    Tensor

    """
    return lb + x*(ub - lb)

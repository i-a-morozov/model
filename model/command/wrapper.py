"""
Wrapper
-------

Generate element/line wrappers

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
            alignment:bool=True,
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
    alignment: bool, default=True
        flag to include the alignment parameters in the default deviation table
    verbose: bool, default=False
        flag to return the deviation table

    Returns
    -------
    Callable[[Tensor, ...], Tensor] | tuple[Callable[[Tensor, ...], Tensor], dict]

    """
    data = element.table(name=name, alignment=alignment)
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
          start:int|str,
          end:int|str,
          *groups:tuple[str, list[str]|None, list[str]|None, list[str|None]],
          root:bool=False,
          name:str='LINE',
          alignment:bool=True) -> tuple[Callable[[Tensor, ...], Tensor], list[tuple[None, list[str], str]], Line]:
    """
    Generate group wrapper (one or more elements/lines)

    Parameters
    ----------
    line: Line
        input line
    start: int|str
        start element index or name (first occurance position)
    end: int|str
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

    """
    if isinstance(start, str):
        count = 0
        for element in line.sequence:
            if element.name == start:
                start = count
                break
            count +=1

    if isinstance(end, str):
        count = 0
        for element in line.sequence:
            if element.name == end:
                end = count
                break
            count +=1

    local = Line(name=name,
                 sequence=line.sequence[start:end + 1],
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
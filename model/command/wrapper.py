"""
Wrapper
-------

Generate element/line wrappers

"""
from typing import Optional
from typing import Callable

from torch import Tensor

from model.library.element import Element

def wrapper(element:Element,
            *groups:tuple[list[str]|None, list[str]|None, str],
            name:bool=False,
            alignment:bool=True,
            verbose:bool=False) -> Callable[[Tensor, ...], Tensor] | tuple[Callable[[Tensor, ...], Tensor], dict]:
    """
    Generate wrapper function for an element

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
        return element(state, data=data) if not verbose else (element(state, data=data), data)
    return wrapper


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
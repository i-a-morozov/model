"""
Util
----

Utils module
Configuration table manipulation

Lattice configuration table is abstracted as a dict[str, dict[str, str | int | float | dict]]
Where the first-level keys correspond to location names
The corresponding values contain the second-level keys and values

table = {..., location -> data, ...} = {..., location -> {..., key -> value, ...}, ...}

Functions

mod    : Compute remainder with offset
cast   : Cast string
load   : Load configuration table
save   : Save configuration table
rename : Rename keys
select : Select key
insert : Insert key
remove : Remove key
mingle : Mingle two configuration tables

"""
from __future__ import annotations

import yaml
from yaml import SafeDumper

from typing import Optional
from typing import Callable

from pathlib import Path
from copy import deepcopy

type Table = dict[str,dict[str,str|int|float|dict]]

def _yaml_float(dumper, value):
    formatted = "{:.16E}".format(value)
    return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)

yaml.add_representer(float, _yaml_float)

class _yaml_dumper(SafeDumper):
    def ignore_aliases(self, data):
        return True

def mod(x:float,
        y:float,
        z:float=0.0) -> float:
    """
    Compute remainder with offset

    Parameters
    ----------
    x: float
        numerator
    y: Tensfloator
        denomenator
    z: float, default=0.0
        offset

    Returns
    -------
    float

    """
    return x - ((x - z) - (x - z) % y)


def cast(value:str) -> str|int|float:
    """
    Casr string

    Parameters
    ----------
    value: str
        input string

    Returns
    -------
    str|int|float

    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load(path:Path) -> Table:
    """
    Load configuration table

    Parameters
    ----------
    path: Path
        input path with table to load

    Returns
    -------
    Table

    """
    with path.open() as stream:
        table = yaml.safe_load(stream)
    return table


def save(table:Table,
         path:Path) -> None:
    """
    Save configuration table

    Parameters
    ----------
    table: Table
        table to save
    path: Path
        output file path

    Returns
    -------
    None

    """
    with path.open('w') as stream:
        yaml.dump(table, stream, Dumper=_yaml_dumper, default_flow_style=False, sort_keys=False)


def rename(table:Table,
           rule:dict[str,str], *,
           drop:bool=True) -> Table:
    """
    Rename keys

    Parameters
    ----------
    table: Table
        input table
    rule: dict[str, str]
        keys replacement rule
    drop: bool, default=true
        flag to drop keys not appearing in rule

    Returns
    -------
    Table

    """
    result:Table = {}
    for location, data in table.items():
        result[location] = {rule.get(key, key): value for key, value in  data.items() if (not drop) or (key in rule)}
    return result


def select(table:Table,
           key:str, *,
           keep:bool=False) -> dict[str,str|int|float|dict]:
    """
    Select key

    Parameters
    ----------
    table: Table
        input table
    key: str
        selected key
    keep: bool
        flag to keep key in the output

    Returns
    -------
    dict[str,str|int|float|dict]

    """
    table:Table = rename(table, {key: key}, drop=True)
    if keep:
        return table
    locations:list[str] = list(table.keys())
    values:list[list[str|int|float|dict]] = list(map(list, map(dict.values, table.values())))
    return {location: value for location, (value, *_) in zip(locations, values)}


def insert(table:Table,
           key:str,
           locations:dict[str,str|int|float|dict], *,
           replace:bool=True,
           apply:Optional[Callable]=None) -> Table:
    """
    Insert key

    Parameters
    ----------
    table: Table
        input configuration table
    key: str
        key name
    locations: dict[str,str|int|float|dict]
        values to insert
    replace: bool, default=True
        flag to replace existing values
    apply: Optional[Callable]
        function to apply to shared keys values

    Returns
    -------
    Table

    """
    if not apply:
        apply:Callable = list
    table: Table = deepcopy(table)
    for location, value in locations.items():
        if replace:
            table[location][key] = value
            continue
        table[location][key] = value if key not in table[location] else apply([table[location][key], value])
    return table


def remove(table:Table,
           key:str, *,
           locations:Optional[list[str]]=None) -> Table:
    """
    Remove key

    Parameters
    ----------
    table: Table
        input configuration table
    key: str
        key name
    locations: Optiona[list[str]]
        list of locations

    Returns
    -------
    Table

    """
    table: Table = deepcopy(table)
    for location in (locations if locations else table):
        table[location].pop(key)
    return table


def mingle(probe:Table,
           other:Table,
           replace:bool=True,
           apply:Optional[Callable]=None) -> Table:
    """
    Mingle two configuration tables

    Parameters
    ----------
    probe: Table
        probe table (shared keys values appear first)
    other: Table
        other table
    replace: bool, default=True
        flag to replace existing values
    apply: Optional[Callable]
        function to apply to shared keys values

    Returns
    -------
    Table

    """
    if not apply:
        apply:Callable = list
    probe: Table = deepcopy(probe)
    for location, data in other.items():
        for key, value in data.items():
            probe = insert(probe, key, {location: value}, replace=replace, apply=apply)
    return probe
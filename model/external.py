"""
External
--------

External model
Generate configuration table from MADX or ELEGANT output
Insert locations into MADX or ELEGANT lattice files

Functions

load_tfs     : Load TFS style table
load_sdds    : Load SDDS style table
convert      : Convert (sorted) columns to configuration table
parse        : Parse MADX or ELEGANT element or line
load_lattice : Load MADX or ELEGANT lattice file
text_lattice : Convert MADX or ELEGANT data to text
rift_lattice : Rift MADX or ELEGANT lattice data (insert monitor or virtual locations)
add_rc       : Add RC from lattice to configuration table

External

rename (util)
insert (util)
select (util)
cast   (util)

"""
from __future__ import annotations

from typing import Optional
from typing import Literal

from pathlib import Path
from copy import deepcopy

from math import pi

from model.util import Table
from model.util import cast
from model.util import rename
from model.util import insert

def load_tfs(path:Path, *,
             postfix:str='_') -> tuple[dict[str, str|int|float], dict[str, dict[str, str|int|float]]]:
    """
    Load TFS style table

    Parameters
    ----------
    path: Path
        input path
    postfix: str, default=''
        rename duplicate locations postfix

    Returns
    -------
    tuple[dict[str, str|int|float], dict[str, dict[str, str|int|float]]]
        {key: value} (parameters)
        {location: {key: value}} (columns)
    
    """
    parameters:dict[str, str|int|float] = {}
    columns:dict[str, dict[str, str|int|float]] = {}
    with path.open() as stream:
        for line in stream:
            if line.startswith('@'):
                _, key, unit, *value = line.split()
                unit = str if unit.endswith('s') else float
                parameters[key] = unit((' '.join(value)).replace('"', ''))
                continue
            if line.startswith('*'):
                _, _, *keys = line.split()
                continue
            if line.startswith('$'):
                _, *units = line.split()
                units = [{'%s': str, '%le': float, '%d': float}[unit] for unit in units]
                continue
            values = line.split()
            name, *values = [unit(value.replace('"','')) for unit, value in zip(units, values)]
            if name in columns:
                size = 1
                while f'{name}_{size}' in columns:
                    size +=1
                name = f'{name}{postfix}{size}'
            columns[name] = {key: value for key, value in zip(keys, values)}
    return parameters, columns


def load_sdds(path:Path, *,
              postfix:str='_') -> tuple[dict[str, str|int|float], dict[str, dict[str, str|int|float]]]:
    """
    Load SDDS style table

    Parameters
    ----------
    path: Path
        input path
    postfix: str, default=''
        rename duplicate locations postfix

    Returns
    -------
    tuple[dict[str, str|int|float], dict[str, dict[str, str|int|float]]]
        {key: value} (parameters)
        {location: {key: value}} (columns)
    
    """
    parameters:dict[str, str|int|float] = {}
    columns:dict[str, dict[str, str|int|float]] = {}
    kp:list[str] = []
    kc:list[str] = []
    up:list = []
    uc:list = []
    nl:int = 0
    np:int = 0
    with path.open() as stream:
        for line in stream:
            if line.startswith('SDDS'):
                continue
            if line.startswith('&description'):
                continue
            if line.startswith('&parameter'):
                np += 1
                _, name, *_, unit, _ = line.split()    
                _, name = name.split('=')
                name, _ = name.split(',')
                _, unit = unit.split('=')
                unit, _ = unit.split(',')
                unit = {'string': str, 'long': int, 'double': float}[unit]
                kp.append(name)
                up.append(unit)
                continue
            if line.startswith('&column'):
                _, name, *_, unit, _ = line.split()    
                _, name = name.split('=')
                name, _ = name.split(',')
                _, unit = unit.split('=')
                unit, _ = unit.split(',')
                unit = {'string': str, 'long': int, 'double': float}[unit]
                kc.append(name)
                uc.append(unit)                
                continue
            if line.startswith('&data'):
                continue
            if line.startswith('! page number'):
                continue
            if nl < np:
                try:
                    parameters[kp[nl]] = up[nl](line)
                except ValueError:
                    parameters[kp[nl]] = str(line)
                nl += 1
                continue
            if nl == np:
                nl += 1
                continue
            values = {key: unit(value) for (unit, (key, value)) in zip(uc, dict(zip(kc, line.split())).items())}
            name = values.pop('ElementName')
            if name in columns:
                size = 1
                while f'{name}_{size}' in columns:
                    size +=1
                name = f'{name}{postfix}{size}'            
            columns[name] = values
    return parameters, columns


def convert(columns: dict[str, dict[str, str|int|float]], 
            kind:Literal['TFS', 'SDDS'],
            kind_monitor:list[str],
            kind_virtual:list[str], *, 
            dispersion:bool=False,
            rc:bool=False,
            name_monitor:Optional[list[str]]=None,
            name_virtual:Optional[list[str]]=None,
            monitor:str='MONITOR',
            virtual:str='VIRTUAL',
            rule:Optional[dict[str, str]]=None) -> Table:
    """
    Convert (sorted) columns to configuration table

    TFS parameters (MADX) can be used to setup CS model
    SDDS parameters (ELEGANT) can be used to setup CS and TM models

    Parameters
    ----------
    columns: dict[str, dict[str, str|int|float]]
        columns to convert (sorted locations are assumed)
    kind: Literal['TFS', 'SDDS']
        columns kind
    kind_monitor: list[str]
        list of element types to identify with monitor locations
    kind_virtual: list[str]
        list of element types to identify with virtual locations
    dispersion: bool, default=False
        flag to insert zero dispersion values (original values are kept if present)
    rc: bool, default=False
        flat to add RC key
    name_monitor: Optional[list[str]]
        list of element names to set as monitor locations
    name_virtual: Optional[list[str]]
        list of element names to set as virtual locations
    monitor: str, default='MONITOR'
        monitor kind
    virtual: str, default='VIRTUAL'
        virtual kind
    rule: Optional[dict[str, str]]
        keys rename rule (appended to default rules)
        
    Returns
    -------
    Table
        
    """
    rule_tfs:dict[str, str] = {
        'KEYWORD': 'TYPE',
        'S'      : 'S'  , 
        'ALFX'   : 'AX' , 
        'BETX'   : 'BX' , 
        'MUX'    : 'FX' , 
        'ALFY'   : 'AY' , 
        'BETY'   : 'BY' , 
        'MUY'    : 'FY' , 
        'DX'     : 'DQX', 
        'DPX'    : 'DPX', 
        'DY'     : 'DQY', 
        'DPY'    : 'DPY',      
    }
    rule_sdds:dict[str, str] = {
        'ElementType' : 'TYPE',
        's'           : 'S'  , 
        'alphax'      : 'AX' , 
        'betax'       : 'BX' , 
        'psix'        : 'FX' , 
        'alphay'      : 'AY' , 
        'betay'       : 'BY' , 
        'psiy'        : 'FY' , 
        'etax'        : 'DQX', 
        'etaxp'       : 'DPX', 
        'etay'        : 'DQY', 
        'etayp'       : 'DPY',
        'R11'         : 'T11',
        'R12'         : 'T12',
        'R13'         : 'T13',
        'R14'         : 'T14',
        'R21'         : 'T21',
        'R22'         : 'T22',
        'R23'         : 'T23',
        'R24'         : 'T24',
        'R31'         : 'T31',
        'R32'         : 'T32',
        'R33'         : 'T33',
        'R34'         : 'T34',
        'R41'         : 'T41',
        'R42'         : 'T42',
        'R43'         : 'T43',
        'R44'         : 'T44',         
    }
    rule:dict[str, str] = {**({'TFS': rule_tfs, 'SDDS': rule_sdds}[kind]), **(rule if rule else {})}
    name_monitor:list[str] = name_monitor if name_monitor else []
    name_virtual:list[str] = name_virtual if name_virtual else []    
    table:Table = deepcopy(columns)
    head:str
    tail:str
    head, *_, tail = table.keys()
    table['HEAD'] = table.pop(head)
    table['TAIL'] = table.pop(tail)
    source:str = {'TFS': 'KEYWORD', 'SDDS': 'ElementType'}[kind]
    kinds:list[str] = kind_monitor + kind_virtual
    names:list[str] = name_monitor + name_virtual
    table:Table = {location: data for location, data in table.items() if data[source] in kinds or location in names}
    for location in table:
        value = table[location][source]
        if value in kind_monitor or location in name_monitor:
            table[location][source] = monitor
            continue
        if value in kind_virtual or location in name_virtual:
            table[location][source] = virtual    
            continue
    table = rename(table, rule, drop=True)
    if kind == 'TFS':
        scale:float = 2.0*pi
        for location in table.keys():
            table[location]['FX'] *= scale
            table[location]['FY'] *= scale    
    if dispersion:
        locations:dict[str, float] = {location: 0.0 for location in table.keys()}
        table = insert(table, 'DQX', locations, replace=False, apply=sum)
        table = insert(table, 'DPX', locations, replace=False, apply=sum)
        table = insert(table, 'DQY', locations, replace=False, apply=sum)
        table = insert(table, 'DPY', locations, replace=False, apply=sum)
    if rc:
        locations:dict[str, float] = {location: None for location in table.keys()}
        table = insert(table, 'RC', locations, replace=True)
    table['HEAD']['TYPE'] = virtual
    table['TAIL']['TYPE'] = virtual    
    locations:list[str] = sorted(table.keys(), key=lambda location: table[location]['S'] - (location == 'HEAD') + (location == 'TAIL'))
    return {location: table[location] for location in locations}


def parse(line:str, *, 
          rc:bool=False) -> list[str, dict[str, str|int|float]]:
    """
    Parse MADX or ELEGANT element or line

    Note, full definition is assumed to be on a singe line
    Element or line name should not contain '!', ':' or '"'
    All elements should contain

    name : line=(element, element, ...) [;] [! comment]
    name : kind, [key=value, key=value, ...] [;] [! comment]

    Note, comma after element kind in mandatory
    If comment to be parsed as element it is assumed to match

    name : kind, [key=value, key=value, ...]

    Parameters
    ----------
    line: str
        input line
    rc: bool, default=False
        flag to parse comment as element

    Returns
    -------
    list[str, dict[str, str|int|float]]
    
    """ 
    name, *data = line.split(',')
    *name, kind = name.split(':')    
    name = ':'.join(name).strip()
    if ':LINE=' in line.replace(' ', '').upper():
        _, data = (kind + ',' + ','.join(data)).split('=', 1)
        data, *comment = data.split('!')
        data = data.replace('(', '').replace(')', '').replace(';', '').split(',')
        data = [element.strip() for element in data]
        return [name, {'KIND': 'LINE', 'SEQUENCE': data}]
    kind = kind.strip().rstrip(';')
    data, *comment = ','.join(data).split('!')
    comment = ''.join(comment).strip()
    data = data.strip().rstrip(';').split(',')
    result = {'KIND': kind.strip(), 'RC': parse(comment, rc=False) if rc else comment}
    for pair in data:
        if '=' in pair:
            key, value = pair.split('=')
            result[key.strip()] = cast(value.strip())
    return [name, result]


def load_lattice(path:Path, *,
                 rc:bool=False) -> dict[str,dict[str,str|int|float|dict]]:
    """
    Load MADX or ELEGANT lattice file

    Parameters
    ----------
    path: Path
        input path 
    rc: bool, default=False
        flag to parse comment as element        

    Returns
    -------
    dict[str,dict[str,str|int|float|dict]]
    
    """ 
    lattice: dict[str,dict[str,str|int|float|dict]] = {}
    name:str
    data:dict[str, str|int|float]
    with path.open() as stream:
        for line in stream:
            if ':' in line:
                name, data = parse(line.upper(), rc=rc)
                lattice[name]= data
    return lattice


def text_lattice(kind:Literal['MADX', 'LTE'],
                 lattice:dict[str,dict[str,str|int|float|dict]], *, 
                 rc:bool=False) -> str:
    """
    Convert MADX or ELEGANT data to text

    Parameters
    ----------
    kind: Literal['MADX', 'LTE']
        lattice lind
    lattice: dict[str,dict[str,str|int|float|dict]]
        lattice data
    rc: bool, default=False
        flag to parse RC as comment          

    Returns
    -------
    str
    
    """
    text:str = ''
    line:str
    last:str
    tail:str = {'MADX': ';', 'LTE': ''}[kind]
    name:str
    data:dict[str,str|int|float|dict]
    for name, data in lattice.items():
        line = f'{name}: '
        last = ''
        for key, value in data.items():
            if key == 'RC':
                if rc and value:
                    if isinstance(value, str):
                        last = f' ! {value.rstrip('\n')}'
                        continue
                    name, data = value
                    if not name:
                        continue    
                    last = f' ! {text_lattice(kind, {name: data}, rc=False).rstrip('\n')}'
                continue
            if key == 'KIND':
                line += f'{value}, '
                continue
            if key == 'SEQUENCE':
                line = f'{line.strip().rstrip(",")}=({", ".join(value)})'
                break
            line += f'{key}={value}, '
        text += f'{line.strip()}{tail}{last}\n'
    return text


def rift_lattice(lattice:dict[str,dict[str,str|int|float|dict]],
                 monitor:str,
                 virtual:str,
                 kind_monitor:list[str],
                 kind_virtual:list[str], *, 
                 include_monitor:Optional[list[str]]=None,
                 include_virtual:Optional[list[str]]=None,
                 exclude_monitor:Optional[list[str]]=None,
                 exclude_virtual:Optional[list[str]]=None,                 
                 prefix_monitor:str='M',
                 prefix_virtual:str='V') -> dict[str,dict[str,str|int|float|dict]]:
    """
    Rift MADX or ELEGANT lattice data (insert monitor or virtual locations)

    Elements with matching kinds or names are splitted in half (angle and length are halved)
    Observaton (monitor or virtual) location is inserted between parts
    Elements with zero length are also splitted
    Original parameters are added to location RC
    Old element is replaced by a line

    Parameters
    ----------
    monitor: str
        monitor type to use (e.g. MONITOR or MONI)
    virtual: str
        virtual type to use (e.g. MARKER or MARK)
    kind_monitor: list[str]
        list of element types to insert monitor locations
    kind_virtual: list[str]
        list of element types to insert virtual locations
    include_monitor: Optional[list[str]]
        list of element names to insert monitor locations
    include_virtual: Optional[list[str]]
        list of element names to insert virtual locations
    exclude_monitor: Optional[list[str]]
        list of element names to exclude from monitor locations
    exclude_virtual: Optional[list[str]]
        list of element names to exclude form virtual locations
    prefix_monitor: str, default='M'
        monitor rename prefix
    prefix_virtual: str, default='V' 
        virtual rename prefix

    Returns
    -------
    dict[str,dict[str,str|int|float|dict]]
    
    """
    result:dict[str,dict[str,str|int|float|dict]] = {}
    lattice:dict[str,dict[str,str|int|float|dict]] = deepcopy(lattice)
    include_monitor:list[str] = include_monitor if include_monitor else []
    include_virtual:list[str] = include_virtual if include_virtual else []  
    exclude_monitor:list[str] = exclude_monitor if exclude_monitor else []
    exclude_virtual:list[str] = exclude_virtual if exclude_virtual else []  
    element:str
    data:dict[str,str|int|float|dict]
    for element, data in lattice.items():
        kind:str = data['KIND']
        select:str = ''
        prefix:str = ''
        if (kind in kind_monitor and element not in exclude_monitor) or element in include_monitor:
            select = monitor
            prefix = prefix_monitor
        if (kind in kind_virtual and element not in exclude_virtual) or element in include_virtual:
            select = virtual
            prefix = prefix_virtual
        if not prefix:
            result[element] = data
            continue
        data.pop('RC')
        result[f'{prefix}_{element}'] = {'KIND': select, 'RC': [element, data]}
        data = data.copy()
        if 'L' in data: 
            data['L'] /= 2.0
        if 'ANGLE' in data: 
            data['ANGLE'] /= 2.0
        result[f'H_{element}'] = data
        result[element] = {'KIND': 'LINE', 'SEQUENCE': [f'H_{element}', f'{prefix}_{element}', f'H_{element}']} 
    return result


def add_rc(table:Table, 
           lattice:dict[str,dict[str,str|int|float|dict]], *,
           monitor:str='MONITOR',
           virtual:str='VIRTUAL') -> Table:
    """
    Add RC from lattice to configuration table

    Parameters
    ----------
    table: Table
        configuration table
    lattice: dict[str,dict[str,str|int|float|dict]]
        lattice data
    monitor: str, default='MONITOR'
        monitor kind
    virtual: str, default='VIRTUAL'        

    Returns
    -------
    Table
    
    """
    table:Table = deepcopy(table)
    location:str
    data:dict[str,str|int|float|dict]
    for location, data in lattice.items():
        if 'RC' in data:
            root:str
            root, data = data['RC']
            if root:
                if 'RC' in data:
                    data.pop('RC')
                names:list[str] = [name for name in table if root in name]
                for name in names:
                    table[name]['RC'] = [root, data]
    return table
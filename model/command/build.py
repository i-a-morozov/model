"""
Build
-----

Build MADX or ELEGANT lattice
Save and load YAML lattice

Parameters and functions

_translation_madx    : MADX translation rules
_translation_lte     : ELEGANT translation rules
save_line            : save line to YAML
load_line            : load line from YAML
_traverse            : recursive line construction (mutates input table)
build                : build MADX or ELEGANT lattice
_build_seq           : build MADX lattice
_build_lte           : build ELEGANT lattice

"""
from typing import Literal

from pathlib import Path

from model.command.util import save
from model.command.util import load

from model.library.element    import Element
from model.library.drift      import Drift
from model.library.quadrupole import Quadrupole
from model.library.sextupole  import Sextupole
from model.library.octupole   import Octupole
from model.library.multipole  import Multipole
from model.library.dipole     import Dipole
from model.library.corrector  import Corrector
from model.library.gradient   import Gradient
from model.library.kick       import Kick
from model.library.linear     import Linear
from model.library.bpm        import BPM
from model.library.marker     import Marker
from model.library.line       import Line


_translation_madx: dict[str, dict[str, str]] = {
    'DRIFT'     : {'length': 'L'},
    'QUADRUPOLE': {'length': 'L', 'kn': 'K1', 'ks': 'K1S'},
    'SEXTUPOLE' : {'length': 'L', 'ms': 'K2'},
    'OCTUPOLE'  : {'length': 'L', 'mo': 'K3'},
    'SBEND'     : {'length': 'L', 'angle': 'ANGLE', 'e1': 'E1', 'e2': 'E2', 'kn': 'K1', 'ks': 'K1S', 'ms': 'K2'},
    'MONITOR'   : {},
    'MARKER'    : {}
}


_translation_lte: dict[str, dict[str, str]] = {
    'DRIF'      : {'length': 'L'},
    'QUAD'      : {'length': 'L', 'kn': 'K1', 'ks': 'K1S'},
    'SEXT'      : {'length': 'L', 'ms': 'K2'},
    'OCTU'      : {'length': 'L', 'mo': 'K3'},
    'CSBEND'    : {'length': 'L', 'angle': 'ANGLE', 'e1': 'E1', 'e2': 'E2', 'kn': 'K1', 'ks': 'K1S', 'ms': 'K2'},
    'SBEN'      : {'length': 'L', 'angle': 'ANGLE', 'e1': 'E1', 'e2': 'E2', 'kn': 'K1', 'ks': 'K1S', 'ms': 'K2'},
    'MONI'      : {},
    'MARK'      : {}
}


def save_line(line:Line,
              path:Path) -> None:
    """
    Save line to YAML

    Parameters
    ----------
    path: Path
        path
    line: Line
        line

    Returns
    -------
    None

    """
    save(line.serialize, path)


def load_line(path:Path) -> Line:
    """
    Load line from YAML

    Parameters
    ----------
    path: Path
        path

    Returns
    -------
    Line

    """
    table = load(path)
    return _traverse(table)


def _traverse(table:dict) -> Line:
    """
    Recursive line construction (mutates input table)

    Note, unmatched kind is casted to drift

    Parameters
    ----------
    table: dict
        table

    Returns
    -------
    Line

    """
    table.get('kind')
    match table.get('kind'):
        case 'Line':
            table.pop('kind')
            sequence = [_traverse(element) for element in table.pop('sequence')]
            return Line(sequence=sequence, **table)
        case 'Drift':
            table.pop('kind')
            return Drift(**table)
        case 'Quadrupole':
            table.pop('kind')
            return Quadrupole(**table)
        case 'Sextupole':
            table.pop('kind')
            return Sextupole(**table)
        case 'Octupole':
            table.pop('kind')
            return Octupole(**table)
        case 'Multipole':
            table.pop('kind')
            return Multipole(**table)
        case 'Dipole':
            table.pop('kind')
            return Dipole(**table)
        case 'Corrector':
            table.pop('kind')
            return Corrector(**table)
        case 'Gradient':
            table.pop('kind')
            return Gradient(**table)
        case 'Kick':
            table.pop('kind')
            return Kick(**table)
        case 'Linear':
            table.pop('kind')
            return Linear(**table)
        case 'BPM':
            table.pop('kind')
            return BPM(**table)
        case 'Marker':
            table.pop('kind')
            return Marker(**table)
        case _:
            return Drift(name=table['name'], length=table['length'])


def build(target:str,
          source:Literal['MADX', 'ELEGANT'],
          table:dict[str,dict[str,str|int|float|dict]]):
    """
    Build MADX or ELEGANT lattice

    Parameters
    ----------
    target: str
        element or line name
    source: Literal['MADX', 'ELEGANT']
        source type
    table: dict[str,dict[str,str|int|float|dict]]
        table to build

    Returns
    -------
    Element

    """
    lattice:Element = {'MADX': _build_seq, 'ELEGANT': _build_lte}[source](target, table)
    return lattice


def _build_seq(target:str,
               table:dict[str,dict[str,str|int|float|dict]]):
    """
    Build MADX lattice

    Parameters
    ----------
    target: str
        element or line name
    table: dict[str,dict[str,str|int|float|dict]]
        table to build

    Returns
    -------
    Element

    """
    data: dict[str,str|int|float|dict] = table[target]
    kind: str = data['KIND']
    match kind:
        case 'DRIFT' as case:
            select = _translation_madx[case]
            return Drift(
                name=target,
                length=data.get(select['length'], 0.0)
            )
        case 'QUADRUPOLE' as case:
            select = _translation_madx[case]
            return Quadrupole(
                name=target,
                length=data.get(select['length'], 0.0),
                kn=data.get(select['kn'], 0.0),
                ks=data.get(select['ks'], 0.0)
            )
        case 'SEXTUPOLE' as case:
            select = _translation_madx[case]
            return Sextupole(
                name=target,
                length=data.get(select['length'], 0.0),
                ms=data.get(select['ms'], 0.0)
            )
        case 'OCTUPOLE' as case:
            select = _translation_madx[case]
            return Octupole(
                name=target,
                length=data.get(select['length'], 0.0),
                mo=data.get(select['mo'], 0.0)
            )
        case 'SBEND' as case:
            select = _translation_madx[case]
            return Dipole(
                name=target,
                length=data.get(select['length'], 0.0),
                angle=data.get(select['angle'], 0.0),
                e1=data.get(select['e1'], 0.0),
                e2=data.get(select['e2'], 0.0),
                kn=data.get(select['kn'], 0.0),
                ks=data.get(select['ks'], 0.0),
                ms=data.get(select['ms'], 0.0)
            )
        case 'MONITOR' as case:
            select = _translation_madx[case]
            return BPM(name=target)
        case 'MARKER' as case:
            select = _translation_madx[case]
            return Marker(name=target)
        case 'LINE':
            sequence = data['SEQUENCE']
            elements = [_build_seq(name, table) for name in sequence]
            return Line(name=target, sequence=elements)
        case _:
            select = _translation_madx['DRIFT']
            return Drift(
                name=target,
                length=data.get(select['length'], 0.0)
            )


def _build_lte(target:str,
               table:dict[str,dict[str,str|int|float|dict]]):
    """
    Build ELEGANT lattice

    Parameters
    ----------
    target: str
        element or line name
    table: dict[str,dict[str,str|int|float|dict]]
        table to build

    Returns
    -------
    Element

    """
    data: dict[str,str|int|float|dict] = table[target]
    kind: str = data['KIND']
    match kind:
        case 'DRIF' as case:
            select = _translation_lte[case]
            return Drift(
                name=target,
                length=data.get(select['length'], 0.0)
            )
        case 'QUAD' as case:
            select = _translation_lte[case]
            return Quadrupole(
                name=target,
                length=data.get(select['length'], 0.0),
                kn=data.get(select['kn'], 0.0),
                ks=data.get(select['ks'], 0.0)
            )
        case 'SEXT' as case:
            select = _translation_lte[case]
            return Sextupole(
                name=target,
                length=data.get(select['length'], 0.0),
                ms=data.get(select['ms'], 0.0)
            )
        case 'OCTU' as case:
            select = _translation_lte[case]
            return Octupole(
                name=target,
                length=data.get(select['length'], 0.0),
                mo=data.get(select['mo'], 0.0)
            )
        case 'CSBEND' | 'SBEN' as case:
            select = _translation_lte[case]
            return Dipole(
                name=target,
                length=data.get(select['length'], 0.0),
                angle=data.get(select['angle'], 0.0),
                e1=data.get(select['e1'], 0.0),
                e2=data.get(select['e2'], 0.0),
                kn=data.get(select['kn'], 0.0),
                ks=data.get(select['ks'], 0.0),
                ms=data.get(select['ms'], 0.0)
            )
        case 'MONI' as case:
            select = _translation_lte[case]
            return BPM(name=target)
        case 'MARK' as case:
            select = _translation_lte[case]
            return Marker(name=target)
        case 'LINE':
            sequence = data['SEQUENCE']
            elements = [_build_lte(name, table) for name in sequence]
            return Line(name=target, sequence=elements)
        case _:
            select = _translation_lte['DRIF']
            return Drift(
                name=target,
                length=data.get(select['length'], 0.0)
            )

"""
Build
-----

Build MADX or ELEGANT lattice

"""
from typing import Literal

from model.library.element    import Element
from model.library.drift      import Drift
from model.library.quadrupole import Quadrupole
from model.library.sextupole  import Sextupole
from model.library.octupole   import Octupole
from model.library.dipole     import Dipole
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
"""
AT ID insertion
---------------

Insert thin matrix or kick-map insertion devices

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy
from numpy import ndarray as Array

import at

from model.library.kickmap import KM
from model.library.kickmap import load


def insert(ring:at.Lattice,
           marker:str,
           element:at.Element,
           periods:int=1) -> at.Lattice:
    """
    Insert repeated copies of an AT element after a marker

    Parameters
    ----------
    ring: at.Lattice
        source lattice
    marker: str
        name of the insertion marker
    element: at.Element
        one-period ID element
    periods: int, positive, default=1
        number of element repetitions

    Returns
    -------
    at.Lattice

    """
    result = ring.deepcopy()
    index = next(index for index, target in enumerate(result) if target.FamName == marker)
    length = periods*float(getattr(element, 'Length', 0.0))
    if length:
        left = (index - 1) % len(result)
        while not isinstance(result[left], at.Drift):
            left = (left - 1) % len(result)
        right = (index + 1) % len(result)
        while not isinstance(result[right], at.Drift):
            right = (right + 1) % len(result)
        if left == right:
            result[left].Length -= length
        else:
            result[left].Length -= 0.5*length
            result[right].Length -= 0.5*length
    sequence = [element.deepcopy() for _ in range(periods)]
    for offset, target in enumerate(sequence, start=1):
        result.insert(index + offset, target)
    return result


def matrix(ring:at.Lattice,
           marker:str,
           m44:Array,
           length:float=0.0,
           periods:int=1, *,
           name:str='ID') -> at.Lattice:
    """
    Insert 4x4 ID matrix

    Parameters
    ----------
    ring: at.Lattice
        source lattice
    marker: str
        name of the insertion marker
    m44: Array
        4x4 transverse transfer matrix
    length: float, default=0.0
        ID period length
    periods: int, positive, default=1
        number of periods
    name: str, default='ID'
        AT element family name

    Returns
    -------
    at.Lattice

    """
    m44 = numpy.asarray(m44, dtype=float)
    m66 = numpy.identity(6)
    m66[:4, :4] = m44
    element = at.M66(name, m66, Length=float(length))
    return insert(ring, marker, element, periods)


def kickmap(ring:at.Lattice,
            marker:str,
            path:str|Path,
            periods:int, *,
            energy:Optional[float]=None,
            factor_x:float=1.0,
            factor_y:float=1.0,
            scale:float=1.0,
            sign_x:float=-1.0,
            sign_y:float=-1.0,
            name:str='ID') -> at.Lattice:
    """
    Insert ID kick map

    Parameters
    ----------
    ring: at.Lattice
        source lattice
    marker: str
        name of the insertion marker
    path: str | Path
        path to the one-period MATLAB kick-map table
    periods: int, positive
        number of periods
    energy: Optional[float], default=None
        kick-map normalization energy in GeV
        if None, the map is assumed to be energy scaled
    factor_x: float, default=1.0
        horizontal kick multiplication factor
    factor_y: float, default=1.0
        vertical kick multiplication factor
    scale: float, default=1.0
        common kick multiplication factor
    sign_x: float, default=-1.0
        horizontal kick sign
    sign_y: float, default=-1.0
        vertical kick sign
    name: str, default='ID'
        AT element family name

    Returns
    -------
    at.Lattice

    """
    path = Path(path)
    xgrid, ygrid, xkick, ykick, period = load(path)
    energy_scale = 1.0 if energy is None else (KM.rigidity/energy)**2
    xkick = -factor_x*sign_x*scale*energy_scale*xkick.T
    ykick = -factor_y*sign_y*scale*energy_scale*ykick.T
    zeros = numpy.zeros_like(xkick)
    element = at.InsertionDeviceKickMap(
        name,
        'IdTablePass',
        str(path),
        0.0 if energy is None else energy,
        1,
        period,
        xkick,
        ykick,
        zeros,
        zeros,
        xgrid,
        ygrid
    )
    return insert(ring, marker, element, periods)

"""
AT
--

Convert model lattice to pyAT

"""
from __future__ import annotations

from typing import Any
from typing import Optional

import numpy
from numpy import ndarray as Array

import torch
from torch import Tensor

import at

from model.library.bpm import BPM
from model.library.corrector import Corrector
from model.library.dipole import Dipole
from model.library.drift import Drift
from model.library.element import Element
from model.library.gradient import Gradient
from model.library.kick import Kick
from model.library.kickmap import KM
from model.library.line import Line
from model.library.linear import Linear
from model.library.marker import Marker
from model.library.matrix import Matrix
from model.library.multipole import Multipole
from model.library.octupole import Octupole
from model.library.quadrupole import Quadrupole
from model.library.sextupole import Sextupole


def item(value) -> float:
    return value.item() if isinstance(value, Tensor) else float(value)


def polynomial(element) -> tuple[Array, Array]:
    kn = numpy.array([0.0, item(getattr(element, 'kn', 0.0)), item(getattr(element, 'ms', 0.0))/2.0, item(getattr(element, 'mo', 0.0))/6.0])
    ks = numpy.array([0.0, item(getattr(element, 'ks', 0.0)), 0.0, 0.0])
    return kn, ks


def steps(element:Element, default:int=1) -> int:
    return max(element.ns, default)


def apply(source:Element, target:Any, at:Any, alignment:bool) -> Any:
    if not alignment or not source.alignment:
        return target
    at.transform_elem(target, dx=item(source.dx), dy=item(source.dy), dz=item(source.dz), pitch=item(source.wx), yaw=item(source.wy), tilt=item(source.wz))
    return target


def linear(element:Linear) -> tuple[Array, Array]:
    state = torch.zeros(4, dtype=element.dtype, device=element.device)
    matrix = torch.func.jacrev(element)(state).detach().cpu().numpy()
    vector = element(state).detach().cpu().numpy()
    m66 = numpy.identity(6)
    m66[:4, :4] = matrix
    t2 = numpy.zeros(6)
    t2[:4] = vector
    return m66, t2


def matrix(element:Matrix) -> Array:
    a11, a12, a13, a14, a22, a23, a24, a33, a34, a44 = element.A
    matrix = torch.stack([a11, a12, a13, a14, a12, a22, a23, a24, a13, a23, a33, a34, a14, a24, a34, a44]).reshape(4, 4)
    identity = torch.tensor([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]], dtype=element.dtype, device=element.device)
    direction = -1.0 if element.is_inversed else 1.0
    m66 = numpy.identity(6)
    m66[:4, :4] = torch.matrix_exp(direction*identity @ matrix).detach().cpu().numpy()
    return m66


def kickmap(element:KM, at:Any, alignment:bool) -> list[Any]:
    xgrid = element._xgrid.detach().cpu().numpy()
    ygrid = element._ygrid.detach().cpu().numpy()
    xkick = element._xkick.detach().cpu().numpy().T
    ykick = element._ykick.detach().cpu().numpy().T
    energy_scale = 1.0 if element.energy is None else (element.rigidity/element.energy)**2
    direction = -1.0 if element.is_inversed else 1.0
    xkick = -direction*element.factor_x*element.sign_x*element.scale*energy_scale*xkick
    ykick = -direction*element.factor_y*element.sign_y*element.scale*energy_scale*ykick
    zeros = numpy.zeros_like(xkick)
    period = item(element.period)
    length = element.count*period
    targets = []
    for _ in range(element.count):
        target = at.InsertionDeviceKickMap(element.name, "IdTablePass", element.path, 0.0 if element.energy is None else element.energy, 1, period, xkick, ykick, zeros, zeros, xgrid, ygrid)
        targets.append(apply(element, target, at, alignment))
    if not element.insertion:
        return targets
    half = -0.5*length
    return [
        at.Drift(f"{element.name}_ENTRY", half),
        *targets,
        at.Drift(f"{element.name}_EXIT", half)
    ]


def parse(element:Element, at:Any, alignment:bool) -> list[Any]:
    name = element.name
    length = item(element.length)
    if isinstance(element, KM):
        return kickmap(element, at, alignment)
    if isinstance(element, Drift):
        target = at.Drift(name, length)
    elif isinstance(element, Dipole):
        normal, skew = polynomial(element)
        target = at.Dipole(name, length, item(element.angle), item(element.kn), EntranceAngle=item(element.e1) if element.e1_on else 0.0, ExitAngle=item(element.e2) if element.e2_on else 0.0, PolynomA=skew, NumIntSteps=steps(element, 10))
        target.PolynomB = normal
    elif isinstance(element, Quadrupole):
        normal, skew = polynomial(element)
        target = at.Quadrupole(name, length, item(element.kn), PolynomA=skew, PolynomB=normal, NumIntSteps=steps(element, 10))
    elif isinstance(element, Sextupole):
        target = at.Sextupole(name, length, item(element.ms)/2.0, NumIntSteps=steps(element))
    elif isinstance(element, Octupole):
        normal, skew = polynomial(element)
        target = at.Octupole(name, length, skew, normal, NumIntSteps=steps(element))
    elif isinstance(element, Multipole):
        normal, skew = polynomial(element)
        target = at.Multipole(name, length, skew, normal, NumIntSteps=steps(element, 10))
    elif isinstance(element, Corrector):
        target = at.Corrector(name, 0.0, [element.factor*item(element.cx), element.factor*item(element.cy)])
    elif isinstance(element, Gradient):
        target = at.ThinMultipole(name, [0.0, item(element.ks)], [0.0, item(element.kn)])
    elif isinstance(element, Kick):
        target = at.ThinMultipole(name, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, item(element.ms)/2.0, item(element.mo)/6.0])
    elif isinstance(element, BPM):
        target = at.Monitor(name)
    elif isinstance(element, Marker):
        target = at.Marker(name)
    elif isinstance(element, Linear):
        m66, t2 = linear(element)
        target = at.M66(name, m66, T2=t2, Length=length)
    elif isinstance(element, Matrix):
        target = at.M66(name, matrix(element), Length=length)
    else:
        raise TypeError(f'unsupported model element: {element.__class__.__name__}')
    return [apply(element, target, at, alignment)]


def convert(line:Line, *,
            energy:Optional[float]=None,
            name:Optional[str]=None,
            alignment:bool=True,
            direction:list[str]=['forward'],
            **kwargs) -> Any:
    """
    Convert a model line to a pyAT lattice

    Parameters
    ----------
    line: Line
        model lattice
    energy: Optional[float]
        beam energy in GeV
    name: Optional[str]
        AT lattice name, defaults to the model line name
    alignment: bool, default=True
        flag to convert element alignment parameters
    direction: list[str], default=['forward']
        BPM directions to exclude from the converted lattice
    **kwargs:
        additional pyAT Lattice keyword arguments

    Returns
    -------
    at.Lattice

    """
    elements = []
    for element in line.scan('name'):
        if isinstance(element, BPM) and element.direction in direction:
            continue
        elements.extend(parse(element, at, alignment))
    parameters = {'name': line.name if name is None else name, **kwargs}
    if energy is not None:
        parameters['energy'] = energy*10**9
    return at.Lattice(elements, **parameters)

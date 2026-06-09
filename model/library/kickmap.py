"""
KM
--

Insertion device kick-map element

"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Optional

import numpy
from scipy.io import loadmat
import torch
from torch import Tensor

from model.library.keys import KEY_DP
from model.library.keys import KEY_DL

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


def load(path:Path) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
    """ Load kick-map table """
    table = loadmat(path)
    x = numpy.asarray(table['xtable'], dtype=float).reshape(-1)
    y = numpy.asarray(table['ytable'], dtype=float).reshape(-1)
    xkick = numpy.asarray(table['xkick1'], dtype=float).squeeze()
    ykick = numpy.asarray(table['ykick1'], dtype=float).squeeze()
    length, *_ = numpy.asarray(table['Len'], dtype=float).reshape(-1)
    xorder = numpy.argsort(x)
    yorder = numpy.argsort(y)
    x = x[xorder]
    y = y[yorder]
    xkick = xkick[numpy.ix_(xorder, yorder)]
    ykick = ykick[numpy.ix_(xorder, yorder)]
    return x, y, xkick, ykick, length


def interpolate(table:Tensor, xgrid:Tensor, ygrid:Tensor, x:Tensor, y:Tensor) -> Tensor:
    """Linear interpolation."""
    ix = torch.searchsorted(xgrid, x).clamp(1, xgrid.numel() - 1)
    iy = torch.searchsorted(ygrid, y).clamp(1, ygrid.numel() - 1)
    x0, x1 = xgrid[ix - 1], xgrid[ix]
    y0, y1 = ygrid[iy - 1], ygrid[iy]
    tx = (x - x0)/(x1 - x0)
    ty = (y - y0)/(y1 - y0)
    f00 = table[ix - 1, iy - 1]
    f10 = table[ix, iy - 1]
    f01 = table[ix - 1, iy]
    f11 = table[ix, iy]
    return (1 - tx)*(1 - ty)*f00 + tx*(1 - ty)*f10 + (1 - tx)*ty*f01 + tx*ty*f11


class KM(Element):
    """
    Kick-map element
    ----------------

    Insertion device kick map loaded from a MATLAB table

    Returns
    -------
    KM

    """
    flag: bool = False
    keys: list[str] = [KEY_DP, KEY_DL]
    rigidity: float = 0.299792458

    def __init__(self,
                 name:str,
                 path:str|Path,
                 energy:Optional[float]=None,
                 count:int=1,
                 factor_x:float=1.0,
                 factor_y:float=1.0,
                 scale:float=1.0,
                 sign_x:float=-1.0,
                 sign_y:float=-1.0,
                 dp:float=0.0, *,
                 alignment:bool=True,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
                 insertion:bool=True,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        KM instance initialization

        Parameters
        ----------
        name: str
            name
        path: str | Path
            path to MATLAB kick-map table
        energy: Optional[float], default=None
            reference energy in GeV
            if None, the map is assumed to be energy scaled
        count: int, positive, default=1
            number of periods
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
        dp: float, default=0.0
            momentum deviation
        alignment: bool, default=True
            flag to use alignment errors
        dx: float, default=0.0
            dx alignment error
        dy: float, default=0.0
            dy alignment error
        dz: float, default=0.0
            dz alignment error
        wx: float, default=0.0
            wx alignment error
        wy: float, default=0.0
            wy alignment error
        wz: float, default=0.0
            wz alignment error
        insertion: bool, default=True
            flag to compensate physical drifts and treat the element as a
            thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step

        Returns
        -------
        None

        """
        self._path = str(path)
        self._energy = energy
        self._count = count
        self._factor_x = factor_x
        self._factor_y = factor_y
        self._scale = scale
        self._sign_x = sign_x
        self._sign_y = sign_y

        x, y, xkick, ykick, period = load(Path(path))
        self._period = period

        super().__init__(name=name,
                         length=0.0 if insertion else count*period,
                         dp=dp,
                         alignment=alignment,
                         dx=dx,
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz,
                         insertion=insertion,
                         output=output,
                         matrix=matrix)

        self._xgrid = torch.tensor(x, dtype=self.dtype, device=self.device)
        self._ygrid = torch.tensor(y, dtype=self.dtype, device=self.device)
        self._xkick = torch.tensor(xkick, dtype=self.dtype, device=self.device)
        self._ykick = torch.tensor(ykick, dtype=self.dtype, device=self.device)

        self._lmatrix, self._rmatrix = self.make_matrix()
        self._data = None
        self._step = self.make_step()


    @property
    def serialize(self) -> dict[str, str|int|float|bool|None]:
        table = super().serialize
        for key in ('length', 'ns', 'order', 'exact'):
            table.pop(key, None)
        return {**table,
                'path': self.path,
                'energy': self.energy,
                'count': self.count,
                'factor_x': self.factor_x,
                'factor_y': self.factor_y,
                'scale': self.scale,
                'sign_x': self.sign_x,
                'sign_y': self.sign_y}


    def make_matrix(self) -> tuple[Tensor, Tensor]:
        state = torch.zeros(4, dtype=self.dtype, device=self.device)
        matrix = torch.func.jacrev(lambda value: value)(state)
        return matrix, matrix


    def make_step(self) -> Mapping:
        _ns: int = self.count
        _xgrid: Tensor = self._xgrid
        _ygrid: Tensor = self._ygrid
        _xkick: Tensor = self._xkick
        _ykick: Tensor = self._ykick
        _energy: Optional[float] = self.energy
        _factor_x: float = self.factor_x*self.sign_x*self.scale
        _factor_y: float = self.factor_y*self.sign_y*self.scale
        _dp: Tensor = self.dp
        _length: Tensor = self.count*self.period
        _direction: float = -1.0 if self.is_inversed else 1.0
        insertion = self.insertion
        output = self.output
        matrix = self.matrix

        def integrator(state:State, dp:Tensor, length:Tensor) -> State:
            qx, px, qy, py = state
            momentum = 1.0 + dp
            pz = torch.sqrt(momentum**2 - px**2 - py**2)
            half = 0.5*length/_ns

            qx = qx + half*px/pz
            qy = qy + half*py/pz

            xp = px/pz
            yp = py/pz
            normalization = 1.0/momentum**2 if _energy is None else (self.rigidity/(_energy*momentum))**2
            xp = xp - _direction*_factor_x*normalization*interpolate(_xkick, _xgrid, _ygrid, qx, qy)
            yp = yp - _direction*_factor_y*normalization*interpolate(_ykick, _xgrid, _ygrid, qx, qy)

            denominator = torch.sqrt(1.0 + xp**2 + yp**2)
            px = momentum*xp/denominator
            py = momentum*yp/denominator
            pz = momentum/denominator

            qx = qx + half*px/pz
            qy = qy + half*py/pz
            return torch.stack([qx, px, qy, py])

        def step(state:State, dp:Tensor, dl:Tensor) -> State:
            local_dp = _dp + dp
            local_length = _direction*(_length + dl)
            if output:
                container_output = []
            if matrix:
                container_matrix = []

            if insertion:
                qx, px, qy, py = state
                momentum = 1.0 + local_dp
                pz = torch.sqrt(momentum**2 - px**2 - py**2)
                qx = qx - 0.5*local_length*px/pz
                qy = qy - 0.5*local_length*py/pz
                state = torch.stack([qx, px, qy, py])

            for _ in range(_ns):
                if matrix:
                    container_matrix.append(torch.func.jacrev(integrator)(state, local_dp, local_length))
                state = integrator(state, local_dp, local_length)
                if output:
                    container_output.append(state)

            if insertion:
                qx, px, qy, py = state
                pz = torch.sqrt(momentum**2 - px**2 - py**2)
                qx = qx - 0.5*local_length*px/pz
                qy = qy - 0.5*local_length*py/pz
                state = torch.stack([qx, px, qy, py])

            if output:
                self.container_output = torch.stack(container_output)
            if matrix:
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step

    @property
    def path(self) -> str:
        return self._path

    @property
    def energy(self) -> Optional[float]:
        return self._energy

    @property
    def count(self) -> int:
        return self._count

    @property
    def period(self) -> Tensor:
        return torch.tensor(self._period, dtype=self.dtype, device=self.device)

    @property
    def insertion(self) -> bool:
        return self._insertion

    @insertion.setter
    def insertion(self, insertion:bool) -> None:
        self._insertion = insertion
        self._length = 0.0 if insertion else self._count*self._period
        self._step = self.make_step()

    @property
    def factor_x(self) -> float:
        return self._factor_x

    @property
    def factor_y(self) -> float:
        return self._factor_y

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def sign_x(self) -> float:
        return self._sign_x

    @property
    def sign_y(self) -> float:
        return self._sign_y

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", path="{self._path}", length={self._length}, count={self._count}, energy={self._energy}, dp={self._dp})'

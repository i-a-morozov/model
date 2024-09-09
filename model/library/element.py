"""
Element
-------

Abstract element

"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import Callable

from math import ceil

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from model.library.keys import KEY_DP, KEY_DL, KEY_DW
from model.library.keys import KEY_DX, KEY_DY, KEY_DZ
from model.library.keys import KEY_WX, KEY_WY, KEY_WZ

from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

class Element(ABC):
    """
    Abstract element
    -------------

    """
    _tolerance: float = 1.0E-16
    _alignment: list[str] = [KEY_DX, KEY_DY, KEY_DZ, KEY_WX, KEY_WY, KEY_WZ]

    dtype:DataType = Float64
    device:DataDevice = DataDevice('cpu')

    @property
    @abstractmethod
    def flag(self) -> bool:
        pass


    @property
    @abstractmethod
    def keys(self) -> list[str]:
        pass


    def __init__(self,
                 name:str,
                 length:float=0.0,
                 dp:float=0.0, *,
                 ns:int=1,
                 ds:Optional[float]=None,
                 order:int=0,
                 exact:bool=False,
                 insertion:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Element instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            length
        dp: float, default=0.0
            momentum deviation
        ns: int, positive, default=1
            number of integrtion steps
        ds: Optional[float], positive
            integration step length
            if given, input ns value is ignored and ds is used to compute ns = ceil(length/ds)
            actual integration step is not ds, but length/ns
        order: int, default=0, non-negative
            Yoshida integration order
        exact: bool, default=False
            flag to include kinematic term
        insertion: bool, default=False
            flat to treat element as thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step if output is true


        Returns
        -------
        None

        """
        self._name: str = name
        self._length: float = length
        self._dp: float = dp
        self._ns: int = ceil(self._length/ds) if ds else ns
        self._order: int = order
        self._exact: bool = exact
        self._insertion: bool = insertion
        self._output: bool = output
        self._matrix: bool = matrix

        self._lmatrix: Tensor
        self._rmatrix: Tensor

        self._data: list[list[int], list[float]]
        self._step: Mapping
        self._knob: ParametricMapping

        self.container_output:Tensor
        self.container_matrix:Tensor


    def table(self, *,
              name:bool=False,
              alignment:bool=True) -> dict[str, dict[str,Tensor]] | dict[str,Tensor]:
        """
        Generate default deviation table

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, dict[str,Tensor]] | dict[str,Tensor]

        """
        zeros: Tensor = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        table: dict[str, Tensor] = {key: value for key, value in zip(self.keys, zeros)}
        if alignment:
            keys:list[str] = self._alignment
            zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
            table = {**table, **{key: value for key, value in zip(keys, zeros)}}
        return table if not name else {self.name: table}


    @abstractmethod
    def make_matrix(self) -> tuple[Tensor, Tensor]:
        """
        Generate transformation matrices (error element)

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Tensor, Tensor]

        """
        pass


    @abstractmethod
    def make_step(self) -> tuple[Mapping, ParametricMapping]:
        """
        Generate integration step

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Mapping, ParametricMapping]

        """
        pass


    @property
    def name(self) -> str:
        """
        Get name

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._name


    @name.setter
    def name(self, name:str) -> None:
        """
        Set name

        Parameters
        ----------
        name: str
            name

        Returns
        -------
        None

        """
        self._name = name


    @property
    def length(self) -> Tensor:
        """
        Get length

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._length, dtype=self.dtype, device=self.device)


    @length.setter
    def length(self, length:float) -> None:
        """
        Set length

        Parameters
        ----------
        length: float
            length

        Returns
        -------
        None

        """
        self._length = length
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()


    @property
    def dp(self) -> Tensor:
        """
        Get momentum deviation

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._dp, dtype=self.dtype, device=self.device)


    @dp.setter
    def dp(self, dp:float) -> None:
        """
        Set momentum deviation

        Parameters
        ----------
        dp: float
            momentum deviation

        Returns
        -------
        None

        """
        self._dp = dp
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()


    @property
    def ns(self) -> int:
        """
        Get number of integration steps

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return self._ns


    @ns.setter
    def ns(self, ns:int) -> None:
        """
        Set number of integration steps

        Parameters
        ----------
        ns: int, positive
            number of intergration steps

        Returns
        -------
        None

        """
        self._ns = ns
        self._step, self._knob = self.make_step()


    @property
    def order(self) -> int:
        """
        Get integration order

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return self._order


    @order.setter
    def order(self, order:int) -> None:
        """
        Set integration order

        Parameters
        ----------
        order: int, non-negative
            integration order

        Returns
        -------
        None

        """
        self._order = order
        self._step, self._knob = self.make_step()


    @property
    def exact(self) -> bool:
        """
        Get exact flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._exact


    @exact.setter
    def exact(self, exact:bool) -> None:
        """
        Set exact flag

        Parameters
        ----------
        exact: bool
            exact

        Returns
        -------
        None

        """
        self._exact = exact
        self._step, self._knob = self.make_step()


    @property
    def insertion(self) -> bool:
        """
        Get insertion flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._insertion


    @insertion.setter
    def insertion(self, insertion:bool) -> None:
        """
        Set insertion flag

        Parameters
        ----------
        insertion: bool
            insertion

        Returns
        -------
        None

        """
        self._insertion = insertion
        self._step, self._knob = self.make_step()


    @property
    def output(self) -> bool:
        """
        Get output flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._output


    @output.setter
    def output(self, output:bool) -> None:
        """
        Set output flag

        Parameters
        ----------
        output: bool
            output

        Returns
        -------
        None

        """
        self._output = output
        self._step, self._knob = self.make_step()

    @property
    def matrix(self) -> bool:
        """
        Get matrix flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._matrix


    @matrix.setter
    def matrix(self, matrix:bool) -> None:
        """
        Set matrix flag

        Parameters
        ----------
        matrix: bool
            matrix

        Returns
        -------
        None

        """
        self._matrix = matrix
        self._step, self._knob = self.make_step()


    def __call__(self,
                 state:State, *,
                 data:Optional[dict[str, Tensor]]=None,
                 alignment:bool=False) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations and alignment valurs are passed in data
        Deviations are added to corresponding parameters

        Parameters
        ----------
        state: State
            initial input state
        data: Optional[dict[str, Tensor]]
            deviation and alignment table
        alignment: bool, default=False
            flag to apply alignment error

        Returns
        -------
        State

        """
        data: dict[str, Tensor] = data if data else {}
        knob: dict[str, Tensor] = {key: data[key] for key in self.keys if key in data}
        step: Mapping | ParametricMapping
        step = self._knob if knob else self._step
        if not alignment:
            state = step(state, **knob)
            return state
        state = transform(self, state, data)
        return state


    @abstractmethod
    def __repr__(self) -> str:
        pass


def transform(element:Element,
              state:State,
              data:dict[str, Tensor]) -> State:
    """
    Apply alignment errors

    element: Element
        element to apply error to
    state: State
        initial input state
    data: dict[str, Tensor]
        deviation and alignment table

    """
    dp:Tensor = element.dp + data.get(KEY_DP, 0.0)
    length:Tensor = element.length + data.get(KEY_DL, 0.0)
    if element.flag:
        angle:Tensor = element.angle + data.get(KEY_DW, 0.0)

    dx:Tensor
    dy:Tensor
    dz:Tensor
    dx, dy, dz = [data[key] for key in [KEY_DX, KEY_DY, KEY_DZ]]

    wx:Tensor
    wy:Tensor
    wz:Tensor
    wx, wy, wz = [data[key] for key in [KEY_WX, KEY_WY, KEY_WZ]]

    state = tx(state, +dx)
    state = ty(state, +dy)
    state = tz(state, +dz, dp)

    state = rx(state, +wx, dp)
    state = ry(state, +wy, dp)
    state = rz(state, +wz)

    state = element(state, data=data, alignment=False)

    if element.flag:
        state = ry(state, +angle/2, dp)
        state = tz(state, -2.0*length/angle*(angle/2.0).sin(), dp)
        state = ry(state, +angle/2, dp)
    else:
        state = tz(state, -length, dp)

    state = rz(state, -wz)
    state = ry(state, -wy, dp)
    state = rx(state, -wx, dp)

    state = tz(state, -dz, dp)
    state = ty(state, -dy)
    state = tx(state, -dx)

    if element.flag:
        state = ry(state, -angle/2, dp)
        state = tz(state, +2.0*length/angle*(angle/2.0).sin(), dp)
        state = ry(state, -angle/2, dp)
    else:
        state = tz(state, +length, dp)

    return state
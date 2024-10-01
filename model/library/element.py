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
from copy import deepcopy

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from model.library.keys import KEY_DP
from model.library.keys import KEY_DL
from model.library.keys import KEY_DW

from model.library.keys import KEY_DX
from model.library.keys import KEY_DY
from model.library.keys import KEY_DZ

from model.library.keys import KEY_WX
from model.library.keys import KEY_WY
from model.library.keys import KEY_WZ

from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]

class Element(ABC):
    """
    Abstract element
    -------------

    """
    _tolerance: float = 1.0E-15
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
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
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
        ns: int, positive, default=1
            number of integrtion steps
        ds: Optional[float], positive
            integration step length
            if given, input ns value is ignored and ds is used to compute ns = ceil(length/ds) or 1
            actual integration step is not ds, but length/ns
        order: int, default=0, non-negative
            Yoshida integration order
        exact: bool, default=False
            flag to use exact Hamiltonian
        insertion: bool, default=False
            flat to treat element as thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step


        Returns
        -------
        None

        """
        self._name: str = name
        self._length: float = length
        self._dp: float = dp
        self._dx: float = dx
        self._dy: float = dy
        self._dz: float = dz
        self._wx: float = wx
        self._wy: float = wy
        self._wz: float = wz
        self._ns: int = (ceil(self._length/ds) or 1) if ds else ns
        self._order: int = order
        self._exact: bool = exact
        self._insertion: bool = insertion
        self._output: bool = output
        self._matrix: bool = matrix

        self.is_inversed: bool = False

        self._lmatrix: Tensor
        self._rmatrix: Tensor

        self._data: list[list[int], list[float]]
        self._step: Mapping

        self.container_output: Tensor
        self.container_matrix: Tensor


    def clone(self) -> Element:
        """
        Clone element

        Parameters
        ----------
        None

        Returns
        -------
        Element

        """
        return deepcopy(self)


    def inverse(self) -> Element:
        """
        Inverse element

        Parameters
        ----------
        None

        Returns
        -------
        Element

        """
        element = self.clone()
        element.is_inversed = not element.is_inversed
        element.length = - element.length.item()
        return element


    def data(self, *,
             name:bool=False,
             alignment:bool=True) -> dict[str, dict[str,Tensor]] | dict[str,Tensor]:
        """
        Generate default deviation data

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, dict[str,Tensor]] | dict[str,Tensor]

        """
        zeros: Tensor = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        data: dict[str, Tensor] = {key: value for key, value in zip(self.keys, zeros)}
        if alignment:
            keys:list[str] = self._alignment
            zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
            data = {**data, **{key: value for key, value in zip(keys, zeros)}}
        return data if not name else {self.name: data}


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
    def make_step(self) -> Mapping:
        """
        Generate integration step

        Parameters
        ----------
        None

        Returns
        -------
        Mapping

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
        self._step = self.make_step()


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
        self._step = self.make_step()


    @property
    def dx(self) -> Tensor:
        """
        Get dx aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._dx, dtype=self.dtype, device=self.device)


    @dx.setter
    def dx(self, dx:float) -> None:
        """
        Set dx aligment error

        Parameters
        ----------
        dx: float
            dx aligment error

        Returns
        -------
        None

        """
        self._dx = dx


    @property
    def dy(self) -> Tensor:
        """
        Get dy aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._dy, dtype=self.dtype, device=self.device)


    @dy.setter
    def dy(self, dy:float) -> None:
        """
        Set dy aligment error

        Parameters
        ----------
        dy: float
            dy aligment error

        Returns
        -------
        None

        """
        self._dy = dy


    @property
    def dz(self) -> Tensor:
        """
        Get dz aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._dz, dtype=self.dtype, device=self.device)


    @dz.setter
    def dz(self, dz:float) -> None:
        """
        Set dz aligment error

        Parameters
        ----------
        dz: float
            dz aligment error

        Returns
        -------
        None

        """
        self._dz = dz


    @property
    def wx(self) -> Tensor:
        """
        Get wx aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._wx, dtype=self.dtype, device=self.device)


    @wx.setter
    def wx(self, wx:float) -> None:
        """
        Set wx aligment error

        Parameters
        ----------
        wx: float
            wx aligment error

        Returns
        -------
        None

        """
        self._wx = wx


    @property
    def wy(self) -> Tensor:
        """
        Get wy aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._wy, dtype=self.dtype, device=self.device)


    @wy.setter
    def wy(self, wy:float) -> None:
        """
        Set wy aligment error

        Parameters
        ----------
        wy: float
            wy aligment error

        Returns
        -------
        None

        """
        self._wy = wy


    @property
    def wz(self) -> Tensor:
        """
        Get wz aligment error

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._wz, dtype=self.dtype, device=self.device)


    @wz.setter
    def wz(self, wz:float) -> None:
        """
        Set wz aligment error

        Parameters
        ----------
        wz: float
            wz aligment error

        Returns
        -------
        None

        """
        self._wz = wz


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        data: dict[str, Tensor] = data if data else self.data()
        knob: dict[str, Tensor] = {key: data[key] for key in self.keys if key in data}
        step: Mapping = self._step
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

    dx:Tensor = element.dx + data.get(KEY_DX, 0.0)
    dy:Tensor = element.dy + data.get(KEY_DY, 0.0)
    dz:Tensor = element.dz + data.get(KEY_DZ, 0.0)

    wx:Tensor = element.wx + data.get(KEY_WX, 0.0)
    wy:Tensor = element.wy + data.get(KEY_WY, 0.0)
    wz:Tensor = element.wz + data.get(KEY_WZ, 0.0)

    if not element.is_inversed:

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

    else:

        if element.flag:
            state = ry(state, +angle/2, dp)
            state = tz(state, +2.0*length/angle*(angle/2.0).sin(), dp)
            state = ry(state, +angle/2, dp)
        else:
            state = tz(state, +length, dp)

        state = tx(state, +dx)
        state = ty(state, +dy)
        state = tz(state, +dz, dp)

        state = rx(state, +wx, dp)
        state = ry(state, +wy, dp)
        state = rz(state, +wz)

        if element.flag:
            state = ry(state, -angle/2, dp)
            state = tz(state, -2.0*length/angle*(angle/2.0).sin(), dp)
            state = ry(state, -angle/2, dp)
        else:
            state = tz(state, -length, dp)

        state = element(state, data=data, alignment=False)

        state = rz(state, -wz)
        state = ry(state, -wy, dp)
        state = rx(state, -wx, dp)

        state = tz(state, -dz, dp)
        state = ty(state, -dy)
        state = tx(state, -dx)

    return state

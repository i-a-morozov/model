"""
BPM
---

BPM element

Represents a BPM element
Calibration errors are deviation variables (not specified on initialization)
Transforms input to BPM or to beam frame (initialization flag)
To represent the full transformation (beam-BPM-beam), two elements with the same name can be used (or change the direction switch)

"""
from __future__ import annotations

from typing import Callable
from typing import Literal

import torch
from torch import Tensor

from model.library.keys import KEY_XX
from model.library.keys import KEY_XY
from model.library.keys import KEY_YX
from model.library.keys import KEY_YY
from model.library.keys import KEY_DP

from model.library.element import Element

from model.library.transformations import calibration_forward
from model.library.transformations import calibration_inverse

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


class BPM(Element):
    """
    BPM element
    -----------

    Zero lenght element, can't be used in insertion mode

    Returns
    -------
    BPM

    """
    flag: bool = False
    keys: list[str] = [KEY_XX, KEY_XY, KEY_YX, KEY_YY, KEY_DP]


    def __init__(self,
                 name:str,
                 direction:Literal['forward', 'inverse']='forward',
                 dp:float=0.0, *,
                 alignment:bool=True,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        BPM instance initialization

        Parameters
        ----------
        name: str
            name
        direction: Literal['forward', 'inverse'], default='forward'
            transformation direction
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
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step

        Returns
        -------
        None

        """
        super().__init__(name=name,
                         dp=dp,
                         alignment=alignment,
                         dx=dx,
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz,
                         output=output,
                         matrix=matrix)

        self._direction: Literal['forward', 'inverse'] = direction

        self._lmatrix: Tensor
        self._rmatrix: Tensor
        self._lmatrix, self._rmatrix = self.make_matrix()

        self._data: list[list[int], list[float]] = None
        self._step: Mapping = self.make_step()


    @property
    def serialize(self) -> dict[str, str|int|float|bool]:
        """
        Serialize element

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, str|int|float|bool]

        """
        table:dict[str, str|int|float|bool] = super().serialize
        table.pop('length', None)
        table.pop('ns', None)
        table.pop('order', None)
        table.pop('exact', None)
        table.pop('insertion', None)
        return {**table, 'direction': self.direction}


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
        element.direction = {'forward': 'inverse', 'inverse': 'forward'}[element.direction]
        return element


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
        state: State = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)

        matrix: Tensor = torch.func.jacrev(lambda state: state)(state)

        lmatrix: Tensor = matrix
        rmatrix: Tensor = matrix

        return lmatrix, rmatrix


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
        output:bool = self.output
        matrix:bool = self.matrix

        calibration = {'forward': calibration_forward, 'inverse': calibration_inverse}[self.direction]

        def integrator(state:State, xx:Tensor, xy:Tensor, yx:Tensor, yy:Tensor) -> State:
            return calibration(state, xx, xy, yx, yy)


        def step(state:State, xx:Tensor, xy:Tensor, yx:Tensor, yy:Tensor, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state, xx, xy, yx, yy)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, xx, xy, yx, yy))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step

    @property
    def direction(self) -> Tensor:
        """
        Get direction

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._direction


    @direction.setter
    def direction(self,
                  direction:Literal['forward', 'inverse']) -> None:
        """
        Set direction

        Parameters
        ----------
        direction: Literal['forward', 'inverse']
            direction

        Returns
        -------
        None

        """
        self._direction = direction
        self._step = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", direction="{self._direction}", dp={self._dp})'

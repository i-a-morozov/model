"""
Corrector
---------

Corrector element (thin dipole kick)

px -> px + kx
py -> py + ky

"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_CX
from model.library.keys import KEY_CY
from model.library.keys import KEY_DP

from model.library.element import Element

from model.library.transformations import corrector

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

class Corrector(Element):
    """
    Corrector element
    -----------------

    Zero lenght element, can't be used in insertion mode

    Returns
    -------
    Corrector

    """
    flag: bool = False
    keys: list[str] = [KEY_CX, KEY_CY, KEY_DP]

    def __init__(self,
                 name:str,
                 cx:float=0.0,
                 cy:float=0.0,
                 dp:float=0.0, *,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Corrector instance initialization

        Parameters
        ----------
        name: str
            name
        cx: float, default=0.0
            px -> px + cx
        cy: float, default=0.0
            py -> py + cy
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
                         dx=dx,
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz,
                         output=output,
                         matrix=matrix)

        self._cx: float = cx
        self._cy: float = cy

        self._lmatrix: Tensor
        self._rmatrix: Tensor
        self._lmatrix, self._rmatrix = self.make_matrix()

        self._data: list[list[int], list[float]] = None
        self._step: Mapping
        self._knob: ParametricMapping
        self._step, self._knob = self.make_step()


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
        _cx: Tensor = self.cx
        _cy: Tensor = self.cy

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State, cx:Tensor, cy:Tensor) -> State:
            return corrector(state, cx, cy)

        def step(state:State) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            state = integrator(state, _cx, _cy)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _cx, _cy))
                self.container_matrix = torch.stack(container_matrix)
            return state

        def knob(state:State, cx:Tensor, cy:Tensor, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state, _cx + cx, _cy + cy)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _cx + cx, _cy + cy))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step, knob


    @property
    def cx(self) -> Tensor:
        """
        Get cx

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._cx, dtype=self.dtype, device=self.device)


    @cx.setter
    def cx(self,
           cx:float) -> None:
        """
        Set cx

        Parameters
        ----------
        cx: float
            cx

        Returns
        -------
        None

        """
        self._cx = cx
        self._step, self._knob = self.make_step()


    @property
    def cy(self) -> Tensor:
        """
        Get cy

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._cy, dtype=self.dtype, device=self.device)


    @cy.setter
    def cy(self,
           cy:float) -> None:
        """
        Set cy

        Parameters
        ----------
        cy: float
            cy

        Returns
        -------
        None

        """
        self._cy = cy
        self._step, self._knob = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", cx={self._cx}, cy={self._cy})'
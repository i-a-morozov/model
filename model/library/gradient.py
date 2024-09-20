"""
Gradient
--------

Gradient element (thin quadrupole kick)

px -> px - kn*qx + ks*qy
py -> py + kn*qy + ks*qx

"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_KN
from model.library.keys import KEY_KS
from model.library.keys import KEY_DP

from model.library.element import Element

from model.library.transformations import gradient

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

class Gradient(Element):
    """
    Gradient element
    ----------------

    Zero lenght element, can't be used in insertion mode

    Returns
    -------
    Gradient

    """
    flag: bool = False
    keys: list[str] = [KEY_KN, KEY_KS, KEY_DP]

    def __init__(self,
                 name:str,
                 kn:float=0.0,
                 ks:float=0.0,
                 dp:float=0.0,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Gradient instance initialization

        Parameters
        ----------
        name: str
            name
        kn: float, default=0.0
            px -> px - kn*qx + ks*qy
            py -> py + kn*qy + ks*qx
        ks: float, default=0.0
            px -> px - kn*qx + ks*qy
            py -> py + kn*qy + ks*qx
        dp: float, default=0.0
            momentum deviation
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
                         output=output,
                         matrix=matrix)

        self._kn: float = kn
        self._ks: float = ks

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

        matrix: Tensor = torch.func.jacrev(gradient)(state, self.kn, self.ks, -0.5)

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
        _kn: Tensor = self.kn
        _ks: Tensor = self.ks

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State, kn:Tensor, ks:Tensor) -> State:
            return gradient(state, kn, ks, 1.0)

        def step(state:State) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            state = integrator(state, _kn, _ks)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _kn, _ks))
                self.container_matrix = torch.stack(container_matrix)
            return state

        def knob(state:State, kn:Tensor, ks:Tensor, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state, _kn + kn, _ks + ks)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _kn + kn, _ks + ks))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step, knob


    @property
    def kn(self) -> Tensor:
        """
        Get kn

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._kn, dtype=self.dtype, device=self.device)


    @kn.setter
    def kn(self,
           kn:float) -> None:
        """
        Set kn

        Parameters
        ----------
        kn: float
            kn

        Returns
        -------
        None

        """
        self._kn = kn
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()


    @property
    def ks(self) -> Tensor:
        """
        Get ks

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._ks, dtype=self.dtype, device=self.device)


    @ks.setter
    def ks(self,
           ks:float) -> None:
        """
        Set ks deviation

        Parameters
        ----------
        ks: float
            ks

        Returns
        -------
        None

        """
        self._ks = ks
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", kn={self._kn}, ks={self._ks})'
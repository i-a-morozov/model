"""
Linear
------

4D linear transformation element

(qx, px, qy, py) -> M @ (qx, px, qy, py) + (cqx, cpx, cqy, cpy)

"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.element import Element

from model.library.transformations import linear

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]


class Linear(Element):
    """
    Linear  matrix element
    ----------------------

    Zero lenght element, can't be used in insertion mode
    Note, linear transformation has no deviation variables


    Returns
    -------
    Linear

    """
    flag: bool = False
    keys: list[str] = []

    def __init__(self,
                 name:str,
                 v:list[float],
                 m:list[list[float]],
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Gradient instance initialization

        Parameters
        ----------
        name: str
            name
        v: list[float]
            constant vector
        m: list[list[float]]
            matrix
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step

        Returns
        -------
        None

        """
        super().__init__(name=name,
                         output=output,
                         matrix=matrix)

        self._v: list[float] = v
        self._m: list[list[float]] = m

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
        _v: Tensor = self.v
        _m: Tensor = self.m

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State, vector:Tensor, matrix:Tensor) -> State:
            return linear(state, vector, matrix)

        def step(state:State) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            state = integrator(state, _v, _m)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _v, _m))
                self.container_matrix = torch.stack(container_matrix)
            return state

        def knob(state:State) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            state = integrator(state, _v, _m)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _v, _m))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step, knob


    @property
    def v(self) -> Tensor:
        """
        Get vector

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._v, dtype=self.dtype, device=self.device)


    @v.setter
    def v(self,
          v:list[float]) -> None:
        """
        Set vector

        Parameters
        ----------
        v: list[float]
            vector

        Returns
        -------
        None

        """
        self._v = v
        self._step, self._knob = self.make_step()


    @property
    def m(self) -> Tensor:
        """
        Get matrix

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._m, dtype=self.dtype, device=self.device)


    @m.setter
    def m(self,
          m:list[list[float]]) -> None:
        """
        Set matrix

        Parameters
        ----------
        m: list[float]
            matrix

        Returns
        -------
        None

        """
        self._m = m
        self._step, self._knob = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", v={self._v}, m={self._m})'
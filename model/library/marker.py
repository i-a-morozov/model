"""
Marker
------

Marker element (identiy transformation)

"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_DP

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]


class Marker(Element):
    """
    Marker element
    -----------

    Zero lenght element, can't be used in insertion mode

    Returns
    -------
    Marker

    """
    flag: bool = False
    keys: list[str] = [KEY_DP]


    def __init__(self,
                 name:str,
                 dp:float=0.0,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Marker instance initialization

        Parameters
        ----------
        name: str
            name
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
        output:bool = self.output
        matrix:bool = self.matrix

        def step(state:State) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(lambda state: state)(state))
                self.container_matrix = torch.stack(container_matrix)
            return state

        def knob(state:State, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                 container_matrix = []
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(lambda state: state)(state))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step, knob


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}")'
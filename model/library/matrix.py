"""
Matrix
------

4D linear transport matrix

(qx, px, qy, py) -> exp(S @ A) @ (qx, px, qy, py)

"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_DP

from model.library.keys import KEY_A11
from model.library.keys import KEY_A12
from model.library.keys import KEY_A13
from model.library.keys import KEY_A14
from model.library.keys import KEY_A22
from model.library.keys import KEY_A23
from model.library.keys import KEY_A24
from model.library.keys import KEY_A33
from model.library.keys import KEY_A34
from model.library.keys import KEY_A44

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


class Matrix(Element):
    """
    Matrix element
    ----------------------

    Set A22 and A44 to the element lenthg to model a linear drift

    Returns
    -------
    Linear

    """
    flag: bool = False
    keys: list[str] = [KEY_A11, KEY_A12, KEY_A13, KEY_A14, KEY_A22, KEY_A23, KEY_A24, KEY_A33, KEY_A34, KEY_A44, KEY_DP]

    def __init__(self,
                 name:str,
                 length:float=0.0,
                 elements:list[float]=10*[0.0],
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
        Gradient instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            length
        elements: list[float]
            symmetric matrix elements
            A11, A12, A13, A14, A22, A23, A24, A33, A34, A44
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
                         length=length,
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

        self._elements: list[float] = elements

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
        table.pop('ns', None)
        table.pop('order', None)
        table.pop('exact', None)
        table.pop('insertion', None)
        return {**table, 'elements': self.elements.tolist()}


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
        _identity: Tensor = torch.tensor([[0.,  1.,  0.,  0.],
                                          [-1., 0.,  0.,  0.],
                                          [0.,  0.,  0.,  1.],
                                          [0.,  0., -1.,  0.]], dtype=self.dtype, device=self.device)
        _a11: Tensor
        _a12: Tensor
        _a13: Tensor
        _a14: Tensor
        _a22: Tensor
        _a23: Tensor
        _a24: Tensor
        _a33: Tensor
        _a34: Tensor
        _a44: Tensor
        _a11, _a12, _a13, _a14, _a22, _a23, _a24, _a33, _a34, _a44 = self._elements
        _direction = 1.0

        if self.is_inversed:
            _direction = - 1.0

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State, a11, a12, a13, a14, a22, a23, a24, a33, a34, a44) -> State:
            matrix = _direction*torch.stack([a11, a12, a13, a14, a12, a22, a23, a24, a13, a23, a33, a34, a14, a24, a34, a44]).reshape(4, 4)
            return torch.linalg.matrix_exp(_identity @ matrix) @ state
            
        def step(state:State, a11, a12, a13, a14, a22, a23, a24, a33, a34, a44, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state, _a11 + a11, _a12 + a12, _a13 + a13, _a14 + a14, _a22 + a22, _a23 + a23, _a24 + a24, _a33 + a33, _a34 + a34, _a44 + a44)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _a11 + a11, _a12 + a12, _a13 + a13, _a14 + a14, _a22 + a22, _a23 + a23, _a24 + a24, _a33 + a33, _a34 + a34, _a44 + a44))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step


    @property
    def elements(self) -> Tensor:
        """
        Get elements

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._elements, dtype=self.dtype, device=self.device)


    @elements.setter
    def elements(self,
                 elements:list[float]) -> None:
        """
        Set vector

        Parameters
        ----------
        elements: list[float]
            elements

        Returns
        -------
        None

        """
        self._elements = elements
        self._step = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, elements={self._elements}, m={self._m}, dp={self._dp})'
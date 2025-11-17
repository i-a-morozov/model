"""
Matrix
------

4D linear transport matrix

(qx, px, qy, py) -> exp(S @ A) @ exp(delta * S @ B) @ (qx, px, qy, py)

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

from model.library.keys import KEY_B11
from model.library.keys import KEY_B12
from model.library.keys import KEY_B13
from model.library.keys import KEY_B14
from model.library.keys import KEY_B22
from model.library.keys import KEY_B23
from model.library.keys import KEY_B24
from model.library.keys import KEY_B33
from model.library.keys import KEY_B34
from model.library.keys import KEY_B44

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


class Matrix(Element):
    """
    Matrix element
    --------------

    M = exp(S @  A) @ exp(delta * S @ B)

    Set A22 and A44 to the element length to model a linear drift

    Returns
    -------
    Linear

    """
    flag: bool = False
    keys: list[str] = [
        KEY_A11, KEY_A12, KEY_A13, KEY_A14, KEY_A22, KEY_A23, KEY_A24, KEY_A33, KEY_A34, KEY_A44,
        KEY_B11, KEY_B12, KEY_B13, KEY_B14, KEY_B22, KEY_B23, KEY_B24, KEY_B33, KEY_B34, KEY_B44,
        KEY_DP
    ]

    def __init__(self,
                 name:str,
                 length:float=0.0,
                 A:list[float]=10*[0.0],
                 B:list[float]=10*[0.0],
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
        Matrix instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            length
        A: list[float]
            symmetric A matrix elements
            A11, A12, A13, A14, A22, A23, A24, A33, A34, A44
        B: list[float]
            symmetric B matrix elements
            B11, B12, B13, B14, B22, B23, B24, B33, B34, B44
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

        self._A: list[float] = A
        self._B: list[float] = B

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
        _a11, _a12, _a13, _a14, _a22, _a23, _a24, _a33, _a34, _a44 = self._A

        _b11: Tensor
        _b12: Tensor
        _b13: Tensor
        _b14: Tensor
        _b22: Tensor
        _b23: Tensor
        _b24: Tensor
        _b33: Tensor
        _b34: Tensor
        _b44: Tensor
        _b11, _b12, _b13, _b14, _b22, _b23, _b24, _b33, _b34, _b44 = self._B

        _dp: Tensor = self.dp
        _direction = 1.0

        if self.is_inversed:
            _direction = - 1.0

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State,
                       a11, a12, a13, a14, a22, a23, a24, a33, a34, a44,
                       b11, b12, b13, b14, b22, b23, b24, b33, b34, b44,
                       dp:Tensor) -> State:
            A = _direction*torch.stack([a11, a12, a13, a14, a12, a22, a23, a24, a13, a23, a33, a34, a14, a24, a34, a44]).reshape(4, 4)
            B = _direction*torch.stack([b11, b12, b13, b14, b12, b22, b23, b24, b13, b23, b33, b34, b14, b24, b34, b44]).reshape(4, 4)
            return torch.linalg.matrix_exp(_identity @ A) @ torch.linalg.matrix_exp(dp * _identity @ B) @ state
            
        def step(state:State,
                 a11, a12, a13, a14, a22, a23, a24, a33, a34, a44,
                 b11, b12, b13, b14, b22, b23, b24, b33, b34, b44,
                 dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state,
                              _a11 + a11, _a12 + a12, _a13 + a13, _a14 + a14, _a22 + a22, _a23 + a23, _a24 + a24, _a33 + a33, _a34 + a34, _a44 + a44,
                              _b11 + b11, _b12 + b12, _b13 + b13, _b14 + b14, _b22 + b22, _b23 + b23, _b24 + b24, _b33 + b33, _b34 + b34, _b44 + b44,
                              _dp + dp)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state,
                                                                     _a11 + a11, _a12 + a12, _a13 + a13, _a14 + a14, _a22 + a22, _a23 + a23, _a24 + a24, _a33 + a33, _a34 + a34, _a44 + a44,
                                                                     _b11 + b11, _b12 + b12, _b13 + b13, _b14 + b14, _b22 + b22, _b23 + b23, _b24 + b24, _b33 + b33, _b34 + b34, _b44 + b44,
                                                                     _dp + dp))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step


    @property
    def A(self) -> Tensor:
        """
        Get A elements

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._A, dtype=self.dtype, device=self.device)


    @A.setter
    def A(self,
          A:list[float]) -> None:
        """
        Set vector

        Parameters
        ----------
        A: list[float]
            A elements

        Returns
        -------
        None

        """
        self._A = A
        self._step = self.make_step()


    @property
    def B(self) -> Tensor:
        """
        Get B elements

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._B, dtype=self.dtype, device=self.device)


    @B.setter
    def B(self,
          B:list[float]) -> None:
        """
        Set vector

        Parameters
        ----------
        B: list[float]
            B elements

        Returns
        -------
        None

        """
        self._B = B
        self._step = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, A={self._A}, B={self._B}, dp={self._dp})'
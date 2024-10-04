"""
Gradient
--------

Kick element (thin sextuple and/or octupole kick)


"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_MS
from model.library.keys import KEY_MO
from model.library.keys import KEY_DP

from model.library.element import Element

from model.library.transformations import sextupole
from model.library.transformations import octupole

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]

class Kick(Element):
    """
    Kick element
    ------------

    Zero lenght element, can't be used in insertion mode

    Returns
    -------
    Gradient

    """
    flag: bool = False
    keys: list[str] = [KEY_MS, KEY_MO, KEY_DP]

    def __init__(self,
                 name:str,
                 ms:float=0.0,
                 mo:float=0.0,
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
        Gradient instance initialization

        Parameters
        ----------
        name: str
            name
        ms: float, default=0.0
            sextupole strength (knl)
        mo: float, default=0.0
            octupole strength (knl)
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
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz,
                         output=output,
                         matrix=matrix)

        self._ms: float = ms
        self._mo: float = mo

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
        return {**table, 'ms': self.ms.item(), 'mo': self.mo.item()}


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
        _ms: Tensor = self.ms
        _mo: Tensor = self.mo

        if self.is_inversed:
            _ms = -_ms
            _mo = -_mo

        output:bool = self.output
        matrix:bool = self.matrix

        def integrator(state:State, ms:Tensor, mo:Tensor) -> State:
            return octupole(sextupole(state, ms, 1.0), mo, 1.0)

        def step(state:State, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
            if output:
                container_output = []
            if matrix:
                container_matrix = []
            state = integrator(state, _ms + ms, _mo + mo)
            if output:
                container_output.append(state)
                self.container_output = torch.stack(container_output)
            if matrix:
                container_matrix.append(torch.func.jacrev(integrator)(state, _ms + ms, _mo + mo))
                self.container_matrix = torch.stack(container_matrix)
            return state

        return step


    @property
    def ms(self) -> Tensor:
        """
        Get ms

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._ms, dtype=self.dtype, device=self.device)


    @ms.setter
    def ms(self,
           ms:float) -> None:
        """
        Set ms

        Parameters
        ----------
        ms: float
            ms

        Returns
        -------
        None

        """
        self._ms = ms
        self._step = self.make_step()


    @property
    def mo(self) -> Tensor:
        """
        Get mo

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._mo, dtype=self.dtype, device=self.device)


    @mo.setter
    def mo(self,
           mo:float) -> None:
        """
        Set mo deviation

        Parameters
        ----------
        mo: float
            mo

        Returns
        -------
        None

        """
        self._mo = mo
        self._step = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", ms={self._ms}, mo={self._mo}, dp={self._dp})'
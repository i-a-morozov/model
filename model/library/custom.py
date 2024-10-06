"""
Custom
------

Custom element (custom transformation wrapper)

"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from model.library.keys import KEY_DP

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]

class Custom(Element):
    """
    Custom element
    --------------

    Note, input parameters are used only with alignment errors
    Custom element is not serializable and not invertible
    Slicing is not supported

    Returns
    -------
    Custom

    """
    flag: bool = False
    keys: list[str] = [KEY_DP]

    def __init__(self,
                 name:str,
                 step:Mapping,
                 keys:list[str],
                 length:float=0.0,
                 angle:float=0.0,
                 dp:float=0.0, *,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0) -> None:
        """
        Custom instance initialization

        Parameters
        ----------
        name: str
            name
        step: Mapping
            transformation step
        keys: list[str]
            deviation keys
        length: float, default=0.0
            length
        angle: float, default=0.0
            angle
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

        Returns
        -------
        None

        """
        super().__init__(name=name,
                         length=length,
                         dp=dp,
                         dx=dx,
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz)

        self._step: Mapping = step
        self.keys: list[str] = keys + self.keys
        self._angle: float = angle


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
        keys:list[str] = self.keys
        zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
        data: dict[str, Tensor] = {key: value for key, value in zip(keys, zeros)}
        if alignment:
            keys:list[str] = self._alignment
            zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
            data = {**data, **{key: value for key, value in zip(keys, zeros)}}
        return data if not name else {self.name: data}


    @property
    def serialize(self) -> dict[str, str|int|float|bool]:
        raise NotImplementedError


    def make_step(self):
        raise NotImplementedError


    def make_matrix(self):
        raise NotImplementedError


    @property
    def angle(self) -> Tensor:
        """
        Get angle

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._angle, dtype=self.dtype, device=self.device)


    @angle.setter
    def angle(self,
              angle:float) -> None:
        """
        Set angle

        Parameters
        ----------
        angle: float
            angle

        Returns
        -------
        None

        """
        self._angle = angle


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, angle={self._angle}, dp={self._dp})'
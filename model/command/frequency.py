"""
Frequency
---------

Frequency factory

"""
from typing import Callable

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

def filter(
    length:int,
    degree:float=1.0, *,
    dtype:DataType=Float64,
    device:DataDevice=DataDevice('cpu')) -> Tensor:
    """
    Generate normalized exponential analytical filter

    Parameters
    ----------
    length: int
        length
    degree: float, default=1.0
        degree
    dtype: DataType, default=Float64
        data type
    device: DataDevice, default=DataDevice('cpu')
        device

    Returns
    -------
    Tensor

    """
    s:Tensor = torch.linspace(0.0, (length - 1.0)/length, length, dtype=dtype, device=device)
    w:Tensor = torch.exp(-1.0/((1.0 - s)**degree*s**degree))
    return w/w.sum()


def frequency_factory(trajectory:Callable[[int, Tensor, ...], Tensor]) -> Callable[[Tensor, Tensor, ...], Tensor]:
    """
    Generate frequency computation function from given trajectory generator (single location)

    Parameters
    ----------
    trajectory: Callable[[int, Tensor, ...], Tensor]
        trajectory function

    Returns
    -------
    Callable[[Tensor, Tensor, ...], Tensor]

    """
    def frequency(window:Tensor, state:Tensor, *args:tuple):
        qx, px, qy, py = trajectory(len(window), state, *args).T
        qs = torch.stack([qx, qy])
        ps = torch.stack([px, py])
        return ((window @ (torch.diff(torch.atan2(qs, ps)) % (2.0*torch.pi)).T)/(2.0*torch.pi)).squeeze()
    return frequency
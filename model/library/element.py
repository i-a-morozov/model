"""
Element protocol
----------------

Complient element is required to have several attibutes and (overloaded) __call__ method


__call__(self, state:State) -> State
    Transform initial input state using attibutes


__call__(self, state:State, dxyz:list[Tensor, Tensor, Tensor], rxyz:list[Tensor, Tensor, Tensor]) -> State
    Transform initial input state using attibutes
    Apply alignment errors (dxyz, wzyz)


__call__(self, state:State, *knobs:tuple, error:bool=False) -> State
    Transform initial input state using attibutes and deviations
    Deviations are added to corresponding parameters
    Expected to be differentiable with respect to deviations
    Treat as thin insertion (error flag), if deviations are zero, full transformation is identity

    
__call__(self, state:State, *knobs:tuple, dxyz:list[Tensor, Tensor, Tensor], rxyz:list[Tensor, Tensor, Tensor], error:bool=False) -> State
    Transform initial input state using attibutes and deviations
    Deviations are added to corresponding parameters
    Expected to be differentiable with respect to deviations
    Apply alignment errors (dxyz, wzyz)
    Treat as thin insertion (error flag), if deviations are zero, full transformation is identity


"""
from typing import Protocol

from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice

type State = Tensor
type Knobs = Tensor

class Element(Protocol):
    name: str
    length: Tensor
    dtype: DataType
    device: DataDevice
    def __call__(self, state:State, *args:tuple, **kwargs:dict) -> State:
        pass
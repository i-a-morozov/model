"""
Element protocol
----------------

Compliant element is required to have several attibutes and __call__ method

__call__(self, state:State, insertion:bool=False, alignment:bool=False, **kwargs:dict[str, Tensor]) -> State
    Deviation and alignment errors are passed in kwargs
    Apply alignment errors (alignment flag)
    Treat as thin insertion (insertion flag), if deviations are zero, full transformation is identity

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
    def __call__(self, state:State, insertion:bool=False, alignment:bool=False, **kwargs:dict) -> State:
        pass
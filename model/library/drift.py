"""
Drift
-----

Drift element

"""
from __future__ import annotations

from typing import Optional
from typing import Callable

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from ndmap.yoshida import yoshida

from model.library.element import Element
from model.library.transformations import drift
from model.library.transformations import kinematic

type State = Tensor

class Drift(Element):
    """
    Drift element
    -------------
    
    Returns
    -------
    Drift
    
    """
    flag: bool = False
    keys: list[str] = ['dp', 'dl']

    def __init__(self, 
                 name:str, 
                 length:float=0.0,
                 dp:float=0.0, *,
                 ns:int=1,
                 ds:Optional[float]=None,
                 order:int=0,
                 exact:bool=False,
                 dtype:DataType = Float64,
                 device:DataDevice = DataDevice('cpu')) -> None:
        """
        Drift instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            length
        dp: float, default=0.0
            momentum deviation
        ns: int, positive, default=1
            number of integrtion steps
        ds: Optional[float], positive
            integration step length
            if given, input ns value is ignored and ds is used to compute ns = ceil(length/ds)
            actual integration step is not ds, but length/ns
        order: int, default=0, non-negative
            Yoshida integration order
        exact: bool, default=False
            flag to include kinematic term
        dtype: DataType, default=Float64
            data type
        device: DataDevice, default=DataDevice('cpu')
            data device        

        Returns
        -------
        None
        
        """
        super().__init__(name=name, 
                         length=length, 
                         dp=dp, 
                         ns=ns,
                         ds=ds, 
                         order=order, 
                         exact=exact, 
                         dtype=dtype, 
                         device=device)
        
        self._data: list[list[int], list[float]] = None
        self._step: Callable[[State], State]
        self._knob: Callable[[State, Tensor, ...], State]
        self._step, self._knob = self.make_step()

        self._lmat: Tensor
        self._rmat: Tensor
        self._lmat, self._rmat = self.make_matrix()    

    def make_step(self) -> tuple[Callable[[State], State], Callable[[State, Tensor, ...], State]]:
        """
        Generate integration step

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Callable[[State], State], Callable[[State, Tensor, ...], State]]
        
        """        
        def drif_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
            return drift(state, dp, ds)
            
        def sqrt_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
            return kinematic(state, dp, ds) if self.exact else state
            
        integrator: Callable[[State], State, Tensor, ...]
        integrator = yoshida(0, self.order, True, [drif_wrapper, sqrt_wrapper])

        self._data: list[list[int], list[float]] = integrator.table
        
        def step(state:State) -> State:
            return integrator(state, self.length/self.ns, self.dp)
            
        def knob(state:State, dp:Tensor, dl:Tensor) -> State:
            return integrator(state, (self.length + dl)/self.ns, self.dp + dp)
            
        return step, knob  


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
        
        matrix: Tensor = torch.func.jacrev(drift)(state, 0.0*self.dp, -0.5*self.length)
        
        lmat: Tensor = matrix
        rmat: Tensor = matrix
        
        return lmat, rmat

    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
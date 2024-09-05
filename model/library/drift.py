"""
Drift
-----

Drift element

"""
from typing import Optional
from typing import Callable

from math import ceil

from multimethod import multimethod

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from ndmap.yoshida import yoshida

from model.library.transformations import drift
from model.library.transformations import kinematic
from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor
type Knobs = Tensor


class Drift:
    """
    Drift element
    -------------
    
    Returns
    -------
    Drift
    
    """
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
        self._name: str = name
        self._length: float = length
        self._dp: float = dp
        self._ns: int = ceil(self._length/ds) if ds else ns        
    
        self.order: bool = order
        self.exact: bool = exact
        self.dtype: DataType = dtype
        self.device: DataDevice = device

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


    @property
    def name(self) -> str:
        """
        Get name

        Parameters
        ----------
        None

        Returns
        -------
        str
        
        """        
        return self._name
        
    
    @name.setter
    def name(self, 
             name:str) -> None:
        """
        Set name

        Parameters
        ----------
        name: str
            name

        Returns
        -------
        None
        
        """        
        self._name = name
        
        
    @property
    def length(self) -> Tensor:
        """
        Get length

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
        
        """     
        return torch.tensor(self._length, dtype=self.dtype, device=self.device)
    
    
    @length.setter
    def length(self, 
               length:float) -> None:
        """
        Set length

        Parameters
        ----------
        length: float
            length

        Returns
        -------
        None
        
        """       
        self._length = length
        self._step, self._knob = self.make_step()
        self._lmat, self._rmat = self.make_matrix()
    
    
    @property
    def dp(self) -> Tensor:
        """
        Get momentum deviation

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
        
        """       
        return torch.tensor(self._dp, dtype=self.dtype, device=self.device)
    
    
    @dp.setter
    def dp(self, 
           dp:float) -> None:
        """
        Set momentum deviation

        Parameters
        ----------
        dp: float
            momentum deviation

        Returns
        -------
        None
        
        """       
        self._dp = dp
        self._step, self._knob = self.make_step()
        self._lmat, self._rmat = self.make_matrix()


    @property
    def ns(self) -> int:
        """
        Get number of integration steps

        Parameters
        ----------
        None

        Returns
        -------
        int
        
        """       
        return self._ns
    
    
    @ns.setter
    def ns(self, 
           ns:int) -> None:
        """
        Set number of integration steps

        Parameters
        ----------
        ns: int, positive
            number of intergration steps

        Returns
        -------
        None
        
        """          
        self._ns = ns
        self._step, self._knob = self.make_step()
        self._lmat, self._rmat = self.make_matrix()


    @multimethod
    def __call__(self, 
                 state:State) -> State:
        """
        Transform initial input state using attibutes

        Parameters
        ----------
        state: State
            initial input state

        Returns
        -------
        State
        
        """
        for _ in range(self.ns):
            state = self._step(state)
        return state


    @multimethod
    def __call__(self, 
                 state:State, 
                 dxyz:list[Tensor, Tensor, Tensor],
                 wxyz:list[Tensor, Tensor, Tensor]) -> State:
        """
        Transform initial input state using attibutes
        Apply alignment errors (dxyz, wzyz)

        Parameters
        ----------
        state: State
            initial input state
        dxyz: list[Tensor, Tensor, Tensor]
            translation errors
        wxyz: list[Tensor, Tensor, Tensor]
            rotation errors

        Returns
        -------
        State
        
        """      
        dx:Tensor
        dy:Tensor
        dz:Tensor        
        wx:Tensor
        wy:Tensor
        wz:Tensor   
        dx, dy, dz = dxyz
        wx, wy, wz = wxyz
        
        state = tx(state, +dx)
        state = ty(state, +dy)
        state = tz(state, +dz, self.dp)
        
        state = rx(state, +wx, self.dp)
        state = ry(state, +wy, self.dp)
        state = rz(state, +wz)
        
        state = self(state)
        state = tz(state, -self.length, self.dp)
        
        state = rz(state, -wz)
        state = ry(state, -wy, self.dp)
        state = rx(state, -wx, self.dp)
        
        state = tz(state, -dz, self.dp)
        state = ty(state, -dy)
        state = tx(state, -dx)
        
        state = tz(state, +self.length, self.dp)
        
        return state


    @multimethod            
    def __call__(self, 
                 state:State,
                 dp:Tensor, 
                 dl:Tensor, *,
                 error:bool=False) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations are added to corresponding parameters
        Expected to be differentiable with respect to deviations
        Treat as thin insertion (error flag), if deviations are zero, full transformation is identity

        Parameters
        ----------
        state: State
            initial input state
        dp: Tensor
            momentum deviation
        dl: Tensor
            length deviation

        Returns
        -------
        State
        
        """       
        if error:
            state = self._lmat @ state
            
        for _ in range(self.ns):
            state = self._knob(state, dp, dl)
            
        if error:
            state = self._rmat @ state
            
        return state


    @multimethod            
    def __call__(self, 
                 state:State,
                 dp:Tensor, 
                 dl:Tensor, 
                 dxyz:list[Tensor, Tensor, Tensor],
                 wxyz:list[Tensor, Tensor, Tensor], *,
                 error:bool=False) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations are added to corresponding parameters
        Expected to be differentiable with respect to deviations
        Apply alignment errors (dxyz, rzyz)
        Treat as thin insertion (error flag), if deviations are zero, full transformation is identity

        Parameters
        ----------
        state: State
            initial input state
        dp: Tensor
            momentum deviation
        dl: Tensor
            length deviation            
        dxyz: list[Tensor, Tensor, Tensor]
            translation errors
        wxyz: list[Tensor, Tensor, Tensor]
            rotation errors            

        Returns
        -------
        State
        
        """       
        dx:Tensor
        dy:Tensor
        dz:Tensor        
        wx:Tensor
        wy:Tensor
        wz:Tensor        
        dx, dy, dz = dxyz
        wx, wy, wz = wxyz
        
        if error:
            state = self._lmat @ state   
            
        state = tx(state, +dx)
        state = ty(state, +dy)
        state = tz(state, +dz, self.dp + dp)
        
        state = rx(state, +wx, self.dp + dp)
        state = ry(state, +wy, self.dp + dp)
        state = rz(state, +wz)
        
        state = self(state, dp, dl)
        state = tz(state, -(self.length + dl), self.dp + dp)
        
        state = rz(state, -wz)
        state = ry(state, -wy, self.dp + dp)
        state = rx(state, -wx, self.dp + dp)
        
        state = tz(state, -dz, self.dp + dp)
        state = ty(state, -dy)
        state = tx(state, -dx)
        
        state = tz(state, +(self.length + dl), self.dp + dp)
        
        if error:
            state = self._rmat @ state      
            
        return state 
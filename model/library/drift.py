"""
Drift
-----

Drift element

"""
from typing import Optional
from typing import Callable

from math import ceil

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


    def __call__(self, 
                 state:State, 
                 insertion:bool=False,
                 alignment:bool=False,
                 **kwargs:dict[str, Tensor]) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations and alignment valurs are passed in kwargs
        Deviations are added to corresponding parameters
        
        Parameters
        ----------
        state: State
            initial input state
        insertion: bool, default=False
            flag to treat eleemnt as error insertion
        alignment: bool, default=False
            flag to apply alignment error
        **kwargs: dict[str, Tensor]
            deviation and alignment values

        Returns
        -------
        State
        
        """   
        knob: dict[str, Tensor] = {key: kwargs[key] for key in self.keys if key in kwargs}

        step: Callable[[State], State] | Callable[[State, Tensor, ...], State]
        step = self._knob if knob else self._step

        if not alignment:
            if insertion:
                state = self._lmat @ state
            for _ in range(self.ns):
                state = step(state, **knob)
            if insertion:
                state = self._rmat @ state
            return state

        dp:Tensor = self.dp
        dl:Tensor = self.length

        if knob:
            dp = dp + knob['dp']
            dl = dl + knob['dl']
            
        dx:Tensor
        dy:Tensor
        dz:Tensor         
        dx, dy, dz = [kwargs[key] for key in ['dx', 'dy', 'dz']]

        wx:Tensor
        wy:Tensor
        wz:Tensor  
        wx, wy, wz = [kwargs[key] for key in ['wx', 'wy', 'wz']]

        if insertion:
            state = self._lmat @ state
        
        state = tx(state, +dx)
        state = ty(state, +dy)
        state = tz(state, +dz, dp)

        state = rx(state, +wx, dp)
        state = ry(state, +wy, dp)
        state = rz(state, +wz)
        
        for _ in range(self.ns):
            state = step(state, **knob)

        state = tz(state, -dl, dp)
        
        state = rz(state, -wz)
        state = ry(state, -wy, dp)
        state = rx(state, -wx, dp)
        
        state = tz(state, -dz, dp)
        state = ty(state, -dy)
        state = tx(state, -dx)
        
        state = tz(state, +dl, dp)

        if insertion:
            state = self._rmat @ state
        
        return state
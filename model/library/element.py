"""
Element
-------

Abstract element

"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import Callable

from math import ceil

import torch
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor

class Element(ABC):
    """
    Abstract element
    -------------
    
    """
    _tolerance:float = 1.0E-16
    _alignment:list[str] = ['dx', 'dy', 'dz', 'wx', 'wy', 'wz']    
    
    @property
    @abstractmethod
    def flag(self) -> bool:
        pass

    
    @property
    @abstractmethod
    def keys(self) -> list[str]:
        pass

    
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
        Element instance initialization

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
        self._order: bool = order
        self._exact: bool = exact
        
        self.dtype: DataType = dtype
        self.device: DataDevice = device


    def table(self, *, 
              name:bool=False,
              alignment:bool=True) -> dict[str, dict[str,Tensor]] | dict[str,Tensor]:
        """
        Generate default deviation table

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, dict[str,Tensor]] | dict[str,Tensor]
        
        """
        zeros: Tensor = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        table: dict[str, Tensor] = {key: value for key, value in zip(self.keys, zeros)}
        if alignment:
            keys:list[str] = self._alignment
            zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
            table = {**table, **{key: value for key, value in zip(keys, zeros)}}
        return table if not name else {self.name: table}


    @abstractmethod
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
        pass


    @abstractmethod
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
        pass


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
    def name(self, name:str) -> None:
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
    def length(self, length:float) -> None:
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
    def dp(self, dp:float) -> None:
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
    def ns(self, ns:int) -> None:
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


    @property
    def order(self) -> int:
        """
        Get integration order

        Parameters
        ----------
        None

        Returns
        -------
        int
        
        """       
        return self._order
    
    
    @order.setter
    def order(self, order:int) -> None:
        """
        Set integration order

        Parameters
        ----------
        order: int, non-negative
            integration order

        Returns
        -------
        None
        
        """          
        self._order = order
        self._step, self._knob = self.make_step()


    @property
    def exact(self) -> bool:
        """
        Get exact flag

        Parameters
        ----------
        None

        Returns
        -------
        bool
        
        """        
        return self._exact
        
    
    @exact.setter
    def exact(self, exact:bool) -> None:
        """
        Set exact flag

        Parameters
        ----------
        exact: bool
            exact

        Returns
        -------
        None
        
        """        
        self._exact = exact
        self._step, self._knob = self.make_step()


    def __call__(self, 
                 state:State, *,
                 data:Optional[dict[str, Tensor]]=None,
                 insertion:bool=False,
                 alignment:bool=False) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations and alignment valurs are passed in kwargs
        Deviations are added to corresponding parameters
        
        Parameters
        ----------
        state: State
            initial input state
        data: Optional[dict[str, Tensor]]
            deviation and alignment table            
        insertion: bool, default=False
            flag to treat element as error insertion
        alignment: bool, default=False
            flag to apply alignment error

        Returns
        -------
        State
        
        """   
        data: dict[str, Tensor] = data if data else {}
        knob: dict[str, Tensor] = {key: data[key] for key in self.keys if key in data}
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
        if insertion:
            state = self._lmat @ state
        state = transform(self, state, data)
        if insertion:
            state = self._rmat @ state
        return state

    
    @abstractmethod
    def __repr__(self) -> str:
        pass


def transform(element:Element, 
              state:State, 
              data:dict[str, Tensor]) -> State:
    """
    Apply alignment errors

    element: Element
        element to apply error to
    state: State
        initial input state
    data: dict[str, Tensor]
        deviation and alignment table

    """
    dp:Tensor = element.dp
    if 'dp' in data: 
        dp = dp + data['dp']

    length:Tensor = element.length
    if 'dl' in data: 
        length = length + data['dl']

    if element.flag:
        angle:Tensor = element.angle 
        if 'dw' in data:
            angle = angle + data['dw']

    dx:Tensor
    dy:Tensor
    dz:Tensor         
    dx, dy, dz = [data[key] for key in ['dx', 'dy', 'dz']]

    wx:Tensor
    wy:Tensor
    wz:Tensor  
    wx, wy, wz = [data[key] for key in ['wx', 'wy', 'wz']]

    state = tx(state, +dx)
    state = ty(state, +dy)
    state = tz(state, +dz, dp)

    state = rx(state, +wx, dp)
    state = ry(state, +wy, dp)
    state = rz(state, +wz)
    
    state = element(state, data=data, alignment=False, insertion=False)

    if element.flag:
        state = ry(state, +angle/2, dp)
        state = tz(state, -2.0*length/angle*(angle/2.0).sin(), dp)
        state = ry(state, +angle/2, dp)            
    else:
        state = tz(state, -length, dp)
    
    state = rz(state, -wz)
    state = ry(state, -wy, dp)
    state = rx(state, -wx, dp)
    
    state = tz(state, -dz, dp)
    state = ty(state, -dy)
    state = tx(state, -dx)

    if element.flag:
        state = ry(state, -angle/2, dp)
        state = tz(state, +2.0*length/angle*(angle/2.0).sin(), dp)
        state = ry(state, -angle/2, dp)
    else:
        state = tz(state, +length, dp)

    return state
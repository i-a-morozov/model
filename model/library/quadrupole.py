"""
Quadrupole
----------

Quadrupole element

"""
from __future__ import annotations

from typing import Optional
from typing import Callable

import torch
from torch import Tensor

from ndmap.yoshida import yoshida

from model.library.element import Element
from model.library.transformations import quadrupole
from model.library.transformations import kinematic

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

class Quadrupole(Element):
    """
    Quadrupole element
    ------------------
    
    Returns
    -------
    Quadrupole
    
    """
    flag: bool = False
    keys: list[str] = ['kn', 'ks', 'dp', 'dl']

    def __init__(self, 
                 name:str, 
                 length:float=0.0,
                 kn:float=0.0,
                 ks:float=0.0,
                 dp:float=0.0, *,
                 ns:int=1,
                 ds:Optional[float]=None,
                 order:int=0,
                 exact:bool=False,
                 insertion:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Quadrupole instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            length
        kn: float
            normal quadrupole strength (epsilon is added)
        ks: float
            skew quadrupole strength 
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
        insertion: bool, default=False
            flat to treat element as thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step            
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
                         insertion=insertion,
                         output=output,
                         matrix=matrix)

        self._kn: float = kn + self._tolerance
        self._ks: float = ks        

        self._lmatrix: Tensor
        self._rmatrix: Tensor
        self._lmatrix, self._rmatrix = self.make_matrix() 
        
        self._data: list[list[int], list[float]] = None
        self._step: Callable[[State], State]
        self._knob: Callable[[State, Tensor, ...], State]
        self._step, self._knob = self.make_step()


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
        
        matrix: Tensor = torch.func.jacrev(quadrupole)(state, self.kn, self.ks, self.dp, -0.5*self.length)
        
        lmatrix: Tensor = matrix
        rmatrix: Tensor = matrix
        
        return lmatrix, rmatrix
        

    def make_step(self) -> tuple[Mapping, ParametricMapping]:
        """
        Generate integration step

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Mapping, ParametricMapping]
        
        """        
        ns: int = self.ns
        exact:bool = self.exact
        order:int = self.order
        _ds: Tensor = self.length/self.ns
        _kn: Tensor = self.kn
        _ks: Tensor = self.ks
        _dp: Tensor = self.dp
        insertion:bool = self.insertion
        if insertion:
            lmatrix: Tensor = self._lmatrix
            rmatrix: Tensor = self._rmatrix
        output:bool = self.output
        if output:
            container_output = []
        matrix:bool = self.matrix
        if matrix:
            container_matrix = []        
        
        def quad_wrapper(state:State, ds:Tensor, kn:Tensor, ks:Tensor, dp:Tensor) -> State:
            return quadrupole(state, kn, ks, dp, ds)
            
        def sqrt_wrapper(state:State, ds:Tensor, kn:Tensor, ks:Tensor, dp:Tensor) -> State:
            return kinematic(state, dp, ds) if exact else state
            
        integrator: Callable[[State], State, Tensor, ...]
        integrator = yoshida(0, order, True, [quad_wrapper, sqrt_wrapper])
        
        self._data: list[list[int], list[float]] = integrator.table
        
        def step(state:State) -> State:
            state = lmatrix @ state if insertion else state
            for _ in range(ns):
                state = integrator(state, _ds, _kn, _ks, _dp)
                if output:
                    container_output.append(state)
                if matrix:
                    container_matrix.append(torch.func.jacrev(integrator)(state, _ds, _kn, _ks, _dp))                
            if output:
                self.container_output = torch.stack(container_output)
            if matrix:
                self.container_matrix = torch.stack(container_matrix)                    
            state = rmatrix @ state if insertion else state
            return state
            
        def knob(state:State, kn:Tensor, ks:Tensor, dp:Tensor, dl:Tensor) -> State:
            state = lmatrix @ state if insertion else state
            for _ in range(ns):
                state = integrator(state, (_ds + dl/ns), _kn + kn, _ks + ks, _dp + dp)
                if output:
                    container_output.append(state)
                if matrix:
                    container_matrix.append(torch.func.jacrev(integrator)(state, (_ds + dl/ns), _kn + kn, _ks + ks, _dp + dp))                
            if output:
                self.container_output = torch.stack(container_output)
            if matrix:
                self.container_matrix = torch.stack(container_matrix)                    
                
            state = rmatrix @ state if insertion else state
            return state
            
        return step, knob


    @property
    def kn(self) -> Tensor:
        """
        Get kn

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
        
        """       
        return torch.tensor(self._kn, dtype=self.dtype, device=self.device)
    
    
    @kn.setter
    def kn(self, 
           kn:float) -> None:
        """
        Set momentum deviation

        Parameters
        ----------
        kn: float
            kn

        Returns
        -------
        None
        
        """       
        self._kn = kn
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()
        
    
    @property
    def ks(self) -> Tensor:
        """
        Get ks

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
        
        """       
        return torch.tensor(self._ks, dtype=self.dtype, device=self.device)
    
    
    @ks.setter
    def ks(self, 
           ks:float) -> None:
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
        self._ks = ks
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()

    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, kn={self._kn}, ks={self._ks}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
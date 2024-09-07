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

from ndmap.yoshida import yoshida

from model.library.element import Element
from model.library.transformations import drift
from model.library.transformations import kinematic

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

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
                 insertion:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
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
        insertion: bool, default=False
            flat to treat element as thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step if output is true

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

        matrix: Tensor = torch.func.jacrev(drift)(state, self.dp, -0.5*self.length)

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
        _dp: Tensor = self.dp
        insertion:bool = self.insertion
        if insertion:
            lmatrix: Tensor = self._lmatrix
            rmatrix: Tensor = self._rmatrix
        output:bool = self.output
        matrix:bool = self.matrix

        def drif_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
            return drift(state, dp, ds)

        if exact:
            def sqrt_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
                return kinematic(state, dp, ds)
        else:
            def sqrt_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
                return state

        if insertion:
            def lmatrix_wrapper(state:State) -> State:
                return lmatrix @ state
            def rmatrix_wrapper(state:State) -> State:
                return rmatrix @ state
        else:
            def lmatrix_wrapper(state:State) -> State:
                return state
            def rmatrix_wrapper(state:State) -> State:
                return state

        integrator: Callable[[State], State, Tensor, ...]
        integrator = yoshida(0, order, True, [drif_wrapper, sqrt_wrapper])

        self._data: list[list[int], list[float]] = integrator.table

        if output and matrix:
            def step(state:State) -> State:
                container_output = []
                container_matrix = []
                state = lmatrix_wrapper(state)
                for _ in range(ns):
                    state = integrator(state, _ds, _dp)
                    container_output.append(state)
                    container_matrix.append(torch.func.jacrev(integrator)(state, _ds, _dp))
                self.container_output = torch.stack(container_output)
                self.container_matrix = torch.stack(container_matrix)
                state = rmatrix_wrapper(state)
                return state
            def knob(state:State, dp:Tensor, dl:Tensor) -> State:
                container_output = []
                container_matrix = []
                state = lmatrix_wrapper(state)
                for _ in range(ns):
                    state = integrator(state, (_ds + dl/ns), _dp + dp)
                    container_output.append(state)
                    container_matrix.append(torch.func.jacrev(integrator)(state, (_ds + dl/ns), _dp + dp))
                self.container_output = torch.stack(container_output)
                self.container_matrix = torch.stack(container_matrix)
                state = rmatrix_wrapper(state)
                return state
            return step, knob

        if output:
            def step(state:State) -> State:
                container_output = []
                state = lmatrix_wrapper(state)
                for _ in range(ns):
                    state = integrator(state, _ds, _dp)
                    container_output.append(state)
                self.container_output = torch.stack(container_output)
                state = rmatrix_wrapper(state)
                return state
            def knob(state:State, dp:Tensor, dl:Tensor) -> State:
                container_output = []
                state = lmatrix_wrapper(state)
                for _ in range(ns):
                    state = integrator(state, (_ds + dl/ns), _dp + dp)
                    container_output.append(state)
                self.container_output = torch.stack(container_output)
                state = rmatrix_wrapper(state)
                return state
            return step, knob


        def step(state:State) -> State:
            state = lmatrix_wrapper(state)
            for _ in range(ns):
                state = integrator(state, _ds, _dp)
            state = rmatrix_wrapper(state)
            return state
        def knob(state:State, dp:Tensor, dl:Tensor) -> State:
            state = lmatrix_wrapper(state)
            for _ in range(ns):
                state = integrator(state, (_ds + dl/ns), _dp + dp)
            state = rmatrix_wrapper(state)
            return state
        return step, knob


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
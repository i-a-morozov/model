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

from model.library.keys import KEY_DP
from model.library.keys import KEY_DL

from model.library.element import Element

from model.library.transformations import drift
from model.library.transformations import kinematic

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]

class Drift(Element):
    """
    Drift element
    -------------

    Returns
    -------
    Drift

    """
    flag: bool = False
    keys: list[str] = [KEY_DP, KEY_DL]

    def __init__(self,
                 name:str,
                 length:float=0.0,
                 dp:float=0.0, *,
                 alignment:bool=True,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
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
        alignment: bool, default=True
            flag to use alignment errors
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
        ns: int, positive, default=1
            number of integrtion steps
        ds: Optional[float], positive
            integration step length
            if given, input ns value is ignored and ds is used to compute ns = ceil(length/ds) or 1
            actual integration step is not ds, but length/ns
        order: int, default=0, non-negative
            Yoshida integration order
        exact: bool, default=False
            flag to use exact Hamiltonian
        insertion: bool, default=False
            flat to treat element as thin insertion
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step

        Returns
        -------
        None

        """
        super().__init__(name=name,
                         length=length,
                         dp=dp,
                         alignment=alignment,
                         dx=dx,
                         dy=dy,
                         dz=dz,
                         wx=wx,
                         wy=wy,
                         wz=wz,
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
        self._step: Mapping = self.make_step()


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


    def make_step(self) -> Mapping:
        """
        Generate integration step

        Parameters
        ----------
        None

        Returns
        -------
        Mapping

        """
        _ns: int = self.ns
        _ny:int = self.order
        _ds: Tensor = self.length/self.ns
        _dp: Tensor = self.dp

        exact:bool = self.exact
        insertion:bool = self.insertion
        if insertion:
            lmatrix: Tensor = self._lmatrix
            rmatrix: Tensor = self._rmatrix
        output:bool = self.output
        matrix:bool = self.matrix

        integrator: Callable[[State, Tensor, ...], State]

        if exact:
            def drif_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
                return drift(state, dp, ds)
            def sqrt_wrapper(state:State, ds:Tensor, dp:Tensor) -> State:
                return kinematic(state, dp, ds)
            integrator = yoshida(0, _ny, True, [drif_wrapper, sqrt_wrapper])
            self._data: list[list[int], list[float]] = integrator.table

        if not exact:
            def integrator(state:State, ds:Tensor, dp:Tensor) -> State:
                return drift(state, dp, ds)

        if insertion:
            def lmatrix_wrapper(state:State) -> State:
                return lmatrix @ state
            def rmatrix_wrapper(state:State) -> State:
                return rmatrix @ state
            def step(state:State, dp:Tensor, dl:Tensor) -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                state = lmatrix_wrapper(state)
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                        container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                state = rmatrix_wrapper(state)
                return state

        if not insertion:
            def step(state:State, dp:Tensor, dl:Tensor) -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                        container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                return state

        return step

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
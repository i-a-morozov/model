"""
Multipole
---------

Multipole element

"""
from __future__ import annotations

from typing import Optional
from typing import Callable

import torch
from torch import Tensor

from ndmap.yoshida import yoshida

from model.library.keys import KEY_KN, KEY_KS, KEY_MS, KEY_MO, KEY_DP, KEY_DL
from model.library.element import Element
from model.library.transformations import quadrupole
from model.library.transformations import sextupole
from model.library.transformations import octupole
from model.library.transformations import kinematic

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]

class Multipole(Element):
    """
    Multipole element
    -----------------

    Returns
    -------
    Multipole

    """
    flag: bool = False
    keys: list[str] = [KEY_KN, KEY_KS, KEY_MS, KEY_MO, KEY_DP, KEY_DL]

    def __init__(self,
                 name:str,
                 length:float=0.0,
                 kn:float=0.0,
                 ks:float=0.0,
                 ms:float=0.0,
                 mo:float=0.0,
                 dp:float=0.0, *,
                 ns:int=1,
                 ds:Optional[float]=None,
                 order:int=0,
                 exact:bool=False,
                 insertion:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Multipole instance initialization

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
        ms: float
            sextupole strength
        mo: float
            octupole strength
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

        self._kn: float = kn + self._tolerance
        self._ks: float = ks
        self._ms: float = ms
        self._mo: float = mo

        self._lmatrix: Tensor
        self._rmatrix: Tensor
        self._lmatrix, self._rmatrix = self.make_matrix()

        self._data: list[list[int], list[float]] = None
        self._step: Mapping
        self._knob: ParametricMapping
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
        _ns: int = self.ns
        _order:int = self.order
        _ds: Tensor = self.length/self.ns
        _kn: Tensor = self.kn
        _ks: Tensor = self.ks
        _ms: Tensor = self.ms
        _mo: Tensor = self.mo
        _dp: Tensor = self.dp

        exact:bool = self.exact
        insertion:bool = self.insertion
        if insertion:
            lmatrix: Tensor = self._lmatrix
            rmatrix: Tensor = self._rmatrix
        output:bool = self.output
        matrix:bool = self.matrix

        integrator: Callable[[State, Tensor, ...], State]

        def quad_wrapper(state:State, ds:Tensor, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor) -> State:
            return quadrupole(state, kn, ks, dp, ds)

        def mult_wrapper(state:State, ds:Tensor, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor) -> State:
            return octupole(sextupole(state, ms, ds), mo, ds)

        if exact:
            def sqrt_wrapper(state:State, ds:Tensor, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor) -> State:
                return kinematic(state, dp, ds)
            integrator = yoshida(0, _order, True, [quad_wrapper, mult_wrapper, sqrt_wrapper])

        if not exact:
            integrator = yoshida(0, _order, True, [quad_wrapper, mult_wrapper])

        self._data: list[list[int], list[float]] = integrator.table

        if insertion:
            def lmatrix_wrapper(state:State) -> State:
                return lmatrix @ state
            def rmatrix_wrapper(state:State) -> State:
                return rmatrix @ state
            def step(state:State) -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                state = lmatrix_wrapper(state)
                for _ in range(_ns):
                    state = integrator(state, _ds, _kn, _ks, _ms, _mo, _dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                         container_matrix.append(torch.func.jacrev(integrator)(state, _ds, _kn, _ks, _ms, _mo, _dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                state = rmatrix_wrapper(state)
                return state
            def knob(state:State, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor, dl:Tensor)  -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                state = lmatrix_wrapper(state)
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns, _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                        container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns, _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                state = rmatrix_wrapper(state)
                return state

        if not insertion:
            def step(state:State) -> State:
                if output:
                    container_output = []
                if matrix:
                     container_matrix = []
                for _ in range(_ns):
                    state = integrator(state, _ds, _kn, _ks, _ms, _mo, _dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                        container_matrix.append(torch.func.jacrev(integrator)(state, _ds, _kn, _ks, _ms, _mo, _dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                return state
            def knob(state:State, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor, dl:Tensor)  -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns, _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                        container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns, _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
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
        Set kn

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
        Set ks

        Parameters
        ----------
        dp: float
            ks

        Returns
        -------
        None

        """
        self._ks = ks
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step, self._knob = self.make_step()


    @property
    def ms(self) -> Tensor:
        """
        Get ms

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._ms, dtype=self.dtype, device=self.device)


    @ms.setter
    def ms(self,
           ms:float) -> None:
        """
        Set ms

        Parameters
        ----------
        ms: float
            ms

        Returns
        -------
        None

        """
        self._ms = ms
        self._step, self._knob = self.make_step()


    @property
    def mo(self) -> Tensor:
        """
        Get mo

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._mo, dtype=self.dtype, device=self.device)


    @mo.setter
    def mo(self,
           mo:float) -> None:
        """
        Set mo

        Parameters
        ----------
        mo: float
            mo

        Returns
        -------
        None

        """
        self._mo = mo
        self._step, self._knob = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, kn={self._kn}, ks={self._ks}, ms={self._ms}, mo={self._mo}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
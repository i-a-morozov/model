"""
Dipole
------

Dipole element

"""
from __future__ import annotations

from typing import Optional
from typing import Callable

import torch
from torch import Tensor

from ndmap.yoshida import yoshida

from model.library.keys import KEY_DW
from model.library.keys import KEY_E1
from model.library.keys import KEY_E2
from model.library.keys import KEY_KN
from model.library.keys import KEY_KS
from model.library.keys import KEY_MS
from model.library.keys import KEY_MO
from model.library.keys import KEY_DP
from model.library.keys import KEY_DL

from model.library.element import Element

from model.library.transformations import bend
from model.library.transformations import wedge
from model.library.transformations import cylindrical_error
from model.library.transformations import kinematic
from model.library.transformations import polar
from model.library.transformations import sector_bend_fringe
from model.library.transformations import sector_bend_wedge

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


class Dipole(Element):
    """
    Dipole element
    --------------

    Note, this element has curved layout
    Input length corresponds to arc length
    Cylindrical multipoles are truncated to octupole degree
    Wedges do not account for non-zero multipoles

    Returns
    -------
    Dipole

    """
    flag: bool = True
    keys: list[str] = [KEY_DW, KEY_E1, KEY_E2, KEY_KN, KEY_KS, KEY_MS, KEY_MO, KEY_DP, KEY_DL]


    def __init__(self,
                 name:str,
                 length:float=0.0,
                 angle:float=0.0,
                 e1:float=0.0,
                 e2:float=0.0,
                 kn:float=0.0,
                 ks:float=0.0,
                 ms:float=0.0,
                 mo:float=0.0,
                 dp:float=0.0, *,
                 dx:float=0.0,
                 dy:float=0.0,
                 dz:float=0.0,
                 wx:float=0.0,
                 wy:float=0.0,
                 wz:float=0.0,
                 e1_on:bool=True,
                 e2_on:bool=True,
                 linear:bool=False,
                 ns:int=1,
                 ds:Optional[float]=None,
                 order:int=0,
                 exact:bool=False,
                 insertion:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Dipole instance initialization

        Parameters
        ----------
        name: str
            name
        length: float, default=0.0
            arc length
        angle: float, default=0.0
            angle (epsilon is added)
        e1: float, default=0.0
            entrance wedge angle
        e2: float, default=0.0
            exit wedge angle
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
        e1_on: bool, default=True
            flag to include entrance wedge
        e2_on: bool, default=True
            flag to include exit wedge
        linear: bool, default=False
            flat to use linear approximation
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

        self._angle: float = angle + self._tolerance
        self._e1: float = e1
        self._e2: float = e2
        self._kn: float = kn + self._tolerance
        self._ks: float = ks
        self._ms: float = ms
        self._mo: float = mo

        self._e1_on : bool = e1_on
        self._e2_on : bool = e2_on

        self._linear: bool = linear

        self._lmatrix: Tensor
        self._rmatrix: Tensor
        self._lmatrix, self._rmatrix = self.make_matrix()

        self._data: list[list[int], list[float]] = None
        self._step: Mapping = self.make_step()


    @property
    def serialize(self) -> dict[str, str|int|float|bool]:
        """
        Serialize element

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, str|int|float|bool]

        """
        table:dict[str, str|int|float|bool] = super().serialize
        return {**table, 'angle': self.angle.item(), 'e1': self.e1.item(), 'e2': self.e2.item(), 'kn': self.kn.item(), 'ks': self.ks.item(), 'ms': self.ms.item(), 'mo': self.mo.item(), 'e1_on': self.e1_on, 'e2_on': self.e2_on, 'linear': self.linear}


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

        radius: Tensor = self.length.abs()/self.angle

        matrix: Tensor = torch.func.jacrev(bend)(state, radius, self.kn, self.ks, self.dp, -0.5*self.length)

        if self.is_inversed:
            lwedge: Tensor = torch.func.jacrev(wedge)(state, self.e1, radius)
            rwedge: Tensor = torch.func.jacrev(wedge)(state, self.e2, radius)
            rmatrix: Tensor = matrix @ lwedge
            lmatrix: Tensor = rwedge @ matrix

        if not self.is_inversed:
            lwedge: Tensor = torch.func.jacrev(wedge)(state, self.e1, radius).inverse()
            rwedge: Tensor = torch.func.jacrev(wedge)(state, self.e2, radius).inverse()
            lmatrix: Tensor = lwedge @ matrix
            rmatrix: Tensor = matrix @ rwedge

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
        _ny: int = self.order
        _ds: Tensor = self.length/self.ns
        _dw: Tensor = self.angle
        _e1: Tensor = self.e1
        _e2: Tensor = self.e2
        _kn: Tensor = self.kn
        _ks: Tensor = self.ks
        _ms: Tensor = self.ms
        _mo: Tensor = self.mo
        _dp: Tensor = self.dp
        _dl: Tensor = self.length.abs()
        _r : Tensor = _dl/_dw

        if self.is_inversed:
            _e1 = -self.e2
            _e2 = -self.e1

        exact:bool = self.exact
        insertion:bool = self.insertion
        if insertion:
            lmatrix: Tensor = self._lmatrix
            rmatrix: Tensor = self._rmatrix
        output:bool = self.output
        matrix:bool = self.matrix

        integrator: Callable[[State, Tensor, ...], State]

        def bend_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
            return bend(state, r, kn, ks, dp, ds)

        if self.linear:
            def mult_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
                return state
        else:
            def mult_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
                return cylindrical_error(state, r, kn, ks, ms, mo, ds)

        if exact:
            if self.is_inversed:
                if self.e1_on:
                    def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = polar(state, epsilon, dp)
                        state = sector_bend_fringe(state, +r, dp)
                        state = sector_bend_wedge(state, -epsilon, r, dp)
                        return state
                else:
                    def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_fringe(state, +r, dp)
                        return state
                if self.e2_on:
                    def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_wedge(state, -epsilon, r, dp)
                        state = sector_bend_fringe(state, -r, dp)
                        state = polar(state, epsilon, dp)
                        return state
                else:
                    def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_fringe(state, -r, dp)
                        return state
            if not self.is_inversed:
                if self.e1_on:
                    def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = polar(state, epsilon, dp)
                        state = sector_bend_fringe(state, +r, dp)
                        state = sector_bend_wedge(state, -epsilon, r, dp)
                        return state
                else:
                    def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_fringe(state, +r, dp)
                        return state
                if self.e2_on:
                    def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_wedge(state, -epsilon, r, dp)
                        state = sector_bend_fringe(state, -r, dp)
                        state = polar(state, epsilon, dp)
                        return state
                else:
                    def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                        state = sector_bend_fringe(state, -r, dp)
                        return state
            def sqrt_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
                return kinematic(state, dp, ds)
            def drif_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
                return polar(state, ds/r, dp)
            def kick_wrapper(state:Tensor, ds:Tensor, r:Tensor, kn:Tensor, ks:Tensor, ms:Tensor, mo:Tensor, dp:Tensor) -> State:
                qx, px, qy, py = state
                return torch.stack([qx, px - (1 + dp)/r*ds, qy, py])
            integrator = yoshida(0,  _ny, True, [bend_wrapper, sqrt_wrapper, kick_wrapper, drif_wrapper, mult_wrapper])

        if not exact:
            if self.e1_on:
                def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                    return wedge(state, epsilon, r)
            else:
                def lwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                    return state
            if self.e2_on:
                def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                    return wedge(state, epsilon, r)
            else:
                def rwedge_wrapper(state:State, epsilon:Tensor, r:Tensor, dp:Tensor) -> State:
                    return state
            integrator = yoshida(0, _ny, True, [bend_wrapper, mult_wrapper])

        self._data: list[list[int], list[float]] = integrator.table

        if insertion:
            def lmatrix_wrapper(state:State) -> State:
                return lmatrix @ state
            def rmatrix_wrapper(state:State) -> State:
                return rmatrix @ state
            def step(state:State, dw:Tensor, e1:Tensor, e2:Tensor, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor, dl:Tensor)  -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                state = lmatrix_wrapper(state)
                state = lwedge_wrapper(state, _e1 + e1,(_dl + dl)/(_dw + dw), _dp + dp)
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns,(_dl + dl)/(_dw + dw), _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                         container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns,(_dl + dl)/(_dw + dw), _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                state = rwedge_wrapper(state, _e2 + e2,(_dl + dl)/(_dw + dw), _dp + dp)
                state = rmatrix_wrapper(state)
                return state

        if not insertion:
            def step(state:State, dw:Tensor, e1:Tensor, e2:Tensor, kn:Tensor, ks:Tensor, ms: Tensor, mo:Tensor, dp:Tensor, dl:Tensor)  -> State:
                if output:
                    container_output = []
                if matrix:
                    container_matrix = []
                state = lwedge_wrapper(state, _e1 + e1,(_dl + dl)/(_dw + dw), _dp + dp)
                for _ in range(_ns):
                    state = integrator(state, _ds + dl/_ns,(_dl + dl)/(_dw + dw), _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp)
                    if output:
                        container_output.append(state)
                    if matrix:
                         container_matrix.append(torch.func.jacrev(integrator)(state, _ds + dl/_ns,(_dl + dl)/(_dw + dw), _kn + kn, _ks + ks, _ms + ms, _mo + mo, _dp + dp))
                if output:
                    self.container_output = torch.stack(container_output)
                if matrix:
                    self.container_matrix = torch.stack(container_matrix)
                state = rwedge_wrapper(state, _e2 + e2,(_dl + dl)/(_dw + dw), _dp + dp)
                return state

        return step


    @property
    def angle(self) -> Tensor:
        """
        Get angle

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._angle, dtype=self.dtype, device=self.device)


    @angle.setter
    def angle(self,
              angle:float) -> None:
        """
        Set angle

        Parameters
        ----------
        angle: float
            angle

        Returns
        -------
        None

        """
        self._angle = angle
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step = self.make_step()


    @property
    def e1(self) -> Tensor:
        """
        Get e1

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._e1, dtype=self.dtype, device=self.device)


    @e1.setter
    def e1(self,
           e1:float) -> None:
        """
        Set e1

        Parameters
        ----------
        e1: float
            e1

        Returns
        -------
        None

        """
        self._e1 = e1
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step = self.make_step()


    @property
    def e2(self) -> Tensor:
        """
        Get e2

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return torch.tensor(self._e2, dtype=self.dtype, device=self.device)


    @e2.setter
    def e2(self,
           e2:float) -> None:
        """
        Set e2

        Parameters
        ----------
        e2: float
            e2

        Returns
        -------
        None

        """
        self._e2 = e2
        self._lmatrix, self._rmatrix = self.make_matrix()
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


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
        self._step = self.make_step()


    @property
    def e1_on(self) -> bool:
        """
        Get e1 flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._e1_on


    @e1_on.setter
    def e1_on(self,
              flag:bool) -> None:
        """
        Set e1 flag

        Parameters
        ----------
        flag: bool
            flag

        Returns
        -------
        None

        """
        self._e1_on = flag
        self._step = self.make_step()


    @property
    def e2_on(self) -> bool:
        """
        Get e1 flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._e2_on


    @e2_on.setter
    def e2_on(self,
              flag:bool) -> None:
        """
        Set e1 flag

        Parameters
        ----------
        flag: bool
            flag

        Returns
        -------
        None

        """
        self._e2_on = flag
        self._step = self.make_step()


    @property
    def linear(self) -> bool:
        """
        Get linear flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._linear


    @linear.setter
    def linear(self,
               flag:bool) -> None:
        """
        Set linear flag

        Parameters
        ----------
        flag: bool
            flag

        Returns
        -------
        None

        """
        self._linear = flag
        self._step = self.make_step()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self._name}", length={self._length}, angle={self._angle}, e1={self._e1}, e1_on={self._e1_on}, e2={self._e2}, e2_on={self._e2_on}, kn={self._kn}, ks={self._ks}, ms={self._ms}, mo={self._mo}, dp={self._dp}, exact={self.exact}, ns={self._ns}, order={self.order})'
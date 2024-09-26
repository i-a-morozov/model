"""
Line
----

Group ordered sequence of elements or (nested) lines

"""
from __future__ import annotations

from math import ceil

from typing import Callable
from typing import Optional
from typing import Any

import torch
from torch import Tensor

from model.library.element import Element

from model.library.keys import KEY_DP, KEY_DL, KEY_DW
from model.library.keys import KEY_DX, KEY_DY, KEY_DZ
from model.library.keys import KEY_WX, KEY_WY, KEY_WZ

from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor
type Mapping = Callable[[State], State]
type ParametricMapping = Callable[[State, Tensor, ...], State]


class Line(Element):
    """
    Line element
    ------------

    Returns
    -------
    Line

    """
    keys: list[str] = [KEY_DP, KEY_DL, KEY_DW]


    def __init__(self,
                 name:str,
                 sequence:list[Element|Line],
                 propagate:bool=False,
                 dp:float=0.0,
                 exact:bool=False,
                 output:bool=False,
                 matrix:bool=False) -> None:
        """
        Line instance initialization

        Parameters
        ----------
        name: str
            name
        sequence: list[Element|Line]
            line sequence (ordered sequence of elements and/or (nested) lines)
        propagate: bool, default=False
            flat to propagate flags to elements
        dp: float, default=0.0
            momentum deviation
        exact: bool, default=False
            flag to include kinematic term
        output: bool, default=False
            flag to save output at each step
        matrix: bool, default=False
            flag to save matrix at each step

        Returns
        -------
        None

        """
        super().__init__(name=name,
                         dp=dp,
                         exact=exact,
                         output=output,
                         matrix=matrix)

        self._sequence:list[Element|Line] = sequence

        self._exact: bool = exact
        self._output: bool = output
        self._matrix: bool = matrix

        self.propagate: bool = propagate
        if self.propagate:
            self.set('dp', dp)
            self.set('exact', exact)
            self.set('output', output)
            self.set('matrix', matrix)
            for element in self.sequence:
                if isinstance(element, Line):
                    element.output = output
                    element.matrix = matrix


    def make_step(self):
        raise NotImplementedError


    def make_matrix(self):
        raise NotImplementedError


    def table(self, *,
              name:bool=False,
              alignment:bool=True) -> dict[str,dict[str,Tensor]] | dict[str,dict[str,dict[str,Tensor]]]:
        """
        Generate default deviation table for all unique elements

        Parameters
        ----------
        None

        Returns
        -------
        dict[str,dict[str,Tensor]] | dict[str,dict[str,dict[str,Tensor]]]

        """
        table: dict[str, dict[str, Tensor]]
        table = {element.name: element.table(name=False, alignment=alignment) for element in self.sequence}
        zeros: Tensor = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        table = {**table, **{key: value for key, value in zip(self.keys, zeros)}}
        if alignment:
            keys:list[str] = self._alignment
            zeros: Tensor = torch.zeros(len(keys), dtype=self.dtype, device=self.device)
            table = {**table, **{key: value for key, value in zip(keys, zeros)}}
        return table if not name else {self.name: table}


    def scan(self,
             attribute:str):
        """
        Scan line and yeild (with duplicates) all elements with given attribute

        Parameters
        ----------
        attribute: str
            target attribute name

        Yeilds
        ------
        Element

        """
        for element in self.sequence:
            if isinstance(element, Line):
                yield from element.scan(attribute)
            elif hasattr(element, attribute):
                yield element


    @staticmethod
    def select(elements:list[Element], *,
               kinds:Optional[list[str]]=None,
               names:Optional[list[str]]=None) -> list[Element]:
        """
        Select (filter) elements with given kinds and/or names

        Parameters
        ----------
        kinds: Optional[list[str]]
            list of kinds to select
        names: Optional[list[str]]
            list of names to select

        Returns
        -------
        list[Element]

        """
        elements = [element for element in elements if not kinds or element.__class__.__name__ in kinds]
        elements = [element for element in elements if not names or element.name in names]
        return elements


    def get(self,
            attribute:str, *,
            kinds:Optional[list[str]]=None,
            names:Optional[list[str]]=None) -> list[tuple[str, Any]]:
        """
        Get given attribute from selected elements

        Parameters
        ----------
        attribute: str
            target attribute name
        kinds: Optional[list[str]]
            list of kinds to select
        names: Optional[list[str]]
            list of names to select

        Returns
        -------
        list[tuple[str, Any]]

        """
        elements:list[Element] = [*self.scan(attribute)]
        elements = self.select(elements, kinds=kinds, names=names)
        return [(element.name, getattr(element, attribute)) for element in elements]


    def set(self,
            attribute:str,
            value:Any, *,
            kinds:Optional[list[str]]=None,
            names:Optional[list[str]]=None) -> None:
        """
        Set value to a given attribute for selected elements

        Parameters
        ----------
        attribute: str
            target attribute name
        value: Any
            value to set
        kinds: Optional[list[str]]
            list of kinds to select
        names: Optional[list[str]]
            list of names to select

        Returns
        -------
        None

        """
        elements:list[Element] = [*self.scan(attribute)]
        elements = self.select(elements, kinds=kinds, names=names)
        for element in {element.name: element for element in elements}.values():
            setattr(element, attribute, value)


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
    def sequence(self) -> list[Element|Line]:
        """
        Get sequence

        Parameters
        ----------
        None

        Returns
        -------
        list[Element|Line]

        """
        return self._sequence


    @sequence.setter
    def sequence(self,
                 sequence:list[Element|Line]) -> None:
        """
        Set sequence

        Parameters
        ----------
        sequence: sequence:list[Element|Line]
            sequence

        Returns
        -------
        None

        """
        self._sequence = sequence


    @property
    def unique(self) -> dict[str, tuple[str, Tensor, Tensor]]:
        default:Tensor = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return {
            element.name : (
                element.__class__.__name__,
                getattr(element, 'length', default),
                getattr(element, 'angle', default)
            ) for element in self.scan('name')
        }


    def itemize(self, select:str) -> list[str]:
        """
        Get list of all elements with matching kind

        Parameters
        ----------
        select: str
            kind

        Returns
        -------
        list[str]

        """
        return [key for key, (kind, *_) in self.unique.items() if kind == select]


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
        self.set('dp', dp)


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
    def exact(self,
              exact:bool) -> None:
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
        self.set('exact', exact)


    @property
    def length(self) -> Tensor:
        """
        Get sequence length

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return  sum(value for name, value in self.get('length'))


    @property
    def angle(self) -> Tensor:
        """
        Get sequence angle

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        angle: Tensor = torch.zeros_like(self.length)
        return sum(value for _, value in self.get('angle')) or angle


    @property
    def flag(self) -> bool:
        """
        Get layout flag

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        return bool(self.angle)


    @property
    def ns(self) -> dict[str, int]:
        """
        Get number of integration steps for all unique elements

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, int]

        """
        return {name: value for name, value in self.get('ns')}


    @ns.setter
    def ns(self,
           value:int|float|tuple[tuple[str,int|float],...]) -> None:
        """
        Set number of integration steps

        value: int
            set given number of steps to all unique elements
        value: float
            set ceil(length/value) number of steps to all unique elements
        value: tuple[str,int],...]
            set given number of steps to given names and/or elements types
        value: tuple[str,float],...]
            set ceil(length/value) number of steps to given names and/or elements types

        Returs
        ------
        Note

        """
        elements:dict[str,Element] = {element.name: element for element in set(self.scan('name'))}

        if isinstance(value, int):
            for element in elements.values():
                setattr(element, 'ns', value)
            return

        if isinstance(value, float):
            for element in elements.values():
                setattr(element, 'ns', ceil(element.length/value))
            return

        for key, parameter in value:
            if key in elements:
                element = elements[key]
                setattr(element, 'ns', parameter if isinstance(parameter, int) else ceil(element.length/parameter))
                continue
            for element in self.select(elements.values(), kinds=[key]):
                setattr(element, 'ns', parameter if isinstance(parameter, int) else ceil(element.length/parameter))


    @property
    def order(self) -> dict[str, int]:
        """
        Get integration order for all unique elements

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, int]

        """
        return {name: value for name, value in self.get('order')}


    @order.setter
    def order(self,
              value:int|tuple[tuple[str,int],...]) -> None:
        """
        Set number of integration steps

        value: int
            set given order to all unique elements
        value: tuple[str,int],...]
            set given order to given names and/or elements types

        Returs
        ------
        Note

        """
        elements:dict[str,Element] = {element.name: element for element in set(self.scan('name'))}

        if isinstance(value, int):
            for element in elements.values():
                setattr(element, 'order', value)
            return

        for key, parameter in value:
            if key in elements:
                element = elements[key]
                setattr(element, 'order', parameter)
                continue
            for element in self.select(elements.values(), kinds=[key]):
                setattr(element, 'order', parameter)


    @property
    def output(self) -> bool:
        """
        Get output flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._output


    @output.setter
    def output(self,
               output:bool) -> None:
        """
        Set output flag

        Parameters
        ----------
        exact: bool
            exact

        Returns
        -------
        None

        """
        self._output = output
        if self.propagate:
            self.set('output', output)
            for element in self.sequence:
                if isinstance(element, Line):
                    element.output = output

    @property
    def matrix(self) -> bool:
        """
        Get matrix flag

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return self._matrix


    @matrix.setter
    def matrix(self,
               matrix:bool) -> None:
        """
        Set matrix flag

        Parameters
        ----------
        exact: bool
            exact

        Returns
        -------
        None

        """
        self._matrix = matrix
        if self.propagate:
            self.set('matrix', matrix)
            for element in self.sequence:
                if isinstance(element, Line):
                    element.matrix = matrix


    def __call__(self,
                 state:State, *,
                 data:Optional[dict[str, Tensor | dict[str, Tensor]]]=None,
                 alignment:bool=False) -> State:
        """
        Transform initial input state using attibutes and deviations
        Deviations and alignment values are passed in data
        Deviations are added to corresponding parameters

        Parameters
        ----------
        state: State
            initial input state
        data: Optional[dict[str, Tensor]]
            deviation and alignment table
        alignment: bool, default=False
            flag to apply alignment error

        Returns
        -------
        State

        """
        data: Optional[dict[str, Tensor | dict[str, Tensor]]] = data if data else {}

        if self.output:
            container_output: list[Tensor] = []
        if self.matrix:
            container_matrix: list[Tensor] = []

        if not alignment:
            for element in self.sequence:
                state = element(state, alignment=alignment, data=data.get(element.name))
                if self.output:
                    if self.propagate:
                        container_output.append(element.container_output)
                    else:
                        container_output.append(state)
                if self.matrix:
                    if self.propagate:
                        container_matrix.append(element.container_matrix)
                    else:
                        matrix = torch.func.jacrev(lambda state: element(state, alignment=alignment, data=data.get(element.name)))(state)
                        container_matrix.append(matrix)
            if self.output:
                self.container_output = torch.vstack(container_output)
            if self.matrix:
                self.container_matrix = torch.vstack(container_matrix)
            return state

        dp:Tensor = self.dp + data.get(KEY_DP, 0.0)
        length:Tensor = self.length + data.get(KEY_DL, 0.0)
        if self.flag:
            angle:Tensor = self.angle + data.get(KEY_DW, 0.0)

        dx:Tensor
        dy:Tensor
        dz:Tensor
        dx, dy, dz = [data[key] for key in [KEY_DX, KEY_DY, KEY_DZ]]

        wx:Tensor
        wy:Tensor
        wz:Tensor
        wx, wy, wz = [data[key] for key in [KEY_WX, KEY_WY, KEY_WZ]]

        state = tx(state, +dx)
        state = ty(state, +dy)
        state = tz(state, +dz, dp)

        state = rx(state, +wx, dp)
        state = ry(state, +wy, dp)
        state = rz(state, +wz)

        for element in self.sequence:
            state = element(state, alignment=alignment, data=data.get(element.name))
            if self.output:
                if self.propagate:
                    container_output.append(element.container_output)
                else:
                    container_output.append(state)
            if self.matrix:
                if self.propagate:
                    container_matrix.append(element.container_matrix)
                else:
                    matrix = torch.func.jacrev(lambda state: element(state, alignment=alignment, data=data.get(element.name)))(state)
                    container_matrix.append(matrix)
        if self.output:
            self.container_output = torch.vstack(container_output)
        if self.matrix:
            self.container_matrix = torch.vstack(container_matrix)

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


    def __len__(self) -> int:
        """
        Get number of (first level) elements (sequence length)

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return len(self.sequence)


    def __getitem__(self, index: int) -> Element:
        """
        Get (first level) element by index

        Parameters
        ----------
        index: int
            element index|name

        Returns
        -------
        Element

        """
        if isinstance(index, int):
            return self.sequence[index]
        for element in self.sequence:
            if element.name == index:
                return element

    def __setitem__(self, index: int|str, element: Element) -> None:
        """
        Set (first level) element by index|name

        Parameters
        ----------
        index: int|str
            element index|name to replace
        element: Element
            element

        Returns
        -------
        None

        """
        if isinstance(index, int):
            self.sequence[index] = element
        for element in self.sequence:
            if element.name == index:
                self.sequence[index] = element


    def __repr__(self) -> str:
        return '\n'.join([str(element) for element in self.sequence])


    def layout(self) -> list[tuple[str,str,Tensor,Tensor]]:
        """
        Generate data for layout plotting

        Parameters
        ----------
        None

        Returns
        -------
        list[tuple[str,str,Tensor,Tensor]]
            (name,type,length,angle)

        """
        default:Tensor = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return [
            (
                element.name,
                element.__class__.__name__,
                getattr(element, 'length', default),
                getattr(element, 'angle', default)
            ) for element in self.scan('name')
        ]


def accumulate(data:dict[str, Tensor|dict[str, Tensor]],
               attribute:str) -> float|Tensor:
    """
    Accumulate selected attibute value

    Parameters
    ----------
    data: dict[str, Tensor|dict[str, Tensor]]
        input data
    attibute: str
        selected attribute

    """
    total = 0.0
    for key, value in data.items():
        if key == attribute:
            total += value
        elif isinstance(value, dict):
            total += accumulate(value, attribute)
    return total
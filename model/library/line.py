"""
Line
----

Group ordered sequence of elements or (nested) lines

Methods and properties

__init__    : line instance initialization
serialize   : (property) serialize line
inverse     : inverse line
data        : generate default deviation data for all unique elements
scan        : scan line and yeild (with duplicates) all elements with given attribute
select      : (static) select elements
get         : get given attribute from selected elements
set         : set value to a given attribute for selected elements
name        : (property) get/set name of the line
sequence    : (property) get/set sequence
flatten     : flatten line (all levels)
rename      : rename first level element
append      : append element
extend      : extend line
insert      : insert element
remove      : remove first occurrance of element with given name
replace     : replace first occurrance of element with given name
names       : (property) get list of first level element names
layout      : generate data for layout plotting
locations   : first level element/line entrance frame locations
position    : get element position in sequence
positions   : get all element position in sequence
start       : (property) set/get the first element
roll        : roll first level sequence
unique      : (property) get unique elements
duplicate   : (property) get duplicate elements
itemize     : get list of all elements with matching kind
describe    : (property) return number of elements (with unique names) for each kind
split       : split elements
clean       : clean first level sequence
mangle      : mangle elements
merge       : merge drift elements
splice      : splice line
group       : replace sequence part (first level) from probe to other with a line
dp          : (property) get/set momentum deviation
exact       : (property) get/set exact flag
length      : (property) get line length
angle       : (property) get line angle
flag        : (property) get layout flag
ns          : (property) get/set number of integration steps
order       : (property) get/set integration order
output      : (property) get/set output flag
matrix      : (property) get/set matrix flag
__call__    : transform state
__len__     : get number of elements (first level)
__getitem__ : get (first level) element by key
__setitem__ : set (first level) element by key
__delitem__ : del (first level) element by key
__repr__    : print line


"""
from __future__ import annotations

from math import ceil
from collections import Counter

from typing import Callable
from typing import Optional
from typing import Any

import torch
from torch import Tensor

from model.library.element import Element

type State = Tensor
type Mapping = Callable[[State, Tensor, ...], State]


class Line(Element):
    """
    Line element
    ------------

    Returns
    -------
    Line

    """
    keys: list[str] = []


    def __init__(self,
                 name:str,
                 sequence:list[Element|Line],
                 propagate:bool=False,
                 dp:float=0.0, *,
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


    @property
    def serialize(self) -> dict:
        """
        Serialize line

        Parameters
        ----------
        None

        Returns
        -------
        dict

        """
        sequence = []
        for element in self.sequence:
            sequence.append({'kind': element.__class__.__name__, **element.serialize})
        return {'kind': 'Line', 'name': self.name, 'sequence': sequence, 'propagate': self.propagate, 'dp': self.dp.item(), 'exact': self.exact, 'output': self.output, 'matrix': self.matrix}


    def inverse(self) -> Line:
        """
        Inverse line

        Parameters
        ----------
        None

        Returns
        -------
        Element

        """
        line = self.clone()
        line.is_inversed = not line.is_inversed
        sequence = []
        for element in reversed(line.sequence):
            sequence.append(element.inverse())
        line.sequence = sequence
        return line


    def make_step(self):
        raise NotImplementedError


    def make_matrix(self):
        raise NotImplementedError


    def data(self, *,
             name:bool=False,
             alignment:bool=True) -> dict[str,dict[str,Tensor]] | dict[str,dict[str,dict[str,Tensor]]]:
        """
        Generate default deviation data for all unique elements

        Parameters
        ----------
        None

        Returns
        -------
        dict[str,dict[str,Tensor]] | dict[str,dict[str,dict[str,Tensor]]]

        """
        data: dict[str, dict[str, Tensor]]
        data = {element.name: element.data(name=False, alignment=alignment) for element in self.sequence}
        zeros: Tensor = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        data = {**data, **{key: value for key, value in zip(self.keys, zeros)}}
        return data if not name else {self.name: data}


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


    def flatten(self) -> None:
        """
        Flatten line (all levels)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.sequence = [*self.scan('name')]


    def rename(self,
               old:str,
               new:str) -> None:
        """
        Rename first level element

        Parameters
        ----------
        old: str
            old name
        new: str
            new name

        Returns
        -------
        None

        """
        self[old].name = new


    def append(self,
               element:Element) -> None:
        """
        Append element

        Parameters
        ----------
        element: Element
            element

        Returns
        -------
        None

        """
        self.sequence.append(element)


    def extend(self,
               line:Line) -> None:
        """
        Extend line

        Parameters
        ----------
        line: Line
            line

        Returns
        -------
        None

        """
        self.sequence.extend(line.sequence)


    def insert(self,
               element:Element,
               name:str) -> None:
        """
        Insert element after the element with given name

        Parameters
        ----------
        element: Element
            element
        name: str
            name

        Returns
        -------
        None

        """
        self.sequence.insert(self.position(name) + 1, element)


    def remove(self,
               name:str) -> None:
        """
        Remove first occurrance of element with given name

        Parameters
        ----------
        name: str
            element name

        Returns
        -------
        None

        """
        self.sequence.remove(self[name])

    def replace(self, name:str, element:Element) -> None:
        """
        Replace first occurrance of element with given name

        Parameters
        ----------
        name: str
            element name
        element: Element
            element

        Returns
        -------
        None

        """
        self.sequence[self.position(name)] = element


    @property
    def names(self) -> list[str]:
        """
        Get list of first level element names

        Parameters
        ----------
        None

        Returns
        -------
        list[str]

        """
        return [element.name for element in self.sequence]


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


    @property
    def locations(self) -> Tensor:
        """
        First level element/line entrance frame locations

        Parameters
        ----------
        None

        Returns
        -------
        Tensor

        """
        lengths = [element.length for element in self.sequence]
        return (torch.cumsum(torch.stack(lengths), dim=-1) % self.length).roll(1).abs()


    def position(self,
                 name:str) -> int:
        """
        Get element position in sequence

        Parameters
        ----------
        name: str
            element name

        Returns
        -------
        int

        """
        for index, element in enumerate(self.sequence):
            if element.name == name:
                return index


    def positions(self,
                  name:str) -> list[int]:
        """
        Get all element position in sequence

        Parameters
        ----------
        name: str
            element name

        Returns
        -------
        list[int]

        """
        return [index for index, element in enumerate(self.sequence) if element.name == name]


    @property
    def start(self) -> str:
        """
        Get the first element

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        element, *_ = self.sequence
        return element.name


    @start.setter
    def start(self,
              start:int|str) -> None:
        """
        Set the first element

        Parameters
        ----------
        start: int|str
            element position|name

        Returns
        -------
        None

        """
        index = self.position(start) if isinstance(start, str) else start
        self.sequence = self.sequence[index:] + self.sequence[:index]


    def roll(self,
             shift:int) -> None:
        """
        Roll first level sequence

        Parameters
        ----------
        shift: int
            shift value

        Returns
        -------
        None

        """
        self.sequence = self.sequence[shift:] + self.sequence[:shift]


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


    @property
    def duplicates(self) -> bool:
        """
        Check for first level duplicates (elements with same name)

        Parameters
        ----------
        None

        Returns
        -------
        bool

        """
        return len(self.names) != len(set(self.names))


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
    def describe(self) -> dict[str, int]:
        """
        Return number of elements (with unique names) for each kind

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, int]

        """
        kinds, *_ = zip(*self.unique.values())
        return dict(Counter(kinds))


    def split(self,
              group:tuple[int, list[str]|None, list[str]|None, list[str]|None], *,
              paste:Optional[list[Element]]=None) -> None:
        """
        Split elements

        Note, BPM elements are splitted into two parts
        The first part has original direction, in the second direction is switched
        Other zero length elements are not intended to be splitted and are skipped

        Elements with nonzero length are splitted by length (and angle if present)

        Parameters
        ----------
        group: tuple[int, list[str]|None, list[str]|None, list[str]|None]
            split group
            count, kinds, names to include, names to exclude
            count is the number of elements, not splits
        paste: Optional[list[Element]]
            elements to paste between parts

        Returns
        -------
        None

        """
        sequence:list[Element] = []
        count, kinds, names, clean = group
        count = count or 1
        kinds = kinds or []
        names = names or []
        clean = clean or []
        paste = paste or []
        for index, element in enumerate(self.sequence):
            kind = element.__class__.__name__
            if (kind in kinds or element.name in names) and (element.name not in clean):
                if kind == 'BPM':
                    head = element.clone()
                    tail = element.clone()
                    tail.direction = {'forward': 'inverse', 'inverse': 'forward'}[tail.direction]
                    sequence.extend([head, *paste, tail])
                    continue
                if count == 1:
                    sequence.append(element)
                    continue
                if element.length == 0.0:
                    continue
                element = element.clone()
                element.length = element.length.item()/count
                if element.flag:
                    element.angle = element.angle.item()/count
                local = [element.clone() for _ in range(count)]
                if kind == 'Dipole':
                    head, *local, tail = local
                    tail.e1_on = False
                    head.e2_on = False
                    for part in local:
                        part.e1_on = False
                        part.e2_on = False
                    local = [head, *local, tail]
                if paste:
                    *local, _ = [item for pair in zip(local, count*[*paste]) for item in pair]
                sequence.extend(local)
                continue
            sequence.append(element)
        self.sequence = sequence


    def clean(self,
              group:tuple[float|None, list[str]|None, list[str]|None, list[str]|None]) -> None:
        """
        Clean first level sequence (remove elements by length/kind/name)

        Parameters
        ----------
        group: tuple[float|None, list[str]|None, list[str]|None, list[str]|None]
            clean group
            length, kinds, names to include, names to exclude

        Returns
        -------
        None

        """
        length, kinds, names, clean = group
        kinds = kinds or []
        names = names or []
        clean = clean or []
        sequence:list[Element] = []
        for index, element in enumerate(self.sequence):
            if (element.__class__.__name__ in kinds or element.name in names) and (element.name not in clean):
                continue
            if length and element.length and element.length.abs() <= length:
                continue
            sequence.append(element)
        self.sequence = sequence


    def mangle(self,
               kind:str, *,
               names:Optional[list[str]]=None,
               size:int=3) -> None:
        """
        Mangle names

        Parameters
        ----------
        kind: str
            element kind to mangle
        names: Optional[list[str]]
            element names to skip
        size: int
            number of zeros to prepend

        Returns
        -------
        None

        """
        names = names or []
        sequence:list[Element] = []
        total:dict[str, int] = dict(Counter(self.names))
        table:dict[str, int] = dict.fromkeys(self.names, 1)
        for element in self.sequence:
            current = element.clone()
            if element.name not in names and element.__class__.__name__ == kind and total[element.name] != 1:
                current.name = f'{element.name}{table[element.name]:0{size}}'
                table[element.name] += 1
            sequence.append(current)
        self.sequence = sequence


    def merge(self, *,
              name:str='DR',
              size:int=3) -> None:
        """
        Merge and rename drifts

        Parameters
        ----------
        name: str, default='DR'
            root name
        size: int, default=3
            number of zeros to prepend

        Returns
        -------
        float

        """
        sequence:list[Element] = []
        length:float = 0.0
        for index, element in enumerate(self.sequence):
            if element.__class__.__name__ == 'Drift':
                current = element.clone()
                length += current.length.item()
                current.name = name
                current.length = length
                continue
            if length:
                sequence.append(current)
            sequence.append(element)
            length = 0.0
        if length:
            sequence.append(current)
        self.sequence = sequence
        self.mangle('Drift', size=size)


    def splice(self) -> None:
        """
        Splice line

        Given a line with splitted BPMs, create lines between them
        Note, sequence is expected to start and end with BPMs
        [BPM_I, ..., BPM_F, BPM_I, ..., BPM_F] -> [[BPM_I, ..., BPM_F], ..., [BPM_I, ..., BPM_F]]

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        line:list[Element] = []
        sequence:list[Element] = []
        flag:bool = True
        for element in self.sequence:
            if element.__class__.__name__ == 'BPM':
                flag = not flag
            line.append(element)
            if flag:
                head, *_, tail = line
                sequence.append(Line(name=f'{head.name}_{tail.name}', sequence=line))
                line = []
        self.sequence = sequence


    def group(self,
              name:str,
              probe:str|int,
              other:str|int, *,
              include:bool=True) -> None:
        """
        Replace sequence part (first level) from probe to other with a line

        Parameters
        ----------
        name: str
            line name
        probe: str|int
            probe name
        other: str|int
            other name
        include: bool, default=True
            flag to include last element

        Returns
        -------
        None

        """
        if isinstance(probe, str):
            probe = self.position(probe)
        if isinstance(other, str):
            other = self.position(other)
        count:int = 0
        sequence:list[Element] = []
        for element in self.sequence:
            if count < probe or count >= other + include:
                sequence.append(element)
            elif probe == count:
                line = self.__class__(name=name,
                                      sequence=[],
                                      dp=self.dp.item(),
                                      exact=self.exact,
                                      output=self.output,
                                      matrix=self.matrix)
                sequence.append(line)
                local = [element]
            else:
                local.append(element)
            count += 1
        line.sequence = local
        self.sequence = sequence


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
        return sum(value for name, value in self.get('length'))


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
                setattr(element, 'ns', ceil(element.length/value) or 1)
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
            flag to apply alignment error (passed to elements)

        Returns
        -------
        State

        """
        data: Optional[dict[str, Tensor | dict[str, Tensor]]] = data if data else {}

        if self.output:
            container_output: list[Tensor] = []
        if self.matrix:
            container_matrix: list[Tensor] = []

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


    def __getitem__(self, key: int|str|slice|tuple) -> Element | list[Element]:
        """
        Get (first level) element by key

        Parameters
        ----------
        key: int|str|slice|tuple
            element index|name|slice|tuple

        Returns
        -------
        Element | list[Element]

        """
        if isinstance(key, int):
            return self.sequence[key]
        if isinstance(key, str):
            for element in self.sequence:
                if element.name == key:
                    return element
        if isinstance(key, slice):
            return [self[index] for index in range(*key.indices(len(self)))]
        if isinstance(key, tuple):
            result = self
            for index in key:
                result = result[index]
            return result


    def __setitem__(self, key: int|str|slice|tuple, element: Element) -> None:
        """
        Set (first level) element by index|name|slice|tuple

        Parameters
        ----------
        key: int|str|slice
            element index|name|slice|tuple to replace
        element: Element
            element

        Returns
        -------
        None

        """
        if isinstance(key, int):
            self.sequence[key] = element
            return
        if isinstance(key, str):
            self.sequence[self.position(key)] = element
            return
        if isinstance(key, slice):
            for index, item in zip(range(*key.indices(len(self.sequence))), element):
                self.sequence[index] = item
            return
        if isinstance(key, tuple):
            current = self
            for index in key[:-1]:
                current = current[index]
            current[key[-1]] = element
            return


    def __delitem__(self, key: int|str|slice|tuple) -> None:
        """
        Del (first level) element by index|name|slice|tuple

        Parameters
        ----------
        key: int|str|slice
            element index|name|slice|tuple to replace
        element: Element
            element

        Returns
        -------
        None

        """
        if isinstance(key, int):
            del self.sequence[key]
            return
        if isinstance(key, str):
            del self.sequence[self.position(key)]
            return
        if isinstance(key, slice):
            for index in range(*key.indices(len(self.sequence))):
                del self.sequence[index]
        if isinstance(key, tuple):
            current = self
            for index in key[:-1]:
                current = current[index]
            del current[key[-1]]


    def __repr__(self) -> str:
        return '\n'.join([str(element) for element in self.sequence])


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

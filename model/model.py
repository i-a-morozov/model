"""
Model module

Set linear model from CS Twiss parameters, normalization matrices or transport matrices
Generate model with errors
Compute closed orbit and twiss parameters

External

mod          (util)
load         (util)
save         (util)
insert       (util)

load_tfs     (external)
oad_sdds     (external)
convert      (external)
load_lattice (external)
add_rc       (external)

"""
from __future__ import annotations

from typing import Literal
from typing import Optional
from typing import Callable

from pathlib import Path

from multimethod import multimethod
from torch import Tensor
from torch import dtype as DataType
from torch import device as DataDevice
from torch import float64 as Float64

from pandas.core.frame import DataFrame
from pandas.testing import assert_frame_equal

import numpy
import torch

from twiss import cs_normal
from twiss import transport
from twiss import normal_to_wolski
from twiss import wolski_to_cs
from twiss import twiss
from twiss import propagate
from twiss import advance

from model.util import mod
from model.util import load
from model.util import save
from model.util import insert

from model.external import load_tfs
from model.external import load_sdds
from model.external import convert
from model.external import load_lattice
from model.external import add_rc


class Model:    
    """
    
    Returns
    ----------
    Model class instance
    
    """
    _epsilon:float = 1.0E-12
    
    _virtual:str = 'VIRTUAL'
    _monitor:str = 'MONITOR'
    
    _head:str = 'HEAD'
    _tail:str = 'TAIL'

    _model:list[str] = ['CS', 'NM', 'TM']

    _rc:list[str] = ['TYPE', 'S']

    _rc_mu:list[str] = ['FX', 'FY']

    _rc_cs:list[str] = [
        'AX', 'BX', 'AY', 'BY'
    ]
    
    _rc_nm:list[str] = [
        'N11', 'N12', 'N13', 'N14', 
        'N21', 'N22', 'N23', 'N24', 
        'N31', 'N32', 'N33', 'N34', 
        'N41', 'N42', 'N43', 'N44'
    ]

    _rc_tm:list[str] = [
        'T11', 'T12', 'T13', 'T14', 
        'T21', 'T22', 'T23', 'T24', 
        'T31', 'T32', 'T33', 'T34', 
        'T41', 'T42', 'T43', 'T44'
    ]

    _rc_dp:list[str] = [
        'DQX', 'DPX', 'DQY', 'DPY'
    ]
    
    _index:dict[str, list[str]] = {
        'CS': [*_rc, *_rc_mu, *_rc_cs, *_rc_dp, 'RC'],
        'NM': [*_rc, *_rc_mu, *_rc_nm, *_rc_dp, 'RC'],
        'TM': [*_rc, *_rc_tm, *_rc_dp, 'RC']
    }


    def __init__(self, 
                 model:Literal['CS', 'NM', 'TM'] = 'CS', *,
                 rc:bool=True,
                 table:Optional[Path | dict[str, dict[str, str | int | float | dict]]] = None,
                 dtype:DataType = Float64,
                 device:DataDevice = DataDevice('cpu')) -> None:
        """
        Model instance initialization
        
        Parameters
        ----------
        model: Literal['CS', 'NM', 'TM'], default='CS'
            model type
        rc: bool, default=True
            flag to set default RC
        table: Optional[Path | dict[str, dict[str, str | int | float | dict]]]
            configuration file path or dictionary
        dtype: DataType, default=Float64
            data type
        device: DataDevice, default=DataDevice('cpu')
            data device

        Returns
        -------
        None

        """
        self.dtype:DataType = dtype
        self.device:DataDevice = device
            
        self.model:str = model
        
        self.table:Path|dict[str,dict[str,str|int|float|dict]]|None = table
        self.empty:bool = False
        if not self.table:
            self.empty = True
            return

        self.dict:dict[str,dict[str,str|int|float|dict]]
        if isinstance(self.table, dict):
            self.dict = self.table   
        if isinstance(self.table, Path):
            self.dict = load(self.table)
            if rc:
                def apply(x):
                    x, *_ = x
                    return x
                self.dict = insert(self.dict, 'RC', dict.fromkeys(self.dict.keys()), replace=False, apply=apply)

        self.data_frame:DataFrame = DataFrame.from_dict(self.dict)

        self.size:int
        *_, self.size = self.data_frame.shape

        self.name:list[str] = [*self.data_frame.columns]
        
        self.type:list[str] = [*self.data_frame.loc['TYPE'].values]

        self.s:list[float] = [*self.data_frame.loc['S'].values]

        self.c:float
        *_, self.c = self.s

        self.mi:list[int] = [index for index, kind in enumerate(self.type) if kind == self._monitor]
        self.vi:list[int] = [index for index, kind in enumerate(self.type) if kind == self._virtual]

        self.mc:int = len(self.mi)
        self.vc:int = len(self.vi)

        self.mn:list[str] = [name for name, kind in zip(self.name, self.type) if kind == self._monitor]
        self.vn:list[str] = [name for name, kind in zip(self.name, self.type) if kind == self._virtual]

        set_attibutes:dict[str, Callable] = {
            'CS': self._set_cs_attributes,
            'NM': self._set_nm_attributes,
            'TM': self._set_tm_attributes
        }
        set_attibutes[self.model]()
        
        self.mux:Tensor
        self.muy:Tensor
        *_, (self.mux, self.muy) = self.mu

        self.nux:Tensor = self.mux/(2.0*torch.pi)
        self.nuy:Tensor = self.muy/(2.0*torch.pi)
        self.nu:Tensor = torch.stack([self.nux, self.nuy])

        self.probe:Tensor = torch.tensor(self.mi, dtype=torch.int64, device=self.device)
        self.other:Tensor = self.probe.roll(-1)
        for location, _ in enumerate(self.other):
            while self.other[location] < self.other[location - 1]: 
                self.other[location:].add_(self.size)
        phase:Tensor = torch.vmap(self.advance)(self.probe, self.other)
        self.dmu:Tensor =  mod(phase*(phase.abs() > self._epsilon), 2.0*torch.pi).abs()
        self.dmux:Tensor
        self.dmuy:Tensor
        self.dmux, self.dmuy = self.dmu.T

        for element in self._rc_dp:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )
        
        self.rc:list[dict] = [*self.data_frame.loc['RC'].values]

        self.update()

    
    def _set_cs_attributes(self) -> None:
        """
        Set CS model attributes

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for element in self._rc_mu:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )

        self.mu:Tensor = torch.stack([self.__dict__[element.lower()] for element in self._rc_mu]).T

        for element in self._rc_cs:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )
        
        self.cs:Tensor = torch.stack([self.__dict__[element.lower()] for element in self._rc_cs]).T
                
        self.nm:Tensor = torch.vmap(cs_normal)(self.cs)

        for element, value in zip(self._rc_nm, self.nm.reshape(self.size, -1).swapaxes(0, 1)):
            self.__dict__[element.lower()] = value

        start:Tensor
        start, *_ = self.nm
        self.tm:Tensor = torch.vmap(lambda probe, fx, fy: transport(start, probe, fx, fy))(self.nm, *self.mu.T)

        for element, value in zip(self._rc_tm, self.tm.reshape(self.size, -1).swapaxes(0, 1)):
            self.__dict__[element.lower()] = value


    def _set_nm_attributes(self) -> None:
        """
        Set NM model attributes

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for element in self._rc_mu:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )

        self.mu:Tensor = torch.stack([self.__dict__[element.lower()] for element in self._rc_mu]).T

        for element in self._rc_nm:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )

        self.nm:Tensor = torch.tensor(
            self.data_frame.loc[self._rc_nm].to_numpy(dtype=numpy.float64), 
            dtype=self.dtype, device=self.device
        ).T.reshape(self.size, 4, 4)

        self.cs:Tensor = torch.vmap(lambda normal: wolski_to_cs(normal_to_wolski(normal)))(self.nm)

        for element, value in zip(self._rc_cs, self.cs.T):
            self.__dict__[element.lower()] = value
            
        start:Tensor
        start, *_ = self.nm
        self.tm:Tensor = torch.vmap(lambda probe, fx, fy: transport(start, probe, fx, fy))(self.nm, *self.mu.T)

        for element, value in zip(self._rc_tm, self.tm.reshape(self.size, -1).swapaxes(0, 1)):
            self.__dict__[element.lower()] = value

    
    def _set_tm_attributes(self) -> None:
        """
        Set TM model attributes

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for element in self._rc_tm:
            self.__dict__[element.lower()] = torch.tensor(
                self.data_frame.loc[element].to_numpy(dtype=numpy.float64), 
                dtype=self.dtype, 
                device=self.device
            )

        self.tm:Tensor = torch.tensor(
            self.data_frame.loc[self._rc_tm].to_numpy(dtype=numpy.float64), 
            dtype=self.dtype, device=self.device
        ).T.reshape(self.size, 4, 4)

        matrix:Tensor
        *_, matrix = self.tm

        normal:Tensor
        wolski:Tensor
        _, normal, wolski = twiss(matrix, epsilon=self._epsilon)

        self.cs:Tensor = torch.vmap(lambda matrix: wolski_to_cs(propagate(wolski, matrix)))(self.tm)

        for element, value in zip(self._rc_cs, self.cs.T):
            self.__dict__[element.lower()] = value

        mu:Tensor
        nm:Tensor
        mu, nm = torch.vmap(lambda matrix: advance(normal, matrix))(self.tm)

        for element, value in zip(self._rc_mu, mu.T):
            self.__dict__[element.lower()] = value
            for location in range(1, self.size):
                while self.__dict__[element.lower()][location] < self.__dict__[element.lower()][location - 1]: 
                    self.__dict__[element.lower()][location:].add_(2.0*torch.pi)

        self.mu:Tensor = torch.stack([self.__dict__[element.lower()] for element in self._rc_mu]).T
        self.nm:torch.Tensor = nm

        for element, value in zip(self._rc_nm, self.nm.reshape(self.size, -1).swapaxes(0, 1)):
            self.__dict__[element.lower()] = value


    @multimethod
    def advance(self,
                probe:Tensor,
                other:Tensor) -> Tensor:
        """
        Compute phase advance from probe to other
            
        Both indices can have any integer (tensor) values
        Advance is negative, if the other location is before the probe location

        Parameters
        ----------
        probe: Tensor
            probe location
        other: Tensor
            other location

        Returns
        -------
        Tensor
            advance

        """
        group = torch.stack([probe, other])
        index = mod(group, self.size).to(torch.int64)
        shift = torch.div(group - index, self.size, rounding_mode='floor')
        phase = self.mu[index] + 2.0*numpy.pi*self.nu*shift.unsqueeze(-1)
        delta = phase.T.diff().squeeze()
        return delta

    
    @multimethod
    def advance(self, probe:int, other:int) -> Tensor:
        probe = torch.tensor(probe, dtype=torch.int64, device=self.device)
        other = torch.tensor(other, dtype=torch.int64, device=self.device)
        return self.advance(probe, other)

    
    @multimethod
    def advance(self, probe:str, other:str) -> Tensor:
        probe = self.name.index(probe)
        other = self.name.index(other)
        return self.advance(probe, other)

    
    @multimethod
    def matrix(self,
               probe:Tensor,
               other:Tensor) -> Tensor:
        """
        Compute transport matrix from probe to other
            
        Both indices can have any integer (tensor) values
        Inverse of the matrix is computed, if the other location is before the probe location        
        One-turn matrix at the probe location is returned if other == probe + size
        If other == probe, identity matrix is returned

        Parameters
        ----------
        probe: Tensor
            probe location
        other: Tensor
            other location

        Returns
        -------
        Tensor
            transport matrix

        """
        group = torch.stack([probe, other])
        index = mod(group, self.size).to(torch.int64)
        return transport(*self.nm[index], *self.advance(probe, other))
        
        
    @multimethod
    def matrix(self, probe:int, other:int) -> Tensor:
        probe = torch.tensor(probe, dtype=torch.int64, device=self.device)
        other = torch.tensor(other, dtype=torch.int64, device=self.device)
        return self.matrix(probe, other)
        
    
    @multimethod
    def matrix(self, probe:str, other:str) -> Tensor:
        probe = self.name.index(probe)
        other = self.name.index(other)
        return self.matrix(probe, other)

    
    def update(self) -> None: 
        """
        Update configuration

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        key:str
        for key in self._rc:
            self.data_frame.loc[key] = self.__dict__[key.lower()]
        for key in [*self._rc_cs, *self._rc_mu, *self._rc_nm, *self._rc_tm, *self._rc_dp]:
            self.data_frame.loc[key] = self.__dict__[key.lower()].cpu().numpy()
        self.dict = self.data_frame.to_dict()


    def export(self, path:Path) -> None:
        """
        Export configuration

        Parameters
        ----------
        path: Path
            output file path

        Returns
        -------
        None
        
        """
        self.update()
        save(self.data_frame.to_dict(), path)


    def compare(self, 
                model:Model, *,
                rc:bool=False,
                atol:Optional[float]=None,
                rtol:Optional[float]=None,
                **kwargs) -> bool:
        """
        Compare configuration

        Parameters
        ----------
        model: Model
            model to compare with
        rc: bool, default=False
            flag to compare RC
        atol: Optional[float]
            absolute tolerance (abs(a - b) <= atol)
        rtol: Optional[float]
            relative tolerance (abs(a - b) < rtol*max(abs(a), abs(b)))
        **kwargs: dict
             assert_frame_equal options

        Returns
        -------
        bool
        
        """
        if not atol:
            atol = self._epsilon

        if not rtol:
            rtol = self._epsilon

        ldf:DataFrame = (self.data_frame if rc else self.data_frame.drop('RC')).sort_index()
        rdf:DataFrame = (model.data_frame if rc else model.data_frame.drop('RC')).sort_index()
        
        try:
            assert_frame_equal(ldf, rdf, atol=atol, rtol=rtol, **kwargs)
        except AssertionError:
            return False
        
        return True


    @classmethod
    def from_table(cls,
                   kind:Literal['TFS', 'SDDS'],
                   path:Path, 
                   kind_monitor:list[str],
                   kind_virtual:list[str], *,
                   model:Literal['CS', 'NM', 'TM'] = 'CS',
                   dispersion:bool=False,
                   name_monitor:Optional[list[str]]=None,
                   name_virtual:Optional[list[str]]=None,
                   rule:Optional[dict[str, str]]=None,
                   postfix:str='_',
                   rc:bool=False,
                   lattice:Optional[Path]=None,
                   dtype:DataType = Float64,
                   device:DataDevice = DataDevice('cpu')) -> Model:
        """
        Generate model from (TFS or SDDS) table

        Parameters
        ----------
        kind: Literal['TFS', 'SDDS']
            table kind
        path: Path
            input path
        kind_monitor: list[str]
            list of element types for monitor locations
        kind_virtual: list[str]
            list of element types for virtual locations
        model: Literal['CS', 'NM', 'TM'], default='CS'
            model type
        dispersion: bool, default=False
            flag to insert zero dispersion   
        name_monitor: Optional[list[str]]
            list of element names for monitor locations
        name_virtual: Optional[list[str]]
            list of element names for virtual locations 
        rule: dict[str, str]
            rename rule        
        postfix: str, default=''
            rename duplicate postfix   
        rc: bool, default=True
            flag to set RC
        lattice: Optional[Path]
            lattice with RC data
        dtype: DataType, default=Float64
            data type
        device: DataDevice, default=DataDevice('cpu')
            data device

        Returns
        -------
        Model
        
        """
        parameters: dict[str, str|int|float]
        columns: dict[str, dict[str, str|int|float]]
        parameters, columns = {'TFS': load_tfs, 'SDDS': load_sdds}[kind](path, postfix=postfix)

        table: dict[str, dict[str, str|int|float]] = convert(columns, 
                                                             kind, 
                                                             kind_monitor, 
                                                             kind_virtual,
                                                             dispersion=dispersion,
                                                             rc=rc,
                                                             name_monitor=name_monitor,
                                                             name_virtual=name_virtual,
                                                             monitor=cls._monitor,
                                                             virtual=cls._virtual,
                                                             rule=rule)
        if lattice:
            lattice:dict[str,dict[str,str|int|float|dict]] = load_lattice(lattice, rc=True)
            table = add_rc(table, lattice)
        
        result:Model = Model(model, rc=rc, table=table, dtype=dtype, device=device)
        
        result.parameters: dict[str, str|int|float] = parameters
        result.columns: dict[str, dict[str, str|int|float]] = columns
        
        if lattice:
            result.lattice:dict[str,dict[str,str|int|float|dict]] = lattice
        
        return result
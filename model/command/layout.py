"""
Layout
------

Graphical representation of planar accelerator lattice

"""
from __future__ import annotations

from typing import Optional

from math import ceil

import torch
from torch import Tensor

from model.library.line import Line


class Layout:
    """
    Layout
    ------

    Returns
    -------
    Layout

    """
    radian: float = 180/torch.pi

    config: [dict[str, dict[str, str | float]]] = {
        'Drift'     : {'color': 'white', 'width': 0.25, 'height': 0.25, 'opacity': 0.00},
        'Quadrupole': {'color': 'red'  , 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Sextupole' : {'color': 'green', 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Octupole'  : {'color': 'green', 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Multipole' : {'color': 'green', 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Dipole'    : {'color': 'blue' , 'width': 0.50, 'height': 0.50, 'opacity': 0.20},
        'Corrector' : {'color': 'gray' , 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Gradient'  : {'color': 'gray' , 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'Linear'    : {'color': 'gray' , 'width': 0.25, 'height': 0.25, 'opacity': 0.20},
        'BPM'       : {'color': 'gray' , 'width': 1.00, 'height': 0.75, 'opacity': 0.10},
        'Marker'    : {'color': 'gray' , 'width': 1.00, 'height': 0.75, 'opacity': 0.10}
    }

    def __init__(self, line:Line) -> None:
        """
        Layout instance initialization

        Parameters
        ----------
        line: Line
            input line

        Returns
        -------
        None

        """
        self.line = line

        self.dtype  = self.line.dtype
        self.device  = self.line.device

        self.one = torch.tensor(1.0, dtype=self.line.dtype, device=self.line.device)
        self.nul = torch.tensor(0.0, dtype=self.line.dtype, device=self.line.device)


    def orbit(self, *,
              start:tuple[float,float] = (0.0, 0.0),
              step:float=0.01,
              flat:bool=False,
              flag:bool=True,
              lengths:Optional[Tensor]=None,
              angles:Optional[Tensor]=None) -> tuple[Tensor,Tensor]|tuple[Tensor,Tensor,Tensor]:
        """
        Generate planar reference orbit (anti-clockwise rotation)

        Reference orbit is formed by straight lines and circe arcs (sampled by angle)

        Parameters
        ----------
        start: tuple[float,float], default=(0.0,0.0)
            starting point relative to global origin
        step: float, default=0.01
            arc slicing step size (not performed if None is passes)
        flat: bool, default=True
            flag to (x, y) instead of (x, y, z) with z = 0
        flag: bool, default=True
            flag to extract lengths and angles from line (use to generate dense set of points)
        lengths: Optional[Tensor]
            input lenghts
        angles: Optional[Tensor]
            input angles (not accumulated)


        Returns
        -------
        tuple[Tensor,Tensor]|tuple[Tensor,Tensor,Tensor]
            (x, y) or (x, y, z) with z = 0

        """
        start = torch.tensor(start, dtype=self.dtype, device=self.device)
        x, y = start
        orbit = [torch.stack([x, y])]
        if flag:
            *_, lengths, angles = zip(*self.line.layout())
            lengths = torch.stack(lengths)
            angles = torch.stack(angles)
        current_angle = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for length, angle in zip(lengths, angles):
            if angle == 0:
                dx = length*current_angle.cos()
                dy = length*current_angle.sin()
                x += dx
                y += dy
                orbit.append(torch.stack([x, y]))
                continue
            radius = length/angle
            xc = x - radius*current_angle.sin()
            yc = y + radius*current_angle.cos()
            if step and step < angle.abs():
                arc_angles = torch.linspace(0, angle, ceil(angle.abs()/step), dtype=self.dtype, device=self.device)
            else:
                arc_angles = [angle]
            for arc_angle in arc_angles:
                arc_x = xc + radius*(current_angle + arc_angle).sin()
                arc_y = yc - radius*(current_angle + arc_angle).cos()
                orbit.append(torch.stack([arc_x, arc_y]))
            x = arc_x
            y = arc_y
            current_angle += angle
        x, y = torch.stack(orbit).T
        return (x, y) if flat else (x, y, torch.zeros_like(x))


    def transform(self,
                  points:Tensor,
                  shifts:Tensor,
                  angle:Tensor) -> Tensor:
        """
        Geometical transformation (3D shift and in-plane rotation)

        Parameters
        ----------
        points: Tensor
            set of points to transform
        shifts: Tensor
            shifts valurs
        angle: Tensor
            rotation angle (anti-clockwise)

        Retuns
        ------
        Tensor

        """
        cos = angle.cos()
        sin = angle.sin()
        matrix = torch.stack([torch.stack([cos, +sin, self.nul]), torch.stack([-sin, cos, self.nul]), torch.stack([self.nul, self.nul, self.one])])
        return points @ matrix + shifts


    def make_face(self,
                  width:float=1.0,
                  height:float=1.0) -> Tensor:
        """
        Generate block face points

        Parameters
        ----------
        width: float, default=1.0
            width
        height: float, default=1.0
            height

        Returns
        -------
        Tensor

        """
        return torch.tensor([[
                [0, -width/2, -height/2],
                [0, +width/2, -height/2],
                [0, +width/2, +height/2],
                [0, -width/2, +height/2]
            ]], dtype=self.dtype, device=self.device)


    def make_straight_block(self,
                            length:Tensor,
                            width:float=1.0,
                            height:float=1.0,
                            count:int=1) -> Tensor:
        """
        Generate straight block

        Parameters
        ----------
        length: Tensor
            block length
        width: float, default=1.0
            width
        height: float, default=1.0
            height
        count: int, default=1
            number of slices

        Returns
        -------
        Tensor

        """
        face = self.make_face(width, height).squeeze()
        return torch.stack([face + i*torch.stack([length/count, self.nul, self.nul]) for i in range(0, count + 1)])


    def make_curved_block(self,
                          length:Tensor,
                          angle:Tensor,
                          width:float=1.0,
                          height:float=1.0,
                          count:int=1) -> Tensor:
        """
        Generate straight block

        Parameters
        ----------
        length: Tensor
            block length
        angle: Tensor
            total angle (midplane rotation)
        width: float, default=1.0
            width
        height: float, default=1.0
            height
        count: int, default=1
            number of (angular) slices

        Returns
        -------
        Tensor

        """
        face = self.make_face(width, height).squeeze()
        delta = angle/count
        radius = length/angle
        blocks = [face]
        for i in range(1, count + 1):
            total = i*delta
            dx = radius*total.sin()
            dy = radius*(1 - total.cos())
            blocks.append(self.transform(face, torch.stack([dx, dy, self.nul]), total))
        return torch.stack(blocks)


    def mesh(self,
             blocks:Tensor) -> tuple[Tensor, Tensor]:
        """
        Genereate mesh indices

        Parameters
        ----------
        blocks: Tensor
            blocks

        Returns
        -------
        tuple[Tensor, Tensor]
            (i, j, k), (x, y, z)

        """
        if len(blocks) == 1:
            index = torch.tensor([[0, 0], [1, 2], [2, 3]], dtype=torch.int64, device=self.device)
            return blocks.squeeze().T, index
        index = torch.tensor([
            [7, 0, 0, 0, 4, 4, 6, 1, 4, 0, 3, 6],
            [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            [0, 7, 2, 3, 6, 7, 1, 6, 5, 5, 7, 2]], dtype=torch.int64, device=self.device).T
        count = 0
        edges = []
        for _ in range(len(blocks) - 1):
            edges.append(index + count)
            count += 4
        return torch.vstack([*blocks]).T, torch.vstack(edges).T


    def profile_1d(self, *,
                   scale:float=1.0,
                   exclude:list[str]=['Drift'],
                   shift:float=0.0,
                   alpha:float=0.75,
                   text:bool=True,
                   delta:float=0.0,
                   rotation:float=90.0,
                   fontsize:int=10) -> list[dict]|tuple[list[dict],list[dict]]:
        """
        Generate data for 1D profile plot

        Parameters
        ----------
        scale: float, default=1.0
            rectangles height scaling factor
        exclude: list[str], default=['Drift']
            list of line kinds and/or names to exclude
        shift: float, default=0.0
            profile absolute vertical shift
        alpha: float, default=0.75
            rectangle fill opacity
        text: bool, default=True
            flag to include labels
        delta: float, default=0.0
            label vertical shift
        rotation: float, default=90.0
            label rotation
        fontsize: int, default=10
            label font size

        Returns
        -------
        list[dict] | tuple[list[dict],list[dict]]
            rectangles or (rectangles, labels)

        """
        location = 0.0
        rectangles = []
        labels = []
        for element in self.line.scan('name'):
            name = element.name
            kind = element.__class__.__name__
            if kind not in exclude:
                rectangles.append(
                    dict(
                        xy=(location , shift),
                        width=element.length.item(),
                        height=scale*self.config[kind]['height'],
                        fill=True,
                        color=self.config[kind]['color'],
                        alpha=alpha,
                        edgecolor=None
                    )
                )
                if text:
                    labels.append(
                        dict(
                            x=location + 0.5*element.length.item(),
                            y=delta,
                            s=name,
                            fontsize=fontsize,
                            rotation=rotation,
                            color='black',
                            horizontalalignment='center',
                            verticalalignment='center'
                        )
                    )
            location += element.length.item()
        return rectangles if not text else (rectangles, labels)


    def profile_2d(self, *,
                   start:tuple[float,float] = (0.0, 0.0),
                   exclude:list[str]=['Drift'],
                   scale:float=1.0,
                   linestyle:str='solid',
                   linewidth:float=0.5,
                   text:bool=True,
                   delta:float=0.5,
                   rotation:float=90.0,
                   fontsize:int=10) -> list[dict]|tuple[list[dict],list[dict]]:
        """
        Generate data for 2D profile plot

        Parameters
        ----------
        start: tuple[float,float], default=(0.0, 0.0)
            starting point relative to global origin
        exclude: list[str], default=['Drift']
            list of line kinds and/or names to exclude
        scale: float, default=1.0
            width and height scaling factor
        linestyle: str, default='solid'
            line style
        linewidth: float, default=0.5
            line width
        text: bool, default=True
            flag to include labels
        delta: float, default=0.5
            label shift along radius
        rotation: float, default=90.0
            additional label rotation
        fontsize: int, default=10
            label font size

        Returns
        -------
        list[dict] | tuple[list[dict],list[dict]]
            blocks or (blocks, labels)
        """
        blocks = []
        labels = []
        x, y, z = self.orbit(start=start, flat=False, step=None)
        *points, _ = torch.stack([x, y, z]).T
        *_, angles = zip(*self.line.layout())
        delta = torch.tensor([0.0, delta, 0.0], dtype=self.dtype, device=self.device)
        total = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for i, (element, point, angle) in enumerate(zip(self.line.scan('name'), points, angles)):
            name = element.name
            kind = element.__class__.__name__
            if kind not in exclude and name not in exclude:
                if not element.flag:
                    if element.length:
                        block = self.make_straight_block(
                            element.length,
                            width=scale*self.config[kind]['width'],
                            height=scale*self.config[kind]['height'],
                            count=element.ns
                        )
                    else:
                        block = self.make_face(
                            width=scale*self.config[kind]['width'],
                            height=scale*self.config[kind]['height']
                        )
                else:
                    block = self.make_curved_block(
                        element.length,
                        element.angle,
                        width=scale*self.config[kind]['width'],
                        height=scale*self.config[kind]['height'],
                        count=element.ns
                    )
                block = self.transform(block, point, total)
                x, y, z = torch.vstack([*block]).T
                blocks.append(
                    dict(x=x.cpu().numpy(),
                         y=y.cpu().numpy(),
                         color=self.config[kind]['color'],
                         linestyle=linestyle,
                         linewidth=linewidth)
                )
                if text:
                    x, y, _ = self.transform(delta, point, total)
                    labels.append(
                        dict(x=x.cpu().numpy(),
                             y=y.cpu().numpy(),
                             s=element.name,
                             fontsize=fontsize,
                             color='black',
                             rotation=rotation + self.radian*total.item(),
                             horizontalalignment='center',
                             verticalalignment='center')
                    )
            total += angle
        return blocks if not text else (blocks, labels)


    def profile_3d(self, *,
                   start:tuple[float,float] = (0.0, 0.0),
                   exclude:list[str]=['Drift'],
                   scale:float=1.0) -> list[dict]:
        """
        Generate data for 3D profile plot

        Parameters
        ----------
        start: tuple[float,float], default=(0.0, 0.0)
            starting point relative to global origin
        exclude: list[str], default=['Drift']
            list of line kinds and/or names to exclude
        scale: float, default=1.0
            width and height scaling factor

        Returns
        -------
        list[dict]
            blocks

        """
        blocks = []
        x, y, z = self.orbit(start=start, flat=False, step=None)
        *points, _ = torch.stack([x, y, z]).T
        *_, angles = zip(*self.line.layout())
        total = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for i, (element, point, angle) in enumerate(zip(self.line.scan('name'), points, angles)):
            name = element.name
            kind = element.__class__.__name__
            if kind not in exclude and name not in exclude:
                if not element.flag:
                    if element.length:
                        block = self.make_straight_block(
                            element.length,
                            width=scale*self.config[kind]['width'],
                            height=scale*self.config[kind]['height'],
                            count=element.ns
                        )
                    else:
                        block = self.make_face(
                            width=scale*self.config[kind]['width'],
                            height=scale*self.config[kind]['height']
                        )
                else:
                    block = self.make_curved_block(
                        element.length,
                        element.angle,
                        width=scale*self.config[kind]['width'],
                        height=scale*self.config[kind]['height'],
                        count=element.ns
                    )
                block = self.transform(block, point, total)
                (x, y, z), (i, j, k) = self.mesh(block)
                blocks.append(
                    dict(x=x.cpu().numpy(),
                         y=y.cpu().numpy(),
                         z=z.cpu().numpy(),
                         i=i.cpu().numpy(),
                         j=j.cpu().numpy(),
                         k=k.cpu().numpy(),
                         color=self.config[kind]['color'],
                         opacity=self.config[kind]['opacity'],
                         name=name,
                         showscale=False,
                         legendgroup=kind,
                         hovertext=str(element).replace(',', ',<br>  ').replace('(', '(<br>   '),
                         legendgrouptitle_text=kind,
                         hoverinfo='text',
                         showlegend=True)
                )
            total += angle
        return blocks


    def slicing_table(self) -> tuple[list[str], list[str], Tensor, Tensor, Tensor]:
        """
        Generate slicing table

        Parameters
        ----------
        None

        Returns
        -------
        tuple[list[str], list[str], Tensor, Tensor, Tensor]
            names, kinds, lengths (not accumulated), angles (not accumulated), points (slice starting points relative to global frame)

            |           |
            |           |
        (x, y, z)---l---|
            |           |
            |           |

        """
        names, kinds, lengths, angles = [], [], [], []
        for element in self.line.scan('name'):
            name = element.name
            kind = element.__class__.__name__
            length = element.length/element.ns
            angle = element.angle/element.ns if element.flag else torch.zeros_like(length)
            for _ in range(element.ns):
                names.append(name)
                kinds.append(kind)
                lengths.append(length)
                angles.append(angle)
        x, y, z = self.orbit(step=None, flat=False, flag=False, lengths=lengths, angles=angles)
        *points, _ = torch.stack([x, y, z]).T
        return names, kinds, torch.stack(lengths), torch.stack(angles), torch.stack(points)
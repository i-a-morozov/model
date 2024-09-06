"""
Alignment
---------

Apply alignment errors

"""
from torch import Tensor

from model.library.element import Element

from model.library.transformations import tx, ty, tz
from model.library.transformations import rx, ry, rz

type State = Tensor

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
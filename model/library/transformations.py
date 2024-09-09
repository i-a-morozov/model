"""
Transformations module
----------------------

Collection of symplectic transformations, building blocks for accelerator elements

calibration_forward[_{knobs, track}] -- calibration transformation (from beam frame to bpm frame)
calibration_inverse[_{knobs, track}] -- calibration transformation (from bpm frame to beam frame)
corrector[_{knobs, track}]   -- corrector transformation
drift[_{knobs, track}]       -- drift transformation
kinematic[_{knobs, track}]   -- kinematic correction transformation
fquad[_{knobs, track}]       -- focusing quadrupole transformation
dquad[_{knobs, track}]       -- defocusing quadrupole transformation
quadrupole[_{knobs, track}]  -- generic quadrupole transformation
linear                       -- generic linear transformation
gradient[_{knobs, track}]    -- thin quadrupole transformation
sextupole[_{knobs, track}]   -- thin sextupole transformation
octupole[_{knobs, track}]    -- thin octupole transformation
dipole[_{knobs, track}]      -- dipole transformation
bend[_{knobs, track}]        -- generic bend transformation
wedge[_{knobs, track}]       -- dipole wedge transformation
multipole[_{knobs, track}]   -- cylindrical multipole error kick upto octupole degree
tx[_{knobs, track}]          -- TX translation
ty[_{knobs, track}]          -- TY translation
tz[_{knobs, track}]          -- TZ translation
rx[_{knobs, track}]          -- RX rotation
ry[_{knobs, track}]          -- RY rotation
rz[_{knobs, track}]          -- RZ rotation

"""
import torch
from torch import Tensor

type State = Tensor
type Knobs = Tensor


def calibration_forward(state:State, 
                        gxx:Tensor, 
                        gxy:Tensor, 
                        gyx:Tensor, 
                        gyy:Tensor) -> State:
    """
    Calibration transformation (from beam frame to bpm frame)

    qx -> gxx qx + gxy qy
    qy -> gyx qx + gyy qy    

    Parameters
    ----------
    state: State
        initial state
    gxx: Tensor
        qx scaling
    gxy: Tensor
        qx coupling
    gyx: Tensor
        qy coupling
    gyy: Tensor
        qy scaling
        
    Returns
    -------
    State
    
    """        
    knobs: Knobs = calibration_forward_knobs(gxx, gxy, gyx, gyy)
    return calibration_forward_track(state, knobs)


def calibration_forward_knobs(gxx:Tensor, gxy:Tensor, gyx:Tensor, gyy:Tensor) -> Knobs:   
    det =  gxx*gyy - gxy*gyx
    qxqx = gxx
    qxqy = gxy
    pxpx = gyy/det
    pxpy = -gyx/det
    qyqx = gyx
    qyqy = gyy
    pypx = -gxy/det
    pypy = gxx/det
    return torch.stack([qxqx, qxqy, pxpx, pxpy, qyqx, qyqy, pypx, pypy])


def calibration_forward_track(state:State, knobs:Knobs) -> State:    
    qxqx, qxqy, pxpx, pxpy, qyqx, qyqy, pypx, pypy, *_ = knobs
    qx, px, qy, py = state
    Qx = qx*qxqx + qy*qxqy
    Px = px*pxpx + py*pxpy
    Qy = qx*qyqx + qy*qyqy
    Py = px*pypx + py*pypy
    return torch.stack([Qx, Px, Qy, Py])


def calibration_inverse(state:State, 
                        gxx:Tensor, 
                        gxy:Tensor, 
                        gyx:Tensor, 
                        gyy:Tensor) -> State:
    """
    Calibration transformation (from bpm frame to beam frame)

    Parameters
    ----------
    state: State
        initial state
    gxx: Tensor
        qx scaling
    gxy: Tensor
        qx coupling
    gyx: Tensor
        qy coupling
    gyy: Tensor
        qy scaling
        
    Returns
    -------
    State
    
    """        
    knobs: Knobs = calibration_inverse_knobs(gxx, gxy, gyx, gyy)
    return calibration_inverse_track(state, knobs)


def calibration_inverse_knobs(gxx:Tensor, gxy:Tensor, gyx:Tensor, gyy:Tensor) -> Knobs:   
    det =  gxx*gyy - gxy*gyx
    qxqx = gyy/det
    qxqy = -gxy/det
    pxpx = gxx
    pxpy = gyx
    qyqx = -gyx/det
    qyqy = gxx/det
    pypx = gxy
    pypy = gyy
    return torch.stack([qxqx, qxqy, pxpx, pxpy, qyqx, qyqy, pypx, pypy])


def calibration_inverse_track(state:State, knobs:Knobs) -> State:    
    qxqx, qxqy, pxpx, pxpy, qyqx, qyqy, pypx, pypy, *_ = knobs
    qx, px, qy, py = state
    Qx = qx*qxqx + qy*qxqy
    Px = px*pxpx + py*pxpy
    Qy = qx*qyqx + qy*qyqy
    Py = px*pypx + py*pypy
    return torch.stack([Qx, Px, Qy, Py])


def corrector(state:State, 
              kx:Tensor, 
              ky:Tensor) -> State:
    """
    Corrector transformation

    px -> px + kx
    py -> py + ky

    Parameters
    ----------
    state: State
        initial state
    kx: Tensor
        px kick
    ky: Tensor
        py kick
        
    Returns
    -------
    State
      
    """ 
    knobs: Knobs = corrector_knobs(kx, ky)
    return corrector_track(state, knobs)


def corrector_knobs(kx:Tensor, ky:Tensor) -> Knobs:    
    return torch.stack([kx, ky])


def corrector_track(state:State, knobs:Knobs) -> State:    
    kx, ky, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px + kx
    Qy = qy
    Py = py + ky
    return torch.stack([Qx, Px, Qy, Py])


def drift(state:State, 
          dp:Tensor, 
          length:Tensor) -> State:
    """
    Drift transformation

    Parameters
    ----------
    state: State
        initial state
    dp: Tensor
        momentum deviation
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """       
    knobs: Knobs = drift_knobs(dp, length)
    return drift_track(state, knobs)


def drift_knobs(dp:Tensor, length:Tensor) -> Knobs:
    return torch.stack([length/(1 + dp)])


def drift_track(state:State, knobs:Knobs) -> State:
    ds, *_ = knobs
    qx, px, qy, py = state
    Qx = qx + ds*px
    Px = px
    Qy = qy + ds*py
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def kinematic(state:State, 
              dp:Tensor, 
              length:Tensor) -> State:
    """
    Kinematic correction transformation

    Parameters
    ----------
    state: State
        initial state
    dp: Tensor
        momentum deviation
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """      
    knobs: Knobs = kinematic_knobs(dp, length)    
    return kinematic_track(state, knobs)   
    

def kinematic_knobs(dp:Tensor, length:Tensor) -> Knobs:
    return torch.stack([length, 1 + dp])


def kinematic_track(state:State, knobs:Knobs) -> State:
    kl, kp, *_ = knobs
    qx, px, qy, py = state
    pz = (kp**2 - px**2 - py**2).sqrt()
    Qx = qx + kl*px*(1/pz - 1/kp)
    Px = px
    Qy = qy + kl*py*(1/pz - 1/kp)
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def fquad(state:State, 
          kn:Tensor, 
          dp:Tensor, 
          length:Tensor) -> State:
    """
    Focusing quadrupole transformation

    Parameters
    ----------
    state: State
        initial state
    kn: Tensor, positive
        quadrupole strength
    dp: Tensor
        momentum deviation
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """        
    knobs: Knobs = fquad_knobs(kn, dp, length)
    return fquad_track(state, knobs)


def fquad_knobs(kn:Tensor, dp:Tensor, length:Tensor) -> Knobs:
    dw = length*(kn/(1 + dp)).sqrt()
    kp = (kn*(1 + dp)).sqrt()
    cos, cosh = dw.cos(), dw.cosh()
    sin, sinh = dw.sin(), dw.sinh()
    qxqx = cos
    qxpx = sin/kp
    pxpx = cos
    pxqx = -sin*kp
    qyqy = cosh
    qypy = sinh/kp
    pypy = cosh
    pyqy = sinh*kp
    return torch.stack([qxqx, qxpx, pxpx, pxqx, qyqy, qypy, pypy, pyqy])


def fquad_track(state:State, knobs:Knobs) -> State:      
    qxqx, qxpx, pxpx, pxqx, qyqy, qypy, pypy, pyqy, *_ = knobs
    qx, px, qy, py = state
    Qx = qx*qxqx + px*qxpx
    Px = px*pxpx + qx*pxqx
    Qy = qy*qyqy + py*qypy
    Py = py*pypy + qy*pyqy
    return torch.stack([Qx, Px, Qy, Py])


def dquad(state:State, 
          kn:Tensor, 
          dp:Tensor, 
          length:Tensor) -> State:
    """
    Defocusing quadrupole transformation

    Parameters
    ----------
    state: State
        initial state
    kn: Tensor, positive
        quadrupole strength
    dp: Tensor
        momentum deviation
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """        
    knobs: Knobs = dquad_knobs(kn, dp, length)
    return dquad_track(state, knobs)


def dquad_knobs(kn:Tensor, dp:Tensor, length:Tensor) -> Knobs:  
    dw = length*(kn/(1 + dp)).sqrt()
    kp = (kn*(1 + dp)).sqrt()
    cos, cosh = dw.cos(), dw.cosh()
    sin, sinh = dw.sin(), dw.sinh()
    qxqx = cosh
    qxpx = sinh/kp
    pxpx = cosh
    pxqx = sinh*kp
    qyqy = cos
    qypy = sin/kp
    pypy = cos
    pyqy = -sin*kp
    return torch.stack([qxqx, qxpx, pxpx, pxqx, qyqy, qypy, pypy, pyqy])


def dquad_track(state:State, knobs:Knobs) -> State:
    qxqx, qxpx, pxpx, pxqx, qyqy, qypy, pypy, pyqy, *_ = knobs
    qx, px, qy, py = state
    Qx = qx*qxqx + px*qxpx
    Px = px*pxpx + qx*pxqx
    Qy = qy*qyqy + py*qypy
    Py = py*pypy + qy*pyqy
    return torch.stack([Qx, Px, Qy, Py])


def quadrupole(state:State, 
               kn:Tensor, 
               ks:Tensor, 
               dp:Tensor, 
               length:Tensor) -> State:
    """
    Generic quadrupole transformation

    Note, singular if kn = ks = 0

    Parameters
    ----------
    state: State
        initial state
    kn: Tensor
        normal quadrupole strength
    ks: Tensor
        skew quadrupole strength   
    dp: Tensor
        momentum deviation
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """   
    knobs: Knobs = quadrupole_knobs(kn, ks, dp, length)
    return quadrupole_track(state, knobs)


def quadrupole_knobs(kn:Tensor, ks:Tensor, dp:Tensor, length:Tensor) -> Knobs:    
    kq = (kn**2 + ks**2).sqrt()
    kp = (kq/(1 + dp)).sqrt()
    dw = kp*length    
    ka = kq + kn
    kb = kq - kn
    ka, kb, ks, kp = ka/kq, kb/kq, ks/kq, kp/kq
    cos, cosh = dw.cos(), dw.cosh()
    sin, sinh = dw.sin(), dw.sinh()
    qxqx = (kb*cosh + ka*cos)
    qxpx = (kb*sinh + ka*sin)*kp
    qxqy = (cosh - cos)*ks
    qxpy = (sinh - sin)*ks*kp
    pxqx = (kb*sinh - ka*sin)/kp
    pxpx = (kb*cosh + ka*cos)
    pxqy = (sinh + sin)*ks/kp
    pxpy = (cosh - cos)*ks
    qyqx = (cosh - cos)*ks
    qypx = (sinh - sin)*ks*kp
    qyqy = (ka*cosh + kb*cos)
    qypy = (ka*sinh + kb*sin)*kp
    pyqx = (sinh + sin)*ks/kp
    pypx = (cosh - cos)*ks
    pyqy = (ka*sinh - kb*sin)/kp
    pypy = (ka*cosh + kb*cos)
    return torch.stack([qxqx, qxpx, qxqy, qxpy, 
                        pxqx, pxpx, pxqy, pxpy, 
                        qyqx, qypx, qyqy, qypy, 
                        pyqx, pypx, pyqy, pypy])



def quadrupole_track(x:Tensor, knobs:Tensor) -> Tensor:       
    qx, px, qy, py = 0.5*x
    qxqx, qxpx, qxqy, qxpy, \
    pxqx, pxpx, pxqy, pxpy, \
    qyqx, qypx, qyqy, qypy, \
    pyqx, pypx, pyqy, pypy = knobs
    Qx = qx*qxqx + px*qxpx + qy*qxqy + py*qxpy
    Px = qx*pxqx + px*pxpx + qy*pxqy + py*pxpy
    Qy = qx*qyqx + px*qypx + qy*qyqy + py*qypy
    Py = qx*pyqx + px*pypx + qy*pyqy + py*pypy
    return torch.stack([Qx, Px, Qy, Py])


def linear(state:State, 
           vector:Tensor, 
           matrix:Tensor) -> State:
    """
    Generic linear transformation

    Parameters
    ----------
    state: State
        initial state
    vector: Tensor
        constant vector (dispersion)
    matrix: Tensor
        matrix
        
    Returns
    -------
    State
    
    """       
    return vector + matrix @ state


def gradient(state:State, 
             kn:Tensor, 
             ks:Tensor, 
             length:Tensor) -> State:
    """
    Thin quadrupole transformation

    Parameters
    ----------
    state: State
        initial state
    kn: Tensor
        skew strength
    ks: Tensor
        skew strength
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """     
    knobs: Knobs = gradient_knobs(kn, ks, length)
    return gradient_track(state, knobs)  


def gradient_knobs(kn:Tensor, ks:Tensor, length:Tensor) -> Knobs:      
    return torch.stack([length*kn, length*ks])


def gradient_track(state:State, knobs:Knobs) -> State:       
    kn, ks, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px - kn*qx + ks*qy
    Qy = qy
    Py = py + kn*qy + ks*qx
    return torch.stack([Qx, Px, Qy, Py])


def sextupole(state:State, 
              ks:Tensor, 
              length:Tensor) -> State:
    """
    Thin sextupole transformation

    Parameters
    ----------
    state: State
        initial state
    ks: Tensor
        strength
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """     
    knobs: Knobs = sextupole_knobs(ks, length)
    return sextupole_track(state, knobs)  


def sextupole_knobs(ks:Tensor, length:Tensor) -> Knobs:      
    kl = ks*length
    return torch.stack([kl/2.0, kl])


def sextupole_track(state:State, knobs:Knobs) -> State:       
    kx, ky, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px - kx*(qx*qx - qy*qy)
    Qy = qy
    Py = py + ky*qx*qy
    return torch.stack([Qx, Px, Qy, Py])


def octupole(state:State, 
             ko:Tensor, 
             length:Tensor) -> State:
    """
    Thin octupole transformation

    Parameters
    ----------
    state: State
        initial state
    ko: Tensor
        strength
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """        
    knobs: Knobs = octupole_knobs(ko, length)
    return octupole_track(state, knobs)


def octupole_knobs(ko:Tensor, length:Tensor) -> Knobs:        
    kl = ko*length
    return torch.stack([-kl/6.0])


def octupole_track(state:State, knobs:Knobs) -> State:      
    kl, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px + kl*(qx*qx*qx - 3*qx*qy*qy)
    Qy = qy
    Py = py + kl*(qy*qy*qy - 3*qx*qx*qy)
    return torch.stack([Qx, Px, Qy, Py])


def dipole(state:State, 
           r:Tensor, 
           dp:Tensor, 
           length:Tensor) -> State:
    """
    Dipole transformation

    Parameters
    ----------
    state: State
        initial state
    r: Tensor
        bending radius
    dp: Tensor
        momentum deviation        
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """       
    knobs: Knobs = dipole_knobs(r, dp, length)
    return dipole_track(state, knobs)


def dipole_knobs(r:Tensor, dp:Tensor, length:Tensor) -> Knobs:
    dq = (1 + dp).sqrt()
    dw = length/(dq*r)
    cos = dw.cos()
    sin = dw.sin()
    qxcx = dp*r*(1 - cos)
    qxqx = cos
    qxpx = r*sin/dq 
    pxcx = dp*dq*sin
    pxqx = - dq*sin/r
    pxpx = cos
    qypy = length/(1 + dp)
    return torch.stack([qxcx, qxqx, qxpx, pxcx, pxqx, pxpx, qypy])


def dipole_track(state:State, knobs:Knobs) -> State:     
    qxcx, qxqx, qxpx, pxcx, pxqx, pxpx, qypy, *_ = knobs
    qx, px, qy, py = state
    Qx = qxcx + qx*qxqx + px*qxpx
    Px = pxcx + qx*pxqx + px*pxpx
    Qy = qy + py*qypy
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def bend(state:State, 
         r:Tensor, 
         kn:Tensor, 
         ks:Tensor, 
         dp:Tensor, 
         length:Tensor) -> State:
    """
    Combined function bend transformation

    Note, singular if kn = ks = 0

    Parameters
    ----------
    state: State
        initial state
    r: Tensor
        bending radius
    kn: Tensor
        normal quadrupole strength
    ks: Tensor
        skew quadrupole strength           
    dp: Tensor
        momentum deviation        
    length: Tensor
        length
        
    Returns
    -------
    State
    
    """       
    knobs: Knobs = bend_knobs(r, kn, ks, dp, length)
    return bend_track(state, knobs)


def bend_knobs(r:Tensor, kn:Tensor, ks:Tensor, dp:Tensor, length:Tensor) -> Knobs:
    kq = kn + r**2*(kn**2 + ks**2)
    kr = (1 + 4*kq*r**2).sqrt()
    kra = (kr + 1).sqrt()
    krb = (kr - 1).sqrt()
    dpsq = (2*(1 + dp)).sqrt()
    ka = kr + 2*kn*r**2 + 1
    kb = kr - 2*kn*r**2 - 1
    wa = (kra*length)/(dpsq*r)
    wb = (krb*length)/(dpsq*r)
    kakr = ka/kr
    kbkr = kb/kr
    kskr = ks/kr
    ksdr = kskr*r**2
    qa = r/(kra*dpsq)
    qb = r/(krb*dpsq)
    kx = krb**2*kn + 2*kq
    ky = kn - 2*kq + kn*kr
    cos, cosh = wa.cos(), wb.cosh()
    sin, sinh = wa.sin(), wb.sinh()    
    qxcx = dp*r/kq*(kn - (kx*cos + ky*cosh)/(2*kr))
    pxcx = dp*dpsq/(4*kq*kr)*(kra*kx*sin - krb*ky*sinh)    
    qycy = dp*kskr*r/kq*(-kr + krb**2*cos/2 + kra**2*cosh/2)
    pycy = dp*kskr*dpsq*kra*krb/(4*kq)*(kra*sinh - krb*sin)
    qxqx = (kakr*cos + kbkr*cosh)/2
    qxpx = (kakr*qa*sin + kbkr*qb*sinh)
    qxqy = ksdr*(cosh - cos)
    qxpy = 2*ksdr*(qb*sinh - qa*sin)
    pxqx = (kbkr*sinh/qb - kakr*sin/qa)/4
    pxpx = (kakr*cos + kbkr*cosh)/2
    pxqy = (ksdr*(qb*sin + qa*sinh))/(2*qa*qb)
    pxpy = (ksdr*(-cos + cosh))    
    qyqx = (ksdr*(cosh - cos))
    qypx = (2*ksdr*(qb*sinh- qa*sin))
    qyqy = (kbkr*cos + kakr*cosh)/2
    qypy = (kbkr*qa*sin + kakr*qb*sinh)    
    pyqx = (ksdr*(qb*sin + qa*sinh))/(2.*qa*qb)
    pypx = (ksdr*(cosh - cos))
    pyqy = (kakr*sinh/qb - kbkr*sin/qa)/4
    pypy = (kbkr*cos + kakr*cosh)/2
    return torch.stack([qxcx, qxqx, qxpx, qxqy, qxpy, 
                        pxcx, pxqx, pxpx, pxqy, pxpy, 
                        qycy, qyqx, qypx, qyqy, qypy, 
                        pycy, pyqx, pypx, pyqy, pypy])


def bend_track(state:Tensor, knobs:Tensor) -> Tensor:      
    qx, px, qy, py = state
    qxcx, qxqx, qxpx, qxqy, qxpy, \
    pxcx, pxqx, pxpx, pxqy, pxpy, \
    qycy, qyqx, qypx, qyqy, qypy, \
    pycy, pyqx, pypx, pyqy, pypy = knobs
    Qx = qxcx + qx*qxqx + px*qxpx + qy*qxqy + py*qxpy
    Px = pxcx + qx*pxqx + px*pxpx + qy*pxqy + py*pxpy
    Qy = qycy + qx*qyqx + px*qypx + qy*qyqy + py*qypy
    Py = pycy + qx*pyqx + px*pypx + qy*pyqy + py*pypy
    return torch.stack([Qx, Px, Qy, Py])


def wedge(state:State,
          epsilon:Tensor, 
          r:Tensor) -> State:
    """
    Dipole wedge transformation
    
    Parameters
    ----------
    state: State
        initial state
    epsilon: Tensor
        wedge angle
    r: Tensor
        bending radius
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = wedge_knobs(epsilon, r)
    return wedge_track(state, knobs)


def wedge_knobs(epsilon:Tensor, r:Tensor) -> Knobs:
    return torch.stack([epsilon.tan()/r])


def wedge_track(state:State, knobs:Knobs) -> State:  
    kw, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px + qx*kw
    Qy = qy
    Py = py - qy*kw
    return torch.stack([Qx, Px, Qy, Py])


def multipole(state:State, 
              r:Tensor, 
              kqn:Tensor, 
              kqs:Tensor, 
              ks:Tensor, 
              ko:Tensor, 
              length:Tensor) -> State:
    """ 
    Cylindrical multipole kick upto octupole degree

    Parameters
    ----------
    state: State
        initial state
    r: Tensor
        bending radius
    kqn: Tensor
        normal quadrupole strength
    kqs: Tensor
        skew quadrupole strength
    ks: Tensor
        sextupole strength
    ks: Tensor
        octupole strength
    length: Tensor
        length
        
    Returns
    -------
    State    
    
    """
    knobs: Knobs = multipole_knobs(r, kqn, kqs, ks, ko, length)
    return multipole_track(state, knobs)


def multipole_knobs(r:Tensor, kqn:Tensor, kqs:Tensor, ks:Tensor, ko:Tensor, length:Tensor) -> Knobs:
    kqnlr =  kqn*length/r
    kqslr =  kqs*length/r
    kslr = ks*length/r
    kol = ko*length
    return torch.stack([r, kqnlr, kqslr, kslr, kol])


def multipole_track(state:State, knobs:Knobs) -> State:
    r, kqnlr, kqslr, kslr, kol, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px - kqnlr*(qx**2 - qy**2/2) + kqslr*(2*qx*qy + 1/r*qy**3/6) - kslr*(r*(qx**2 - qy**2)/2 + (qx**3/2 - qx*qy**2))  - kol*qx*(qx**2 - 3*qy**2)/6
    Qy = qy
    Py = py + kqnlr*(1/r*qy**3/6 + qx*qy) + kqslr*(1/r*qx*qy**2/2 + (qx**2 - qy**2/2)) + kslr*(r*qx*qy + (qx**2*qy - qy**3/6)) + kol*qy*(3*qx**2 - qy**2)/6
    return torch.stack([Qx, Px, Qy, Py])


def tx(state:State,
       dx:Tensor) -> State:
    """
    TX translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    dx: Tensor
        qx translation error
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = tx_knobs(dx)
    return tx_track(state, knobs)


def tx_knobs(dx:Tensor) -> Knobs:
    return torch.stack([dx])


def tx_track(state:State, knobs:Knobs) -> State:
    dx, *_ = knobs
    qx, px, qy, py = state
    Qx = qx - dx
    Px = px
    Qy = qy
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def ty(state:State,
       dy:Tensor) -> State:
    """
    TY translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    dy: Tensor
        qy translation error
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = ty_knobs(dy)
    return ty_track(state, knobs)


def ty_knobs(dy:Tensor) -> Knobs:
    return torch.stack([dy])


def ty_track(state:State, knobs:Knobs) -> State:
    dy, *_ = knobs
    qx, px, qy, py = state
    Qx = qx
    Px = px
    Qy = qy - dy
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def tz(state:State,
       dz:Tensor,
       dp:Tensor) -> State:
    """
    TZ translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    dz: Tensor
        qz translation error
    dp: Tensor
        momentum deviation           
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = tz_knobs(dz, dp)
    return tz_track(state, knobs)


def tz_knobs(dz:Tensor, dp:Tensor) -> Knobs:
    kp = (1 + dp)**2
    return torch.stack([dz, kp])


def tz_track(state:State, knobs:Knobs) -> State:
    dz, kp, *_ = knobs
    qx, px, qy, py = state
    pz = (kp - px**2 - py**2).sqrt()
    Qx = qx + px*dz/pz
    Px = px
    Qy = qy + py*dz/pz
    Py = py
    return torch.stack([Qx, Px, Qy, Py])


def rx(state:State,
       wx:Tensor,
       dp:Tensor) -> State:
    """
    RX translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    wx: Tensor
        qx rotation angle
    dp: Tensor
        momentum deviation           
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = rx_knobs(wx, dp)
    return rx_track(state, knobs)


def rx_knobs(wx:Tensor, dp:Tensor) -> Knobs:
    kp = (1 + dp)**2
    cos = (-wx).cos()
    sin = (-wx).sin()
    tan = sin/cos
    return torch.stack([kp, cos, sin, tan])


def rx_track(state:State, knobs:Knobs) -> State:
    kp, cos, sin, tan, *_ = knobs
    qx, px, qy, py = state
    pz = (kp - px**2 - py**2).sqrt()
    Qx = qx + px*qy*tan/(pz - py*tan)
    Px = px
    Qy = qy/cos/(1 - py*tan/pz)
    Py = py*cos + pz*sin
    return torch.stack([Qx, Px, Qy, Py])


def ry(state:State,
       wy:Tensor,
       dp:Tensor) -> State:
    """
    RY translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    wz: Tensor
        qy rotation angle
    dp: Tensor
        momentum deviation           
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = ry_knobs(wy, dp)
    return ry_track(state, knobs)


def ry_knobs(wy:Tensor, dp:Tensor) -> Knobs:
    kp = (1 + dp)**2
    cos = (-wy).cos()
    sin = (-wy).sin()
    tan = sin/cos
    return torch.stack([kp, cos, sin, tan])


def ry_track(state:State, knobs:Knobs) -> State:
    kp, cos, sin, tan, *_ = knobs
    qx, px, qy, py = state
    pz = (kp - px**2 - py**2).sqrt()
    Qx = qx/cos/(1 - px/pz*tan)
    Px = px*cos + pz*sin 
    Qy = qy + py*qx*tan/(pz - px*tan)
    Py = py 
    return torch.stack([Qx, Px, Qy, Py])


def rz(state:State,
       wz:Tensor) -> State:
    """
    RZ translation (sign matches MADX)
    
    Parameters
    ----------
    state: State
        initial state
    wz: Tensor
        qz rotation angle
    dp: Tensor
        momentum deviation           
    
    Returns
    -------
    State
    
    """      
    knobs: Knobs = rz_knobs(wz)
    return rz_track(state, knobs)


def rz_knobs(wz:Tensor) -> Knobs:
    cos = wz.cos()
    sin = wz.sin()    
    return torch.stack([cos, sin])


def rz_track(state:State, knobs:Knobs) -> State:
    cos, sin, *_ = knobs
    qx, px, qy, py = state
    Qx = qx*cos + qy*sin
    Px = px*cos + py*sin
    Qy = qy*cos - qx*sin
    Py = py*cos - px*sin
    return torch.stack([Qx, Px, Qy, Py])
"""
Optimize
--------

"""
from typing import Callable
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

def adam(objective:Callable[[Tensor, Tensor, Tensor], Tensor],
         knobs:Tensor,
         dataloader:DataLoader, *,
         count:int=32,
         lr:float=0.005,
         betas:tuple[float, float]=(0.900, 0.999),
         epsilon:float=1.0E-9) -> Tensor:
    """
    Adam optimizer

    Parameters
    ----------
    objective: Callable[[Tensor, Tensor, Tensor], Tensor]
        batch objective
        (knobs, X, y) -> value
    knobs: Tensor
        initial knobs
    dataloader: Dataloader
        dataloader
    count: int, positive, default=32
        number of iterations
    lr: float, positive, default=0.01
        learning rate
    betas: tuple[float, float], positive, default=(0.900, 0.999)
        coefficients used for computing running averages of gradient and its square
    epsilon: float, positive, default=1.0E-9
        numerical stability epsilon

    Returns
    -------
    Tensor

    """
    b1, b2 = betas
    m1 = torch.zeros_like(knobs)
    m2 = torch.zeros_like(knobs)
    for i in range(count):
        f1 = 1 / (1 - b1 ** (i + 1))
        f2 = 1 / (1 - b2 ** (i + 1))
        for batch, (X, y) in enumerate(dataloader):
            gradient = torch.func.grad(objective)(knobs, X, y)
            m1 = b1 * m1 + (1.0 - b1) * gradient
            m2 = b2 * m2 + (1.0 - b2) * gradient ** 2
            m1_hat = m1 / f1
            m2_hat = m2 / f2
            knobs = knobs - lr * m1_hat / (torch.sqrt(m2_hat) + epsilon)
    return knobs


def newton(objective:Callable[[Tensor, Tensor, Tensor], Tensor],
           knobs:Tensor,
           dataloader:DataLoader, *,
           count:int=8,
           lr:float=1.0,
           epsilon:float=1.0E-6,
           jacobian:Optional[Callable]=None,) -> Tensor:
    """
    Newton optimizer

    Note, use to improve on a good initial guess

    Parameters
    ----------
    objective: Callable[[Tensor, Tensor, Tensor], Tensor]
        batch objective
        (knobs, X, y) -> value
    knobs: Tensor
        initial knobs
    dataloader: Dataloader
        dataloader
    count: int, positive, default=8
        number of iterations
    lr: float, positive, default=1.0
        learning rate
    epsilon: float, positive, default=1.0E-9
        numerical stability epsilon
    jacobian: Optional[Callable]
        torch.func.jacfwd or torch.func.jacrev (default)

    Returns
    -------
    Tensor

    """
    jacobian = torch.func.jacrev if jacobian is None else jacobian
    def wrapper(knobs, X, y):
        value = objective(knobs, X, y)
        return value, value
    for i in range(count):
        hessian_sum = torch.zeros((len(knobs), len(knobs)), dtype=knobs.dtype, device=knobs.device)
        gradient_sum = torch.zeros(len(knobs), dtype=knobs.dtype, device=knobs.device)
        length = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            hessian_batch, gradient_batch = jacobian(jacobian(wrapper, has_aux=True))(knobs, X, y)
            hessian_sum += hessian_batch
            gradient_sum += gradient_batch
        hessian = hessian_sum/length
        gradient = gradient_sum/length
        knobs = knobs - lr * torch.pinverse(hessian + epsilon * torch.eye(len(gradient))) @ gradient
    return knobs
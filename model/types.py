""" Data types (abstraction) """

from __future__ import annotations

from typing import TypeAlias
from typing import Callable
from torch import Tensor

State: TypeAlias = Tensor
Knobs: TypeAlias = Tensor
Mapping: TypeAlias = Callable[[State, Knobs], State]
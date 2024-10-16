from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy.typing as npt


@dataclass
class KdTreeNode:
    axis: int = -1
    location_point: npt.NDArray = field(default_factory=npt.NDArray)
    left_child: Optional[KdTreeNode] = None
    right_child: Optional[KdTreeNode] = None

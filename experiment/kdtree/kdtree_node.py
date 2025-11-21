from __future__ import annotations

from dataclasses import dataclass, field

import numpy.typing as npt


@dataclass
class KdTreeNode:
    axis: int = -1
    location_point: npt.NDArray = field(default_factory=npt.NDArray)
    left_child: KdTreeNode | None = None
    right_child: KdTreeNode | None = None

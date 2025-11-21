from __future__ import annotations

import numpy as np
import numpy.typing as npt
from kdtree_node import KdTreeNode


class KdTree:
    @staticmethod
    def create(points, depth: int = 0) -> KdTreeNode | None:
        # Accept any list type by converting NDArray.
        points_list = np.array(points)
        if len(points_list) == 0:
            return None

        # Convert to python list.
        points_list: list = points_list.tolist()

        # Assume that all elements are the same dimension.
        dim = len(points_list[0])

        # Decide which axis for splitting.
        axis = depth % dim

        # Sort with the specified axis value. So, key to be sorted is the axis indexed element.
        points_list.sort(key=lambda values: values[axis])

        median = int(len(points_list) / 2)

        # Convert back to NDArray.
        points_list = np.array(points_list)

        node = KdTreeNode(
            axis=axis,
            location_point=np.array(points_list[median]).reshape(1, 2),
            left_child=KdTree.create(points_list[0:median], depth + 1),
            right_child=KdTree.create(points_list[median + 1 :], depth + 1),
        )
        return node

    @staticmethod
    def search_nearest_neighbors(
        base_point: npt.NDArray,
        max_dist2: float,
        found_points: list,
        node: KdTreeNode | None,
    ):
        if node is None:
            return

        axis = node.axis
        delta = base_point[0][axis] - node.location_point[0][axis]

        dist = np.linalg.norm(base_point - node.location_point)
        dist2 = dist * dist

        if dist2 < max_dist2:
            found_points.append(node.location_point)

        if delta < 0:
            # Left
            KdTree.search_nearest_neighbors(
                base_point,
                max_dist2,
                found_points,
                node.left_child,
            )
            if delta * delta < max_dist2:
                KdTree.search_nearest_neighbors(
                    base_point,
                    max_dist2,
                    found_points,
                    node.right_child,
                )
        else:
            # Right
            KdTree.search_nearest_neighbors(
                base_point,
                max_dist2,
                found_points,
                node.right_child,
            )
            if delta * delta < max_dist2:
                KdTree.search_nearest_neighbors(
                    base_point,
                    max_dist2,
                    found_points,
                    node.left_child,
                )

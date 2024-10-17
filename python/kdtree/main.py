import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from kdtree import KdTree


def main():
    np.random.seed(19)

    points = np.random.random_sample((10, 2))
    target_point = np.array([0.5, 0.5]).reshape((1, 2))

    kdtree_root = KdTree.create(points)

    found_points = []
    max_dist = 0.3
    KdTree.search_nearest_neighbors(
        target_point, max_dist**2, found_points, kdtree_root
    )

    c = patches.Circle(
        (0.5, 0.5), radius=max_dist, edgecolor="green", facecolor="none", linewidth=1
    )
    ax = plt.axes()
    ax.add_patch(c)

    if True:
        plt.scatter([i[0] for i in points], [i[1] for i in points])
        plt.scatter(target_point[0][0], target_point[0][1])
        plt.scatter([i[0][0] for i in found_points], [i[0][1] for i in found_points])
        plt.axis("square")
        x, y = 1.1, 1.1
        plt.xlim(0, x)
        plt.ylim(0, y)
        plt.xticks(np.arange(0, x + 0.1, step=0.1))
        plt.yticks(np.arange(0, y + 0.1, step=0.1))
        plt.axhline(0, linewidth=2, color="gray")
        plt.axvline(0, linewidth=2, color="gray")
        plt.show()
    else:
        points = np.append(points, target_point, axis=0)

        colors = np.full((10, 3), [0, 255, 0])
        colors = np.append(colors, np.array([255, 0, 0]).reshape(1, 3), axis=0)

        rr.init("knn", spawn=True)
        rr.log("points", rr.Points2D(points, colors=colors, radii=0.02))

        while True:
            pass


if __name__ == "__main__":
    main()

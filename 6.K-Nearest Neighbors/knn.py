from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# 2-Dimensional dataset
# points = {
#     "blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
#     "red": [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]],
# }
# new_point = [3, 3]

# 3-Dimensional dataset
points = {
    "blue": [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
    "red": [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]],
}
new_point = [3, 3, 4]


def euclidean_distance(p, q):
    result = np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))
    return result


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.points = None

    # no exquisite fitting since points are themselves the model to be tested.
    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[: self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result


# custom model fitting
clf = KNN()
clf.fit(points)
print(clf.predict(new_point))

# 2-Dimensional implementation
# ax = plt.subplot()
# ax.grid(True, color="#323232")

# ax.set_facecolor("black")
# ax.figure.set_facecolor("#121212")
# ax.tick_params(axis="x", colors="white")
# ax.tick_params(axis="y", colors="white")

# [ax.scatter(point[0], point[1], color="#104DCA", s=60) for point in points["blue"]]
# [ax.scatter(point[0], point[1], color="#FF0000", s=60) for point in points["red"]]

# new_class = clf.predict(new_point)
# color = "#FF0000" if new_class == "red" else "#104DCA"
# ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)

# [
#     ax.plot(
#         [new_point[0], point[0]],
#         [new_point[1], point[1]],
#         color="#104DCA",
#         linestyle="--",
#         linewidth=1,
#     )
#     for point in points["blue"]
# ]
# [
#     ax.plot(
#         [new_point[0], point[0]],
#         [new_point[1], point[1]],
#         color="#EF6C35",
#         linestyle="--",
#         linewidth=1,
#     )
#     for point in points["red"]
# ]

# plt.show()

# 3-Dimensional implementation
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection="3d")
ax.grid(True, color="#323232")

ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", colors="orange")
ax.tick_params(axis="y", colors="orange")
ax.tick_params(axis="z", colors="orange")

[
    ax.scatter(point[0], point[1], point[2], color="#104DCA", s=60)
    for point in points["blue"]
]
[
    ax.scatter(point[0], point[1], point[2], color="#FF0000", s=60)
    for point in points["red"]
]

new_class = clf.predict(new_point)
color = "#FF0000" if new_class == "red" else "#104DCA"
ax.scatter(
    new_point[0], new_point[1], new_point[2], color=color, marker="*", s=200, zorder=100
)

[
    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        color="#104DCA",
        linestyle="--",
        linewidth=1,
    )
    for point in points["blue"]
]
[
    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        color="#EF6C35",
        linestyle="--",
        linewidth=1,
    )
    for point in points["red"]
]

plt.show()

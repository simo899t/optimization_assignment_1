import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class search_space():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.trajectory = []

    def add_obstacle(self, object: circular_object):
        self.obstacles.append(object)
        
    def straigt_path(self, start, goal, steps=20):
        dir = goal - start
        self.trajectory = [start + t * dir for t in np.linspace(0, 1, steps)]

    def pathlength(self, path: list):
        return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    def smoothness(self, path: list):
        # TODO
        pass

    def avoidance(self, path: list):
        # TODO
        pass

    def obj_func(self, x, lam, mu):
        path_length = self.pathlength(x)
        path_smoothness = lam * self.smoothness(x)
        obs_avoidance = mu * self.avoidance(x)
        return path_length + path_smoothness + obs_avoidance

    def plot(self):
        _, ax = plt.subplots()
        ax.set_aspect("equal")

        for i, obs in enumerate(self.obstacles):
            radius = obs.diameter / 2
            circle = patches.Circle(obs.center_point, radius, color="red", alpha=0.5)
            ax.add_patch(circle)
            ax.text(obs.center_point[0], obs.center_point[1], f"O{i+1}",
                    ha="center", va="center", fontsize=9, fontweight="bold")

        if self.trajectory:
            tx = [p[0] for p in self.trajectory]
            ty = [p[1] for p in self.trajectory]
            ax.plot(tx, ty, "k.-", linewidth=1.5, markersize=6, label="trajectory")

        ax.plot(*self.start, "go", markersize=8, label="start")
        ax.plot(*self.goal, "bs", markersize=8, label="goal")

        all_points = [self.start, self.goal] + [o.center_point for o in self.obstacles]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        margin = 2
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        ax.legend()
        plt.show()

class circular_object():
    def __init__(self, center: np.array, diameter: float):
        self.center_point = center
        self.diameter = diameter

    def __repr__(self):
        return f"circular_object(center={self.center_point}, diameter={self.diameter})"

    

x0= np.array([1, 1])
xn = np.array([100, 100])

search = search_space(x0, xn)

object1 = circular_object(np.array([40,30]), 20.0)
search.add_obstacle(object1)
object2 = circular_object(np.array([60,80]), 30.0)
search.add_obstacle(object2)

search.straigt_path(search.start, search.goal)
search.plot()


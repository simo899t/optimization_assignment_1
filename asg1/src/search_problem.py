import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy.optimize import approx_fprime
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
        
    def straigt_path(self, start, goal, steps=50):
        dir = goal - start
        return [start + t * dir for t in np.linspace(0, 1, steps)]

    def gradient_descent_path(self, lam=0, mu=0, alpha_step=0.01, alpha=0.1, tol=1e-6, max_iter=1000):
        x = np.array(self.straigt_path(self.start, self.goal)).flatten()    # <- initial straigth path
        convergence_vals = []

        for i in range(max_iter):
            val, g = self.obj_func(x, lam, mu, alpha)
            convergence_vals.append(val)

            x_new = x - alpha_step * g
            print(f"stepped towards {x_new}")

            x_new[:2] = self.start
            x_new[-2:] = self.goal

            if anp.linalg.norm(x_new-x) < tol:
                print(f"converged at iteation: {i+1}")
                break
            x = x_new
        self.trajectory = list(x.reshape(-1, 2))
        return x, convergence_vals

    def pathlength(self, x):
        return sum(anp.linalg.norm(x[i+1] - x[i])**2 for i in range(len(x) - 1))

    def smoothness(self, x):
        return sum(anp.linalg.norm(x[i+1] - 2*x[i] - x[i-1]) for i in range(1, len(x) - 1))

    def avoidance1(self, x):
        penalty = 0

        def d(xi, obs):
          return anp.linalg.norm(xi - obs.center_point)
    
        for xi in x:
            for obs in self.obstacles:
                r = obs.radius
                if d(xi, obs) > r:
                    penalty += 1 / (d(xi, obs) - r)**2
                else:
                    penalty += 1e9
        return penalty

    
    def avoidance2(self, x, alpha):
        penalty = 0  

        def d(xi, obs):
            return anp.linalg.norm(xi - obs.center_point)

        for xi in x:
            penalty += sum(anp.exp(-alpha * (
                d(xi, obs)**2 - obs.radius**2))
                    for obs in self.obstacles
            )
        return penalty

    def obj_func(self, x, lam, mu, alpha=None):
        def f(x):
            traj = x.reshape(-1, 2)
            av = self.avoidance2(traj, alpha)
            # av = self.avoidance1(traj)
            return self.pathlength(traj) + lam * self.smoothness(traj) + mu * av
        
        # return f(x), grad(f)(x) # if avoidance2, else use aprox
        return f(x), approx_fprime(x, f, 1e-5) # <- maybe nanograd https://github.com/rasmusbergpalm/nanograd/tree/main

    def plot_convergence(self, vals: list):
        plt.plot(vals)
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")


    def plot(self):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")

        for i, obs in enumerate(self.obstacles):
            circle = patches.Circle(obs.center_point, obs.radius, color="red", alpha=0.5)
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
    def __init__(self, center: np.array, radius: float):
        self.center_point = center
        self.radius = radius

    def __repr__(self):
        return f"circular_object(c={self.center_point}, d={self.radius})"

    

x0= np.array([1, 1])
xn = np.array([100, 100])

search = search_space(x0, xn)

object1 = circular_object(np.array([40,30]), 10.0)
search.add_obstacle(object1)
object2 = circular_object(np.array([50,55]), 7.0)
search.add_obstacle(object2)
object3 = circular_object(np.array([60,80]), 15.0)
search.add_obstacle(object3)

# search.straigt_path(search.start, search.goal)
search.straigt_path(search.start, search.goal)
x, history = search.gradient_descent_path(lam=0.0, mu=10, alpha_step=1e-3)


search.plot_convergence(history)
search.plot()



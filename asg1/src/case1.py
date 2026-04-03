import autograd.numpy as anp
from autograd import grad, hessian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class search_space():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.trajectory = []

    def add_obstacle(self, object):
        self.obstacles.append(object)
        
    def straigt_path(self, start, goal, steps):
        dir = goal - start
        x = [start + t * dir for t in anp.linspace(0, 1, steps)]
        self.trajectory = list(anp.array(x).flatten().reshape(-1, 2))
        return x

    def gradient_descent_path(self, eta=1, lam=0, mu=0, alpha_step=0.01, alpha=0.1, max_iter=1000, steps = 100):
        x = anp.array(self.straigt_path(self.start, self.goal, steps)).flatten()    # <- initial straigth path
        convergence_vals = []
        for i in range(max_iter):
            val, g, h= self.obj_func(x, eta, lam, mu, alpha)
            convergence_vals.append(val)
            # print(f"x = {val}")
            # print(f"first order derivative = {g}")
            # print(f"second order derivative = {h}")
            g = anp.clip(g, -1e4, 1e4) # <- ensure that g in {1e-4, 1e4}
            x_new = x - alpha_step * g
            if i % 10 == 0:
                # print(f"new trajectory {x_new.reshape(-1, 2)}", end='\r', flush=True)
                pass
            print(i)

            x_new[:2] = self.start
            x_new[-2:] = self.goal

            x = x_new
        self.trajectory = list(x.reshape(-1, 2))
        return x, convergence_vals

    def nelder_mead():
        # TODO <- cool algorithm for comparison
        pass

    def third_method_maybe():
        # TODO <- cool algorithm for comparison
        pass

    def pathlength(self, x):
        epsilon = 1e-12
        pathlength = sum(anp.sqrt(anp.dot(x[i+1] - x[i], x[i+1] - x[i]) + epsilon)**2 for i in range(len(x)-1))
        # pathlength = sum(anp.linalg.norm(x[i+1] - x[i])**2 for i in range(len(x) - 1))
        # print(pathlength)
        return pathlength

    def smoothness(self, x):
        epsilon = 1e-12
        smoothness = sum(anp.sqrt(anp.dot((x[i+1] - 2*x[i] + x[i-1]),(x[i+1] - 2*x[i] + x[i-1])) + epsilon)**2 for i in range(1, len(x) - 1))
        # smoothness = sum(anp.linalg.norm((x[i+1] - 2*x[i] + x[i-1]))**2 for i in range(1, len(x) - 1))
        # print(smoothness)
        return smoothness

    def avoidance1(self, x):
        penalty = 0

        def d(xi, obs):
            epsilon = 1e-12
            return anp.sqrt(anp.dot(xi - obs.center_point, xi - obs.center_point) + epsilon)
    
        for xi in x:
            for obs in self.obstacles:
                r = obs.radius
                if d(xi, obs) > r:
                    penalty += 1 / (d(xi, obs) - r)**2
                else:
                    penalty += 1e9
        # print(penalty)
        return penalty

    
    def avoidance2(self, x, alpha):
        penalty = 0  

        def d(xi, obs):
            epsilon = 1e-12
            # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            return anp.sqrt(anp.dot(xi - obs.center_point, xi - obs.center_point) + epsilon) #anp.norm men finite 

        for xi in x:
            penalty += sum(anp.exp(-alpha * (
                d(xi, obs)**2 - obs.radius**2))
                    for obs in self.obstacles
            )
        return penalty

    def obj_func(self, x, eta=1, lam=0, mu=0, alpha=0.1):
        def f(x):
            x = x.reshape(-1, 2)
            av = self.avoidance2(x, alpha)
            # av = self.avoidance1(x)
            return eta * self.pathlength(x) + lam * self.smoothness(x) + mu * av
        
        return f(x), grad(f)(x), hessian(f)(x)

    def plot_convergence(self, vals: list):
        plt.figure()
        plt.plot(vals)
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.show()

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
    def __init__(self, center: anp.array, radius: float):
        self.center_point = center
        self.radius = radius

    def __repr__(self):
        return f"circular_object(c={self.center_point}, d={self.radius})"


x0= anp.array([1, 1])
xn = anp.array([100, 100])

search = search_space(x0, xn)



def random_placement(n_objects = 3):
    def overlaps(new_center, new_radius, existing):
        for obs in existing:
            dist = ((new_center[0] - obs.center_point[0])**2 + (new_center[1] - obs.center_point[1])**2)**0.5
            if dist < new_radius + obs.radius:
                return True
        return False

    print(f"=== creating {n_objects} objects ===")

    dir_vec = xn - x0
    length = (dir_vec[0]**2 + dir_vec[1]**2)**0.5
    unit = dir_vec / length
    perp = anp.array([-unit[1], unit[0]])
    
    for _ in range(n_objects):
        radius = random.randint(10,15)
        while True:
            t = random.uniform(0.15, 0.85)
            offset = random.uniform(-20, 20)
            coord = x0 + t * dir_vec + offset * perp
            coord = [float(coord[0]), float(coord[1])]
            if not overlaps(coord, radius, search.obstacles):
                break
        search.add_obstacle(circular_object(anp.array(coord), radius))




# replicate
search.add_obstacle(circular_object(anp.array([50,50]), 30))

n_objects = 5
steps = 100
max_iterations = 10

# random_placement(n_objects)
print(f"=== searching on {steps} steps ===")
x, history = search.gradient_descent_path(eta = 1,lam=50, mu=10, alpha_step=1e-3, max_iter=max_iterations, steps=steps)
# print(x)
# print(history)

# search.plot_convergence(history)
search.plot()



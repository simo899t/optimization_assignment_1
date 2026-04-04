import autograd.numpy as anp
import numpy as np
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
        # return x

    def gradient_descent_path(self, eta, lam=0, mu=0, alpha_step=0.01, alpha=0.1, max_iter=1000): # <- 1st order
        x = anp.array(self.trajectory).flatten()    # <- initial straigth path
        convergence_vals = []
        i = 0
        last_vals = [0,0]
        val = 1
        # and not all(t == val for t in last_vals)
        while i < max_iter:
            val, g = self.obj_func(x, eta, lam, mu, alpha, order=1)
            convergence_vals.append(val)
            # print(f"x = {val}")
            # print(f"first order derivative = {g}")
            # print(f"second order derivative = {h}")
            g = anp.clip(g, -1e2, 1e2) # <- ensure that g in {1e-4, 1e4}
            x_new = x - alpha_step * g
            if i % 10 == 0:
                # print(f"new trajectory {x_new.reshape(-1, 2)}", end='\r', flush=True)
                pass

            print(f"[{i},{val}]")
            x_new[:2] = self.start
            x_new[-2:] = self.goal

            x = x_new
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return x, convergence_vals

    def newton_method(self, lam, mu, alpha=0.1, max_iter = 1000, step_size = 0.001): # <- 2nd order
        x = anp.array(self.trajectory).flatten()
        convergence_vals = []
        best_solution = [[np.inf],[]]

        for i in range(max_iter):
            val, g, h = self.obj_func(x, 1, lam, mu, alpha, order=2)
            convergence_vals.append(val)
            g = anp.clip(g, -1e3, 1) # <- ensure that g in {1e-4, 1e4}
            # h = anp.clip(h, -1e3, 1) # <- ensure that g in {1e-4, 1e4}
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")
            # print(f"eigenvalues: {np.linalg.eigvals(h)}")
            # print(f"Gradient: {g}")
            # print(f"Hessian: {h}")
            # print(f"Inverted Hessian: {np.linalg.inv(h)}")
            # print(f"x: {x.shape}    delta: {delta.shape}")
            delta = np.linalg.inv(h) @ g
            x_new = x - step_size * delta
            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1], convergence_vals

    def momentum_descent(self, lam, mu, alpha=0.1, beta=0.001, max_iter = 100, step_size=0.001): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        convergence_vals = []
        best_solution = [[np.inf],[]]

        for i in range(max_iter):
            val, g = self.obj_func(x, 1, lam, mu, alpha, order=1)
            convergence_vals.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
            momentum = beta * momentum - step_size * g
            x_new = x + momentum


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1], convergence_vals


    def nesterov_momentum_descent(self, lam, mu, alpha=0.1, beta=0.001, max_iter = 100, step_size=0.001): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        convergence_vals = []
        best_solution = [[np.inf],[]]

        for i in range(max_iter):
            # val, g = self.obj_func(x, 1, lam, mu, alpha, order=1)
            val, g_lookahead = self.obj_func(x+beta*momentum, 1, lam, mu, alpha, order=1)
            convergence_vals.append(val)
            # g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
            momentum = beta * momentum - step_size * g_lookahead
            x_new = x + momentum


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1], convergence_vals


    def adam(self, lam, mu, alpha=0.1, beta=0.01, max_iter = 100, step_size=0.01, gamma_v=0.9, gamma_s=0.999, eps=1e-8): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        squared_gradients = np.zeros(len(x))
        convergence_vals = []
        best_solution = [[np.inf],[]]

        for i in range(max_iter):
            val, g = self.obj_func(x, 1, lam, mu, alpha, order=1)
            convergence_vals.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
            momentum = beta * momentum - step_size * g
            squared_gradients = gamma_s * squared_gradients + (1 - gamma_s) * np.multiply(g,g)
            corrected_decaying_mom = momentum / (1 - gamma_v)
            corrected_squared_gradients = squared_gradients / (1 - gamma_s)
            x_new = x + (step_size / (eps + np.sqrt(corrected_squared_gradients))) * corrected_decaying_mom


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1], convergence_vals


    """
    def nelder_mead(self, ): # <- 0-order
        #def nelder_mead(f, S, eps, max_iterations, alpha=1.0, beta=2.0, gamma=0.5):
        delta = float("inf")
        y_arr = np.array([f(x) for x in S])
        simplex_history = [S.copy()]
        iterations=0
        while delta > eps and iterations <= max_iterations:
            iterations+=1
            # Sort by objective values (lowest to highest)
            p = np.argsort(y_arr)
            S, y_arr = S[p], y_arr[p]
            xl, yl = S[0], y_arr[0] # Lowest
            xh, yh = S[-1], y_arr[-1] # Highest
            xs, ys = S[-2], y_arr[-2] # Second-highest
            xm = np.mean(S[:-1], axis=0) # Centroid
            # Reflection
            xr = xm + alpha * (xm - xh)
            yr = f(xr)
            if yr < yl:
                # Expansion
                xe = xm + beta * (xr - xm)
                ye = f(xe)
                S[-1], y_arr[-1] = (xe, ye) if ye < yr else (xr, yr)
            elif yr >= ys:
                if yr < yh:
                    xh, yh = xr, yr
                    S[-1], y_arr[-1] = xr, yr
                # Contraction
                xc = xm + gamma * (xh - xm)
                yc = f(xc)
                if yc > yh:
                    # Shrink
                    for i in range(1, len(S)):
                        S[i] = (S[i] + xl) / 2
                        y_arr[i] = f(S[i])
                else:
                    S[-1], y_arr[-1] = xc, yc
            else:
                S[-1], y_arr[-1] = xr, yr
            simplex_history.append(S.copy())
            delta = np.std(y_arr, ddof=0)
        return S[np.argmin(y_arr)], simplex_history
        """

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

    def obj_func(self, x, eta, lam=0, mu=0, alpha=0.1, order=1):
        def f(x):
            x = x.reshape(-1, 2)
            av = self.avoidance2(x, alpha)
            # av = self.avoidance1(x)
            return eta * self.pathlength(x) + lam * self.smoothness(x) + mu * av
        
        match order:
            case 0:
                return f(x)
            case 1:
                return f(x), grad(f)(x)
            case 2:
                return f(x), grad(f)(x), hessian(f)(x)

    def random_placements(self, n_objects = 3):
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



## === CONFIG === ##

steps = 100
max_iterations = 1000

x0= anp.array([1, 1])
xn = anp.array([100, 100])
search = search_space(x0, xn)

# replicate our issue
# search.add_obstacle(circular_object(anp.array([50,50]), 30))

# basic test
search.add_obstacle(circular_object(anp.array([55,45]), 20))

#n_objects = 5
#search.random_placements(n_objects)

search.straigt_path(search.start, search.goal, steps)

## === OPTIMIZATION ALGORITHMS === ##

print(f"=== searching on {steps} steps ===")

# x, history = search.gradient_descent_path(eta = 5, lam=1, mu=10, alpha_step=1e-4, max_iter=max_iterations)

# x, history = search.newton_method(lam=10, mu=10, max_iter=max_iterations)

# x, history = search.momentum_descent(lam=10, mu=10, max_iter=max_iterations)
# test -> [999,316.5307994083728] : Best solution: 316.5307994083728


# x, history = search.nesterov_momentum_descent(lam=10, mu=10, max_iter=max_iterations)


x, history = search.adam(lam=10, mu=10, max_iter=max_iterations)


search.plot_convergence(history)
search.plot()



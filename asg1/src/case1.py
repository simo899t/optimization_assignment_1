import autograd.numpy as anp
import time
import numpy as np
from autograd import grad, hessian, jacobian
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class search_space():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.trajectory = []
        self.convergence = []

    def add_obstacle(self, object):
        self.obstacles.append(object)
        
    def straigt_path(self, start, goal, steps):
        dir = goal - start
        x = [start + t * dir for t in anp.linspace(0, 1, steps)]
        self.trajectory = list(anp.array(x).flatten().reshape(-1, 2))
        # return x

    def gradient_descent_path(self, eta=1, obj_alpha=0.05, lam=2, mu=5, alpha_step=0.01, max_iter=1000, sec = 60): # <- 1st order
        t0 = time.perf_counter()
        self.convergence = []
        x = anp.array(self.trajectory).flatten()    # <- initial trajectory
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            val, g = self.obj_func(x, eta, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            
            g = anp.clip(g, -1e5, 1e5) # clip gradient
            x_new = x - alpha_step * g

            x_new[:2] = self.start
            x_new[-2:] = self.goal

            x = x_new
            print(f"[{i},{val}]")
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return x
         
    def strong_bracketing(self, x,  f, nabla, d, alpha=1, beta=1e-4, sigma=1):
        y0, g0, y_prev, alpha_prev = f(x), nabla(x) @ d, None, 0

        # bracket phase
        while True:
            y = f(x + alpha*d)
            if y > y0 + beta*alpha*g0 or (y_prev is not None and y >= y_prev):
                alpha_lo, alpha_hi = alpha_prev, alpha
                break
            dir_gradient = nabla(x + alpha*d) @ d
            if abs(dir_gradient) <= -sigma * g0:
                return alpha
            elif dir_gradient >= 0:
                alpha_lo, alpha_hi = alpha, alpha_prev
                break
            else:
                pass
            y_prev, alpha_prev, alpha = y, alpha, 2 * alpha

        # zoom phase
        ylo = f(x + alpha_lo*d)
        while abs(alpha_hi - alpha_lo) > 1e-10:
            alpha = (alpha_lo + alpha_hi)/2
            y = f(x + alpha*d)
            if y > y0 + beta*alpha*g0 or y >= ylo:
                alpha_hi = alpha
            else:
                g = nabla(x + alpha*d) @ d
                if abs(g) <= -sigma*g0:
                    return alpha
                elif g*(alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                else:
                    pass
                alpha_lo = alpha
    
        return alpha_lo
    
    def gradient_descent_sb(self, alpha=1, obj_alpha=1, 
                                beta=1e-4, sigma=1, eta=1, lam=1, mu=1, max_iter=1000, sec = 60):
        t0 = time.perf_counter()
    
        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)

        f = lambda v: obj(v)
        nabla = grad(obj)
        self.convergence = []
        x = anp.array(self.trajectory).flatten()    # <- initial straigth path
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            g = nabla(x)
            self.convergence.append(f(x))
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            dir = -g
            alpha_step = self.strong_bracketing(x, f, nabla, d=dir, alpha=alpha, beta=beta, sigma=sigma)
            x_new = x - alpha_step * g

            x_new[:2] = self.start
            x_new[-2:] = self.goal
        
            x = x_new
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return x
        
# <<<<<<< HEAD
    def newton_method(self, lam, mu, obj_alpha=0.1, alpha=1, max_iter = 1000, beta=1e-4, sigma=1, eta=10, sec = 60): # <- 2nd order
        x = anp.array(self.trajectory).flatten()
        self.convergence = []
        best_solution = [[np.inf],[]]

        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)
        f = lambda v: obj(v)
        nabla = grad(obj)

        for i in range(max_iter):
            val, g, h = self.obj_func(x, eta, lam, mu, obj_alpha, order=2)
            self.convergence.append(val)
            eig, eig_vec = np.linalg.eigh(h)
            new_h = eig_vec @ (np.diag(np.full(np.size(eig),np.absolute(eig))))  @ np.transpose(eig_vec)
            #new_h = eig_vec @ np.diag(np.maximum(np.abs(eig), 1e-6)) @ np.transpose(eig_vec)
            p_k = np.linalg.solve(new_h, (-g))
            alpha_step = self.strong_bracketing(x, f, nabla, d=p_k, alpha=alpha, beta=beta, sigma=sigma)
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")


            x_new = x + alpha_step * p_k

            x_new[:2] = self.start
            x_new[-2:] = self.goal
# =======
    def newton_method(self, lam, mu, obj_alpha=0.1, alpha=1, max_iter = 1000, beta=1e-4, sigma=1, eta=10, sec = 60): # <- 2nd order
        t0 = time.perf_counter()
        x = anp.array(self.trajectory).flatten()
        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)
        f = lambda v: obj(v)
        nabla = grad(obj)
        self.convergence = []
        best_solution = [[np.inf],[]]
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)

            val, g, h = self.obj_func(x, eta, lam, mu, obj_alpha, order=2)
            self.convergence.append(val)
            eig, eig_vec = np.linalg.eigh(h)
            new_h = eig_vec @ (np.diag(np.full(np.size(eig),np.absolute(eig))))  @ np.transpose(eig_vec)
            #new_h = eig_vec @ np.diag(np.maximum(np.abs(eig), 1e-6)) @ np.transpose(eig_vec)
            p_k = np.linalg.solve(new_h, (-g))
            alpha_step = self.strong_bracketing(x, f, nabla, d=p_k, alpha=alpha, beta=beta, sigma=sigma)
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")


            x_new = x + alpha_step * p_k

            x_new[:2] = self.start
            x_new[-2:] = self.goal


            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        # return best_solution[1]
        return x

    def momentum_descent(self, lam, mu, obj_alpha=0.01, beta=0.01, max_iter = 100, alpha_step=1, sec = 60): # <- 1st order
        t0 = time.perf_counter()
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        best_solution = [[np.inf],[]]
        self.convergence = []
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            momentum = beta * momentum - alpha_step * g
            x_new = x + momentum
            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]


    def nesterov_momentum_descent(self, lam, mu, obj_alpha=0.1, beta=0.001, max_iter = 100, alpha_step=0.001, sec = 60): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        t0 = time.perf_counter()
        momentum = np.zeros(len(x))
        best_solution = [[np.inf],[]]
        self.convergence = []
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            val, g_lookahead = self.obj_func(x+beta*momentum, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            # g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            momentum = beta * momentum - alpha_step * g_lookahead
            x_new = x + momentum
            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]


    def adam(self, lam, mu, obj_alpha=0.1, beta=0.01, max_iter = 100, alpha_step=0.01, gamma_v=0.9, gamma_s=0.999, eps=1e-8, sec = 60): # <- 1st order
        t0 = time.perf_counter()
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        squared_gradients = np.zeros(len(x))
        self.convergence = []
        best_solution = [[np.inf],[]]
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            momentum = beta * momentum - alpha_step * g
            squared_gradients = gamma_s * squared_gradients + (1 - gamma_s) * np.multiply(g,g)
            corrected_decaying_mom = momentum / (1 - gamma_v)
            corrected_squared_gradients = squared_gradients / (1 - gamma_s)
            x_new = x + (alpha_step / (eps + np.sqrt(corrected_squared_gradients))) * corrected_decaying_mom


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]
        
    def nelder_mead(self, eps, alpha=1.0, beta=2.0, eta=1, gamma=0.5, lam=1, mu=1, obj_alpha=0.1, max_iter=1000, sec = 60): # <- 0-order
        t0 = time.perf_counter()
        x = anp.array(self.trajectory).flatten()
        n = len(x)

        def f(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)

        # Build initial simplex: perturb only interior waypoint coordinates.
        # Skip start ([:2]) and end ([-2:]) — fixed boundary conditions.
        step = 5
        S = np.zeros((n + 1, n))
        S[0] = x
        for j in range(n):
            vertex = x.copy()
            if 2 <= j < n - 2:
                vertex[j] += step
            S[j + 1] = vertex
        delta = float("inf")
        y_arr = np.array([f(v) for v in S])
        self.convergence = [f(x)]
        i = 0

        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (delta <= eps and i >= max_iter)
            i += 1
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
                    for j in range(1, len(S)):
                        S[j] = (S[j] + xl) / 2
                        y_arr[j] = f(S[j])
                else:
                    S[-1], y_arr[-1] = xc, yc
            else:
                S[-1], y_arr[-1] = xr, yr
            self.convergence.append(np.min(y_arr))
            delta = np.std(y_arr, ddof=0)
        best = S[np.argmin(y_arr)]
        self.trajectory = list(best.reshape(-1, 2))
        return best

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
                    penalty += np.inf
        return penalty

    
    def avoidance2(self, x, alpha):
        penalty = 0  

        def d(xi, obs):
            epsilon = 1e-12
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
            # av = self.avoidance2(x, alpha)
            av = self.avoidance1(x)
            return eta * self.pathlength(x) + lam * self.smoothness(x) + mu * av
        
        match order:
            case 0:
                return f(x)
            case 1:
                return f(x), grad(f)(x)
            case 2:
                return f(x), grad(f)(x), jacobian(grad(f))(x)

    def plot_convergence(self, vals: list):
        plt.figure()
        plt.plot(vals)
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.show()

    def plot_single_dingle(self, save_path=None):
        fig, (ax) = plt.subplots(figsize=(10, 10))
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
       
        # ax.plot(tx, ty, "k.-", linewidth=1.5, markersize=6, label="trajectory")
        ax.plot(*self.start, "go", markersize=8, label="start")
        ax.plot(*self.goal, "bs", markersize=8, label="goal")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def plot_mult(self, iterations, steps, save_path=None, trajectories=None):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(15, 3.5))

        fig_list = [ax1, ax2, ax3, ax4, ax5]


        for ax in fig_list:
            for i, obs in enumerate(self.obstacles):
                circle = patches.Circle(obs.center_point, obs.radius, color="red", alpha=0.5)
                ax.add_patch(circle)
                ax.text(obs.center_point[0], obs.center_point[1], f"O{i+1}",
                        ha="center", va="center", fontsize=9, fontweight="bold")

        for i, ax in enumerate(fig_list):
            if trajectories and i < len(trajectories) and trajectories[i]:
                tx = [p[0] for p in trajectories[i]]
                ty = [p[1] for p in trajectories[i]]
                print(i)
                ax.plot(tx, ty, "k.-", linewidth=1.5, markersize=6, label="trajectory")
                if iterations[i] <= 0:
                    ax.set_title(f'initial')
                else:
                    ax.set_title(f'{iterations[i]} iter, {steps[i]} steps')
                ax.plot(*self.start, "go", markersize=8, label="start")
                ax.plot(*self.goal, "bs", markersize=8, label="goal")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def plot_compare_time(self, trajectories=None, title: str="no title", titles: list[str]=["no titles"]):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(15, 3.5))
        fig.suptitle(title, fontsize=14)

        fig_list = [ax1, ax2, ax3, ax4, ax5]


        for ax in fig_list:
            for i, obs in enumerate(self.obstacles):
                circle = patches.Circle(obs.center_point, obs.radius, color="red", alpha=0.5)
                ax.add_patch(circle)
                ax.text(obs.center_point[0], obs.center_point[1], f"O{i+1}",
                        ha="center", va="center", fontsize=9, fontweight="bold")

        for i, ax in enumerate(fig_list):
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            if trajectories and i < len(trajectories) and trajectories[i]:
                tx = [p[0] for p in trajectories[i]]
                ty = [p[1] for p in trajectories[i]]
                ax.plot(tx, ty, "k.-", linewidth=1.5, markersize=6, label="trajectory")
                ax.set_title(titles[i])
                ax.plot(*self.start, "go", markersize=8, label="start")
                ax.plot(*self.goal, "bs", markersize=8, label="goal")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()
        plt.show()


class circular_object():
    def __init__(self, center: anp.array, radius: float):
        self.center_point = center
        self.radius = radius

    def __repr__(self):
        return f"circular_object(c={self.center_point}, d={self.radius})"

## === OPTIMIZATION ALGORITHMS & CONFIGURATION === ##


def initialize(start, goal, test=1, steps = 100):
    x0= anp.array(start)
    xn = anp.array(goal)
    search = search_space(x0, xn)
    search.straigt_path(search.start, search.goal, steps)
    match test:
        case 1:
            # basic minimum test
            search.add_obstacle(circular_object(anp.array([40,30]), 10))
            search.add_obstacle(circular_object(anp.array([60,70]), 20))
        case 2:
            # replicate problem
            search.add_obstacle(circular_object(anp.array([50,50]), 30))
    return search

### OBJECTIVE FUNCTION PARAMETERS
obj_alpha = 0.02
lam = 2
mu = 5

def initialize_straight_line(start, goal, test=1, steps= 100):
    search = initialize(start, goal, test, steps)
    return search

def basic_GD(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting basic GD at {sec} seconds")
    # best configs -> steps=100, iterations=1000, lam=50, mu=10, alpha=1, obj_alpha=1e-3
    search = initialize(start, goal, test, steps)
    search.gradient_descent_path(eta = 1, lam=lam, mu=mu, obj_alpha=obj_alpha, alpha_step=1e-2, max_iter=max_iter, sec = sec)
    return search

def GD_with_SB(start, goal, test=1, steps= 100, max_iter=200, sec = 60):
    print(f"Starting GDSB at {sec} seconds")
    #  best configs -> steps=100, iterations=200, lam=2, mu=7, alpha=1, obj_alpha=0.005, beta=1e-4
    search = initialize(start, goal, test, steps)
    search.gradient_descent_sb(alpha=1e-3, obj_alpha=obj_alpha, beta=1e-4, sigma=0.9, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def GD_with_nesterov_momentum(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting GD_with_nesterov_momentum at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.nesterov_momentum_descent(obj_alpha=obj_alpha, beta=1e-3, alpha_step=1e-3, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def GD_with_momemtum(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting GD_with_momemtum at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.momentum_descent(alpha_step=1e-3, obj_alpha=obj_alpha, beta=1e-3, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search
def GD_adam(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting GD_adam at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.adam(obj_alpha=obj_alpha, alpha_step=1, beta=0.1,  lam=lam, mu=mu, max_iter=max_iter, gamma_v=0.9, gamma_s=0.999, eps=1e-5) 
    return search

def Newton_method(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting Newton_method at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.newton_method(alpha=obj_alpha, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def Nelder_Mead_Method(start, goal, test=1, steps= 100, max_iter=1000, sec = 60):
    print(f"Starting Nelder_Mead_Method at {sec} seconds")
    # best configs -> steps: 15, iterations: 5000, eps: 1e-6, lam: 1, mu: 10
    # best configs2 -> steps: 100, iterations: 500000 (500k), eps: 1e-6, eps: 1e-6, lam: 1, mu: 10
    search = initialize(start, goal, test, steps)
    search.nelder_mead(eps=1e-6, obj_alpha=obj_alpha, lam=1, mu=10, max_iter=max_iter, sec = sec)
    return search

def main():
    start, goal = [1,1], [100,100]
    
    plot_trajectories = []

    # iterations = [0,500,1500,5000,10000]
    # steps = [50, 50, 50, 50, 50]
    # iterations = [0,100,100,100,100]
    # steps = [1, 10, 50, 100, 200]
    iterations = [100, 1000, 5000]
    steps = [50, 50, 50, 50]
    # iterations = [0, 1, 5, 10, 50]
    # steps = [30, 30, 30, 30, 30]

    title = "Comparing Gradient Descent with Nesterov Momentum"
    

    seconds = 10

    test_all_time = False

    if not test_all_time:
        search = initialize_straight_line(start, goal, test=1, steps=steps[0])
        plot_trajectories.append(search.trajectory)

    i = 0
    while (i < len(iterations) and not test_all_time):
        if iterations[i] > 0:
            search = basic_GD(start, goal, test=1, steps=steps[i], max_iter=iterations[i], sec = seconds)
            # search = GD_with_SB(start, goal, test=1, steps= steps[i], max_iter=iterations[i], sec = seconds)
            # search = GD_with_nesterov_momentum(start, goal, test=1, steps= steps[i], max_iter=iterations[i], sec = seconds)
            # search = GD_with_momemtum(start, goal, test=1, steps= steps[i], max_iter=iterations[i], sec = seconds)
            #search = Newton_method(start, goal, test=1, steps= steps[i], max_iter=iterations[i])
            #search = Nelder_Mead_Method(start = start, goal = goal, steps = steps[i], max_iter=iterations[i], sec = seconds)
            plot_trajectories.append(search.trajectory)
            i += 1

    if not test_all_time:
        search.plot_mult(trajectories=plot_trajectories, iterations = iterations, steps=steps, title=title)
        plot_trajectories = []

    seconds = 20
    steps = 100
    title = "120 seconds"
    titles = ["basic_GD [100s]", "GDSB [100s]", "GS_nest_mom [100s]", "GD_mom [100s]", "Nelder_Mead - [100s]s"]

    if test_all_time:
        search = basic_GD(start, goal, test=1, steps=steps, sec = seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_SB(start, goal, test=1, steps= steps, sec = seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_nesterov_momentum(start, goal, test=1, steps= steps, sec = seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_momemtum(start, goal, test=1, steps= steps, sec = seconds)
        plot_trajectories.append(search.trajectory)
        # search = Newton_method(start, goal, test=1, steps= 15, sec = seconds)
        # plot_trajectories.append(search.trajectory)
        # print(plot_trajectories)
        search = Nelder_Mead_Method(start = start, goal = goal, steps = steps, sec = seconds)
        plot_trajectories.append(search.trajectory)
    
    search.plot_compare_time(trajectories=plot_trajectories, title=title, titles = titles)

if __name__ == "__main__":
    # print(np.transpose(np.diag(np.full(np.size(test),1))))
    # print(np.transpose(np.diag(np.full(np.size(test),test))))
    # print(np.transpose(np.absolute(test)) @ np.diag(np.full(np.size(test),1)))
    main()



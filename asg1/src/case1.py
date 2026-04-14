import autograd.numpy as anp
import time
import numpy as np
from autograd import grad, hessian
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

    def gradient_descent_path(self, eta, lam=0, mu=0, alpha_step=0.01, obj_alpha=0.1, max_iter=1000): # <- 1st order
        self.convergence = []
        x = anp.array(self.trajectory).flatten()    # <- initial trajectory
        i = 0
        while i < max_iter:
            print(i)
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
                                beta=1e-4, sigma=1, eta=1, lam=1, mu=1, max_iter=1000):
        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)

        f = lambda v: obj(v)
        nabla = grad(obj)
        self.convergence = []
        x = anp.array(self.trajectory).flatten()    # <- initial straigth path
        for i in range(max_iter):
            print(i)
            g = nabla(x)
            self.convergence.append(f(x))
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            dir = -g
            alpha_step = self.strong_bracketing(x, f, nabla, d=dir, alpha=alpha, beta=beta, sigma=sigma)
            x_new = x - alpha_step * g
            if i % 10 == 0:
                # print(f"new trajectory {x_new.reshape(-1, 2)}", end='\r', flush=True)
                pass
            print(i)

            x_new[:2] = self.start
            x_new[-2:] = self.goal

            x = x_new
        self.trajectory = list(x.reshape(-1, 2))
        return x
        
    def newton_method(self, lam, mu, obj_alpha=0.1, alpha=1, max_iter = 1000, beta=1e-4, sigma=1, eta=10): # <- 2nd order
        x = anp.array(self.trajectory).flatten()
        self.convergence = []
        best_solution = [[np.inf],[]]

        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)
        f = lambda v: obj(v)
        nabla = grad(obj)

        for i in range(max_iter):
            print(i)
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

        self.trajectory = list(x.reshape(-1, 2))
        # return best_solution[1]
        return x

    def momentum_descent(self, lam, mu, obj_alpha=0.1, beta=0.001, max_iter = 100, alpha_step=0.001): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        best_solution = [[np.inf],[]]
        self.convergence = []

        for i in range(max_iter):
            print(i)
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
            momentum = beta * momentum - alpha_step * g
            x_new = x + momentum


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]


    def nesterov_momentum_descent(self, lam, mu, obj_alpha=0.1, beta=0.001, max_iter = 100, alpha_step=0.001): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        best_solution = [[np.inf],[]]
        self.convergence = []
        for i in range(max_iter):
            print(i)
            # val, g = self.obj_func(x, 1, lam, mu, alpha, order=1)
            val, g_lookahead = self.obj_func(x+beta*momentum, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            # g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
            momentum = beta * momentum - alpha_step * g_lookahead
            x_new = x + momentum


            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]


    def adam(self, lam, mu, obj_alpha=0.1, beta=0.01, max_iter = 100, alpha_step=0.01, gamma_v=0.9, gamma_s=0.999, eps=1e-8): # <- 1st order
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        squared_gradients = np.zeros(len(x))
        self.convergence = []
        best_solution = [[np.inf],[]]

        for i in range(max_iter):
            print(i)
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-4, 1e4}
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
            print(f"[{i},{val}] : Best solution: {best_solution[0][0]}")

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]
        
    def nelder_mead(self, eps, alpha=1.0, beta=2.0, eta=1, gamma=0.5, lam=1, mu=1, obj_alpha=0.1, max_iter=1000): # <- 0-order

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
        while delta > eps and i <= max_iter:
            print(i)
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
            # history.append(f(y_arr.reshape(-1,2)))
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
        # print(penalty)
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
                return f(x), grad(f)(x), hessian(f)(x)

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

def initialize_straight_line(start, goal, test=1, steps= 100):
    search = initialize(start, goal, test, steps)
    return search

def basic_GD(start, goal, test=1, steps= 100, max_iter=1000):
    # best configs -> steps=100, iterations=1000, lam=50, mu=10, alpha=1, obj_alpha=1e-3
    search = initialize(start, goal, test, steps)
    search.gradient_descent_path(eta = 1, lam=2, mu=5, obj_alpha=0.01, alpha_step=1e-2, max_iter=max_iter)
    return search

def GD_with_SB(start, goal, test=1, steps= 100, max_iter=200):
    # best configs -> steps=100, iterations=200, lam=2, mu=7, alpha=1, obj_alpha=0.005, beta=1e-4
    search = initialize(start, goal, test, steps)
    search.gradient_descent_sb(alpha=1, obj_alpha=0.005, beta=1e-4, sigma=0.9, lam=2, mu=5, max_iter=max_iter)
    return search

def GD_with_nesterov_momentum(start, goal, test=1, steps= 100, max_iter=1000):
    search = initialize(start, goal, test, steps)
    search.nesterov_momentum_descent(obj_alpha=0.005, beta=1e-4, alpha_step=0.001, lam=2, mu=7, max_iter=max_iter)
    return search

def GD_with_momemtum(start, goal, test=1, steps= 100, max_iter=1000):
    search = initialize(start, goal, test, steps)
    search.momentum_descent(alpha_step=0.001, obj_alpha=0.1, beta=1e-3, lam=10, mu=10, max_iter=max_iter)
    return search

def GD_adam(start, goal, test=1, steps= 100, max_iter=1000):
    search = initialize(start, goal, test, steps)
    search.adam(obj_alpha=0.005, alpha_step=1, beta=0.1,  lam=2, mu=5, max_iter=max_iter, gamma_v=0.9, gamma_s=0.999, eps=1e-5) 
    return search

def Newton_method(start, goal, test=1, steps= 100, max_iter=1000):
    search = initialize(start, goal, test, steps)
    search.newton_method(lam=2, mu=5, max_iter=max_iter)
    return search

def Nelder_Mead_Method(start, goal, test=1, steps= 100, max_iter=1000):
    # best configs -> steps: 15, iterations: 5000, eps: 1e-6, lam: 1, mu: 10
    # best configs2 -> steps: 100, iterations: 500000 (500k), eps: 1e-6, eps: 1e-6, lam: 1, mu: 10
    search = initialize(start, goal, test, steps)
    search.nelder_mead(eps=1e-6, lam=1, mu=10, max_iter=max_iter)
    return search

def main():
    start, goal = [1,1], [100,100]
    
    plot_trajectories = []

    # iterations = [0,500,1500,5000,10000]
    # steps = [50, 50, 50, 50, 50]
    # iterations = [0,100,100,100,100]
    # steps = [1, 10, 50, 100, 200]
    iterations = [0, 100, 1000, 5000, 10000]
    steps = [50, 50, 50, 50, 50]
    # iterations = [0, 1, 5, 10, 50]
    # steps = [30, 30, 30, 30, 30]

    search = initialize_straight_line(start, goal, test=1, steps= steps[0])
    
    plot_trajectories.append(search.trajectory)

    for i in range(len(iterations)):
        print(i)
        
        if iterations[i] > 0:
            search = basic_GD(start, goal, test=1, steps=steps[i], max_iter=iterations[i])
            plot_trajectories.append(search.trajectory)
    
    
    search.plot_mult(trajectories=plot_trajectories, iterations = iterations, steps=steps)

if __name__ == "__main__":
    main()



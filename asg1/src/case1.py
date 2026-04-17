import autograd.numpy as anp
import time
import numpy as np
from autograd import grad, hessian, jacobian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import line_search, minimize


class search_space():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.trajectory = []
        self.convergence = []
        self.alphas = []
        self.effective_lr = []

    def add_obstacle(self, object):
        self.obstacles.append(object)
        
    def straigt_path(self, start, goal, steps):
        direc = goal - start
        x = [start + t * direc + np.random.normal(0, 0.00001, size=start.shape) for t in anp.linspace(0, 1, steps)]
        x[0], x[-1] = start, goal  # keep endpoints fixed
        self.trajectory = list(anp.array(x).flatten().reshape(-1, 2))

    def gradient_descent_path(self, eta=1, obj_alpha=0.05, lam=2, mu=5, alpha_step=0.01, max_iter=1000, sec = None):
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
            print(i)
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return x
         
    def strong_bracketing(self, x, f, nabla, d, y0, g0_vec, alpha=1, beta=1e-4, sigma=1):
        g0 = g0_vec @ d
        y_prev, alpha_prev = None, 0

        # bracket phase
        while True:
            y = f(x + alpha*d)
            if y > y0 + beta*alpha*g0 or (y_prev is not None and y >= y_prev):
                alpha_lo, alpha_hi = alpha_prev, alpha
                break
            dir_gradient = nabla(x + alpha*d) @ d
            if abs(dir_gradient) <= sigma * abs(g0):
                return alpha
            elif dir_gradient >= 0:
                alpha_lo, alpha_hi = alpha, alpha_prev
                break
            else:
                pass
            y_prev, alpha_prev, alpha = y, alpha, 1.5 * alpha

        # zoom phase
        ylo = f(x + alpha_lo*d)
        while abs(alpha_hi - alpha_lo) > 1e-6:
            alpha = (alpha_lo + alpha_hi)/2
            y = f(x + alpha*d)
            if y > y0 + beta*alpha*g0 or y >= ylo:
                alpha_hi = alpha
            else:
                g = nabla(x + alpha*d) @ d
                if abs(g) <= sigma*abs(g0):
                    return alpha
                elif g*(alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                else:
                    pass
                alpha_lo = alpha
        return alpha_lo
    
    # --- Custom strong bracketing line search ---
    # def gradient_descent_sb(self, alpha=1, obj_alpha=1,
    #                             beta=1e-4, sigma=0.5, eta=1, lam=1, mu=1, max_iter=1000, sec = 60):
    #     t0 = time.perf_counter()
    #
    #     def obj(v):
    #         return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)
    #     f = obj
    #     nabla = grad(f)
    #     self.convergence = []
    #     self.alphas = []
    #     x = anp.array(self.trajectory).flatten()    # <- initial straigth path
    #     i = 0
    #     stop = False
    #     while not stop:
    #         if sec is not None:
    #             stop = (time.perf_counter()-t0 >= sec)
    #         else:
    #             stop = (i >= max_iter)
    #         g = nabla(x)
    #
    #         val = f(x)
    #         g = nabla(x)
    #
    #
    #         self.convergence.append(val)
    #         g = anp.clip(g, -1e5, 1e5) # <- ensure that g in {1e-5, 1e5}
    #         direc = -g
    #         alpha_step = self.strong_bracketing(x, f, nabla, d=direc, y0=val, g0_vec=g, alpha=alpha, beta=beta, sigma=sigma)
    #         self.alphas.append(alpha_step)
    #         x_new = x + alpha_step * direc
    #
    #         x_new[:2] = self.start
    #         x_new[-2:] = self.goal
    #
    #         x = x_new
    #         print(i)
    #         i += 1
    #
    #     self.trajectory = list(x.reshape(-1, 2))
    #     return x

    # --- Scipy line search (Wolfe conditions) ---
    def gradient_descent_sb(self, alpha=1, obj_alpha=1,
                                beta=1e-4, sigma=0.5, eta=1, lam=1, mu=1, max_iter=1000, sec = None):
        t0 = time.perf_counter()

        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)
        nabla = grad(obj)
        def obj_float(v):
            return float(obj(v))
        self.convergence = []
        self.alphas = []
        x = anp.array(self.trajectory).flatten()
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            val = obj_float(x)
            g = nabla(x)
            self.convergence.append(val)
            g = anp.clip(g, -1e5, 1e5)
            direc = -g
            # scipy line_search: c1=Armijo (beta), c2=Wolfe curvature (sigma)
            result = line_search(obj_float, nabla, x, direc, gfk=g, old_fval=val, c1=beta, c2=sigma)
            alpha_step = result[0]
            if alpha_step is None:
                alpha_step = 1e-4  # fallback if scipy line search fails to converge
            self.alphas.append(alpha_step)
            x_new = x + alpha_step * direc

            x_new[:2] = self.start
            x_new[-2:] = self.goal

            x = x_new
            print(i)
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        return x

    def newton_method(self, lam, mu, obj_alpha=0.1, alpha=1, max_iter = 1000, beta=1e-4, sigma=1, eta=10, sec = None): # <- 2nd order
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

    def momentum_descent(self, lam, mu, obj_alpha=0.01, beta=0.01, max_iter = 100, alpha_step=1, sec = None): # <- 1st order
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
            print(i)
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]


    def nesterov_momentum_descent(self, lam, mu, obj_alpha=0.1, beta=0.9, max_iter = 100, alpha_step=0.001, sec = None): # <- 1st order
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
            print(i)
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

    def nest_mom_line_descent(self, lam, mu, eta=1, alpha=1, obj_alpha=0.1, beta=0.9, beta_armijo=1e-4, sigma=1, max_iter = 100, alpha_step=0.001, sec = 60):
        x = anp.array(self.trajectory).flatten()
        t0 = time.perf_counter()

        def obj(v):
            return self.obj_func(v, eta, lam, mu, obj_alpha, order=0)

        f = lambda v: obj(v)
        nabla = grad(obj)

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
            x_lookahead = x + beta * momentum
            val, g_lookahead = self.obj_func(x_lookahead, eta, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g_lookahead = anp.clip(g_lookahead, -1e3, 1e3)
            d = -g_lookahead
            alpha_step = self.strong_bracketing(x_lookahead, f, nabla, d=d, y0=val, g0_vec=g_lookahead, alpha=alpha, beta=beta_armijo, sigma=sigma)
            momentum = beta * momentum + alpha_step * d
            x_new = x + momentum
            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            print(i)
            i += 1
        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]

    def adam(self, lam, mu, obj_alpha=0.1, beta=0.01, max_iter = 100, alpha_step=0.01, gamma_v=0.9, gamma_s=0.999, eps=1e-8, sec = 60): # <- 1st order
        t0 = time.perf_counter()
        x = anp.array(self.trajectory).flatten()
        momentum = np.zeros(len(x))
        squared_gradients = np.zeros(len(x))
        self.convergence = []
        self.effective_lr = []        # <- for trakcing adam adaptive lr
        best_solution = [[np.inf],[]]
        i = 0
        stop = False
        while not stop:
            if sec is not None:
                stop = (time.perf_counter()-t0 >= sec)
            else:
                stop = (i >= max_iter)
            print(i)
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            self.convergence.append(val)
            g = anp.clip(g, -1e3, 1e3) # <- ensure that g in {1e-3, 1e3}
            t = i+1
            momentum = gamma_v * momentum + (1 - gamma_v) * g
            squared_gradients = gamma_s * squared_gradients + (1 - gamma_s) * np.multiply(g,g)
            corrected_decaying_mom = momentum / (1 - gamma_v**t)
            corrected_squared_gradients = squared_gradients / (1 - gamma_s**t)
            
            effective = alpha_step / (np.sqrt(corrected_squared_gradients) + eps)
            self.effective_lr.append(effective.copy())

            x_new = x - effective * corrected_decaying_mom

            x_new[:2] = self.start
            x_new[-2:] = self.goal
            x = x_new
            if val < best_solution[0][0]:
                best_solution = [[val],x]
            i += 1

        self.trajectory = list(x.reshape(-1, 2))
        return best_solution[1]

    def scipy_lbfgsb(self, lam, mu, obj_alpha=0.1, max_iter=100, sec=60): # <- 1st order (scipy baseline)
        t0 = time.perf_counter()
        x0 = anp.array(self.trajectory).flatten()
        self.convergence = []

        def fun_and_grad(x):
            x = x.copy()
            x[:2] = self.start
            x[-2:] = self.goal
            val, g = self.obj_func(x, 1, lam, mu, obj_alpha, order=1)
            g = anp.clip(g, -1e3, 1e3)
            g[:2] = 0.0
            g[-2:] = 0.0
            self.convergence.append(float(val))
            return float(val), np.asarray(g, dtype=float)
        
        res = minimize(
            fun_and_grad, x0, jac=True, method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False},
        )
        x_final = res.x
        x_final[:2] = self.start
        x_final[-2:] = self.goal
        self.trajectory = list(x_final.reshape(-1, 2))
        return x_final

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
                stop = (i >= max_iter)
            i += 1
            print(i)
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
            i += 1
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
            av = self.avoidance2(x, alpha)
            # av = self.avoidance1(x)
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
        ncols = 2 + bool(self.alphas) + bool(self.effective_lr)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                                 constrained_layout=True)
        ax_traj, ax_conv = axes[0], axes[1]
        next_ax = 2

        # --- Trajectory plot ---
        ax_traj.set_aspect("equal")
        ax_traj.set_xlim(-5, 110)
        ax_traj.set_ylim(-5, 110)
        for i, obs in enumerate(self.obstacles):
            circle = patches.Circle(obs.center_point, obs.radius, color="red", alpha=0.5)
            ax_traj.add_patch(circle)
            ax_traj.text(obs.center_point[0], obs.center_point[1], f"O{i+1}",
                    ha="center", va="center", fontsize=9, fontweight="bold")

        if self.trajectory:
            tx = [p[0] for p in self.trajectory]
            ty = [p[1] for p in self.trajectory]
            ax_traj.plot(tx, ty, "k.-", linewidth=1.5, markersize=6, label="trajectory")

        ax_traj.plot(*self.start, "go", markersize=8, label="start")
        ax_traj.plot(*self.goal, "bs", markersize=8, label="goal")
        ax_traj.legend(loc="upper left")
        ax_traj.set_title("Trajectory")

        # --- Convergence plot ---
        if self.convergence:
            ax_conv.plot(range(len(self.convergence)), self.convergence, "b-", linewidth=1.5)
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Objective Value")
            ax_conv.set_title("Convergence")
            ax_conv.grid(True, alpha=0.3)

        # --- Alpha plot ---
        if self.alphas:
            ax_alpha = axes[next_ax]
            next_ax += 1
            ax_alpha.plot(range(len(self.alphas)), self.alphas, "r-", linewidth=1.5)
            ax_alpha.set_xlabel("Iteration")
            ax_alpha.set_ylabel("Alpha")
            ax_alpha.set_title("Step Size (Alpha)")
            ax_alpha.grid(True, alpha=0.3)

        # --- Adam effective learning rate ---
        if self.effective_lr:
            ax_lr = axes[next_ax]
            next_ax += 1
            lr = np.array(self.effective_lr)
            ax_lr.plot(lr.min(axis=1), "c-", linewidth=1, label="min")
            ax_lr.plot(np.median(lr, axis=1), "m-", linewidth=1.5, label="median")
            ax_lr.plot(lr.max(axis=1), "y-", linewidth=1, label="max")
            ax_lr.set_xlabel("Iteration")
            ax_lr.set_ylabel("Effective LR")
            ax_lr.set_yscale("log")
            ax_lr.set_title("Adam Adaptive LR (per-param)")
            ax_lr.legend()
            ax_lr.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def plot_mult(self, iterations, steps, save_path=None, trajectories=None, seconds = None, title=None):
        ncols = 6
        side = 2.0
        fig, (ax1, ax2, ax3, ax4, ax5, ex6) = plt.subplots(
            1, ncols, sharey=True, figsize=(side * ncols, side)
        )
        if title is not None:
            fig.suptitle(title, fontsize=14)
        fig_list = [ax1, ax2, ax3, ax4, ax5, ex6]


        for ax in fig_list:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
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
                if i == 0:
                    ax.set_title(f'initial')
                else:
                    ax.set_title(f'{steps[i-1]} steps - {iterations[i-1]} iterations')
                ax.plot(*self.start, "go", markersize=8, label="start")
                ax.plot(*self.goal, "bs", markersize=8, label="goal")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def plot_compare_time(self, trajectories=None, title: str="no title", titles: list[str]=["no titles"]):
        ncols = 7
        side = 2.5
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
            1, ncols, sharey=True, figsize=(side * ncols, side)
        )
        fig.suptitle(title, fontsize=14)

        fig_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]


        for ax in fig_list:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
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
                ax.set_title(f"{titles[i]}", pad=4)
                ax.plot(*self.start, "go", markersize=8, label="start")
                ax.plot(*self.goal, "bs", markersize=8, label="goal")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()
        plt.show()


class circular_object():
    def __init__(self, center: anp.array, radius: float):
        self.center_point = anp.asarray(center, dtype=float)
        self.radius = float(radius)

    def __repr__(self):
        return f"circular_object(c={self.center_point}, d={self.radius})"

## === OPTIMIZATION ALGORITHMS & CONFIGURATION === ##


def initialize(start, goal, test=1, steps = 100):
    x0 = anp.asarray(start, dtype=float)
    xn = anp.asarray(goal, dtype=float)
    search = search_space(x0, xn)
    search.straigt_path(search.start, search.goal, steps)
    match test:
        case 1:
            # basic test
            #search.add_obstacle(circular_object(anp.array([20,35]), 18))
            #search.add_obstacle(circular_object(anp.array([55,42]), 12))
            #search.add_obstacle(circular_object(anp.array([70,80]), 22))

            #search.add_obstacle(circular_object(anp.array([10,10]), 5))
            #search.add_obstacle(circular_object(anp.array([35,45]), 25))
            #search.add_obstacle(circular_object(anp.array([75,45]), 10))
            #search.add_obstacle(circular_object(anp.array([70,80]), 15))

            #search.add_obstacle(circular_object(anp.array([30,40]), 20))
            #search.add_obstacle(circular_object(anp.array([70,60]), 20))

            # narrow corridor with flanking obstacles
            search.add_obstacle(circular_object(anp.array([25,10]), 12))
            search.add_obstacle(circular_object(anp.array([10,30]), 10))
            search.add_obstacle(circular_object(anp.array([45,55]), 6))
            search.add_obstacle(circular_object(anp.array([40,30]), 8))
            search.add_obstacle(circular_object(anp.array([60,40]), 11))
            search.add_obstacle(circular_object(anp.array([72,78]), 6))
            search.add_obstacle(circular_object(anp.array([85,65]), 10))
            search.add_obstacle(circular_object(anp.array([55,75]), 7))
            search.add_obstacle(circular_object(anp.array([90,92]), 9))
            search.add_obstacle(circular_object(anp.array([30,50]), 6))
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

def basic_GD(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting basic GD at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.gradient_descent_path(eta = 1, lam=lam, mu=mu, obj_alpha=obj_alpha, alpha_step=1e-2, max_iter=max_iter, sec = sec)
    return search

def GD_with_SB(start, goal, test=1, steps= 100, max_iter=100, sec = None):
    print(f"Starting GDSB at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.gradient_descent_sb(alpha=0.1, obj_alpha=obj_alpha, beta=1e-4, sigma=0.5, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def GD_with_nesterov_momentum(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting GD_with_nesterov_momentum at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.nesterov_momentum_descent(obj_alpha=obj_alpha, beta=0.9, alpha_step=1e-2, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def GD_with_momemtum(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting GD_with_momemtum at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.momentum_descent(alpha_step=1e-2, obj_alpha=obj_alpha, beta=0.9, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def GD_adam(start, goal, test=1, steps= 100, max_iter=100, sec = None):
    print(f"Starting GD_adam at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.adam(obj_alpha=obj_alpha, alpha_step=1, beta=0.1,  lam=lam, mu=mu, max_iter=max_iter, gamma_v=0.9, gamma_s=0.999, eps=1e-5, sec = sec) 
    return search

def GD_mix(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting GD_mix at {sec}")
    search = initialize(start, goal, test, steps)
    search.nest_mom_line_descent(lam, mu, alpha=5e-3, obj_alpha=obj_alpha, beta=0.9, beta_armijo=1e-4, sigma=1, max_iter = max_iter, alpha_step=0.001, sec=sec)
    return search

def SciPy_min(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting L-BFGS-B at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.scipy_lbfgsb(lam, mu, obj_alpha, max_iter=max_iter, sec=sec)
    return search

def Newton_method(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
    print(f"Starting Newton_method at {sec} seconds")
    search = initialize(start, goal, test, steps)
    search.newton_method(alpha=obj_alpha, lam=lam, mu=mu, max_iter=max_iter, sec = sec)
    return search

def Nelder_Mead_Method(start, goal, test=1, steps= 100, max_iter=1000, sec = None):
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
    # iterations = [100, 1000, 5000]
    # steps = [50, 50, 50]
    # iterations = [0, 1, 5, 10, 50]
    # steps = [30, 30, 30, 30, 30]
    

    title = "Comparing Gradient Descent with Nesterov Momentum"
    
    sec = 100

    #search = basic_GD(start, goal, steps=100, max_iter=1000)
    #search = GD_with_SB(start, goal, steps=100, max_iter=1000)
    #search = GD_with_nesterov_momentum(start, goal, steps=100, max_iter=1000)
    #search = GD_with_momemtum(start, goal, steps=100, max_iter=1000)
    #search = GD_mix(start, goal, steps=100, max_iter=1000)
    #search = Newton_method(start, goal, steps=10, max_iter=100)
    #search = SciPy_min(start=start, goal=goal, steps=100, max_iter=1000)
    search = GD_adam(start = start, goal = goal, steps=100, max_iter=1000)
    #search = Nelder_Mead_Method(start = start, goal = goal, steps=50, max_iter=50000)
    search.plot_single_dingle()
    


    #seconds = [10, 30, 60]

    test_all_time = True

    title = "Gradient descent with line search compared on number of steps"
    #titles = ["GD_basic", "GD_line_search", "GD_Nest_Mom", "GD_Momentum", "GD_NestMom_LineSearch", "GD_Adam", "Nelder_Mead"]

    iterations = [100, 100, 100, 100, 100]
    steps = [10,25,50,100,200]

    if not test_all_time:
        search = initialize_straight_line(start, goal, test=1, steps=2)
        plot_trajectories.append(search.trajectory)

    i = 0
    while i < len(steps) and (not test_all_time):
        search = basic_GD(start, goal, test=1, steps=steps[i], max_iter=iterations[i])
        #search = GD_with_SB(start, goal, test=1, steps= steps[i], max_iter=iterations[i])
        # search = GD_with_nesterov_momentum(start, goal, test=1, steps= steps[i], max_iter=iterations[i], sec = seconds)
        # search = GD_with_momemtum(start, goal, test=1, steps= steps[i], max_iter=iterations[i], sec = seconds)
        # search = Newton_method(start, goal, test=1, steps= steps[i], max_iter=iterations[i])
        # search = Nelder_Mead_Method(start = start, goal = goal, steps = steps[i], max_iter=iterations[i], sec = seconds)
        # search = GD_adam(start = start, goal = goal, steps = steps[i], max_iter=iterations[i], sec = seconds)
        plot_trajectories.append(search.trajectory)
        i += 1

    if not test_all_time:
        search.plot_mult(trajectories=plot_trajectories, iterations = iterations, steps=steps, title=title)
        plot_trajectories = []

    seconds = 1
    steps = 50
    iter = None
    title = "1 second"
    titles = ["GD_basic\n 100 steps", "GD_line_search\n 100 steps", "GD_Nest_Mom\n 100 steps", "GD_Momentum\n 100 steps", "GD_NestMom_LineSearch\n 100 steps", "GD_Adam\n 100 steps", "Nelder_Mead\n 100 steps"]

    if test_all_time:
        search = basic_GD(start, goal, test=1, steps=100, max_iter=iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_SB(start, goal, test=1, steps=100, max_iter= iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_nesterov_momentum(start, goal, test=1, steps=100, max_iter= iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_with_momemtum(start, goal, test=1, steps= 100, max_iter= iter, sec = seconds)
        plot_trajectories.append(search.trajectory)
        #search = Newton_method(start, goal, test=1, steps=steps, max_iter= iter, sec = seconds)
        search = GD_mix(start, goal, steps=100, max_iter= iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search = GD_adam(start = start, goal = goal, steps = 100, max_iter= iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search = Nelder_Mead_Method(start=start, goal=goal, steps=100, max_iter= iter, sec=seconds)
        plot_trajectories.append(search.trajectory)
        search.plot_compare_time(trajectories=plot_trajectories, title=title, titles=titles)
    

if __name__ == "__main__":
    # print(np.transpose(np.diag(np.full(np.size(test),1))))
    # print(np.transpose(np.diag(np.full(np.size(test),test))))
    # print(np.transpose(np.absolute(test)) @ np.diag(np.full(np.size(test),1)))
    main()



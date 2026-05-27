import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from autograd import grad, hessian, jacobian
import autograd.numpy as anp
import math


def main():
    fig, ax = plt.subplots()
    ax.plot([1, 1], [3, -3], label='$g_1$', linestyle='--', color='b')
    ax.plot([-3, 3], [1, 1], label='$g_2$', linestyle='--', color='b')
    ax.plot([-1, -1], [3, -3], label='$g_3$', linestyle='--', color='b')
    ax.plot([-3, 3], [-1, -1], label='$g_2$', linestyle='--', color='b')
    ax.plot([0, 1.5], [1.5, 0], label='$g_5$', linestyle='--', color='b')
    # ax.plot([0, 1], [1, 0], label='$g_5$', linestyle='--', color='r')
    poly1 = Polygon([(-1, 1), (0.5, 1), (1, 0.5), (1, -1), (-1, -1)],
                    facecolor='blue',
                    linewidth=0,
                    alpha=0.1
                    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    x = anp.array([0.0, 0.0])
    x_newton = newton_method(x)
    x_newton_stop = newton_method_stop(x, 0.001, 100)

    _, _, _, hes = obj_func(x, 2)
    print(f"Hessian for original x: {hes}")
    slack_newton, obj_val, _, _ = obj_func(x_newton, 2)
    slack_newton_stop, obj_val_stop, _, _ = obj_func(x_newton_stop, 2)
    print(f"New x after newton method: {x_newton}")
    print(f"Obj value for x with newton method: {obj_val}")
    print(f"Slack for x with newton method: {slack_newton}")

    print("")
    print("")
    print("")
    print("")
    

    print(f"New x after newton method without stop: {x_newton}")
    print(f"Obj value for x with newton method without stop: {obj_val}")
    print(f"Slack for x after newton method without stop: {slack_newton}")

    print(f"New x after newton method with stop: {x_newton_stop}")
    print(f"Obj value for x with newton method with stop: {obj_val_stop}")
    print(f"Slack for x after newton method with stop: {slack_newton_stop}")

    print("")
    print("")
    print("")
    print(f"Distance from x after first itertaion, to after full newton method: {math.dist(x_newton, x_newton_stop)}")
    print(f"Differece in objective value after first x, to after full newton method: {obj_val - obj_val_stop}")


    print("")
    print("")
    print("")
    x_gam10 = newton_method_stop(x, 0.001, 100, 10)
    slack_gam10, obj_val_gam10, _, _ = obj_func(x_gam10, 2)
    print(f"x: after newton method with stop: {x_newton_stop}")
    print(f"Obj value for x with newton method with stop: {obj_val_stop}")
    print(f"Slack for x after newton method with stop: {slack_newton_stop}")

    print(f"Distance from x after first itertaion, to after full newton method: {math.dist(x_newton_stop, x_gam10)}")
    print(f"Differece in objective value after first x, to after full newton method: {obj_val_gam10 - obj_val_stop}")


    ax.plot(x[0], x[1], 'ro')
    #ax.plot(x_newton[0], x_newton[1], 'go')
    ax.plot(x_newton_stop[0], x_newton_stop[1], color='orange', marker='o',)
    ax.plot(x_gam10[0], x_gam10[1], color='blue', marker='o',)
    """
    ax.plot(x_01[0], x_01[1], 'go')
    ax.plot(x_1[0], x_1[1], 'go')
    ax.plot(x_10[0], x_10[1], 'go')
    ax.plot(x_100[0], x_100[1], 'go')
    ax.plot(x_norm[0], x_norm[1], 'ro')
        """

    ax.add_patch(poly1)



    plt.axhline(0, color='black', linewidth=1) # x-axis
    plt.axvline(0, color='black', linewidth=1) # y-axis
    print("")
    print("")
    print("")
    print("")

    plt.show()

def modify_point(x, count=1, gam=1, norm=False):
    i = 0
    while i < count:
        if norm==True:
            x = newton_method_norm(x, gam)
        else:
            x = newton_method_mod(x, gam)

        i += 1

    return x


def obj_func(x, order=1):
    def f(x):
        return -(
            anp.log( -(x[0] * 1 + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 + x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * (-1) + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 - x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * 1 + x[1] * 1 - 1.5) )  
            )

    def slack(x):
        return anp.array(
            [(x[0] * 1 + x[1] * 0 - 1),
            (x[0] * 0 + x[1] * 1 - 1),
            (x[0] * (-1) + x[1] * 0 - 1),
            (x[0] * 0 - x[1] * 1 - 1), 
            (x[0] * 1 + x[1] * 1 - 1.5)]
        )

    match order:
        case 0:
            return slack(x), f(x)
        case 1:
            return slack(x), f(x), grad(f)(x)
        case 2:
            return slack(x), f(x), grad(f)(x), jacobian(grad(f))(x)
            
def obj_func_modified(x, order=1, gam=1):
    def f(x):
        return -(
            anp.log( -(x[0] * 1 + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 + x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * (-1) + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 - x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * (1*gam) + x[1] * (1*gam) - (1.5*gam)) )  
            )
    def slack(x):
        return anp.array(
            [-(x[0] * 1 + x[1] * 0 - 1),
            -(x[0] * 0 + x[1] * 1 - 1),
            -(x[0] * (-1) + x[1] * 0 - 1),
            -(x[0] * 0 - x[1] * 1 - 1), 
            -(x[0] * (1*gam) + x[1] * (1*gam) - (1*gam))]
        )

    match order:
        case 0:
            return slack(x), f(x)
        case 1:
            return slack(x), f(x), grad(f)(x)
        case 2:
            return slack(x), f(x), grad(f)(x), jacobian(grad(f))(x)



def obj_func_normalized(x, order=1, gam=1):
    def f(x):
        return -(
            anp.log( -(x[0] * 1 + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 + x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * (-1) + x[1] * 0 - 1) ) + 
            anp.log( -(x[0] * 0 - x[1] * 1 - 1) ) + 
            anp.log( -(x[0] * (1*gam) + x[1] * (1*gam) - (1*gam)) )  
            )
    def slack(x):
        return anp.array(
            [-(x[0] * 1 + x[1] * 0 - 1),
            -(x[0] * 0 + x[1] * 1 - 1),
            -(x[0] * (-1) + x[1] * 0 - 1),
            -(x[0] * 0 - x[1] * 1 - 1), 
            -(x[0] * (gam*1) + x[1] * (1*gam) - 1)]
        )

    match order:
        case 0:
            return slack(x), f(x)
        case 1:
            return slack(x), f(x), grad(f)(x)
        case 2:
            return slack(x), f(x), grad(f)(x), jacobian(grad(f))(x)


def newton_method(x):
    _, _, grad, hes = obj_func(x, order=2)
    return x - anp.linalg.inv(hes) @ grad

def newton_method_stop(x, crit, max_iter, gam=1):
    i = 0
    while i < max_iter :
        _, _, grad, hes = obj_func_modified(x, order=2, gam=gam)
        x_new = x - anp.linalg.inv(hes) @ grad
        if math.dist(x_new, x) <= crit:
            return x_new
        else:
            x = x_new
        i += 1
    return x



def newton_method_mod(x, gam=1):
    _, _, grad, hes = obj_func_modified(x, order=2, gam=gam)
    return x - anp.linalg.inv(hes) @ grad

def newton_method_norm(x, gam=1):
    _, _, grad, hes = obj_func_normalized(x, order=2, gam=gam)
    return x - anp.linalg.inv(hes) @ grad


if __name__ == "__main__":
    main()

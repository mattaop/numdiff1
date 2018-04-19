import numpy as np
import matplotlib.pyplot as plt

import constants as c
import simple_lax_vectorized as sl_v
import upwind_vectorized as up_v
import upwind_vectorized_v2 as up_v2
import mac_cormack as mc


def spatial_convergence_vec(solver, T, X, delta_t, delta_x):
    startnumber = 3
    convergence_list = np.zeros((2, c.M-startnumber-1))
    u_exact = solver(T, X, delta_t, delta_x)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(c.M -startnumber-1)

    x_list = np.linspace(-c.L / 2, c.L / 2, len(exact_list))
    plt.plot(x_list,exact_list[:,0])
    plt.show()


    for j in range(startnumber,c.M-1):

        x_points = 2 ** (j + 1)
        new_exact_list = np.zeros((x_points,2))

        ratio = (len(exact_list) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[h] = exact_list[int(h * ratio)]

        delta_x = c.L / (x_points - 1)
        print(delta_x)
        step_length_list[j - startnumber] = delta_x
        u = solver(c.TIME_POINTS, x_points, delta_t, delta_x)
        j_list=u[-1]

        convergence_list[0][j-startnumber] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,0] - j_list[:,0], 2)
        convergence_list[1][j-startnumber] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,1] - j_list[:,1], 2)


        x_list = np.linspace(-c.L/2, c.L/2, len(new_exact_list[:,0]))
        x_list2 = np.linspace(-c.L/2, c.L/2, len(j_list[:,0]))
        plt.plot(x_list,new_exact_list[:,0],label="exact")
        plt.plot(x_list2,j_list[:,0],label="not exact")
        plt.legend()
        plt.show()

        print("Points: ", x_points)

    return convergence_list, step_length_list

def plot_convergence(method):
    conv_list, step_length_list = spatial_convergence_vec(method, c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    print(conv_list[0])
    print(conv_list[1])

    plt.loglog(step_length_list,conv_list[0],label='rho')
    plt.show()
    plt.figure()
    plt.loglog(step_length_list,conv_list[1],label='v')
    plt.xlabel("Steplength ($\delta x$)")
    plt.ylabel("Error")
    plt.grid()

    plt.legend()
    plt.show()

plot_convergence(up_v2.solve_upwind)
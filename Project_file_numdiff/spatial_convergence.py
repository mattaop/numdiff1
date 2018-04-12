import numpy as np
import matplotlib.pyplot as plt

import constants as c
import simple_lax_vectorized as sl_v
import upwind_vectorized as up_v


def spatial_convergence_vec(solver, T, X, delta_t, delta_x):
    convergence_list = np.zeros((2, c.M + 1))
    u_exact = solver(T, X, delta_t, delta_x)
    x_list = np.linspace(-c.L, c.L, len(u_exact[0]))
    exact_list = np.transpose(u_exact[-1])
    step_length_list = np.zeros(c.M + 1)

    for j in range(c.M):
        print("inside for loop")
        x_points = 2 ** (j + 1)
        new_exact_list = np.zeros((2, x_points))
        ratio = (len(exact_list[0]) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[:, h] = exact_list[:, int(h * ratio)]

        delta_x = c.L / (x_points - 1)
        step_length_list[j - 1] = delta_x
        u = solver(c.TIME_POINTS, x_points, delta_t, delta_x)
        j_list = np.array([u[:, :, 0][-1], [u[:, :, 1][-1]]])

        convergence_list[0][j - 1] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[0] - j_list[0], 2)
        convergence_list[1][j - 1] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[1] - j_list[1], 2)

    return convergence_list, step_length_list

def plot_convergence():
    conv_list,step_length_list=spatial_convergence_vec(sl_v.solve_simple_lax, c.TIME_POINTS, c.SPACE_POINTS,c.delta_t,c.delta_x)
    print(conv_list[0])
    print(conv_list[1])
    plt.loglog(step_length_list,conv_list[0],label='rho')
    plt.loglog(step_length_list,conv_list[1],label='v')
    plt.legend()
    plt.show()

plot_convergence()
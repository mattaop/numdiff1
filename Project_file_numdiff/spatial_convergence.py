import numpy as np
import matplotlib.pyplot as plt
import constants as c


def spatial_convergence_vec(solver, T, X, MAX_TIME, M, startnumber):
    convergence_list = np.zeros((2, M-startnumber-1))
    u_exact = solver(T, X, MAX_TIME)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(M -startnumber-1)

    for j in range(startnumber,M-1):
        x_points = 2**(j+1)
        new_exact_list = np.zeros((x_points,2))
        ratio = (len(exact_list) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[h] = exact_list[int(h * ratio)]

        delta_x = c.L / (x_points - 1)
        step_length_list[j - startnumber] = delta_x
        u = solver(T, x_points, MAX_TIME)
        j_list=u[-1]
        convergence_list[0][j-startnumber] = np.sqrt(delta_x) * np.linalg.norm(new_exact_list[:,0] - j_list[:,0], 2)
        convergence_list[1][j-startnumber] = np.sqrt(delta_x) * np.linalg.norm(new_exact_list[:,1] - j_list[:,1], 2)
        print("Points: ", x_points)

    return step_length_list,convergence_list


def plot_spatial_convergence(solver1, solver2, solver3,solver4):
    M = 10
    m = 6
    MAX_TIME = 1
    time_points = 100
    space_points = 2 ** M

    delta_x_list1, conv_1 = spatial_convergence_vec(solver1, time_points, space_points, MAX_TIME, M, m)

    M = 8
    m = 3
    MAX_TIME = 40
    time_points = 2000
    space_points = 2**M

    delta_x_list2, conv_2 = spatial_convergence_vec(solver2, time_points, space_points, MAX_TIME, M, m)

    M = 8
    m = 3
    MAX_TIME = 160
    time_points = 1000
    space_points = 2 ** M

    delta_x_list3, conv_3 = spatial_convergence_vec(solver3, time_points, space_points, MAX_TIME, M, m)

    M = 10
    m = 3
    MAX_TIME = 1
    time_points = 800
    space_points = 2 ** M

    delta_x_list4, conv_4 = spatial_convergence_vec(solver4, time_points, space_points, MAX_TIME, M, m)

    plt.figure()
    plt.loglog(delta_x_list1, conv_1[0], label=r"Lax-Friedrichs")
    plt.loglog(delta_x_list2, conv_2[0], label=r"Upwind")
    plt.loglog(delta_x_list3, conv_3[0], label= r"Lax-Wendroff")
    plt.loglog(delta_x_list4, conv_4[0], label= r"MacCormack")
    plt.title("Convergence plot of "+ r'$\rho$' +" in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    #plt.savefig("conv_rho_space.pdf")
    plt.show()

    plt.figure()
    plt.loglog(delta_x_list1, conv_1[1], label=r"Lax-Friedrichs")
    plt.loglog(delta_x_list2, conv_2[1], label=r"Upwind")
    plt.loglog(delta_x_list3, conv_3[1], label=r"Lax-Wendroff")
    plt.loglog(delta_x_list4, conv_4[1], label=r"MacCormack")
    plt.title("Convergence plot of " + r'$v$' + " in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    #plt.savefig("conv_v_space.pdf")
    plt.show()


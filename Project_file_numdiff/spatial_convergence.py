import numpy as np
import matplotlib.pyplot as plt
import constants as c
import upwind_vectorized_v2 as up
import simple_lax_vectorized as slv



def spatial_convergence_vec(solver, T, X, delta_t, delta_x,M):
    startnumber = 3
    convergence_list = np.zeros((2, M-startnumber-1))
    u_exact = slv.solve_simple_lax(T, X, delta_t, delta_x)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(M -startnumber-1)

    for j in range(startnumber,M-1):

        x_points = 2 ** (j + 1)
        new_exact_list = np.zeros((x_points,2))
        ratio = (len(exact_list) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[h] = exact_list[int(h * ratio)]

        delta_x = c.L / (x_points - 1)
        step_length_list[j - startnumber] = delta_x
        u = solver(c.TIME_POINTS, x_points, delta_t, delta_x)
        j_list=u[-1]

        convergence_list[0][j-startnumber] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,0] - j_list[:,0], 2)
        convergence_list[1][j-startnumber] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,1] - j_list[:,1], 2)
        x = np.linspace(-2500,2500, x_points)
        plt.figure()
        plt.plot(x, j_list[:,0], label='Test')
        plt.plot(x, new_exact_list[:, 0], label='Exact')
        plt.legend()
        plt.show()
        print("Points: ", x_points)

    return step_length_list,convergence_list

'''def plot_convergence(method):
    conv_list, step_length_list = spatial_convergence_vec(method, c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)


    plt.loglog(step_length_list,conv_list[0],label='rho')
    plt.show()
    plt.figure()
    plt.loglog(step_length_list,conv_list[1],label='v')
    plt.xlabel("Steplength ($\Delta x$)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.show()'''

def plot_spatial_convergence_lax(solver1, solver2):
    M=14
    time_points=100
    space_points=2**M
    delta_t=0.01
    delta_x=c.L/(space_points-1)
    #delta_x_list1, conv_1 = spatial_convergence_vec(solver1, time_points,space_points,delta_t, delta_x,M)
    delta_x_list2, conv_2 = spatial_convergence_vec(solver2, time_points,space_points,delta_t, delta_x,M)

    plt.figure()
    #plt.loglog(delta_x_list1, conv_1[0], label= r"Lax-Fredrich")
    plt.loglog(delta_x_list2, conv_2[0], label= r"Lax-Fredrich v2")
    plt.title("Convergence plot of "+ r'$\rho$' +" in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("conv_rho_space_lax.pdf")
    plt.show()

    plt.figure()
    plt.loglog(delta_x_list1, conv_1[1], label=r"Lax-Fredrich")
    plt.loglog(delta_x_list2, conv_2[1], label=r"Lax-Fredrich v2")
    plt.title("Convergence plot of " + r'$v$' + " in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.savefig("conv_v_space_lax.pdf")
    plt.legend()
    plt.show()

def plot_spatial_convergence(solver3,solver4):
    M=10
    time_points = 1000
    space_points = 2**M
    delta_t = 0.1
    delta_x = c.L / (space_points - 1)

    #delta_x_list3, conv_3 = spatial_convergence_vec(solver3, time_points, space_points, delta_t, delta_x, M)
    delta_x_list4, conv_4 = spatial_convergence_vec(solver4, time_points, space_points, delta_t, delta_x, M)

    plt.figure()
    #plt.loglog(delta_x_list3, conv_3[0], label= r"Upwind")
    plt.loglog(delta_x_list4, conv_4[0], label= r"Lax-Wendroff")
    plt.title("Convergence plot of "+ r'$\rho$' +" in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_rho_space.pdf")
    plt.show()

    plt.figure()
    #plt.loglog(delta_x_list3, conv_3[1], label=r"Upwind")
    plt.loglog(delta_x_list4, conv_4[1], label=r"Lax-Wendroff")
    plt.title("Convergence plot of " + r'$v$' + " in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_v_space.pdf")
    plt.show()
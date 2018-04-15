from time import time
import numpy as np
import matplotlib.pyplot as plt
import simple_lax_vectorized as sl_v
import simple_lax_vectorized_v2 as sl_v2
import upwind_vectorized as up_v
import upwind_vectorized_v2 as up_v2
import constants as c
import lax_wendroff as lw
import mac_cormack as mc

def time_error(solver, space_points, delta_x):
    m = 5  #2^m points for first iteration
    n = 12  #2^n points for last iteration
    T_max = 1 * 20  # Time (minutes?) until we stop the simulation
    T_ex = 2**(n+1)  # Number of time steps in the reference (exact) solution

    delta_t_min = T_max / (T_ex - 1)  # The delta T-value used in the exact solution
    u_ex = solver(T_ex, c.SPACE_POINTS, delta_t_min, c.delta_x)
    error_list_rho = np.zeros(n-m)
    error_list_v = np.zeros(n-m)
    delta_t_list = np.zeros(n-m)
    print(delta_x)
    for i in range(m,n):
        t0 = time()
        time_points = 2**(i+1) #Number of time points in each iteration
        delta_t = T_max/(time_points-1) #delta t in each iteration
        #print(delta_t)
        u = solver(time_points, space_points, delta_t,delta_x)
        error_rho = u_ex[-1,:,0]-u[-1,:,0]
        error_v = u_ex[-1,:,1]-u[-1,:,1]
        error_list_rho[i-m] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_rho,2)
        error_list_v[i-m] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_v,2)
        delta_t_list[i-m] = delta_t
        t1 = time()
        print("Points: ", time_points, " , Time: ", t1 - t0)
        x_list = np.linspace(-c.L/2, c.L/2, len(u_ex[-1,:,0]))
        x_list2 = np.linspace(-c.L/2, c.L/2, len(u[-1,:,0]))
        #plt.plot(x_list,u_ex[-1,:,0],label="exact")
        #plt.plot(x_list2,u[-1,:,0],label="not exact")
        #plt.legend()
        #plt.show()
    return delta_t_list,error_list_rho,error_list_v
   
    
def plot_time_convergence(solver):
    delta_t_list, error_rho, error_v = time_error(solver, c.SPACE_POINTS, c.delta_x)
    plt.figure()
    plt.plot(delta_t_list, error_rho, label=r"$\rho$")
    plt.plot(delta_t_list, error_v, label= "v")
    plt.title("Convergence plot in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()

def plot_time_convergence_2(solver1, solver2, solver3, solver4):
    delta_t_list1, error_rho1, error_v1 = time_error(solver1, c.SPACE_POINTS, c.delta_x)
    delta_t_list2, error_rho2, error_v2 = time_error(solver2, c.SPACE_POINTS, c.delta_x)
    delta_t_list3, error_rho3, error_v3 = time_error(solver3, c.SPACE_POINTS, c.delta_x)
    delta_t_list4, error_rho4, error_v4 = time_error(solver4, c.SPACE_POINTS, c.delta_x)
    plt.figure()
    plt.plot(delta_t_list1, error_rho1, label= r"Lax-Fredrich")
    plt.plot(delta_t_list2, error_rho2, label= r"Lax-Fredrich v2")
    plt.plot(delta_t_list3, error_rho3, label= r"Upwind")
    plt.plot(delta_t_list4, error_rho4, label= r"Lax-Wendroff")
    plt.title("Convergence plot of $\rho$ in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()


#print("Lax-Friedrich")
#plot_time_convergence(sl_v.solve_simple_lax)
#print("Lax-Friedrich V2")
#plot_time_convergence(sl_v2.solve_simple_lax_v2)
#print("Upwind")
#plot_time_convergence(up_v.solve_upwind)
#print("Upwind V2")
#plot_time_convergence(up_v2.solve_upwind)
#print("MacCormack")
#plot_time_convergence(mc.solve_mac_cormack)
#print("Lax-Wendroff")
#plot_time_convergence(lw.solve_lax_wendroff)
#print("Lax-Wendroff V2")
#plot_time_convergence(lw.solve_lax_wendroff_v2)
plot_time_convergence_2(sl_v.solve_simple_lax, sl_v2.solve_simple_lax, up_v2.solve_upwind, lw.solve_lax_wendroff)

